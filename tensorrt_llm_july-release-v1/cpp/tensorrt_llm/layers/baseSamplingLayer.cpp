/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/samplingPenaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
void BaseSamplingLayer<T>::allocateBuffer(size_t batch_size)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    curandstate_buf_ = allocator_->reMalloc(curandstate_buf_, sizeof(curandState_t) * batch_size, false);
    random_seeds_buf_ = allocator_->reMalloc(random_seeds_buf_, sizeof(unsigned long long) * batch_size, false);
    temperature_buf_ = allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false);
    repetition_penalty_buf_ = allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false);
    min_lengths_buf_ = allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false);
    runtime_logits_buf_ = allocator_->reMalloc(runtime_logits_buf_, sizeof(T) * batch_size * vocab_size_padded_, false);
    skip_decode_buf_ = allocator_->reMalloc(skip_decode_buf_, sizeof(bool) * batch_size, false);

    // host buffers.
    skip_decode_ = new bool[batch_size];

    is_allocate_buffer_ = true;
}

template <typename T>
void BaseSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&curandstate_buf_));
        allocator_->free((void**) (&random_seeds_buf_));
        allocator_->free((void**) (&temperature_buf_));
        allocator_->free((void**) (&repetition_penalty_buf_));
        allocator_->free((void**) (&min_lengths_buf_));
        allocator_->free((void**) (&runtime_logits_buf_));
        allocator_->free((void**) (&skip_decode_buf_));
        delete[] skip_decode_;
        is_allocate_buffer_ = false;
    }
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper, IAllocator* allocator, bool is_free_buffer_after_forward,
    cudaDeviceProp* cuda_device_prop)
    : BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& sampling_layer)
    : BaseLayer(sampling_layer)
    , vocab_size_(sampling_layer.vocab_size_)
    , vocab_size_padded_(sampling_layer.vocab_size_padded_)
    , sampling_workspace_size_(sampling_layer.sampling_workspace_size_)
{
}

template <typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer()
{
}

template <typename T>
void BaseSamplingLayer<T>::setupBase(const size_t batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    allocateBuffer(batch_size);

    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batch_size] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    if (setupParams.random_seed)
    {
        if (setupParams.random_seed->size() == 1)
        {
            invokeCurandInitialize(curandstate_buf_, batch_size, setupParams.random_seed->front(), stream_);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams.random_seed->size() == batch_size, "Random seed vector size mismatch.");
            cudaAutoCpy(random_seeds_buf_, setupParams.random_seed->data(), batch_size, stream_);
            invokeCurandBatchInitialize(curandstate_buf_, batch_size, random_seeds_buf_, stream_);
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(curandstate_buf_, batch_size, 0, stream_);
    }

    // Setup penalties.
    auto fillBuffers
        = [this, &batch_size](auto const& optParam, auto const defaultValue, auto& hostBuffer, auto& deviceBuffer)
    {
        hostBuffer.resize(batch_size);
        if (!optParam)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), defaultValue);
        }
        else if (optParam->size() == 1)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), optParam->front());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(optParam->size() == batch_size, "Argument vector size mismatch.");
            std::copy(optParam->begin(), optParam->end(), std::begin(hostBuffer));
        }
        cudaAutoCpy(deviceBuffer, hostBuffer.data(), batch_size, stream_);
    };

    fillBuffers(setupParams.temperature, 1.0f, mTemperature, temperature_buf_);
    fillBuffers(setupParams.min_length, 1, mMinLengths, min_lengths_buf_);

    if ((setupParams.repetition_penalty) || (setupParams.presence_penalty))
    {
        TLLM_CHECK_WITH_INFO(!((setupParams.repetition_penalty) && (setupParams.presence_penalty)),
            "Found ambiguous parameters repetition_penalty and presence_penalty "
            "which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        repetition_penalty_type_ = (setupParams.repetition_penalty) ? RepetitionPenaltyType::Multiplicative
                                                                    : RepetitionPenaltyType::Additive;
        auto const& repetition_penalty = (repetition_penalty_type_ == RepetitionPenaltyType::Multiplicative)
            ? setupParams.repetition_penalty
            : setupParams.presence_penalty;

        // repetition_penalty is not empty, so default value 0.0f has no effect
        fillBuffers(repetition_penalty, 0.0f, mRepetitionPenalty, repetition_penalty_buf_);
    }
    else
    {
        repetition_penalty_type_ = RepetitionPenaltyType::None;
    }
}

template <typename T>
void BaseSamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams const& params)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto const batch_size = outputs.output_ids.shape[1];
    auto const local_batch_size = params.logits.shape[0];
    auto const ite = params.ite;
    auto const step = params.step;
    auto const max_input_length = params.max_input_length;

    auto* logits = params.logits.template getPtr<T>();

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    bool* skip_decode = skip_decode_ + ite * local_batch_size;
    if (ALL_OF(skip_decode, local_batch_size, bool, true))
    {
        // No sample in the current batch to do TopX sampling.
        return;
    }
    skip_any_ = std::any_of(skip_decode, skip_decode + local_batch_size, [](bool b) { return b; });
    if (skip_any_)
    {
        // A TopX Sampling layer directly changes the logit values. In case of
        // skip_any==true, meaning topk and topp layers will run simultaneously for
        // a batch in the same step. We copy the logits to an internal buffer, not
        // affecting the other sampling layers.
        TLLM_CHECK(params.logits.size() == local_batch_size * vocab_size_padded_);
        cudaD2Dcpy(runtime_logits_buf_, logits, params.logits.size());
        logits = runtime_logits_buf_;
    }

    auto* embedding_bias = params.embedding_bias ? params.embedding_bias->template getPtr<T const>() : nullptr;
    if (embedding_bias != nullptr
        || !ALL_OF(std::begin(mTemperature) + ite * local_batch_size, local_batch_size, float, 1.0f))
    {
        invokeBatchApplyTemperaturePenalty(logits, embedding_bias, temperature_buf_ + ite * local_batch_size,
            local_batch_size, vocab_size_, vocab_size_padded_, stream_);
    }
    sync_check_cuda_error();

    if (step > 1 && repetition_penalty_type_ != RepetitionPenaltyType::None)
    {
        float default_value = getDefaultPenaltyValue(repetition_penalty_type_);
        if (!ALL_OF(std::begin(mRepetitionPenalty) + ite * local_batch_size, local_batch_size, float, default_value))
        {
            auto* const input_lengths
                = params.input_lengths ? params.input_lengths->template getPtr<const int>() : nullptr;
            invokeBatchApplyRepetitionPenalty(logits, repetition_penalty_buf_ + ite * local_batch_size,
                outputs.output_ids.getPtrWithOffset<int>(ite * local_batch_size), batch_size, local_batch_size,
                vocab_size_padded_, input_lengths, max_input_length, step, repetition_penalty_type_, stream_);
            sync_check_cuda_error();
        }
    }

    const int num_generated_tokens = step - max_input_length;
    const auto min_lengths = std::begin(mMinLengths) + ite * local_batch_size;
    const bool invoke_min_length_penalty = std::any_of(
        min_lengths, min_lengths + local_batch_size, [&](int min_length) { return min_length > num_generated_tokens; });
    if (invoke_min_length_penalty)
    {
        auto* end_ids = params.end_ids.template getPtr<const int>();
        invokeMinLengthPenalty(logits, min_lengths_buf_ + ite * local_batch_size, end_ids,
            outputs.sequence_length->getPtr<const int>(), max_input_length, local_batch_size, vocab_size_padded_,
            stream_);
        sync_check_cuda_error();
    }
#undef ALL_OF

    runSampling(outputs, params);

    if (is_free_buffer_after_forward_)
    {
        freeBuffer();
    }
    sync_check_cuda_error();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
