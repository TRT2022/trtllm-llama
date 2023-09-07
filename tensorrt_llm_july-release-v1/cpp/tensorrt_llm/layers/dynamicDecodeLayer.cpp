/*
 * Copyright (c) 2022-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/kernels/banBadWords.h"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mOnlineBeamsearchDecode = std::make_unique<OnlineBeamSearchLayer<T>>(
        vocab_size_, vocab_size_padded_, stream_, cublas_wrapper_, allocator_, is_free_buffer_after_forward_);

    mTopKDecode = std::make_unique<TopKSamplingLayer<T>>(
        vocab_size_, vocab_size_padded_, stream_, cublas_wrapper_, allocator_, false);

    mTopPDecode = std::make_unique<TopPSamplingLayer<T>>(
        vocab_size_, vocab_size_padded_, stream_, cublas_wrapper_, allocator_, false, cuda_device_prop_);
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper, IAllocator* allocator, bool is_free_buffer_after_forward,
    cudaDeviceProp* cuda_device_prop)
    : BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
    , cuda_device_prop_(cuda_device_prop)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template <typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer)
    : BaseLayer(dynamic_decode_layer)
    , vocab_size_(dynamic_decode_layer.vocab_size_)
    , vocab_size_padded_(dynamic_decode_layer.vocab_size_padded_)
    , cuda_device_prop_(dynamic_decode_layer.cuda_device_prop_)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

namespace
{
template <typename T>
bool allSame(std::optional<std::vector<T>> const& vOpt)
{
    if (!vOpt)
    {
        return true;
    }

    auto const& v = *vOpt;

    if (v.size() <= 1)
    {
        return true;
    }
    auto first = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] != first)
        {
            return false;
        }
    }
    return true;
}

bool hasDiffRuntimeArgs(DecodingSetupParams const& params)
{
    return !allSame(params.presence_penalty) || !allSame(params.repetition_penalty) || !allSame(params.temperature)
        || !allSame(params.min_length);
}
} // namespace

template <typename T>
void DynamicDecodeLayer<T>::setup(size_t batch_size, size_t beam_width, SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (beam_width == 1)
    { // sampling layers
        typename TopPSamplingLayer<T>::SetupParams samplingParams;

        samplingParams.temperature = setupParams.temperature;
        samplingParams.min_length = setupParams.min_length;
        samplingParams.repetition_penalty = setupParams.repetition_penalty;
        samplingParams.presence_penalty = setupParams.presence_penalty;

        samplingParams.runtime_top_k = setupParams.runtime_top_k;
        samplingParams.runtime_top_p = setupParams.runtime_top_p;
        samplingParams.random_seed = setupParams.random_seed;

        samplingParams.top_p_decay = setupParams.top_p_decay;
        samplingParams.top_p_min = setupParams.top_p_min;
        samplingParams.top_p_reset_ids = setupParams.top_p_reset_ids;

        mTopKDecode->setup(batch_size, samplingParams);
        mTopPDecode->setup(batch_size, samplingParams);
    }
    else
    { // beam search layer
        typename OnlineBeamSearchLayer<T>::SetupParams beamSearchParams;

        beamSearchParams.temperature = setupParams.temperature;
        beamSearchParams.min_length = setupParams.min_length;
        beamSearchParams.repetition_penalty = setupParams.repetition_penalty;
        beamSearchParams.presence_penalty = setupParams.presence_penalty;

        beamSearchParams.beam_search_diversity_rate = setupParams.beam_search_diversity_rate;
        beamSearchParams.length_penalty = setupParams.length_penalty;

        has_diff_runtime_args_ = hasDiffRuntimeArgs(beamSearchParams);
        mOnlineBeamsearchDecode->setup(beamSearchParams);
    }
}

template <typename T>
void DynamicDecodeLayer<T>::forward(OutputParams& outputs, ForwardParams const& params)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const auto ite = params.ite;
    const auto step = params.step;
    auto const& logits = params.logits;
    TLLM_CHECK(logits.shape.size() == 3);

    auto const batch_size = logits.shape[0];
    auto const beam_width = logits.shape[1];
    auto const local_batch_size = static_cast<std::size_t>(params.local_batch_size);

    if (params.bad_words_list)
    {
        const auto& bad_words = params.bad_words_list.value();
        const int* bad_words_ptr = bad_words.template getPtr<const int>();
        TLLM_CHECK_WITH_INFO(
            bad_words.shape.size() == 2 || bad_words.shape.size() == 3, "Bad words dimension must be 2 or 3.");

        const bool is_matrix = bad_words.shape.size() == 2;
        if (bad_words.shape.size() == 3)
        {
            TLLM_CHECK_WITH_INFO(bad_words.shape[0] == batch_size,
                fmtstr("Shape of dim 0 of bad words is invalid. It "
                       "must be equal to batch size."
                       " However, it is %ld and the batch size is %ld.",
                    bad_words.shape[0], batch_size));
        }

        const bool shared_bad_words = is_matrix || bad_words.shape[0] == 1;
        const size_t bad_words_len = bad_words.shape[is_matrix ? 1 : 2];
        // Add check on batch size of bad words
        const int id_offset = ite * local_batch_size;
        const int decode_vocab_size_units_offset = id_offset * vocab_size_padded_;

        invokeBanBadWords((T*) logits.getPtrWithOffset(decode_vocab_size_units_offset),
            outputs.output_ids.template getPtr<const int>(),
            beam_width > 1 ? outputs.parent_ids->template getPtr<const int>() : nullptr, batch_size, local_batch_size,
            beam_width,
            shared_bad_words
                ? bad_words_ptr
                : bad_words.template getPtrWithOffset<const int>(ite * local_batch_size * 2 * bad_words_len),
            shared_bad_words, bad_words_len, id_offset, vocab_size_padded_, step, stream_);
    }

    // common inputs
    auto const max_input_length = params.max_input_length;
    auto const& end_id = params.end_ids;

    // dynamic decode GPT
    if (beam_width > 1)
    {
        TLLM_CHECK_WITH_INFO(
            params.src_cache_indirection.has_value(), "src_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(
            outputs.tgt_cache_indirection.has_value(), "tgt_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.parent_ids.has_value(), "parent_ids tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(
            outputs.sequence_length.has_value(), "sequence_length tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.finished.has_value(), "finished tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.cum_log_probs.has_value(), "cum_log_probs tensor is mandatory in beam search.");

        // Because we still not support batch beam search now, so we need to compute
        // one by one if there are different runtime arguments.
        const size_t dynamic_decode_batch_size = has_diff_runtime_args_ ? 1 : local_batch_size;
        const int dynamic_decode_total_iteration = local_batch_size / dynamic_decode_batch_size;

        for (uint dynamic_ite = ite * dynamic_decode_total_iteration;
             dynamic_ite < (ite + 1) * dynamic_decode_total_iteration; ++dynamic_ite)
        {
            const int dynamic_id_offset = dynamic_ite * dynamic_decode_batch_size * beam_width;
            const int dynamic_decode_vocab_size_units_offset = dynamic_id_offset * vocab_size_padded_;

            auto const logits_offset = logits.slice(
                {dynamic_decode_batch_size, logits.shape[1], logits.shape[2]}, dynamic_decode_vocab_size_units_offset);
            auto const end_id_offset
                = end_id.slice({dynamic_decode_batch_size}, dynamic_ite * dynamic_decode_batch_size);
            typename BaseBeamSearchLayer<T>::ForwardParams dynamic_decode_input_tensors{
                step, ite, max_input_length, logits_offset, end_id_offset, *params.src_cache_indirection};

            dynamic_decode_input_tensors.embedding_bias = params.embedding_bias;

            if (params.input_lengths)
            {
                dynamic_decode_input_tensors.input_lengths
                    = params.input_lengths->slice({dynamic_decode_batch_size * beam_width}, dynamic_id_offset);
            }

            // common outputs
            typename BaseBeamSearchLayer<T>::BeamSearchOutputParams dynamic_decode_outputs(
                outputs.output_ids, outputs.parent_ids.value(), outputs.tgt_cache_indirection.value());

            dynamic_decode_outputs.sequence_length
                = outputs.sequence_length->slice({dynamic_decode_batch_size * beam_width}, dynamic_id_offset);
            dynamic_decode_outputs.finished
                = outputs.finished->slice({dynamic_decode_batch_size * beam_width}, dynamic_id_offset);
            dynamic_decode_outputs.cum_log_probs
                = outputs.cum_log_probs->slice({dynamic_decode_batch_size * beam_width}, dynamic_id_offset);

            dynamic_decode_outputs.beam_hyps = outputs.beam_hyps;
            dynamic_decode_outputs.output_log_probs = outputs.output_log_probs;

            // only OnlineBeamSearchLayer support beam_search_diversity_rate
            // when beam_hyps is used
            mOnlineBeamsearchDecode->forward(dynamic_decode_outputs, dynamic_decode_input_tensors);
        } // end of dynamic_ite
    }
    else
    { // beam_width == 1
        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        const size_t local_batch_offset = ite * local_batch_size * beam_width;

        Tensor const logits_slice{
            logits.slice({local_batch_size, beam_width, logits.shape[2]}, local_batch_offset * logits.shape[2])};
        Tensor const end_id_slice{end_id.slice({local_batch_size}, ite * local_batch_size)};
        typename BaseSamplingLayer<T>::ForwardParams decode_input_tensors{
            step, ite, max_input_length, logits_slice, end_id_slice};

        decode_input_tensors.embedding_bias = params.embedding_bias;

        if (params.input_lengths)
        {
            auto& input_lengths = params.input_lengths.value();
            decode_input_tensors.input_lengths
                = input_lengths.slice({local_batch_size, beam_width}, local_batch_offset);
        }

        DecodingOutputParams decode_outputs(outputs.output_ids);
        if (outputs.sequence_length)
        {
            decode_outputs.sequence_length
                = outputs.sequence_length->slice({local_batch_size * beam_width}, local_batch_offset);
        }
        if (outputs.finished)
        {
            decode_outputs.finished = outputs.finished->slice({local_batch_size * beam_width}, local_batch_offset);
        }
        if (outputs.cum_log_probs)
        {
            decode_outputs.cum_log_probs
                = outputs.cum_log_probs->slice({local_batch_size * beam_width}, local_batch_offset);
        }
        if (outputs.output_log_probs)
        {
            Tensor& output_log_probs = outputs.output_log_probs.value();
            size_t step_offset = (step - max_input_length) * batch_size * beam_width;
            decode_outputs.output_log_probs = output_log_probs.slice(
                {output_log_probs.shape[0] - (step - max_input_length), local_batch_size * beam_width},
                step_offset + local_batch_offset);
        }

        // Run topk / topp decode layers.
        // Currently, we support batch sampling. If the runtime arguments are like
        // topk = [4, 0, 4]. topp = [0.0, 0.5, 0.5]
        // then topk_decode handles [4, x, 4 + 0.5]
        //      topp_decode handles [x, 0.5, x]
        // where "x" are skipped.
        mTopKDecode->forward(decode_outputs, decode_input_tensors);
        mTopPDecode->forward(decode_outputs, decode_input_tensors);
    }

    if (params.stop_words_list)
    {
        const size_t id_offset = ite * local_batch_size * beam_width;
        const size_t stop_words_length = params.stop_words_list->shape[2];

        invokeStopWordsCriterion(outputs.output_ids.template getPtr<const int>(),
            outputs.parent_ids->template getPtr<const int>(),
            params.stop_words_list->template getPtrWithOffset<const int>(
                ite * local_batch_size * 2 * stop_words_length),
            outputs.finished->template getPtrWithOffset<bool>(id_offset), id_offset, stop_words_length, batch_size,
            beam_width, step, stream_);
    }

    if (params.sequence_limit_length)
    {
        invokeLengthCriterion(outputs.finished->template getPtr<bool>(),
            outputs.finished_sum ? outputs.finished_sum->template getPtr<int>() : nullptr,
            params.sequence_limit_length->template getPtr<const uint32_t>(), batch_size, beam_width, step, stream_);
    }
}

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
