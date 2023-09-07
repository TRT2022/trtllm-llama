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

#pragma once

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
public:
    DynamicDecodeLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        tc::cublasMMWrapper* cublas_wrapper, tc::IAllocator* allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop);

    ~DynamicDecodeLayer() override;
    DynamicDecodeLayer(DynamicDecodeLayer const& dynamic_decode_layer);

    class SetupParams
    {
    public:
        std::optional<std::vector<float>> temperature;       // [1] or [batch_size] on cpu
        std::optional<std::vector<std::int32_t>> min_length; // [1] or [batch_size] on cpu
        // repetition_penalty and presence_penalty are mutually exclusive.
        std::optional<std::vector<float>> repetition_penalty; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> presence_penalty;   // [1] or [batch_size] on cpu

        // baseSamplingLayer
        std::optional<std::vector<std::uint32_t>> runtime_top_k;    // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> runtime_top_p;            // [1] or [batch_size] on cpu
        std::optional<std::vector<unsigned long long>> random_seed; // [1] or [batch_size] on cpu

        // topPSamplingLayer
        std::optional<std::vector<float>> top_p_decay;            // [batch_size], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;              // [batch_size], must between [0, 1]
        std::optional<std::vector<std::int32_t>> top_p_reset_ids; // [batch_size]

        // omlineBeamSearchLayer
        std::optional<float> beam_search_diversity_rate;
        std::optional<float> length_penalty;
    };

    void setup(size_t batch_size, size_t beam_width, SetupParams const& setupParams);

    class ForwardParams
    {
    public:
        ForwardParams(int step, int ite, int maxInputLength, int localBatchSize, tc::Tensor logits, tc::Tensor endIds)
            : step{step}
            , ite{ite}
            , max_input_length{maxInputLength}
            , local_batch_size{localBatchSize}
            , logits{std::move(logits)}
            , end_ids{std::move(endIds)}
        {
        }

        // mandatory parameters
        int step;
        int ite;
        int max_input_length;
        int local_batch_size;
        tc::Tensor logits;  // [batch_size, beam_width, vocab_size_padded], on gpu
        tc::Tensor end_ids; // [batch_size], on gpu

        // optional parameters
        std::optional<tc::Tensor> src_cache_indirection; // [local_batch_size, beam_width, max_seq_len] - the k/v cache
                                                         // index for beam search, mandatory for beam search, on gpu
        std::optional<tc::Tensor> sequence_limit_length; // [batch_size], on gpu
        std::optional<tc::Tensor> embedding_bias;        // [vocab_size_padded], on gpu
        std::optional<tc::Tensor> input_lengths;         // [batch_size, beam_width], on gpu
        std::optional<tc::Tensor> bad_words_list;  // [2, bad_words_length] or [batch_size, 2, bad_words_length], on gpu
        std::optional<tc::Tensor> stop_words_list; // [batch_size, 2, stop_words_length], on gpu
    };

    class OutputParams
    {
    public:
        explicit OutputParams(tc::Tensor outputIds)
            : output_ids{std::move(outputIds)}
        {
        }

        // mandatory parameters
        tc::Tensor output_ids; // [max_seq_len, batch_size]

        // optional parameters
        std::optional<tc::Tensor> finished;        // [batch_size * beam_width], optional
        std::optional<tc::Tensor> finished_sum;    // [1], optional, in pinned host memory
        std::optional<tc::Tensor> cum_log_probs;   // [batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> parent_ids;      // [max_seq_len, batch_size * beam_width], necessary in beam search
        std::optional<tc::Tensor> sequence_length; // [batch_size * beam_width], optional
        std::optional<tc::Tensor>
            output_log_probs;      // [request_ouptut_length, batch_size * beam_width], must be float*, optional
        std::optional<tc::Tensor>
            tgt_cache_indirection; // [local_batch_size, beam_width, max_seq_len], the k/v cache index for beam search
        std::optional<void*> beam_hyps; // [1] on cpu, a special structure which maintains some pointers of beam search
    };

    void forward(OutputParams& outputs, ForwardParams const& params);

private:
    void initialize();

    std::unique_ptr<OnlineBeamSearchLayer<T>> mOnlineBeamsearchDecode;
    std::unique_ptr<TopKSamplingLayer<T>> mTopKDecode;
    std::unique_ptr<TopPSamplingLayer<T>> mTopPDecode;

    size_t vocab_size_;
    size_t vocab_size_padded_;
    cudaDeviceProp* cuda_device_prop_;

    bool has_diff_runtime_args_ = false;
    int* h_pinned_finished_sum_ = nullptr;
};

} // namespace layers
} // namespace tensorrt_llm
