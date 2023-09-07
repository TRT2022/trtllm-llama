/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template <typename T>
struct Multihead_attention_params_base
{

    // The output buffer. Dimensions B x D.
    T* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The indirections to use for cache when beam sampling.
    const int* cache_indir = nullptr;

    // scales
    const float* query_weight_output_scale = nullptr;
    const float* attention_qk_scale = nullptr;
    const float* attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;
    // Switch to enable multi-query attention
    bool multi_query_mode = false;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    // TODO (martinma): change name max_seq_len
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    bool neox_rotary_style = false;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // Used when we have some input context like gpt
    const int* total_padding_tokens = nullptr;

    const int* masked_tokens = nullptr;
    const int* prefix_prompt_lengths = nullptr;
    int max_prefix_prompt_length = 0;

    const T* relative_attention_bias = nullptr;
    int relative_attention_bias_stride = 0;
    // The slope per head of linear position bias to attention score (H).
    const T* linear_bias_slopes = nullptr;

    const T* ia3_key_weights = nullptr;
    const T* ia3_value_weights = nullptr;
    const int* ia3_tasks = nullptr;

    const float* qkv_scale_quant_orig = nullptr;
    const float* attention_out_scale_orig_quant = nullptr;
    int int8_mode = 0;

    const float* kv_scale_orig_quant = nullptr;
    const float* kv_scale_quant_orig = nullptr;
    bool int8_kv_cache = false;
    bool fp8_kv_cache = false;

    // Multi-block setups
    bool multi_block_mode = false;

    mutable int timesteps_per_block = -1;
    mutable int seq_len_tile = -1;

    mutable int max_seq_len_tile = -1;
    // The partial output buffer. Dimensions max_seq_len_tile x B x D. (for each timestep only seq_len_tile x B x D is
    // needed)
    T* partial_out = nullptr;
    // ThreadBlock sum. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_sum = nullptr;
    // ThreadBlock max. Dimensions max_seq_len_tile x 1. (for each timestep only seq_len_tile x 1 is needed)
    float* partial_max = nullptr;
    // threadblock counter to identify the complete of partial attention computations
    int* block_counter = nullptr;
};

template <typename T>
struct Multihead_attention_params : public Multihead_attention_params_base<T>
{
    int max_decoder_seq_len = 0;

    // allows to exit attention early
    bool* finished = nullptr;

    // required in case of masked attention with different length
    const int* length_per_sample = nullptr;
};
template <class T>
using Masked_multihead_attention_params = Multihead_attention_params<T>;

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const KVBlockArray& block_array,
    const cudaStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params,
    const KVBlockArray& block_array, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
    const KVBlockArray& block_array, const cudaStream_t& stream);
#endif

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params,
    const KVLinearBuffer& block_array, const cudaStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params,
    const KVLinearBuffer& block_array, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
    const KVLinearBuffer& block_array, const cudaStream_t& stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
