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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template <typename T>
__global__ void addQKVBiasIA3Transpose(T* q_out, T* k_out, T* v_out, const T* __restrict q_in,
    const T* __restrict bias_q, const T* __restrict k_in, const T* __restrict bias_k, const T* __restrict v_in,
    const T* __restrict bias_v, const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x)
    {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
            + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        T q = ldg(&q_in[src_id]);
        q_out[target_id] = add(q, ldg(&bias_q[col_id]));

        T k = add(ldg(&k_in[src_id]), ldg(&bias_k[col_id]));
        if (use_ia3_key)
        {
            k = k * ia3_key_weights[ia3_task * n + col_id];
        }
        k_out[target_id] = k;

        T v = add(ldg(&v_in[src_id]), ldg(&bias_v[col_id]));
        if (use_ia3_value)
        {
            v = v * ia3_value_weights[ia3_task * n + col_id];
        }
        v_out[target_id] = v;
    }
}

template <typename T>
__global__ void QKVIA3Transpose(T* q_out, T* k_out, T* v_out, const T* __restrict q_in, const T* __restrict k_in,
    const T* __restrict v_in, const int* ia3_tasks, const T* __restrict ia3_key_weights,
    const T* __restrict ia3_value_weights, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x)
    {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
            + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        q_out[target_id] = ldg(&q_in[src_id]);

        T k = ldg(&k_in[src_id]);
        if (use_ia3_key)
        {
            k = k * ia3_key_weights[ia3_task * n + col_id];
        }
        k_out[target_id] = k;

        T v = ldg(&v_in[src_id]);
        if (use_ia3_value)
        {
            v = v * ia3_value_weights[ia3_task * n + col_id];
        }
        v_out[target_id] = v;
    }
}

template <typename T>
void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, const T* bias_Q, T* K, const T* bias_K, T* V,
    const T* bias_V, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
    const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights, cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    dim3 grid(batch_size, seq_len);
    bool is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0)
    {
        dim3 block(min(k, 512));
        if (is_add_bias)
        {
            addQKVBiasIA3Transpose<T><<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, Q, bias_Q, K, bias_K, V, bias_V,
                ia3_tasks, ia3_key_weights, ia3_value_weights, batch_size, seq_len, head_num, size_per_head);
        }
        else
        {
            QKVIA3Transpose<T><<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, Q, K, V, ia3_tasks, ia3_key_weights,
                ia3_value_weights, batch_size, seq_len, head_num, size_per_head);
        }
        sync_check_cuda_error();
    }
    else
    {
        using T2 = typename TypeConverter<T>::Type; // fp16 to half2, bf16 to bf162
        dim3 block(min(k / 2, 512));
        if (is_add_bias)
        {
            addQKVBiasIA3Transpose<T2><<<grid, block, 0, stream>>>((T2*) q_buf, (T2*) k_buf, (T2*) v_buf, (const T2*) Q,
                (const T2*) bias_Q, (const T2*) K, (const T2*) bias_K, (const T2*) V, (const T2*) bias_V, ia3_tasks,
                (const T2*) ia3_key_weights, (const T2*) ia3_value_weights, batch_size, seq_len, head_num,
                size_per_head / 2);
        }
        else
        {
            QKVIA3Transpose<T2><<<grid, block, 0, stream>>>((T2*) q_buf, (T2*) k_buf, (T2*) v_buf, (const T2*) Q,
                (const T2*) K, (const T2*) V, ia3_tasks, (const T2*) ia3_key_weights, (const T2*) ia3_value_weights,
                batch_size, seq_len, head_num, size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

#define INSTANTIATE_ADDQKVBIASIA3_TRANSPOSE(T)                                                                         \
    template void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, const T* bias_Q, T* K,              \
        const T* bias_K, T* V, const T* bias_V, const int batch_size, const int seq_len, const int head_num,           \
        const int size_per_head, const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights,           \
        cudaStream_t stream)
INSTANTIATE_ADDQKVBIASIA3_TRANSPOSE(float);
INSTANTIATE_ADDQKVBIASIA3_TRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATE_ADDQKVBIASIA3_TRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDQKVBIASTRANSPOSE

template <typename T, typename T_IN, int ITEMS_PER_THREAD>
__global__ void softmax_kernel(T* attn_score, const T_IN* qk, const T* attn_mask, const T* linear_bias_slopes,
    const int64_t batch_size, const int64_t head_num, const int64_t q_length, const int64_t k_length,
    const float qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    const int64_t bi = blockIdx.y; // Batch index.
    const int64_t hi = blockIdx.z; // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float) linear_bias_slopes[hi] : 0.0f;

    // Loop along with Q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x)
    {
        float data[ITEMS_PER_THREAD];
        float local_max = -1e20f;

        // Loop along with K dimension.
        int64_t ki{threadIdx.x};
        for (int i = 0; ki < k_length; i++, ki += blockDim.x)
        {
            int64_t qk_offset{((bi * head_num + hi) * q_length + qi) * k_length + ki};

            float qk_val = static_cast<float>(qk[qk_offset]);
            float qk_bias = 0.0f;

            if (linear_bias_slopes != nullptr)
            {
                // We don't handle the upper diagonal (ki > qi) separately, whose values
                // are negligible due to the negative infinity mask. And it matches with
                // the HF's implementation.
                qk_bias += static_cast<float>(linear_bias_slope * (ki - qi));
            }

            int64_t mask_offset = ((int64_t) bi * q_length + qi) * k_length + ki;
            float mask_val = static_cast<float>(ldg(&attn_mask[mask_offset]));
            qk_bias += (1.0f - mask_val) * -10000.0f;

            data[i] = qk_scale * qk_val + qk_bias;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0)
        {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length; i++, ki += blockDim.x)
        {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0)
        {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length; i++, ki += blockDim.x)
        {
            int64_t qk_offset{((bi * head_num + hi) * q_length + qi) * k_length + ki};
            attn_score[qk_offset] = (T) (data[i] * s_mean);
        }
    }
}

template <typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2(T* attn_score, const T* qk_buf, const T* attn_mask, const T* linear_bias_slopes,
    const int64_t batch_size, const int64_t head_num, const int64_t q_length, const int64_t k_length, const T qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    T2* attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2 = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2 = reinterpret_cast<const T2*>(attn_mask);

    const int64_t bi = blockIdx.y; // Batch index
    const int64_t hi = blockIdx.z; // Head index.

    __shared__ float s_mean, s_max;

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE = cuda_cast<T2>(1.0f);
    const T2 ZERO = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale_h2 = cuda_cast<T2>(qk_scale);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;
    int64_t k_length_half = k_length / 2;

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x)
    {
        T2 data[ITEMS_PER_THREAD];
        float local_max = -1e20f;

        // Loop over k dimension.
        int64_t ki{threadIdx.x};
        for (int i = 0; ki < k_length_half && i < ITEMS_PER_THREAD; i++, ki += blockDim.x)
        {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int64_t qk_offset{((bi * head_num + hi) * q_length + qi) * k_length_half + ki};
            int64_t mask_offset = (bi * q_length + qi) * k_length_half + ki;

            // The value of QK^T matrix at (qi, ki).
            T2 qk = qk_buf_h2[qk_offset];
            // The bias value to the position (qi, ki) including both mask and positional bias.
            T2 qk_bias = ZERO;

            if (linear_bias_slopes != nullptr)
            {
                // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                // separately, whose values are negligible due to the negative infinity mask.
                T2 dist(2.0f * ki - qi, 2.0f * ki + 1 - qi);
                qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(linear_bias_slope, dist));
            }

            T2 mask_val = ldg(&attn_mask_h2[mask_offset]);
            qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(hsub2<T2>(ONE, mask_val), NEG_INFTY));

            data[i] = hadd2<T2>(hmul2<T2>(qk, qk_scale_h2), qk_bias);
            local_max = fmax(local_max, fmax((float) data[i].x, (float) data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0)
        {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length_half && i < ITEMS_PER_THREAD; i++, ki += blockDim.x)
        {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float) (data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0)
        {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length_half && i < ITEMS_PER_THREAD; i++, ki += blockDim.x)
        {
            int64_t qk_offset{((bi * head_num + hi) * q_length + qi) * k_length_half + ki};
            attn_score_h2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

template <typename T, int K_ITEMS_PER_THREAD, int Q_ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2_v2(T* attn_score, const T* qk_buf, const T* attn_mask, const T* linear_bias_slopes,
    const int64_t batch_size, const int64_t head_num, const int64_t q_length, const int64_t k_length, const T scalar)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    // QK^T matrix of shape (batch_size, head_num, q_length, k_length / 2)
    T2* attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2 = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2 = reinterpret_cast<const T2*>(attn_mask);

    const int64_t bi = blockIdx.y; // Batch index
    const int64_t hi = blockIdx.z; // Head index.

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE = cuda_cast<T2>(1.0f);
    const T2 ZERO = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale = cuda_cast<T2>(scalar);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;
    const int64_t k_length_half = k_length / 2;
    __shared__ float s_sum[Q_ITEMS_PER_THREAD], s_max[Q_ITEMS_PER_THREAD];

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x * Q_ITEMS_PER_THREAD)
    {
        T2 data[Q_ITEMS_PER_THREAD][K_ITEMS_PER_THREAD];

        int64_t qk_offset[Q_ITEMS_PER_THREAD];

        float local_max[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++)
        {
            local_max[j] = -1e20f;
        }

        // Loop over k dimension.
        const int64_t q_items = min(static_cast<int64_t>((q_length - qi + gridDim.x - 1) / gridDim.x),
            static_cast<int64_t>(Q_ITEMS_PER_THREAD));
        // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
        int64_t ki{threadIdx.x};
        for (int i = 0; ki < k_length_half && i < K_ITEMS_PER_THREAD; ++i, ki += blockDim.x)
        {

            int64_t mask_offset[Q_ITEMS_PER_THREAD];

            for (int j = 0; j < q_items; j++)
            {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * k_length_half + ki;
                mask_offset[j] = (bi * q_length + qi + j * gridDim.x) * k_length_half + ki;
            }

            T2 mask_val[Q_ITEMS_PER_THREAD];
            for (int j = 0; j < q_items; j++)
            {
                mask_val[j] = ldg(&attn_mask_h2[mask_offset[j]]);
            }

            T2 qk[Q_ITEMS_PER_THREAD];
            for (int j = 0; j < q_items; j++)
            {
                qk[j] = qk_buf_h2[qk_offset[j]];
            }

            T2 pos_bias[Q_ITEMS_PER_THREAD];
            if (linear_bias_slopes != nullptr)
            {
                for (int j = 0; j < q_items; j++)
                {
                    // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                    // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                    // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                    // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                    // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                    // separately, whose values are negligible due to the negative infinity mask.
                    int64_t qidx = qi + j * gridDim.x;
                    T2 dist(2.0f * ki - qidx, 2.0f * ki + 1 - qidx);
                    pos_bias[j] = hmul2<T2>(linear_bias_slope, dist);
                }
            }

            for (int j = 0; j < q_items; j++)
            {
                mask_val[j] = hmul2<T2>(hsub2<T2>(ONE, mask_val[j]), NEG_INFTY);
            }

            for (int j = 0; j < q_items; j++)
            {
                T2 val = hadd2<T2>(hmul2<T2>(qk_scale, qk[j]), mask_val[j]);
                if (linear_bias_slopes != nullptr)
                {
                    val = hadd2<T2>(val, pos_bias[j]);
                }
                data[j][i] = val;
                local_max[j] = fmax(local_max[j], fmax((float) data[j][i].x, (float) data[j][i].y));
            }
        }

        if (blockDim.x <= 32)
        {
            warpReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }
        else
        {
            blockReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }

        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++)
            {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++)
        {
            local_sum[j] = {0.f};
        }

        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length_half && i < K_ITEMS_PER_THREAD; ++i, ki += blockDim.x)
        {
            for (int j = 0; j < q_items; ++j)
            {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

            for (int j = 0; j < q_items; j++)
            {
                local_sum[j] += (float) (data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32)
        {
            warpReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }
        else
        {
            blockReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }

        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++)
            {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        ki = (int64_t) threadIdx.x;
        for (int i = 0; ki < k_length_half && i < K_ITEMS_PER_THREAD; ++i, ki += blockDim.x)
        {
            for (int j = 0; j < q_items; j++)
            {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * k_length_half + ki;
            }

            for (int j = 0; j < q_items; j++)
            {
                attn_score_h2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

#define LAUNCH_MASKED_SOFTMAX_(T_, ITEMS_PER_THREAD)                                                                   \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    block.x = divUp(block.x, 32) * 32;                                                                                 \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2)                                                                                                      \
    {                                                                                                                  \
        if (grid.x % 4 == 0)                                                                                           \
        {                                                                                                              \
            grid.x /= 4;                                                                                               \
            softmax_kernel_h2_v2<T_, ITEMS_PER_THREAD, 4><<<grid, block, 0, stream>>>((T_*) param.attention_score,     \
                (const T_*) param.qk, (const T_*) param.attention_mask, (const T_*) param.linear_bias_slopes,          \
                param.batch_size, param.num_heads, param.q_length, param.k_length, (const T_) param.qk_scale);         \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            softmax_kernel_h2<T_, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((T_*) param.attention_score,           \
                (const T_*) param.qk, (const T_*) param.attention_mask, (const T_*) param.linear_bias_slopes,          \
                param.batch_size, param.num_heads, param.q_length, param.k_length, (const T_) param.qk_scale);         \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        softmax_kernel<T, T_IN, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(param.attention_score, param.qk,         \
            param.attention_mask, param.linear_bias_slopes, param.batch_size, param.num_heads, param.q_length,         \
            param.k_length, param.qk_scale);                                                                           \
    }

#define LAUNCH_MASKED_SOFTMAX(ITEMS_PER_THREAD) LAUNCH_MASKED_SOFTMAX_(half, ITEMS_PER_THREAD)

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360)
    {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 32768)
    {
        TLLM_CHECK(false); // Not implemented - it's not clear we want to use the unfused kernel in that case.
    }
    else if (block.x > 16384)
    {
        LAUNCH_MASKED_SOFTMAX(32)
    }
    else if (block.x > 8192)
    {
        LAUNCH_MASKED_SOFTMAX(16)
    }
    else if (block.x > 4096)
    {
        LAUNCH_MASKED_SOFTMAX(8)
    }
    else if (block.x > 2048)
    {
        LAUNCH_MASKED_SOFTMAX(4)
    }
    else if (block.x > 1024)
    {
        LAUNCH_MASKED_SOFTMAX(2)
    }
    else if (block.x > 0)
    {
        LAUNCH_MASKED_SOFTMAX(1)
    }
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, half>& param, cudaStream_t stream);

#ifdef ENABLE_BF16
template <>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, float>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T = __nv_bfloat16;
    using T_IN = float;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360)
    {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 32768)
    {
        TLLM_CHECK(false); // Not implemented - it's not clear we want to use the unfused kernel in that case.
    }
    else if (block.x > 16384)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 32)
    }
    else if (block.x > 8192)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 16)
    }
    else if (block.x > 4096)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 8)
    }
    else if (block.x > 2048)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 4)
    }
    else if (block.x > 1024)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 2)
    }
    else if (block.x > 0)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 1)
    }
}

template <>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, __nv_bfloat16>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T = __nv_bfloat16;
    using T_IN = __nv_bfloat16;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360)
    {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 32768)
    {
        TLLM_CHECK(false); // Not implemented - it's not clear we want to use the unfused kernel in that case.
    }
    else if (block.x > 16384)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 32)
    }
    else if (block.x > 8192)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 16)
    }
    else if (block.x > 4096)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 8)
    }
    else if (block.x > 2048)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 4)
    }
    else if (block.x > 1024)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 2)
    }
    else if (block.x > 0)
    {
        LAUNCH_MASKED_SOFTMAX_(__nv_bfloat16, 1)
    }
}

#endif

#undef LAUNCH_MASKED_SOFTMAX
#undef LAUNCH_MASKED_SOFTMAX_

template <typename T>
__global__ void transpose(const T* src, T* dst, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const float* scale, int int8_mode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
    int id = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    if (int8_mode == 2)
    {
        using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
        using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;

        const Float_Packed_T scale_val = cuda_cast<Float_Packed_T>(*scale);
        reinterpret_cast<Int8_Packed_T*>(dst)[target_id]
            = cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src[tid]) * scale_val);
    }
    else
    {
        dst[target_id] = src[tid];
    }
}

template <>
__global__ void transpose(const float* src, float* dst, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const float* scale, int int8_mode)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;

    const int target_id = batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
        + head_id * size_per_head + threadIdx.x;
    const int src_id = blockIdx.x * size_per_head + threadIdx.x;

    if (int8_mode == 2)
    {
        const float scale_val = *scale;
        reinterpret_cast<int8_t*>(dst)[target_id] = cuda_cast<int8_t>(src[src_id] * scale_val);
    }
    else
    {
        dst[target_id] = src[src_id];
    }
}

template <typename T>
void invokeTransposeQKV(T* dst, T* src, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const float* scale, const int int8_mode, cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2)
    {
        int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0)
        {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        TLLM_CHECK(grid.x * seq_per_block == (size_t) batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0)
        {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value)
            {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2*) src, (half2*) dst, batch_size, seq_len, head_num, size_per_head / 2, scale, int8_mode);
            }
#ifdef ENABLE_BF16
            else
            {
                transpose<__nv_bfloat162><<<grid, block, 0, stream>>>((__nv_bfloat162*) src, (__nv_bfloat162*) dst,
                    batch_size, seq_len, head_num, size_per_head / 2, scale, int8_mode);
            }
#endif
        }
        else
        {
            block.x = seq_per_block * size_per_head;
            transpose<T>
                <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
        }
    }
    else
    {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
    }
}

#define INSTANTIATE_TRANSPOSE_QKV(T)                                                                                   \
    template void invokeTransposeQKV(T* src, T* dst, const int batch_size, const int seq_len, const int head_num,      \
        const int size_per_head, const float* scale, const int int8_mode, cudaStream_t stream)
INSTANTIATE_TRANSPOSE_QKV(float);
INSTANTIATE_TRANSPOSE_QKV(half);
#ifdef ENABLE_BF16
INSTANTIATE_TRANSPOSE_QKV(__nv_bfloat16);
#endif
#undef INSTANTIATE_TRANSPOSE_QKV

template <typename T>
__global__ void add_QKV_bias_rebuild_padding_ia3(const T* Q, const T* bias_Q, const T* K, const T* bias_K, const T* V,
    const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, const int* ia3_tasks, const T* ia3_key_weights,
    const T* ia3_value_weights, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
    const int* mask_offset)
{
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[tgt_batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
            + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = add(ldg(&Q[src_id]), ldg(&bias_Q[idx]));

        T k = ldg(&K[src_id]);
        if (use_ia3_key)
        {
            k = k * ia3_key_weights[ia3_task * n + idx];
        }
        k_buf_[tgt_id] = add(k, ldg(&bias_K[idx]));

        T v = ldg(&V[src_id]);
        if (use_ia3_value)
        {
            v = v * ia3_value_weights[ia3_task * n + idx];
        }
        v_buf_[tgt_id] = add(v, ldg(&bias_V[idx]));
    }
}

template <typename T>
__global__ void rebuild_padding_ia3(const T* Q, const T* K, const T* V, T* q_buf_, T* k_buf_, T* v_buf_,
    const int* ia3_tasks, const T* ia3_key_weights, const T* ia3_value_weights, const int batch_size, const int seq_len,
    const int head_num, const int size_per_head, const int* mask_offset)
{
    const int bid = blockIdx.x;

    const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
    const int n = head_num * size_per_head;

    const bool use_ia3 = ia3_tasks != nullptr;
    const int ia3_task = use_ia3 ? ia3_tasks[tgt_batch_id] : 0;
    const bool use_ia3_key = use_ia3 && (ia3_key_weights != nullptr);
    const bool use_ia3_value = use_ia3 && (ia3_value_weights != nullptr);
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        const int tgt_head_id = idx / size_per_head;
        const int tgt_hidden_id = idx % size_per_head;

        const int src_id = bid * n + idx;
        const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + tgt_head_id * seq_len * size_per_head
            + tgt_seq_id * size_per_head + tgt_hidden_id;

        q_buf_[tgt_id] = ldg(&Q[src_id]);

        T k = ldg(&K[src_id]);
        if (use_ia3_key)
        {
            k = k * ia3_key_weights[ia3_task * n + idx];
        }
        k_buf_[tgt_id] = k;

        T v = ldg(&V[src_id]);
        if (use_ia3_value)
        {
            v = v * ia3_value_weights[ia3_task * n + idx];
        }
        v_buf_[tgt_id] = v;
    }
}

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf,
    T* k_buf, T* v_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
    const int valid_word_num, const int* mask_offset, const int* ia3_tasks, const T* ia3_key_weights,
    const T* ia3_value_weights, cudaStream_t stream)
{
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2 = typename TypeConverter<T>::Type; // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2)
    {
        while (block_size > 512)
        {
            if (block_size % 2 == 0)
            {
                block_size /= 2;
            }
            else
            {
                is_half2 = false;
                block_size = std::min(block_size, 512);
                break;
            }
        }
    }
    else
    {
        block_size = std::min(block_size, 512);
    }

    if (bias_Q == nullptr && bias_K == nullptr && bias_V == nullptr)
    {
        if (is_half2)
        {
            rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>((T2*) Q, (T2*) K, (T2*) V, (T2*) q_buf,
                (T2*) k_buf, (T2*) v_buf, ia3_tasks, (const T2*) ia3_key_weights, (const T2*) ia3_value_weights,
                batch_size, seq_len, head_num, size_per_head / 2, mask_offset);
        }
        else
        {
            rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>(Q, K, V, q_buf, k_buf, v_buf, ia3_tasks,
                ia3_key_weights, ia3_value_weights, batch_size, seq_len, head_num, size_per_head, mask_offset);
        }
    }
    else if (bias_Q != nullptr && bias_K != nullptr && bias_V != nullptr)
    {
        if (is_half2)
        {
            add_QKV_bias_rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>((T2*) Q, (const T2*) bias_Q,
                (T2*) K, (const T2*) bias_K, (T2*) V, (const T2*) bias_V, (T2*) q_buf, (T2*) k_buf, (T2*) v_buf,
                ia3_tasks, (const T2*) ia3_key_weights, (const T2*) ia3_value_weights, batch_size, seq_len, head_num,
                size_per_head / 2, mask_offset);
        }
        else
        {
            add_QKV_bias_rebuild_padding_ia3<<<valid_word_num, block_size, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V,
                q_buf, k_buf, v_buf, ia3_tasks, ia3_key_weights, ia3_value_weights, batch_size, seq_len, head_num,
                size_per_head, mask_offset);
        }
    }
    else
    {
        TLLM_CHECK(false);
    }
}

#define INSTANTIATE_ADDQKVBIASIA3_REBUILD_PADDING(T)                                                                   \
    template void invokeAddQKVBiasIA3RebuildPadding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V,                \
        const T* bias_V, T* q_buf, T* k_buf, T* v_buf, const int batch_size, const int seq_len, const int head_num,    \
        const int size_per_head, const int valid_word_num, const int* mask_offset, const int* ia3_tasks,               \
        const T* ia3_key_weights, const T* ia3_value_weights, cudaStream_t stream)
INSTANTIATE_ADDQKVBIASIA3_REBUILD_PADDING(float);
INSTANTIATE_ADDQKVBIASIA3_REBUILD_PADDING(half);
#ifdef ENABLE_BF16
INSTANTIATE_ADDQKVBIASIA3_REBUILD_PADDING(__nv_bfloat16);
#endif
#undef INSTANTIATEADDQKVBIASREBUILDPADDING

template <typename T>
__global__ void transpose_remove_padding(const T* src, T* dst, const int batch_size, const int seq_len,
    const int head_num, const int size_per_head, const int* mask_offset, const float* scale, const int int8_mode)
{
    // TODO: optimize this kernel?
    // do remove_sequence_length_padding
    const int bid = blockIdx.x; // batch * seq_len or valid_word_num

    const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

    const int dst_seq_id = bid;

    const int src_offset_base = src_batch_id * seq_len * head_num * size_per_head + src_seq_id * size_per_head;
    const int dst_offset_base = dst_seq_id * head_num * size_per_head;

    using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    const Float_Packed_T scale_val
        = int8_mode == 2 ? cuda_cast<Float_Packed_T>(*scale) : cuda_cast<Float_Packed_T>(0.0f);

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x)
    {
        const int head_id = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        const T src_elem = ldg(&src[src_offset_base + head_id * seq_len * size_per_head + hidden_id]);
        if (int8_mode == 2)
        {
            reinterpret_cast<Int8_Packed_T*>(dst)[dst_offset_base + idx]
                = cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src_elem) * scale_val);
        }
        else
        {
            dst[dst_offset_base + idx] = src_elem;
        }
    }
}

// clang-format off
template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream)
{
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2       = typename TypeConverter<T>::Type;  // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2   = false;
                block_size = std::min(block_size, 1024);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 1024);
    }

    if (is_half2) {
        transpose_remove_padding<T2><<<valid_word_num, block_size, 0, stream>>>(
            (T2*)src, (T2*)dst, batch_size, seq_len, head_num, size_per_head / 2, mask_offset, scale, int8_mode);
    }
    else {
        transpose_remove_padding<<<valid_word_num, block_size, 0, stream>>>(
            src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset, scale, int8_mode);
    }
}

// clang-format on

#define INSTANTIATE_TRANSPOSE_ATTENTION_OUT_REMOVE_PADDING(T)                                                          \
    template void invokeTransposeAttentionOutRemovePadding(T* src, T* dst, const int valid_word_num,                   \
        const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int* mask_offset,  \
        const float* scale, const int int8_mode, cudaStream_t stream)
INSTANTIATE_TRANSPOSE_ATTENTION_OUT_REMOVE_PADDING(float);
INSTANTIATE_TRANSPOSE_ATTENTION_OUT_REMOVE_PADDING(half);
#ifdef ENABLE_BF16
INSTANTIATE_TRANSPOSE_ATTENTION_OUT_REMOVE_PADDING(__nv_bfloat16);
#endif
#undef INSTANTIATE_TRANSPOSE_ATTENTION_OUT_REMOVE_PADDING

template <typename T, bool ADD_BIAS>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* __restrict qkv_bias,
    const int* seq_lens, const int* padding_offset, const int batch_size, const int seq_len, const int token_num,
    const int head_num, const int size_per_head, const float* scale, const int int8_mode, const bool multi_query_mode)
{
    // multi_query_mode == false:
    //   QKV: [token_num, 3 * hidden]
    //   qkv_bias: [3 * hidden]
    //   q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]
    //
    // multi_query_mode == true:
    //   QKV: [token_num, hidden + 2 * size_per_head]
    //   qkv_bias: [hidden + 2 * size_per_head]
    //   q_buf: [batch, head_num, seq_len, size_per_head]
    //   k_buf, v_buf: [batch, 1, seq_len, size_per_head]
    T* qkv_ptr[3] = {q_buf, k_buf, v_buf};
    const bool has_padding = padding_offset == nullptr;
    const int hidden = head_num * size_per_head; // hidden dim Q
    int n;
    if (multi_query_mode)
    {
        n = hidden + 2 * size_per_head;
    }
    else
    {
        n = 3 * hidden;
    }

    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < token_num * n; index += gridDim.x * blockDim.x)
    {
        const int bias_id = index % n;

        const int token_idx = index / n;
        const int token_padded_idx = token_idx + (has_padding ? 0 : padding_offset[token_idx]);
        const int target_batch_id = token_padded_idx / seq_len;
        const int actual_seq_len = seq_lens[target_batch_id];
        const int seq_id = token_padded_idx % seq_len;
        const bool valid_seq = seq_id < actual_seq_len || !has_padding;

        int qkv_id;
        int head_id;
        int size_id = index % size_per_head;
        if (multi_query_mode)
        {
            // [token, h + 2, d]
            //  ^^^^^  ^^^^^^^^
            //    m       n
            head_id = (index % n) / size_per_head;
            if (head_id < head_num)
            {
                qkv_id = 0;
            }
            else if (head_id == head_num)
            {
                qkv_id = 1;
                head_id = 0;
            }
            else // head_id == head_num + 1
            {
                qkv_id = 2;
                head_id = 0;
            }
        }
        else
        {
            // [token, 3, h, d]
            //  ^^^^^  ^^^^^^^
            //    m       n
            qkv_id = (index % n) / hidden;
            head_id = (index % hidden) / size_per_head;
        }

        T val = 0.f;
        if (valid_seq)
        {
            if (int8_mode == 2)
            {
                val = cuda_cast<T>(cuda_cast<float>(reinterpret_cast<const int8_t*>(QKV)[index]) * scale[qkv_id]);
            }
            else
            {
                val = ldg(&QKV[index]);
            }
            if (ADD_BIAS)
            {
                val = val + ldg(&qkv_bias[bias_id]);
            }

            if (int8_mode == 2)
            {
                // TODO(mseznec): add support for int8 BMM with FusedAtt
            }
            else
            {
                QKV[index] = val;
            }
        }
        // Write to split QKV buffer
        if (!multi_query_mode || qkv_id == 0) // QKV or Q when multi_query_mode
        {
            const int target_batch_stride = head_num * seq_len * size_per_head;
            const int target_head_stride = seq_len * size_per_head;
            const int target_seq_stride = size_per_head;
            qkv_ptr[qkv_id][target_batch_id * target_batch_stride + head_id * target_head_stride
                + seq_id * target_seq_stride + size_id]
                = val;
        }
        else if (multi_query_mode && qkv_id > 0) // KV when multi_query_mode
        {
            // always head_id == 0 so gone with the head_id term
            const int target_batch_stride = seq_len * size_per_head;
            const int target_seq_stride = size_per_head;
            qkv_ptr[qkv_id][target_batch_id * target_batch_stride + seq_id * target_seq_stride + size_id] = val;
        }
    }
}

template <typename T>
struct Vec_t
{
    static constexpr int size = 0;
};

template <>
struct Vec_t<float>
{
    using Type = float2;
    static constexpr int size = 2;
};

template <>
struct Vec_t<half>
{
    using Type = uint32_t;
    static constexpr int size = 2;
};

#ifdef ENABLE_BF16
template <>
struct Vec_t<__nv_bfloat16>
{
    using Type = __nv_bfloat162;
    static constexpr int size = 2;
};
#endif

template <typename T, bool ADD_BIAS>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* __restrict qkv_bias,
    const int* seq_lens, const int* padding_offset, const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const int rotary_embedding_dim, const bool neox_rotary_style, const bool multi_query_mode)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // When we pass prefix prompt, this kernel also concatenate the prefix prompt and key/value along
    // seq_len dimension like [prompt, key/value].
    // So, the final shape of q is same ([batch_size, head_num, seq_len, size_per_head]), but
    // the shapes of key and values become [batch_size, head_num, max_prefix_prompt_length + seq_len, size_per_head].

    // NOTE:
    // multi_query_mode == false
    //   QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    //   QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    // multi_query_mode == true
    //   QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * size_per_head)
    //                   ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, 1, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[]; // align on largest vector type

    constexpr int vec_size = Vec_t<T>::size;
    using Vec_t = typename Vec_t<T>::Type;
    const int token_idx = blockIdx.x;
    const int token_padding_offset = (padding_offset == nullptr || token_idx < 0) ? 0 : padding_offset[token_idx];
    const int tgt_token_idx = token_idx + token_padding_offset;
    const bool has_padding = padding_offset == nullptr;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx = tgt_token_idx % seq_len;
    const int actual_seq_len = seq_lens[batch_idx];
    const bool valid_seq = seq_idx < actual_seq_len || !has_padding;

    const int head_idx = blockIdx.y;
    const int tidx = threadIdx.x;

    const int total_seq_len = seq_len;

    const bool is_seq_masked = !valid_seq;
    const bool is_head_size_masked = tidx * vec_size >= size_per_head;
    const bool is_masked = is_head_size_masked || is_seq_masked;

    const int hidden_size = head_num * size_per_head;
    const int hidden_idx = head_idx * size_per_head + tidx * vec_size;
    const int hidden_idx_kv = multi_query_mode ? tidx * vec_size : hidden_idx;
    const int n = multi_query_mode ? (head_num + 2) * size_per_head : 3 * head_num * size_per_head;

    const int dst_kv_seq_idx = seq_idx;
    const int src_k_offset = hidden_size;
    const int src_v_offset = multi_query_mode ? hidden_size + size_per_head : 2 * hidden_size;

    // NOTE: q has seq len excluding prefix prompt
    // multi_query_mode == false:
    //   src QKV: [batch, time, 3, head_num, size_per_head]
    // multi_query_mode == true:
    //   src QKV: [batch, time, head_num * size_per_head + 2 * size_per_head]
    const int src_q_idx = token_idx * n + hidden_idx;
    const int src_k_idx = token_idx * n + src_k_offset + hidden_idx_kv;
    const int src_v_idx = token_idx * n + src_v_offset + hidden_idx_kv;

    // destination offset.
    const int dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
        + seq_idx * size_per_head + tidx * vec_size;

    const int kv_head_num = multi_query_mode ? 1 : head_num;
    const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * kv_head_num
        + head_idx * size_per_head * total_seq_len + dst_kv_seq_idx * size_per_head + tidx * vec_size;

    Vec_t q, k, v, zero;
    Vec_t q_bias, k_bias, v_bias;

#pragma unroll
    for (int i = 0; i < sizeof(Vec_t) / sizeof(uint32_t); i++)
    {
        reinterpret_cast<uint32_t*>(&zero)[i] = 0u;
    }

    // load q,k,v and add bias
    if (!is_masked)
    {
        q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);

        if (ADD_BIAS)
        {
            q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx_kv + src_k_offset]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx_kv + src_v_offset]);

            q = mmha::add(q, q_bias);
            k = mmha::add(k, k_bias);
            v = mmha::add(v, v_bias);
        }
    }

    if (!neox_rotary_style)
    {
        mmha::apply_rotary_embedding(q, k, tidx, rotary_embedding_dim, dst_kv_seq_idx);
    }
    else
    {
        const bool do_rotary = !is_masked && vec_size * tidx < rotary_embedding_dim;

        T* q_smem = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem + rotary_embedding_dim;

        const int half_rotary_dim = rotary_embedding_dim / 2;
        const int half_idx = (tidx * vec_size) / half_rotary_dim;
        const int intra_half_idx = (tidx * vec_size) % half_rotary_dim;
        const int smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts?

        if (do_rotary)
        {
            *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;
            *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
        }

        __syncthreads();

        const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = vec_size / 2;
        if (do_rotary)
        {
            mmha::vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
            mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

            mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, rotary_embedding_dim, dst_kv_seq_idx);

            mmha::write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
            mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx);
            k = *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx);
        }
    }
    if (!is_masked)
    {
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
        *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        if (!(multi_query_mode && head_idx > 0))
        {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }
    else if (is_seq_masked && !is_head_size_masked)
    {
        // Set padding to zero in case of potential nan generated.
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = zero;
        *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = zero;
        *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = zero;
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = zero;
        if (!(multi_query_mode && head_idx > 0))
        {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = zero;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = zero;
        }
    }
}

#define FUSED_QKV_BIAS_TRANSPOSE_LAUNCH(T, ADD_BIAS)                                                                   \
    add_fusedQKV_bias_transpose_kernel<T, ADD_BIAS><<<grid, block, 0, stream>>>(q_buf, k_buf, v_buf, QKV, qkv_bias,    \
        seq_lens, padding_offset, batch_size, seq_len, token_num, head_num, size_per_head, scale, int8_mode,           \
        multi_query_mode);

#define FUSED_QKV_BIAS_ROTARY_TRANSPOSE_LAUNCH(T, ADD_BIAS)                                                            \
    add_fusedQKV_bias_transpose_kernel<T, ADD_BIAS><<<grid, block, smem_size, stream>>>(q_buf, k_buf, v_buf, QKV,      \
        qkv_bias, seq_lens, padding_offset, batch_size, seq_len, head_num, size_per_head, rotary_embedding_dim,        \
        neox_rotary_style, multi_query_mode);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* qkv_bias, const int* seq_lens,
    const int* padding_offset, const int batch_size, const int seq_len, const int token_num, const int head_num,
    const int size_per_head, const int rotary_embedding_dim, const int neox_rotary_style, const float* scale,
    const int int8_mode, const bool multi_query_mode, cudaStream_t stream)
{
    // [bs, seq_len, 3, head, Dh]
    if (rotary_embedding_dim == 0)
    {
        const int m = token_num;
        const int n = head_num * size_per_head;
        dim3 block(384);
        dim3 grid((int) (ceil(1.0 * m * n / 384)));

        if (qkv_bias != nullptr)
        {
            FUSED_QKV_BIAS_TRANSPOSE_LAUNCH(T, true);
        }
        else
        {
            FUSED_QKV_BIAS_TRANSPOSE_LAUNCH(T, false);
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(int8_mode != 2, "w8a8 not yet implemented with prefix prompt"); // TODO(mseznec)
        // To implement rotary embeddings, each thread processes two QKV elems:
        dim3 block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
        dim3 grid(token_num, head_num);
        size_t smem_size = neox_rotary_style ? 2 * rotary_embedding_dim * sizeof(T) : 0;
        // NOTE: add offset for rotary embedding
        if (qkv_bias != nullptr)
        {
            FUSED_QKV_BIAS_ROTARY_TRANSPOSE_LAUNCH(T, true);
        }
        else
        {
            FUSED_QKV_BIAS_ROTARY_TRANSPOSE_LAUNCH(T, false);
        }
    }
}

#define INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(T)                                                                       \
    template void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* qkv_bias,              \
        const int* seq_lens, const int* padding_offset, const int batch_size, const int seq_len, const int token_num,  \
        const int head_num, const int size_per_head, const int rotary_embedding_dim, const int neox_rotary_style,      \
        const float* scale, const int int8_mode, const bool multi_query_mode, cudaStream_t stream)
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(float);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE

template <typename T>
__global__ void transpose_4d(T* dst, T* src, const int dim0, const int dim1, const int dim2, const int dim3,
    const int dim0_leading_dim, const int ite)
{
    // transpose from [dim0, dim1, dim2, dim3] to [dim2, X, dim1, dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * dim3; i += blockDim.x * gridDim.x)
    {
        int index = i;
        const int d3 = index % dim3;
        index = (index - d3) / dim3;
        const int d2 = index % dim2;
        index = (index - d2) / dim2;
        const int d1 = index % dim1;
        index = (index - d1) / dim1;
        const int d0 = index % dim0;
        index = (index - d0) / dim0;
        dst[d2 * dim0_leading_dim * dim1 * dim3 + (d0 + dim0 * ite) * dim1 * dim3 + d1 * dim3 + d3] = src[i];
    }
}

template <>
__global__ void transpose_4d(half* dst, half* src, const int dim0, const int dim1, const int dim2, const int dim3,
    const int dim0_leading_dim, const int ite)
{
    half2* dst_ptr = (half2*) dst;
    half2* src_ptr = (half2*) src;
    const int half_dim3 = dim3 / 2;
    // transpose from [dim0, dim1, dim2, half_dim3] to [dim2, dim0, dim1, half_dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * half_dim3; i += blockDim.x * gridDim.x)
    {
        int index = i;
        const int d3 = index % half_dim3;
        index = (index - d3) / half_dim3;
        const int d2 = index % dim2;
        index = (index - d2) / dim2;
        const int d1 = index % dim1;
        index = (index - d1) / dim1;
        const int d0 = index % dim0;
        index = (index - d0) / dim0;
        dst_ptr[d2 * dim0_leading_dim * dim1 * half_dim3 + (d0 + dim0 * ite) * dim1 * half_dim3 + d1 * half_dim3 + d3]
            = src_ptr[i];
    }
}

template <typename T>
void invokeTranspose4d(T* dst, T* src, const int local_batch_size, const int seq_len, const int size_per_head,
    const int local_hidden_units, const int local_head_num, const int batch_size, const int ite, cudaStream_t stream)
{
    transpose_4d<<<local_batch_size * seq_len * local_hidden_units / 512, 512 / (4 / (sizeof(T))), 0, stream>>>(
        dst, src, local_batch_size, local_head_num, seq_len, size_per_head, batch_size, ite);
}

#define INSTANTIATE_TRANSPOSE_4D(T)                                                                                    \
    template void invokeTranspose4d(T* dst, T* src, const int local_batch_size, const int seq_len,                     \
        const int size_per_head, const int local_hidden_units, const int local_head_num, const int batch_size,         \
        const int ite, cudaStream_t stream)
INSTANTIATE_TRANSPOSE_4D(float);
INSTANTIATE_TRANSPOSE_4D(half);
#undef INSTANTIATE_TRANSPOSE_4D

template <typename T, typename T_cache, typename KVCacheBuffer>
__global__ void transpose4dBatchMajorKVCache(const T* kSrc, const T* vSrc, KVCacheBuffer kvCacheBuffer,
    const int headNum, const int sizePerHead, const int seqLen, const float* kvScaleOrigQuant)
{
    // We allow only fp32/fp16/bf16 as input types
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "");

    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    constexpr bool ENABLE_8BITS_CACHE = sizeof(T_cache) == 1;
    using T_dst = T_cache;
    using T_src = typename mmha::packed_type<T, X_ELEMS>::type;

    const int batchIdx = blockIdx.y;
    const int headIdx = blockIdx.z;

    // idx is over output dimension L * sizePerHead / x for values
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // threadIdx.y 0 handles k, while threadIdx.y 1 handles v.
    const bool handle_k = (threadIdx.y == 0);
    const int sizePerHeadDivX = sizePerHead / X_ELEMS;

    if (idx >= sizePerHeadDivX * seqLen)
    {
        return;
    }

    // Get linear token index
    const int tokenIdx = idx / sizePerHeadDivX;
    // Get channel index
    const int channelIdx = idx % sizePerHeadDivX;

    // Get pointer to the dst block given sequence, head and token ids
    auto valDst = handle_k ? reinterpret_cast<T_dst*>(kvCacheBuffer.getKBlockPtr(batchIdx, tokenIdx))
                           : reinterpret_cast<T_dst*>(kvCacheBuffer.getVBlockPtr(batchIdx, tokenIdx));

    // Local to block dst idx
    int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, headIdx, sizePerHeadDivX, channelIdx);

    // 16 byte loads will handle "x" dimension
    const size_t srcOffset = (batchIdx * headNum + headIdx) * sizePerHead * seqLen;
    auto valSrc = reinterpret_cast<const T_src*>((handle_k ? kSrc : vSrc) + srcOffset);

    T_src val = valSrc[idx];
    if (ENABLE_8BITS_CACHE)
    {
        // If T is fp32, T_src is float4 and mmha::num_elems<T_src>::value returns 4
        // If T is fp16/bf16, T_src is uint4 and mmha::num_elems<T_src>::value returns 8
        // mmha::packed_type<int8_t ...>::type becomes uint32_t or uint64_t respectively
        // FIXME(nkorobov) mmha::num_elems semantic is confusing
        inBlockIdx = inBlockIdx * sizeof(mmha::packed_type<T_dst, mmha::num_elems<T_src>::value>::type);
        // Cast float scale to dst data type.
        using T_scale = typename mmha::kv_cache_scale_type_t<T, T_cache>::Type;
        T_scale scaleOrigQuant;
        mmha::convert_from_float(&scaleOrigQuant, kvScaleOrigQuant[0]);
        // Store 8bits kv cache.
        mmha::store_8bits_kv_cache_vec(valDst, val, inBlockIdx, scaleOrigQuant);
    }
    else
    {
        reinterpret_cast<T_src*>(valDst)[inBlockIdx] = val;
    }
}

template <typename T, typename KVCacheBuffer>
void invokeTranspose4dBatchMajor(const T* kSrc, const T* vSrc, KVCacheBuffer& kvTable, const int localBatchSize,
    const int seqLen, const int maxSeqLen, const int sizePerHead, const int localHeadNum,
    const KvCacheDataType cache_type, const float* kvScaleOrigQuant, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Block handles both K and V tile.
    dim3 blockSz(128, 2);
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    dim3 gridSz((seqLen * sizePerHead / x + blockSz.x - 1) / blockSz.x, localBatchSize, localHeadNum);

    TLLM_CHECK_WITH_INFO(sizePerHead % x == 0, "Size per head is not a multiple of X");

    if (cache_type == KvCacheDataType::INT8)
    {
        transpose4dBatchMajorKVCache<T, int8_t, KVCacheBuffer>
            <<<gridSz, blockSz, 0, stream>>>(kSrc, vSrc, kvTable, localHeadNum, sizePerHead, seqLen, kvScaleOrigQuant);
    }
#ifdef ENABLE_FP8
    else if (cache_type == KvCacheDataType::FP8)
    {
        transpose4dBatchMajorKVCache<T, __nv_fp8_e4m3, KVCacheBuffer>
            <<<gridSz, blockSz, 0, stream>>>(kSrc, vSrc, kvTable, localHeadNum, sizePerHead, seqLen, kvScaleOrigQuant);
    }
#endif // ENABLE_FP8
    else
    {
        transpose4dBatchMajorKVCache<T, T, KVCacheBuffer>
            <<<gridSz, blockSz, 0, stream>>>(kSrc, vSrc, kvTable, localHeadNum, sizePerHead, seqLen, kvScaleOrigQuant);
    }
}

#define INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR_KV_CACHE_TYPE(T, KVCacheBuffer)                                           \
    template void invokeTranspose4dBatchMajor(const T* kSrc, const T* vSrc, KVCacheBuffer& kvTable,                    \
        const int localBatchSize, const int seqLen, const int maxSeqLen, const int sizePerHead,                        \
        const int localHeadNum, const KvCacheDataType cache_type, const float* kvScaleOrigQuant, cudaStream_t stream)

#define INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR(T)                                                                        \
    INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR_KV_CACHE_TYPE(T, KVBlockArray);                                               \
    INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR_KV_CACHE_TYPE(T, KVLinearBuffer);

INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR(float)
INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR(half)
#ifdef ENABLE_BF16
INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR(__nv_bfloat16);
#endif

#undef INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR_KV_CACHE_TYPE
#undef INSTANTIATE_TRANSPOSE_4D_BATCH_MAJOR

} // namespace kernels
} // namespace tensorrt_llm
