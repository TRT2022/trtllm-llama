/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_GPT_ATTENTION_COMMON_H
#define TRT_GPT_ATTENTION_COMMON_H
#include "NvInferPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class GPTAttentionPluginCommon : public IPluginV2DynamicExt
{
public:
    GPTAttentionPluginCommon() = delete;

    GPTAttentionPluginCommon(int num_heads, int head_size, int unidirectional, float q_scaling,
        int rotary_embedding_dim, bool neox_rotary_style, tensorrt_llm::kernels::ContextFMHAType context_fmha_type,
        bool multi_block_mode, bool multi_query_mode, bool int8_kv_cache, bool fp8_kv_cache, bool remove_input_padding,
        tensorrt_llm::kernels::AttentionMaskType mask_type, bool paged_kv_cache, nvinfer1::DataType type);

    GPTAttentionPluginCommon(const void* data, size_t length);

    ~GPTAttentionPluginCommon() override = default;

    template <typename T>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    //! This is called on every trt Engine creation
    int initialize() noexcept override;
    //! This is called on every trt Engine destroy
    void terminate() noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    template <typename T>
    T* cloneImpl() const noexcept;

    //! This is called on evert trt Engine or ExecutionContext destroy.
    //! None-cloned plugins will call terminate and then call destroy, while the cloned plugins will call destroy only
    //! So plugin should put the resource release inside destroy.
    void destroy() noexcept override;

    static size_t getCommonSerializationSize() noexcept;
    void serializeCommon(void* buffer) const noexcept;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

protected:
    int getMaxSeqLenTile(int elemSize) const;
    size_t getWorkspaceSizeForContext(DataType type, int32_t nbReq, int32_t max_input_length) const noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    size_t getWorkspaceSizeForGeneration(DataType type, int32_t total_num_seq) const noexcept;
    template <typename T, typename KVCacheBuffer>
    int enqueueContext(T const* attention_input,
        int32_t input_seq_length, // padded input length
        int32_t max_seq_length,   // cache capacity
        int32_t const* input_lengths, float const* kv_scale_orig_quant, float const* kv_scale_quant_orig,
        T* context_buf_, void* key_value_cache, void* block_pointers, int32_t batch_size, int32_t num_tokens,
        int32_t tokens_per_block, int32_t max_blocks_per_sequence, void* workspace, cudaStream_t stream);
    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(T const* attention_input,
        int32_t input_seq_length, // padded input length
        int32_t const* sequence_lengths, int32_t past_kv_length, int32_t beam_width, int32_t const* masked_tokens,
        int32_t const* input_lengths, float const* kv_scale_orig_quant, float const* kv_scale_quant_orig,
        T* context_buf_, void* key_value_cache, void* block_pointers,
        int32_t max_seq_length, // cache capacity
        int32_t num_requests, int32_t tokens_per_block, int32_t max_blocks_per_sequence, int32_t const* cache_indir,
        void* workspace, cudaStream_t stream);

protected:
    const std::string mLayerName;
    std::string mNamespace;

    int mNumHeads;
    int mHeadSize;
    int mUnidirectional;
    float mQScaling;
    int mRotaryEmbeddingDim;
    bool mNeoxRotaryStyle;
    bool mMultiQueryMode = false;
    bool mInt8KVCache = false;
    bool mRemovePadding = false;
    tensorrt_llm::kernels::AttentionMaskType mMaskType;
    bool mPagedKVCache = false;
    bool mFp8KVCache = false;
    nvinfer1::DataType mType;

    // fmha runner (disable by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = tensorrt_llm::common::getSMVersion();
    tensorrt_llm::kernels::MHARunner* mFMHARunner;

    bool mMultiBlockMode;
    int mDeviceId = -1;
    tensorrt_llm::common::cublasAlgoMap* mCublasAlgoMap;
    std::mutex* mCublasWrapperMutex;
    tensorrt_llm::common::cublasMMWrapper* mCublasWrapper;
};

class GPTAttentionPluginCreatorCommon : public IPluginCreator
{
public:
    GPTAttentionPluginCreatorCommon();

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    template <typename T>
    T* deserializePluginImpl(const char* name, const void* serialData, size_t serialLength) noexcept;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

protected:
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GPT_ATTENTION_COMMON_H
