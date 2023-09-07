/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::BertAttentionPluginCreator;
using nvinfer1::plugin::BertAttentionPlugin;
using nvinfer1::plugin::nextWorkspacePtr;

static const char* BERT_ATTENTION_PLUGIN_VERSION{"1"};
static const char* BERT_ATTENTION_PLUGIN_NAME{"BertAttention"};
PluginFieldCollection BertAttentionPluginCreator::mFC{};
std::vector<PluginField> BertAttentionPluginCreator::mPluginAttributes;

BertAttentionPlugin::BertAttentionPlugin(int num_heads, int head_size, float q_scaling, bool qk_half_accum,
    ContextFMHAType context_fmha_type, nvinfer1::DataType type)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mQKHalfAccum(qk_half_accum)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::disabled)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::enabled_with_fp32_acc)
    , mType(type)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    mEnableContextFMHA = mEnableContextFMHA && (mType == DataType::kHALF) && MHARunner::fmha_supported(mHeadSize, mSM);
}

// Parameterized constructor
BertAttentionPlugin::BertAttentionPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mQScaling);
    read(d, mQKHalfAccum);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* BertAttentionPlugin::clone() const noexcept
{
    auto* plugin = new BertAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs BertAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex == 0);
    auto ret = inputs[0];
    ret.d[2] = exprBuilder.constant(ret.d[2]->getConstantValue() / 3);
    return ret;
}

bool BertAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == nbInputs - 1)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void BertAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t BertAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    const int batch_size = inputs[0].dims.d[0];
    const int input_seq_len = inputs[0].dims.d[1];
    const int local_hidden_units_ = inputs[0].dims.d[2] / 3;
    const int beam_width = 1;
    const int max_input_length = input_seq_len;

    size_t size{0U};
    if (inputs[0].type == DataType::kHALF)
    {
        size = sizeof(half);
    }
    else if (inputs[0].type == DataType::kFLOAT)
    {
        size = sizeof(float);
    }

    const size_t attention_mask_size
        = mEnableContextFMHA ? 0 : size * batch_size * beam_width * max_input_length * max_input_length;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size * beam_width + 1);
    const size_t q_buf_2_size = size * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = size * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = sizeof(int) * batch_size * input_seq_len;

    const int NUM_BUFFERS = 10;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size;
    workspaces[3] = q_buf_2_size;
    workspaces[4] = k_buf_2_size;
    workspaces[5] = v_buf_2_size;
    workspaces[6] = qk_buf_size;
    workspaces[7] = qkv_buf_2_size;
    workspaces[8] = qk_buf_float_size;
    workspaces[9] = padding_offset_size;

    return plugin::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int BertAttentionPlugin::enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    // inputs
    //     input_tensor [batch_size, seq_len, local_hidden_size * 3]
    //     input_lengths [batch_size]
    // outputs
    //     output_tensor [batch_size, seq_len, local_hidden_size]

    const int batch_size = inputDesc[0].dims.d[0];
    const int request_batch_size = batch_size;
    const int input_seq_len = inputDesc[0].dims.d[1];
    const int request_seq_len = input_seq_len;
    const int beam_width = 1;
    const int local_hidden_units_ = inputDesc[0].dims.d[2] / 3;
    const float q_scaling = mQScaling;

    const T* attention_input = reinterpret_cast<const T*>(inputs[0]);
    const int* input_lengths = reinterpret_cast<const int*>(inputs[1]);

    T* context_buf_ = (T*) (outputs[0]);

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    PLUGIN_CUBLASASSERT(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(workspace);
    if (inputDesc[0].type == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (inputDesc[0].type == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }

    const size_t attention_mask_size
        = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * beam_width * input_seq_len * input_seq_len;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size * beam_width + 1);
    const size_t q_buf_2_size = sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = sizeof(int) * batch_size * input_seq_len;

    mMaxInputLength = input_seq_len;

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    T* q_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_ = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    int* padding_offset = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    BuildDecoderInfoParams<T> params;
    memset(&params, 0, sizeof(params));
    params.seqOffsets = cu_seqlens;
    params.paddingOffsets = padding_offset;
    params.attentionMask = attention_mask;
    params.seqLengths = input_lengths;
    params.batchSize = batch_size * beam_width;
    params.maxSeqLength = mMaxInputLength;
    params.numTokens = batch_size * beam_width * mMaxInputLength;
    params.attentionMaskType = AttentionMaskType::PADDING;
    invokeBuildDecoderInfo(params, stream);

    // Padding offset = nullptr here (remove padding is not supported).
    invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(attention_input), input_lengths,
        nullptr, request_batch_size, request_seq_len, batch_size * input_seq_len, mNumHeads, mHeadSize, 0, false,
        (float*) nullptr, 0, false, stream);

    const cudaDataType_t gemm_data_type = CudaDataType<T>::value;
    const int attention_seq_len_1 = request_seq_len; // q length
    const int attention_seq_len_2 = request_seq_len; // kv length
    const T qk_scale = static_cast<T>(1.0f / (sqrtf(mHeadSize * 1.0f) * q_scaling));
    T* linear_bias_slopes = nullptr;

    if (mEnableContextFMHA)
    {
        // b, max_seqlen, actual_total_seqlen
        mFMHARunner->setup(request_batch_size, request_seq_len, request_batch_size * request_seq_len);
        mFMHARunner->run(const_cast<T*>(attention_input), cu_seqlens, context_buf_, stream);
    }
    else
    {
        if (!mQKHalfAccum && gemm_data_type != CUDA_R_32F)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,             // n
                attention_seq_len_1,             // m
                mHeadSize,                       // k
                1.0f, k_buf_2_, gemm_data_type,
                mHeadSize,                       // k
                attention_seq_len_2 * mHeadSize, // n * k
                q_buf_2_, gemm_data_type,
                mHeadSize,                       // k
                attention_seq_len_1 * mHeadSize, // m * k
                0.0f, qk_buf_float_, CUDA_R_32F,
                attention_seq_len_2,             // n
                attention_seq_len_2 * attention_seq_len_1,
                request_batch_size * mNumHeads,  // global batch size
                CUDA_R_32F);

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_float_;              // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N, attention_seq_len_2, attention_seq_len_1,
                mHeadSize, k_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, q_buf_2_, mHeadSize,
                attention_seq_len_1 * mHeadSize, qk_buf_, attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1, request_batch_size * mNumHeads);

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_;                    // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }

        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, attention_seq_len_1,
            attention_seq_len_2, v_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, qk_buf_, attention_seq_len_2,
            attention_seq_len_1 * attention_seq_len_2, qkv_buf_2_, mHeadSize, attention_seq_len_1 * mHeadSize,
            request_batch_size * mNumHeads);

        if (padding_offset == nullptr)
        {
            invokeTransposeQKV(context_buf_, qkv_buf_2_, request_batch_size, attention_seq_len_1, mNumHeads, mHeadSize,
                (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, context_buf_, batch_size * input_seq_len,
                request_batch_size, attention_seq_len_1, mNumHeads, mHeadSize, padding_offset, (float*) nullptr, 0,
                stream);
        }
    }
    return 0;
}

template int BertAttentionPlugin::enqueueImpl<half>(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

template int BertAttentionPlugin::enqueueImpl<float>(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

int BertAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType BertAttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* BertAttentionPlugin::getPluginType() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

const char* BertAttentionPlugin::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

int BertAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int BertAttentionPlugin::initialize() noexcept
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasAlgoMap = new cublasAlgoMap(GEMM_CONFIG);
    mCublasWrapperMutex = new std::mutex();
    mCublasWrapper
        = new cublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, mCublasAlgoMap, mCublasWrapperMutex, nullptr);
    if (mEnableContextFMHA)
    {
        mFMHARunner = new FusedMHARunnerV2(DATA_TYPE_FP16, mNumHeads, mHeadSize, mQScaling);
        // set flags: force_fp32_acc, is_s_padded, causal_mask, multi_query_attention
        mFMHARunner->setup_flags(mFMHAForceFP32Acc, true, false, false);
    }

    return 0;
}

void BertAttentionPlugin::destroy() noexcept
{
    delete mCublasAlgoMap;
    delete mCublasWrapperMutex;
    delete mCublasWrapper;
    if (mEnableContextFMHA)
    {
        delete mFMHARunner;
    }

    mCublasAlgoMap = nullptr;
    mCublasWrapperMutex = nullptr;
    mCublasWrapper = nullptr;
    mFMHARunner = nullptr;
    delete this;
}

size_t BertAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQScaling) + sizeof(mQKHalfAccum) + sizeof(mEnableContextFMHA)
        + sizeof(mFMHAForceFP32Acc) + sizeof(mType);
}

void BertAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mQScaling);
    write(d, mQKHalfAccum);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void BertAttentionPlugin::terminate() noexcept {}

void BertAttentionPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BertAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

BertAttentionPluginCreator::BertAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("enable_qk_half_accum", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BertAttentionPluginCreator::getPluginName() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

const char* BertAttentionPluginCreator::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* BertAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BertAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int num_heads, head_size;
    ContextFMHAType context_fmha_type;
    bool qk_half_accum;
    float q_scaling;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            head_size = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "q_scaling"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            q_scaling = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "enable_qk_half_accum"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT8);
            qk_half_accum = static_cast<bool>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "context_fmha_type"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT8);
            context_fmha_type = static_cast<ContextFMHAType>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new BertAttentionPlugin(num_heads, head_size, q_scaling, qk_half_accum, context_fmha_type, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* BertAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BertAttentionPlugin::destroy()
    try
    {
        auto* obj = new BertAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void BertAttentionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* BertAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
