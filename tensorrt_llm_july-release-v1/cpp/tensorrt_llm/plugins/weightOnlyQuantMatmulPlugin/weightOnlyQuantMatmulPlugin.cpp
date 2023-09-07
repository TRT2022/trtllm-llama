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
#include "tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using nvinfer1::plugin::WeightOnlyQuantMatmulPluginCreator;
using nvinfer1::plugin::WeightOnlyQuantMatmulPlugin;

static const char* WOQ_MATMUL_PLUGIN_VERSION{"1"};
static const char* WOQ_MATMUL_PLUGIN_NAME{"WeightOnlyQuantMatmul"};
PluginFieldCollection WeightOnlyQuantMatmulPluginCreator::mFC{};
std::vector<PluginField> WeightOnlyQuantMatmulPluginCreator::mPluginAttributes;

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, int weightTypeId)
{
    init(type, weightTypeId);
}

// Parameterized constructor
WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    nvinfer1::DataType type;
    int weightTypeId = 0;
    read(d, type);
    read(d, weightTypeId);
    init(type, weightTypeId);
    PLUGIN_ASSERT(d == a + length);
}

void WeightOnlyQuantMatmulPlugin::init(nvinfer1::DataType type, int weightTypeId)
{
    mType = type;
    mWeightTypeId = weightTypeId;
    if (mType == nvinfer1::DataType::kHALF && mWeightTypeId == 1)
    {
        m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<half, uint8_t>>();
    }
    else if (mType == nvinfer1::DataType::kHALF && mWeightTypeId == 2)
    {
        m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t>>();
    }
    else
    {
        PLUGIN_ASSERT(false);
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* WeightOnlyQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyQuantMatmulPlugin(mType, mWeightTypeId);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs WeightOnlyQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // input [m1, m2, m3, ... , k]
    // weight [k, n/4] for int8, [k, n/8] for int4

    try
    {
        PLUGIN_ASSERT(nbInputs == 3);
        PLUGIN_ASSERT(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        const int nbDimsB = inputs[1].nbDims;
        PLUGIN_ASSERT(nbDimsA >= 2);
        PLUGIN_ASSERT(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        if (mWeightTypeId == 1)
        {
            // int8 weight only quant
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() * 4);
        }
        else
        {
            // int4 weight only quant
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() * 8);
        }
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyQuantMatmulPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[0].type == mType && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // FIXME
        // Dirty hack to overcome TRT int8/int4 limitatition with plugins
        // Weights are required to be float, but will be reinterpreted as int8/int4 in enqueue
        // Weights stored in checkpoint should have int8/int4 type
        // Because of the reinterpretation, input weights have shape 4/8 times smaller than required
        // in_channels has to be divisable by 4/8

        return inOut[1].type == nvinfer1::DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
        return inOut[2].type == mType && inOut[2].format == TensorFormat::kLINEAR;
    case 3:
        // out
        return inOut[3].type == mType && inOut[3].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }
}

void WeightOnlyQuantMatmulPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    int maxM = 1;
    for (int ii = 0; ii < in[0].max.nbDims - 1; ++ii)
    {
        maxM *= in[0].max.d[ii];
    }
    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[1].max.d[1] * (mWeightTypeId == 1 ? 4 : 8);
    m_workspaceMaxSize = m_weightOnlyGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyQuantMatmulPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyQuantMatmulPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     mat1           [M1, M2,..., K]
    //     mat2           [K, N/4] for int8, [K, N/8] for int4
    //     scale_channels [N]
    // outputs
    //     mat [M, N]

    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[1].dims.d[1];
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    const int ws_size = m_weightOnlyGemmRunner->getWorkspaceSize(m, n, k);
    if (mType == nvinfer1::DataType::kHALF && mWeightTypeId == 1)
    {
        if (m == 1)
        {
            const half* bias = nullptr;
            tensorrt_llm::kernels::weight_only_gemv_launcher(reinterpret_cast<const half*>(inputs[0]),
                reinterpret_cast<const int8_t*>(inputs[1]), reinterpret_cast<const half*>(inputs[2]), bias,
                reinterpret_cast<half*>(outputs[0]), k, n * 4, tensorrt_llm::kernels::ActivationType::Identity,
                tensorrt_llm::kernels::QuantType::INT8_WEIGHT_ONLY, stream);
        }
        else
        {
            m_weightOnlyGemmRunner->gemm(reinterpret_cast<const half*>(inputs[0]),
                reinterpret_cast<const int8_t*>(inputs[1]), reinterpret_cast<const half*>(inputs[2]),
                reinterpret_cast<half*>(outputs[0]), m, n * 4, k, reinterpret_cast<char*>(workspace), ws_size, stream);
        }
    }
    else if (mType == nvinfer1::DataType::kHALF && mWeightTypeId == 2)
    {
        if (m == 1)
        {
            const half* bias = nullptr;
            tensorrt_llm::kernels::weight_only_gemv_launcher(reinterpret_cast<const half*>(inputs[0]),
                reinterpret_cast<const int8_t*>(inputs[1]), reinterpret_cast<const half*>(inputs[2]), bias,
                reinterpret_cast<half*>(outputs[0]), k, n * 8, tensorrt_llm::kernels::ActivationType::Identity,
                tensorrt_llm::kernels::QuantType::PACKED_INT4_WEIGHT_ONLY, stream);
        }
        else
        {
            m_weightOnlyGemmRunner->gemm(reinterpret_cast<const half*>(inputs[0]),
                reinterpret_cast<const cutlass::uint4b_t*>(inputs[1]), reinterpret_cast<const half*>(inputs[2]),
                reinterpret_cast<half*>(outputs[0]), m, n * 8, k, reinterpret_cast<char*>(workspace), ws_size, stream);
        }
    }
    else
    {
        assert(false);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType WeightOnlyQuantMatmulPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return mType;
}

// IPluginV2 Methods

const char* WeightOnlyQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyQuantMatmulPlugin::initialize() noexcept
{
    return 0;
}

void WeightOnlyQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + sizeof(nvinfer1::DataType);
}

void WeightOnlyQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mWeightTypeId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyQuantMatmulPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void WeightOnlyQuantMatmulPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* WeightOnlyQuantMatmulPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

WeightOnlyQuantMatmulPluginCreator::WeightOnlyQuantMatmulPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight_type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* WeightOnlyQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

const PluginFieldCollection* WeightOnlyQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    int weightTypeId;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "weight_type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            weightTypeId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new WeightOnlyQuantMatmulPlugin(type, weightTypeId);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call WeightOnlyQuantMatmulPlugin::destroy()
    try
    {
        auto* obj = new WeightOnlyQuantMatmulPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void WeightOnlyQuantMatmulPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* WeightOnlyQuantMatmulPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
