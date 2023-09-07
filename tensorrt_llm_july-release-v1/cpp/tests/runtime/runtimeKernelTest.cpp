/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <NvInferRuntime.h>

#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

using TensorPtr = std::shared_ptr<ITensor>;
using BufferPtr = std::shared_ptr<IBuffer>;

class RuntimeKernelTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount == 0)
            GTEST_SKIP();

        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    void TearDown() override {}

    int mDeviceCount;
    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
};

TEST_F(RuntimeKernelTest, FillInt32)
{
    SizeType constexpr value{3};
    SizeType constexpr size{123};
    auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT32);
    kernels::invokeFill(*buffer, value, *mStream);
    auto bufferHost = mManager->copyFrom(*buffer, MemoryType::kCPU);
    auto bufferPtr = bufferCast<SizeType>(*bufferHost);
    std::vector<SizeType> expected(buffer->getSize(), value);

    auto anyMismatch = false;
    for (std::size_t i = 0; i < buffer->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected[i]) << "Error at index " << i;
        anyMismatch |= bufferPtr[i] != expected[i];
    }
    buffer.release();
    ASSERT_FALSE(anyMismatch);

    auto tensor = mManager->gpu(ITensor::makeShape({size, size}), nvinfer1::DataType::kINT32);
    kernels::invokeFill(*tensor, value, *mStream);
    auto tensorHost = mManager->copyFrom(*tensor, MemoryType::kCPU);
    auto tensorPtr = bufferCast<SizeType>(*tensorHost);
    expected.clear();
    expected.resize(tensor->getSize(), value);

    anyMismatch = false;
    for (std::size_t i = 0; i < tensor->getSize(); ++i)
    {
        EXPECT_EQ(tensorPtr[i], expected[i]) << "Error at index " << i;
        anyMismatch |= tensorPtr[i] != expected[i];
    }
    tensor.release();
    ASSERT_FALSE(anyMismatch);
}

TEST_F(RuntimeKernelTest, AddInt32)
{
    SizeType constexpr value{3};
    SizeType constexpr size{123};
    auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT32);
    mManager->setZero(*buffer);
    kernels::invokeAdd(*buffer, value, *mStream);
    kernels::invokeAdd(*buffer, value, *mStream);
    auto bufferHost = mManager->copyFrom(*buffer, MemoryType::kCPU);
    auto bufferPtr = bufferCast<SizeType>(*bufferHost);
    std::vector<SizeType> expected(buffer->getSize(), 2 * value);

    auto anyMismatch = false;
    for (std::size_t i = 0; i < buffer->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected[i]) << "Error at index " << i;
        anyMismatch |= bufferPtr[i] != expected[i];
    }
    buffer.release();
    ASSERT_FALSE(anyMismatch);

    auto tensor = mManager->gpu(ITensor::makeShape({size, size}), nvinfer1::DataType::kINT32);
    mManager->setZero(*tensor);
    kernels::invokeAdd(*tensor, value, *mStream);
    kernels::invokeAdd(*tensor, value, *mStream);
    auto tensorHost = mManager->copyFrom(*tensor, MemoryType::kCPU);
    auto tensorPtr = bufferCast<SizeType>(*tensorHost);
    expected.clear();
    expected.resize(tensor->getSize(), 2 * value);

    anyMismatch = false;
    for (std::size_t i = 0; i < tensor->getSize(); ++i)
    {
        EXPECT_EQ(tensorPtr[i], expected[i]) << "Error at index " << i;
        anyMismatch |= tensorPtr[i] != expected[i];
    }
    tensor.release();
    ASSERT_FALSE(anyMismatch);
}

TEST_F(RuntimeKernelTest, Set2Int32)
{
    std::vector<SizeType> const values{3, 5};
    auto size = static_cast<SizeType>(values.size());
    auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT32);
    mManager->setZero(*buffer);
    kernels::invokeSet2(*buffer, values[0], values[1], *mStream);
    auto bufferHost = mManager->copyFrom(*buffer, MemoryType::kCPU);
    auto bufferPtr = bufferCast<SizeType>(*bufferHost);

    for (std::size_t i = 0; i < buffer->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], values[i]) << "Error at index " << i;
    }
    buffer.release();
}

TEST_F(RuntimeKernelTest, Transpose)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    auto input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);

    kernels::invokeTranspose(*output, *input, *mStream);

    auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
    auto outputHostData = bufferCast<SizeType>(*outputHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType i = 0; i < rowSize; ++i)
        {
            auto const inputIndex = tc::flat_index2(b, i, rowSize);
            auto const outputIndex = tc::flat_index2(i, b, batchSize);
            EXPECT_EQ(outputHostData[outputIndex], inputHost[inputIndex]) << "Error at index (" << b << ',' << i << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithOutputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (SizeType sliceId = 0; sliceId < batchSize; ++sliceId)
    {
        auto inputView = ITensor::slice(input, sliceId, 1);
        kernels::invokeTransposeWithOutputOffset(*output, *inputView, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
        auto outputHostData = bufferCast<SizeType>(*outputHost);

        for (SizeType b = 0; b < batchSize; ++b)
        {
            for (SizeType i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(b, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, b, batchSize);
                auto expected = b <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithInputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (SizeType sliceId = 0; sliceId < rowSize; ++sliceId)
    {
        auto outputView = ITensor::slice(output, sliceId, 1);
        kernels::invokeTransposeWithInputOffset(*outputView, *input, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
        auto outputHostData = bufferCast<SizeType>(*outputHost);

        for (SizeType b = 0; b < batchSize; ++b)
        {
            for (SizeType i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(b, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, b, batchSize);
                auto expected = i <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, BuildTokenMask)
{
    SizeType constexpr batchSize{7};
    std::vector<SizeType> inputLengthsVec(batchSize);
    std::iota(inputLengthsVec.begin(), inputLengthsVec.end(), 3);
    auto const maxInputLength = *std::max_element(inputLengthsVec.begin(), inputLengthsVec.end());

    SizeType constexpr maxNewTokens{1};
    auto const maxSeqLength = maxInputLength + maxNewTokens;

    TensorPtr inputLengths = mManager->copyFrom(inputLengthsVec, ITensor::makeShape({batchSize, 1}), MemoryType::kGPU);

    TensorPtr tokenMask = mManager->gpu(ITensor::makeShape({batchSize, maxSeqLength}), nvinfer1::DataType::kINT32);
    kernels::invokeBuildTokenMask(*tokenMask, *inputLengths, maxInputLength, *mStream);

    std::vector<SizeType> tokenMaskVec(tokenMask->getSize());
    mManager->copy(*tokenMask, tokenMaskVec.data());

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < maxSeqLength; ++j)
        {
            auto const index = i * maxSeqLength + j;
            if (j < inputLengthsVec[i])
                EXPECT_EQ(tokenMaskVec[index], 0) << "tokenMask should be 0 up to inputLengths[i]";
            else if (j < maxInputLength)
                EXPECT_EQ(tokenMaskVec[index], 1) << "tokenMask should be 1 up to maxInputLength";
            else
                EXPECT_EQ(tokenMaskVec[index], 0) << "tokenMask should be 0 after maxInputLength";
        }
    }
}

TEST_F(RuntimeKernelTest, BuildAttentionMask)
{
    SizeType constexpr batchSize{1};
    SizeType constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<SizeType>(input.size());

    TensorPtr inputIds = mManager->copyFrom(input, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, MemoryType::kGPU);
    kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    std::vector<SizeType> attentionMaskVec(attentionMask->getSize());
    mManager->copy(*attentionMask, attentionMaskVec.data());

    std::vector<std::int32_t> attentionMaskHost(input);
    std::for_each(attentionMaskHost.begin(), attentionMaskHost.end(), [padId](auto& x) { x = x != padId; });

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < maxInputLength; ++j)
        {
            auto const index = i * maxInputLength + j;
            EXPECT_EQ(attentionMaskVec[index], attentionMaskHost[index]) << "Error at index (" << i << ',' << j << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, ExtendAttentionMask)
{
    SizeType constexpr batchSize{1};
    SizeType constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<SizeType>(input.size());

    TensorPtr inputIds = mManager->copyFrom(input, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, MemoryType::kGPU);
    kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    auto attentionMaskHost = mManager->copyFrom(*attentionMask, MemoryType::kCPU);
    auto const* attentionMaskData = reinterpret_cast<SizeType const*>(attentionMaskHost->data());
    auto const shape = attentionMask->getShape();
    auto const length = shape.d[1];
    auto const newShape = ITensor::makeShape({shape.d[0], length + 1});
    std::vector<SizeType> attentionMaskVec(ITensor::volume(newShape));
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(attentionMaskData + i * length, attentionMaskData + (i + 1) * length,
            std::begin(attentionMaskVec) + i * (length + 1));
        attentionMaskVec[(i + 1) * (length + 1) - 1] = 1;
    }

    TensorPtr newAttentionMask
        = mManager->gpu(ITensor::makeShape({batchSize, maxInputLength + 1}), nvinfer1::DataType::kINT32);
    kernels::invokeExtendAttentionMask(*newAttentionMask, *attentionMask, *mStream);

    std::vector<SizeType> newAttentionMaskVec(newAttentionMask->getSize());
    mManager->copy(*newAttentionMask, newAttentionMaskVec.data());

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < length; ++j)
        {
            auto const oldIndex = i * length + j;
            auto const newIndex = i * (length + 1) + j;
            EXPECT_EQ(attentionMaskVec[oldIndex], newAttentionMaskVec[newIndex])
                << "Error at index (" << i << ',' << j << ')';
        }
        EXPECT_EQ(attentionMaskVec[(i + 1) * (length + 1) - 1], 1) << "Error at index (" << i << ',' << (-1) << ')';
    }
}

TEST_F(RuntimeKernelTest, CopyInputToOutput)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    auto const beamWidth = 5;
    SizeType constexpr maxNewTokens{3};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<std::int32_t> inputsHost(batchSize * maxInputLength);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.end(), inputsHost.begin() + i * maxInputLength);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    kernels::invokeCopyInputToOutput(*outputIds, *inputIds, *inputLengths, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i]) << "Error at index (" << b << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId) << "Error at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, CopyPackedInputToOutput)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    SizeType constexpr maxNewTokens{3};
    auto const beamWidth = 5;
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    std::vector<SizeType> inputOffsetsHost(batchSize + 1);
    std::inclusive_scan(inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
    auto const totalInputSize = inputOffsetsHost.back();

    std::vector<std::int32_t> inputsHost(totalInputSize);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.begin() + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    auto inputOffsets = std::shared_ptr(mManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32));
    mManager->setZero(*inputOffsets);
    kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, *mManager, *mStream);
    auto inputOffsetsHost2 = mManager->copyFrom(*inputOffsets, MemoryType::kCPU);

    for (std::size_t b = 0; b < inputOffsetsHost.size(); ++b)
    {
        EXPECT_EQ(inputOffsetsHost[b], inputOffsetsHost[b]) << "Error at index " << b;
    }

    kernels::invokeCopyPackedInputToOutput(*outputIds, *inputIds, *inputOffsets, maxInputLength, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId)
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, TileInt32)
{
    SizeType const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);

    kernels::invokeTileTensor<SizeType>(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto outputPtr = bufferCast<SizeType>(*outputHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = b * inputLength + i;
                auto const outputIdx = (b * beamWidth + beam) * inputLength + i;
                EXPECT_EQ(outputPtr[outputIdx], input[inputIdx])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, TileHalf)
{
    SizeType const beamWidth{3};

    std::vector<half> const input{
        28524.f, 287.f, 5093.f, 12.f, 23316.f, 4881.f, 11.f, 30022.f, 263.f, 8776.f, 355.f, 257.f};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);

    kernels::invokeTileTensor<half>(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto outputPtr = bufferCast<half>(*outputHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = b * inputLength + i;
                auto const outputIdx = (b * beamWidth + beam) * inputLength + i;
                EXPECT_EQ(outputPtr[outputIdx], input[inputIdx])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}
