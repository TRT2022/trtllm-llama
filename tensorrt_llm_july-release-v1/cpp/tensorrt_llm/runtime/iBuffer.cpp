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

#include "tensorrt_llm/runtime/iBuffer.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferView.h"

#include <cuda_runtime_api.h>

#include <memory>

using namespace tensorrt_llm::runtime;

MemoryType IBuffer::memoryType(void const* data)
{
    cudaPointerAttributes attributes{};
    TLLM_CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));
    switch (attributes.type)
    {
    case cudaMemoryTypeHost: return MemoryType::kPINNED;
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return MemoryType::kGPU;
    case cudaMemoryTypeUnregistered: return MemoryType::kCPU;
    default: TLLM_THROW("Unsupported memory type");
    }
}

IBuffer::UniquePtr IBuffer::slice(IBuffer::SharedPtr const& buffer, std::size_t offset, std::size_t size)
{
    return std::make_unique<BufferView>(buffer, offset, size);
}

IBuffer::UniquePtr IBuffer::wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity)
{
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    IBuffer::UniquePtr result;
    auto const capacityInBytes = capacity * sizeOfType(type);
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericBuffer( // NOLINT(modernize-make-unique)
            capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer(capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer(capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Unknown memory type");
    }
    result->resize(size);
    return result;
}
