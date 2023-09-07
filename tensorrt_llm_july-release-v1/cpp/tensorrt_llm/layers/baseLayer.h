/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/tensor.h"

namespace tensorrt_llm
{
namespace layers
{

class BaseLayer
{
public:
    BaseLayer(cudaStream_t stream, tensorrt_llm::common::cublasMMWrapper* cublas_wrapper,
        tensorrt_llm::common::IAllocator* allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop = nullptr)
        : stream_(stream)
        , cublas_wrapper_(cublas_wrapper)
        , allocator_(allocator)
        , cuda_device_prop_(cuda_device_prop)
        , is_free_buffer_after_forward_(is_free_buffer_after_forward){};
    virtual ~BaseLayer() = default;

    virtual cudaStream_t getStream()
    {
        return stream_;
    }

    virtual void setStream(cudaStream_t stream)
    {
        stream_ = stream;
    }

protected:
    // device environments
    cudaStream_t stream_;
    tensorrt_llm::common::cublasMMWrapper* cublas_wrapper_;
    tensorrt_llm::common::IAllocator* allocator_;
    cudaDeviceProp* cuda_device_prop_ = nullptr;

    bool is_free_buffer_after_forward_;
    bool is_allocate_buffer_ = false; // TODO (bhsueh) to be deprecated
};

} // namespace layers
} // namespace tensorrt_llm
