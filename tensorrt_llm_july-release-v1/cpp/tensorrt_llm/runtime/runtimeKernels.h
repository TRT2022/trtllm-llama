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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime::kernels
{

template <typename T>
void invokeFill(IBuffer& buffer, T value, CudaStream const& stream);

template <typename T>
void invokeAdd(IBuffer& buffer, T value, CudaStream const& stream);

template <typename T>
void invokeSet2(IBuffer& buffer, T value0, T value1, CudaStream const& stream);

void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream);

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType outputOffset, CudaStream const& stream);

void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType inputOffset, CudaStream const& stream);

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream);

void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType maxInputLength, CudaStream const& stream);

void invokeBuildAttentionMask(ITensor& attentionMask, SizeType padId, CudaStream const& stream);

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream);

void invokeCopyInputToOutput(
    ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths, SizeType padId, CudaStream const& stream);

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType maxInputLength, SizeType padId, CudaStream const& stream);

template <typename T>
void invokeTileTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);

} // namespace tensorrt_llm::runtime::kernels
