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

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tmaDescriptor.h"

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

class MHARunner
{
public:
    MHARunner(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    MHARunner() = default;

    virtual ~MHARunner() = default;

    virtual void setup(const int b, const int s, const int total_seqlen) = 0;

    static bool fmha_supported(const int headSize, const int sm);

    virtual bool fmha_supported() = 0;

    virtual void setup_flags(
        const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask, const bool multi_query_attention)
        = 0;

    virtual void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) = 0;

    virtual bool isValid(int s) const = 0;

    virtual int getSFromMaxSeqLen(const int max_seq_len) = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workflow of fmha runner:
// 1. check if FMHA kernels are supported statically.
// 2. construct FMHA runner object.
// 3. setup_flags (used by all kernels).
// 4. setup runtime parameters (used by this specific case).
// 5. run the kernel (with all needed device pointers).

class FusedMHARunnerV2 : public MHARunner
{
public:
    FusedMHARunnerV2(const Data_type dataType, const int numHeads, const int headSize, const float qScaling);

    ~FusedMHARunnerV2() = default; // for pimpl

    void setup(const int b, const int s, const int total_seqlen) override;

    bool fmha_supported() override;

    void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) override;

    void setup_flags(const bool force_fp32_acc, const bool is_s_padded, const bool causal_mask,
        const bool multi_query_attention) override;

    bool isValid(int s) const override;

    int getSFromMaxSeqLen(const int max_seq_len) override;

private:
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

} // namespace kernels
} // namespace tensorrt_llm
