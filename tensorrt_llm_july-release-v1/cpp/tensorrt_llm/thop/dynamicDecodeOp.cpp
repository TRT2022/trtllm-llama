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

#include "tensorrt_llm/thop/dynamicDecodeOp.h"

#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/torchAllocator.h"

namespace th = torch;

namespace tr = tensorrt_llm::runtime;
namespace tcc = tensorrt_llm::common::conversion;

namespace torch_ext
{

template <typename T>
FtDynamicDecode<T>::FtDynamicDecode(
    const size_t vocab_size, const size_t vocab_size_padded, const int tensor_para_size, const int pipeline_para_size)
    : vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
    TLLM_CHECK_WITH_INFO(vocab_size_padded_ % tensor_para_size == 0,
        tensorrt_llm::common::fmtstr(
            "vocab_size (%ld) is not multiple of tensor_para_size (%d).", vocab_size_padded_, tensor_para_size));

    allocator_ = new tensorrt_llm::thop::TorchAllocator();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    cudaDeviceProp prop;
    tensorrt_llm::common::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    dynamic_decode_layer_ = new tensorrt_llm::layers::DynamicDecodeLayer<T>(
        vocab_size_, vocab_size_padded_, stream, nullptr, allocator_, false, &prop_);
}

template <typename T>
FtDynamicDecode<T>::~FtDynamicDecode()
{
    delete dynamic_decode_layer_;
    delete allocator_;
}

namespace
{

template <typename T>
void safeInsert(th::optional<th::Tensor>& tensor, std::optional<std::vector<T>>& arg)
{
    using value_type = T;
    if (tensor.has_value())
    {
        auto ptr = get_ptr<value_type>(tensor.value());
        auto shape = convert_shape(tensor.value());
        size_t const size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        arg = std::vector<value_type>(ptr, ptr + size);
    }
};

template <typename T>
void safeUpdate(th::optional<th::Tensor>& tensor, std::optional<tc::Tensor>& arg)
{
    using value_type = T;
    if (tensor.has_value())
    {
        arg = convert_tensor<T>(tensor.value());
    }
};

template <typename T>
void safeUpdateScalar(th::optional<th::Tensor>& tensor, std::optional<T>& arg, std::string const& name)
{
    if (tensor.has_value())
    {
        auto accessor = tensor->accessor<T, 1>();
        TLLM_CHECK_WITH_INFO(accessor.size(0) == 1, name + " must be a scalar");
        arg = accessor[0];
    }

} // namespace

} // namespace

template <typename T>
void FtDynamicDecode<T>::setup(size_t batch_size, size_t beam_width, th::optional<th::Tensor> runtime_top_k_opt,
    th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
    th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> random_seed_opt,
    th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
    th::optional<th::Tensor> top_p_reset_ids_opt)
{
    // unused: length_penalty_opt, beam_search_diversity_rate_opt

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dynamic_decode_layer_->setStream(stream);

    SetupParams setupParams;

    safeInsert(runtime_top_k_opt, setupParams.runtime_top_k);
    safeInsert(runtime_top_p_opt, setupParams.runtime_top_p);
    safeInsert(temperature_opt, setupParams.temperature);
    safeInsert(repetition_penalty_opt, setupParams.repetition_penalty);
    safeInsert(presence_penalty_opt, setupParams.presence_penalty);
    safeInsert(min_length_opt, setupParams.min_length);
    safeInsert(random_seed_opt, setupParams.random_seed);
    safeInsert(top_p_decay_opt, setupParams.top_p_decay);
    safeInsert(top_p_min_opt, setupParams.top_p_min);
    safeInsert(top_p_reset_ids_opt, setupParams.top_p_reset_ids);

    safeUpdateScalar(
        beam_search_diversity_rate_opt, setupParams.beam_search_diversity_rate, "beam_search_diversity_rate");
    safeUpdateScalar(length_penalty_opt, setupParams.length_penalty, "length_penalty");

    dynamic_decode_layer_->setup(batch_size, beam_width, setupParams);
}

template <typename T>
void FtDynamicDecode<T>::forward(th::Tensor& logits, // (batch_size, beam_width, hidden_size)
    int step, int max_input_length, uint ite, int local_batch_size, th::Tensor end_id,
    th::optional<th::Tensor> runtime_top_k_opt, th::optional<th::Tensor> runtime_top_p_opt,
    th::optional<th::Tensor> temperature_opt, th::optional<th::Tensor> repetition_penalty_opt,
    th::optional<th::Tensor> presence_penalty_opt, th::optional<th::Tensor> min_length_opt,
    th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> beam_search_diversity_rate_opt,
    th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
    th::optional<th::Tensor> top_p_reset_ids_opt, th::optional<th::Tensor> embedding_bias_opt,
    th::optional<th::Tensor> input_lengths_opt, th::optional<th::Tensor> sequence_limit_length_opt,
    th::optional<th::Tensor> stop_words_list_opt, th::optional<th::Tensor> bad_words_list_opt,
    th::optional<th::Tensor> src_cache_indirection_opt,
    // Outputs
    th::Tensor& output_token_ids, th::Tensor& should_stop, th::optional<th::Tensor> finished_opt,
    th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
    th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> parent_ids_opt,
    th::optional<th::Tensor> tgt_cache_indirection_opt)
{
    auto const& logits_converted = convert_tensor<float>(logits);
    auto const& end_ids_converted = convert_tensor<int>(end_id);
    typename tensorrt_llm::layers::DynamicDecodeLayer<T>::ForwardParams forwardParams{
        step, static_cast<int>(ite), max_input_length, local_batch_size, logits_converted, end_ids_converted};

    safeUpdate<int>(src_cache_indirection_opt, forwardParams.src_cache_indirection);
    safeUpdate<int>(sequence_limit_length_opt, forwardParams.sequence_limit_length);
    safeUpdate<T>(embedding_bias_opt, forwardParams.embedding_bias);
    safeUpdate<int>(input_lengths_opt, forwardParams.input_lengths);
    safeUpdate<int>(bad_words_list_opt, forwardParams.bad_words_list);
    safeUpdate<int>(stop_words_list_opt, forwardParams.stop_words_list);

    auto const& output_ids_converted = convert_tensor<int>(output_token_ids);
    typename tensorrt_llm::layers::DynamicDecodeLayer<T>::OutputParams outputParams{output_ids_converted};

    safeUpdate<bool>(finished_opt, outputParams.finished);
    tr::BufferManager::ITensorPtr finished_sum;
    std::int32_t* finished_sum_host = nullptr;
    if (forwardParams.sequence_limit_length && outputParams.finished.has_value())
    {
        finished_sum = tr::BufferManager::pinned(tr::ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
        outputParams.finished_sum = tcc::toTllmTensor(*finished_sum);
        finished_sum_host = tr::bufferCast<std::int32_t>(*finished_sum);
        *finished_sum_host = 0;
    }
    safeUpdate<int>(sequence_lengths_opt, outputParams.sequence_length);
    safeUpdate<int>(parent_ids_opt, outputParams.parent_ids);
    safeUpdate<float>(cum_log_probs_opt, outputParams.cum_log_probs);
    safeUpdate<float>(output_log_probs_opt, outputParams.output_log_probs);
    safeUpdate<int>(tgt_cache_indirection_opt, outputParams.tgt_cache_indirection);

    dynamic_decode_layer_->forward(outputParams, forwardParams);
    if (finished_sum_host)
    {
        auto const numToFinish = outputParams.finished->size();
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(dynamic_decode_layer_->getStream()));
        auto should_stop_accessor = should_stop.accessor<bool, 1>();
        should_stop_accessor[0] = numToFinish == *finished_sum_host;
    }
}

DynamicDecodeOp::DynamicDecodeOp(const int64_t vocab_size, const int64_t vocab_size_padded,
    const int64_t tensor_para_size, const int64_t pipeline_para_size, at::ScalarType scalar_type)
    : vocab_size_(static_cast<size_t>(vocab_size))
    , vocab_size_padded_(static_cast<size_t>(vocab_size_padded))
    , tensor_para_size_(static_cast<int>(tensor_para_size))
    , pipeline_para_size_(static_cast<int>(pipeline_para_size))
    , scalar_type_(scalar_type)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    createInstance();
}

DynamicDecodeOp::~DynamicDecodeOp()
{
    // Do nothing.
}

void DynamicDecodeOp::createInstance()
{
    dynamic_decode_.reset();
    switch (scalar_type_)
    {
    case at::ScalarType::Float:
        dynamic_decode_ = std::make_unique<FtDynamicDecode<float>>(
            vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
        break;
    case at::ScalarType::Half:
        dynamic_decode_ = std::make_unique<FtDynamicDecode<half>>(
            vocab_size_, vocab_size_padded_, tensor_para_size_, pipeline_para_size_);
        break;
    default: throw std::runtime_error("Wrong tensor type.");
    }
}

void DynamicDecodeOp::setup(int64_t batch_size, int64_t beam_width, th::optional<th::Tensor> runtime_top_k_opt,
    th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
    th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> random_seed_opt,
    th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
    th::optional<th::Tensor> top_p_reset_ids_opt)
{
    // TODO: Revise DynamicDecodeLayer and make the decode arguments consistent.
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_k_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_p_opt, torch::kFloat);

    CHECK_OPTIONAL_CPU_INPUT(temperature_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetition_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(presence_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(min_length_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(length_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(beam_search_diversity_rate_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(random_seed_opt, torch::kInt64);
    CHECK_OPTIONAL_INPUT(top_p_decay_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_min_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_reset_ids_opt, torch::kInt32);

    dynamic_decode_->setup(static_cast<size_t>(batch_size), static_cast<size_t>(beam_width), runtime_top_k_opt,
        runtime_top_p_opt, temperature_opt, repetition_penalty_opt, presence_penalty_opt, min_length_opt,
        length_penalty_opt, beam_search_diversity_rate_opt, random_seed_opt, top_p_decay_opt, top_p_min_opt,
        top_p_reset_ids_opt);
}

th::Tensor DynamicDecodeOp::forward(th::Tensor logits, int64_t step, int64_t max_input_length, int64_t ite,
    int64_t local_batch_size, th::Tensor end_id, th::optional<th::Tensor> runtime_top_k_opt,
    th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
    th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
    th::optional<th::Tensor> min_length_opt, th::optional<th::Tensor> length_penalty_opt,
    th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> top_p_decay_opt,
    th::optional<th::Tensor> top_p_min_opt, th::optional<th::Tensor> top_p_reset_ids_opt,
    th::optional<th::Tensor> embedding_bias_opt,
    th::optional<th::Tensor> input_lengths_opt, // length of input contexts.
    th::optional<th::Tensor> sequence_limit_length_opt, th::optional<th::Tensor> stop_words_list_opt,
    th::optional<th::Tensor> bad_words_list_opt, th::optional<th::Tensor> src_cache_indirection_opt,
    // output buffers.
    th::Tensor output_token_ids, th::optional<th::Tensor> finished_opt,
    th::optional<th::Tensor> seuqence_lengths_opt, // length of the current sequences.
    th::optional<th::Tensor> cum_log_probs_opt, th::optional<th::Tensor> output_log_probs_opt,
    th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt)
{
    // Input Arguments:
    //     logits: [batch_size, beam_width, vocab_size_padded], T
    //     end_id: [batch_size], int, optional
    //     runtime_top_k: [batch_size], int, optional
    //     runtime_top_p: [batch_size], float, optional
    //     temperature: [batch_size], float, optional
    //     repetition_penalty: [batch_size], float, optional
    //     presence_penalty: [batch_size], float, optional,
    //         only one of repetition_penalty and presence_penalty is allowed.
    //     min_length: [batch_size], int, optional
    //     length_penalty: [batch_size], float, optional
    //     beam_search_diversity_rate: [batch_size], float, optional
    //     top_p_decay: [batch_size], float, optional
    //     top_p_min: [batch_size], float, optional
    //     top_p_reset_ids: [batch_size], int, optional
    //     embedding_bias: [vocab_size_padded], T, optional
    //     input_lengths: [batch_size * beam_width], int, optional
    //     sequence_limit_length: [batch_size], int, optional
    //     stop_words_list: [batch_size, 2, stop_words_length], int, optional
    //     bad_words_list: [2, stop_words_length], int, optional
    //     src_cache_indirection: [local_batch_size, beam_width, memory_length],
    //     int, optional output_token_ids: [max_seq_length, batch_size,
    //     beam_width], int finished: [batch_size * beam_width], bool, optional
    //     sequence_lengths: [batch_size * beam_width], int, optional
    //     cum_log_probs: [batch_size * beam_width], float, optional
    //     output_log_probs: [gen_length, batch_size, beam_width], float, optional
    //     parent_ids: [gen_length, batch_size, beam_width], float, optional
    //     tgt_cache_indirection: [local_batch_size, beam_width, memory_length],
    //     float, optional

    CHECK_INPUT(logits, scalar_type_);
    TLLM_CHECK_WITH_INFO(logits.dim() == 3,
        tensorrt_llm::common::fmtstr("logits is of shape (batch_size, beam_width, vocab_size_padded), "
                                     "but got dim=%d shape=%s",
            (int) logits.dim(), tensorrt_llm::common::vec2str(convert_shape(logits)).c_str())
            .c_str());
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(logits.size(2)) == vocab_size_padded_,
        tensorrt_llm::common::fmtstr("logits is of shape (batch_size, beam_width, "
                                     "vocab_size(%ld)), but got the last dim=%ld.",
            vocab_size_padded_, static_cast<size_t>(logits.size(2))));

    CHECK_INPUT(end_id, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_k_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(runtime_top_p_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(temperature_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(repetition_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(presence_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(min_length_opt, torch::kInt32);
    CHECK_OPTIONAL_CPU_INPUT(length_penalty_opt, torch::kFloat);
    CHECK_OPTIONAL_CPU_INPUT(beam_search_diversity_rate_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_decay_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_min_opt, torch::kFloat);
    CHECK_OPTIONAL_INPUT(top_p_reset_ids_opt, torch::kInt32);

    CHECK_OPTIONAL_INPUT(input_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(sequence_limit_length_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(stop_words_list_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(bad_words_list_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(src_cache_indirection_opt, torch::kInt32);

    CHECK_INPUT(output_token_ids, torch::kInt32);
    CHECK_OPTIONAL_INPUT(finished_opt, torch::kBool);
    CHECK_OPTIONAL_INPUT(seuqence_lengths_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(cum_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(output_log_probs_opt, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(parent_ids_opt, torch::kInt32);
    CHECK_OPTIONAL_INPUT(tgt_cache_indirection_opt, torch::kInt32);

    th::Tensor should_stop = torch::zeros({1}, torch::dtype(torch::kBool).requires_grad(false));

    dynamic_decode_->forward(
        // Inputs
        logits, static_cast<int>(step), static_cast<int>(max_input_length), static_cast<uint>(ite),
        static_cast<int>(local_batch_size), end_id, runtime_top_k_opt, runtime_top_p_opt, temperature_opt,
        repetition_penalty_opt, presence_penalty_opt, min_length_opt, length_penalty_opt,
        beam_search_diversity_rate_opt, top_p_decay_opt, top_p_min_opt, top_p_reset_ids_opt, embedding_bias_opt,
        input_lengths_opt, sequence_limit_length_opt, stop_words_list_opt, bad_words_list_opt,
        src_cache_indirection_opt,
        // Outputs
        output_token_ids, should_stop, finished_opt, seuqence_lengths_opt, cum_log_probs_opt, output_log_probs_opt,
        parent_ids_opt, tgt_cache_indirection_opt);

    return should_stop;
}

} // namespace torch_ext

static auto fasterTransformerGptContextDecoderTHS
    = torch::jit::class_<torch_ext::DynamicDecodeOp>("FasterTransformer", "DynamicDecodeOp")
          .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, at::ScalarType>())
          .def("setup", &torch_ext::DynamicDecodeOp::setup)
          .def("forward", &torch_ext::DynamicDecodeOp::forward);
