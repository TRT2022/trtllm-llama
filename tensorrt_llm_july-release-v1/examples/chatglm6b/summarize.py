import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModel, AutoTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger

from build import get_engine_name  # isort:skip


def TRTGPT(args, config):
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    world_size = config['builder_config']['tensor_parallel']
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])

    model_config = tensorrt_llm.runtime.ModelConfig(
        model_name="chatglm6b",
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('chatglm6b', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.ChatGLM6BHeadModelGenerationSession(
        model_config, engine_buffer, runtime_mapping)

    return decoder


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    hf_model_location = args.hf_model_location
    tokenizer = AutoTokenizer.from_pretrained(hf_model_location,
                                              padding_side='left',
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_cnn = load_dataset("ccdv/cnn_dailymail",
                               '3.0.0',
                               cache_dir=args.dataset_path)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    max_batch_size = args.batch_size

    # runtime parameters
    # repetition_penalty = 1
    top_k = 1
    output_len = args.output_len
    test_token_num = 923
    # top_p = 0.0
    # random_seed = 5
    temperature = 1

    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    if test_trt_llm:
        tensorrt_llm_gpt = TRTGPT(args, config)

    if test_hf:
        model = AutoModel.from_pretrained(hf_model_location,
                                          trust_remote_code=True)
        model.cuda()
        if args.data_type == 'fp16':
            model.half()

    def summarize_tensorrt_llm(datapoint):
        batch_size = len(datapoint['article'])

        line = copy.copy(datapoint['article'])
        line_encoded = []
        input_lengths = []
        for i in range(batch_size):
            line[i] = line[i] + ' TL;DR: '

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

            input_id = tokenizer.encode(line[i],
                                        return_tensors='pt').type(torch.int32)
            input_id = input_id[:, -test_token_num:]

            line_encoded.append(input_id)
            input_lengths.append(input_id.shape[-1])

        # do padding, should move outside the profling to prevent the overhead
        max_length = max(input_lengths)
        for i in range(batch_size):
            pad_size = max_length - input_lengths[i]

            pad = torch.ones([1, pad_size]).type(torch.int32) * pad_id
            line_encoded[i] = torch.cat([torch.IntTensor(line_encoded[i]), pad],
                                        axis=-1)

        line_encoded = torch.cat(line_encoded, axis=0).cuda()
        input_lengths = torch.IntTensor(input_lengths).type(torch.int32).cuda()

        sampling_config = tensorrt_llm.runtime.SamplingConfig(pad_id=pad_id,
                                                              end_id=end_id,
                                                              top_k=top_k)

        with torch.no_grad():
            tensorrt_llm_gpt.setup(line_encoded.size(0),
                                   max_input_length=line_encoded.size(1),
                                   max_new_tokens=output_len)

            output_ids = tensorrt_llm_gpt.decode(
                line_encoded,
                input_lengths,
                sampling_config,
            )

            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        output_beams_list = [
            tokenizer.batch_decode(output_ids[batch_idx, :,
                                              input_lengths[batch_idx]:],
                                   skip_special_tokens=True)
            for batch_idx in range(batch_size)
        ]
        return output_beams_list, output_ids[:, :, max_length:].tolist()

    def summarize_hf(datapoint):
        batch_size = len(datapoint['article'])
        if batch_size > 1:
            logger.warning(
                f"HF does not support batch_size > 1 to verify correctness due to padding. Current batch size is {batch_size}"
            )

        line = copy.copy(datapoint['article'])
        for i in range(batch_size):
            line[i] = line[i] + ' TL;DR: '

            line[i] = line[i].strip()
            line[i] = line[i].replace(" n't", "n't")

        line_encoded = tokenizer(line,
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True)["input_ids"].type(torch.int64)

        line_encoded = line_encoded[:, -test_token_num:]
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=len(line_encoded[0]) +
                                    output_len,
                                    top_k=top_k,
                                    temperature=temperature,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        tokens_list = output[:, len(line_encoded[0]):].tolist()
        output_lines_list = tokenizer.batch_decode(
            output[:, len(line_encoded[0]):], skip_special_tokens=True)

        return output_lines_list, tokens_list

    if test_trt_llm:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_tensorrt_llm(datapoint)
        if runtime_rank == 0:
            logger.info(
                "---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Article : {datapoint['article']}")
            logger.info(f"\n Highlights : {datapoint['highlights']}")
            logger.info(f"\n Summary : {summary}")
            logger.info(
                "---------------------------------------------------------")

    if test_hf:
        datapoint = dataset_cnn['test'][0:1]
        summary, _ = summarize_hf(datapoint)
        logger.info("---------------------------------------------------------")
        logger.info("HF Generated : ")
        logger.info(f" Article : {datapoint['article']}")
        logger.info(f"\n Highlights : {datapoint['highlights']}")
        logger.info(f"\n Summary : {summary}")
        logger.info("---------------------------------------------------------")

    metric_tensorrt_llm = load_metric("rouge")
    metric_tensorrt_llm.seed = 0
    metric_hf = load_metric("rouge")
    metric_hf.seed = 0

    ite_count = 0
    data_point_idx = 0
    while (data_point_idx < len(dataset_cnn['test'])) and (ite_count <
                                                           args.max_ite):
        if runtime_rank == 0:
            logger.debug(
                f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
            )
        datapoint = dataset_cnn['test'][data_point_idx:(data_point_idx +
                                                        max_batch_size)]

        if test_trt_llm:
            profiler.start('tensorrt_llm')
            summary_tensorrt_llm, tokens_tensorrt_llm = summarize_tensorrt_llm(
                datapoint)
            profiler.stop('tensorrt_llm')

        if test_hf:
            profiler.start('hf')
            summary_hf, tokens_hf = summarize_hf(datapoint)
            profiler.stop('hf')

        if runtime_rank == 0:
            if test_trt_llm:
                for batch_idx in range(len(summary_tensorrt_llm)):
                    metric_tensorrt_llm.add_batch(
                        predictions=[summary_tensorrt_llm[batch_idx]],
                        references=[datapoint['highlights'][batch_idx]])
            if test_hf:
                metric_hf.add_batch(predictions=[summary_hf],
                                    references=[datapoint['highlights']])

            logger.debug('-' * 100)
            logger.debug(f"Article : {datapoint['article']}")
            if test_trt_llm:
                logger.debug(f'TensorRT-LLM Summary: {summary_tensorrt_llm}')
            if test_hf:
                logger.debug(f'HF Summary: {summary_hf}')
            logger.debug(f"highlights : {datapoint['highlights']}")

        data_point_idx += max_batch_size
        ite_count += 1

    if runtime_rank == 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            computed_metrics_tensorrt_llm = metric_tensorrt_llm.compute()
            logger.info(
                f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )
            for key in computed_metrics_tensorrt_llm.keys():
                logger.info(
                    f'{key} : {computed_metrics_tensorrt_llm[key].mid[2]*100}')

            if args.check_accuracy:
                assert computed_metrics_tensorrt_llm['rouge1'].mid[
                    2] * 100 > args.tensorrt_llm_rouge1_threshold
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            computed_metrics_hf = metric_hf.compute()
            logger.info(
                f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)'
            )
            for key in computed_metrics_hf.keys():
                logger.info(f'{key} : {computed_metrics_hf[key].mid[2]*100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_location', type=str, default='pyTorchModel')
    parser.add_argument(
        '--tokenizer',
        default=None,
        help='tokenizer path; defaults to hf_model_location if left unspecified'
    )
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_trt_llm', action='store_true')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp16')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='trtModel')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=20)
    parser.add_argument('--output_len', type=int, default=100)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--tensorrt_llm_rouge1_threshold',
                        type=float,
                        default=15.0)

    args = parser.parse_args()
    main(args)
