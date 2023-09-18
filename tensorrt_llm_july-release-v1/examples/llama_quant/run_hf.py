'''
run the huggingface checkpoint of llama and test the latency
美迪康-北航AI Lab

'''

import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForCausalLM, LlamaTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--hf_model_location', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'info',
    hf_model_location: str = 'llama_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    tokenizer_dir: str = None,
    num_beams: int = 1,
):

    data_type = "fp16"
    top_k = 1
    temperature = 1
    model = AutoModelForCausalLM.from_pretrained(hf_model_location)

    if data_type == 'fp16':
        model.half()
    model.cuda()

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]



    latencys = []
    for i in range(55):
        profiler.start(str(i))
        line_encoded = tokenizer(input_text,
                                return_tensors='pt',
                                padding=True,
                                truncation=True)["input_ids"].type(torch.int64)
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=len(line_encoded[0]) + max_output_len,
                                    top_k=top_k,
                                    temperature=temperature,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=num_beams,
                                    num_return_sequences=num_beams,
                                    early_stopping=True)
        torch.cuda.synchronize()

        tokens_list = output[:, len(line_encoded[0]):].tolist()
        output = output.reshape([1, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(output[:, i, len(line_encoded[0]):],
                                    skip_special_tokens=True)
            for i in range(num_beams)
        ]

        print(f'Input: \"{input_text}\"')
        print(f'Output: \"{output_lines_list}\"')

        profiler.stop(str(i))
        latencys.append(profiler.elapsed_time_in_sec(str(i)))
    
    print(latencys)
    print(f'llama-hf-run (mean latency: {np.mean(latencys[5:])} sec)')
    
    return 

if __name__ == "__main__":
    args = parse_arguments()
    generate(**vars(args))
