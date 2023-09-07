# GPT-NeoX

This document explains how to build the [GPT-NeoX](https://huggingface.co/EleutherAI/gpt-neox-20b) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM GPT-NeoX implementation can be found in [`tensorrt_llm/models/gptneox/model.py`](../../tensorrt_llm/models/gptneox/model.py). The TensorRT-LLM GPT-NeoX example
code is located in [`examples/gptneox`](./). There are three main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the GPT-NeoX model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

### 1. Download weights from HuggingFace (HF) Transformers

```bash
# Weights & config
sh get_weights.sh
```

### 2. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using HF weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose  \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_batch_size=16 \
                 --max_input_len=1024 \
                 --max_output_len=1024  \
                 --output_dir=gptneox_engine \
                 --model_dir=gptneox_model 2>&1 | tee build.log

# Build a float16 engine using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose  \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_batch_size=16 \
                 --max_input_len=1024 \
                 --max_output_len=1024  \
                 --output_dir=gptneox_engine_dummy_weights 2>&1 | tee build.log
```
#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for GPT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_gpt_attention_plugin float16`.

### 3. Run

To run a TensorRT-LLM GPT-NeoX model:

```bash
python3 run.py --max_output_len=50 --engine_dir=gptneox_engine
```

## Summarization using the GPT-NeoX model

The following section describes how to run a TensorRT-LLM GPT-NeoX model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT-NeoX model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`summarize.py`](./summarize.py) script as follows:

```bash
# Run the summarization task.
python3 summarize.py --engine_dir gptneox_engine \
                     --model_dir gptneox_model \
                     --batch_size 1 \
                     --test_trt_llm \
                     --tensorrt_llm_rouge1_threshold 14 \
                     --data_type fp16 \
                     --check_accuracy 2>&1 | tee summary_trt_llm.log

python3 summarize.py --engine_dir gptneox_engine \
                     --model_dir gptneox_model \
                     --batch_size 1 \
                     --test_hf \
                     --tensorrt_llm_rouge1_threshold 14 \
                     --data_type fp16 \
                     --check_accuracy 2>&1 | tee summary_hf.log
```
