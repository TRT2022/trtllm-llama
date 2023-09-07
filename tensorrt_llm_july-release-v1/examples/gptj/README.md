# GPT-J

This document explains how to build the [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM GPT-J implementation can be found in [`tensorrt_llm/models/gptj/model.py`](../../tensorrt_llm/models/gptj/model.py). The TensorRT-LLM GPT-J example
code is located in [`examples/gptj`](./). There are three main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the GPT-J model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

### 1. Download weights from HuggingFace (HF) Transformers

```bash
# 1. Weights & config
git clone https://huggingface.co/EleutherAI/gpt-j-6b gptj_model
pushd gptj_model && \
  rm -f pytorch_model.bin && \
  wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/pytorch_model.bin && \
popd

# 2. Vocab and merge table
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt
```

### 2. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using
dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using HF weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose  \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128  \
                 --output_dir=gptj_engine \
                 --model_dir=gptj_model 2>&1 | tee build.log

# Build a float16 engine using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose  \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128  \
                 --output_dir=gptj_engine_dummy_weights 2>&1 | tee build.log

```
#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for GPT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_gpt_attention_plugin float16`.

### 3. Run


To run a TensorRT-LLM GPT-J model:

```bash
python3 run.py --max_output_len=50 --engine_dir=gptj_engine
```

## Summarization using the GPT-J model

The following section describes how to run a TensorRT-LLM GPT-J model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT-J model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`summarize.py`](./summarize.py) script as follows:

```bash
# Run the summarization task.
python3 summarize.py --engine_dir gptj_engine \
                     --model_dir gptj_model \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --tensorrt_llm_rouge1_threshold 14 \
                     --data_type fp16 \
                     --check_accuracy

```

## Known issues

- You must enable the LayerNorm plugin to build the engine for GPT-J when using TensorRT 8.6, this constraint is removed in TensorRT 9.0. To enable LayerNorm plugin, you should add `--use_layernorm_plugin <float16 or float32>` in the build.py, see build.py commands example above.
