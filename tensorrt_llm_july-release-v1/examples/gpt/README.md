# GPT

This document explains how to build the [GPT](https://huggingface.co/gpt2) model using TensorRT-LLM and run on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Overview

The TensorRT-LLM GPT implementation can be found in [`tensorrt_llm/models/gpt/model.py`](../../tensorrt_llm/models/gpt/model.py). The TensorRT-LLM GPT example
code is located in [`examples/gpt`](./). There are four main files in that folder:

 * [`hf_gpt_convert.py`](./hf_gpt_convert.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
    format to the [FasterTransformer (FT)](https://github.com/NVIDIA/FasterTransformer) format,
 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the GPT model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the FT format. You can skip those two sections if you already have weights in the
FT format.

Note, also, that if your weights are neither in HF Transformers nor in FT formats, you will need to convert to the FT format. The script like
[`hf_gpt_convert.py`](./hf_gpt_convert.py) can serve as a starting point.

### 1. Download weights from HuggingFace Transformers

```bash
# Weights & config
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
```

### 2. Convert weights from HF Tranformers to FT format

TensorRT-LLM can directly load weights from FT. The [`hf_gpt_convert.py`](./hf_gpt_convert.py) script allows you to convert weights from HF Tranformers
format to FT format.

```bash
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
```

This script uses multiple processes to speed-up writing the model to disk. This may saturate your RAM depending on the model you are exporting.
In case that happens, you can reduce the number of processes with `--processes <num_processes>`. Set it to 1 for minimal RAM usage.

### 3. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a checkpoint in FT format. The checkpoint directory provides the model's weights, architecture configuration
and custom tokenizer if specified. If no checkpoint directories are specified, TensorRT-LLM will build engine(s) using random weights. When building with
random weights, you can use command-line arguments to modify the architecture: `--n_layer, --n_head, --n_embd, --hidden_act, --no_bias, ...`
Also, note that the number of TensorRT engines depends on the number of GPUs that will be used to run inference.

The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s). However, if you have more than one GPU in your system (of the same
model), you can enable parallel builds to accelerate the engine building process. For that, add the `--parallel_build` argument to the build command. Please
note that for the moment, the `parallel_build` feature cannot take advantage of more than a single node.

Examples of build invocations:

```bash
# Build a single-GPU float16 engine using FT weights.
# Enable the special TensorRT-LLM GPT Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin

# Build 8-GPU GPT-175B float16 engines using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
python3 build.py --world_size=8 \
                 --log_level=verbose \
                 --n_layer=96 \
                 --n_embd=12288 \
                 --n_head=96 \
                 --max_batch_size=256 \
                 --dtype float16 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir=gpt_175b 2>&1 | tee build.log

# Build 16-GPU GPT-530B float16 engines using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
python3 build.py --world_size=16 \
                 --log_level=info \
                 --n_layer=105 \
                 --n_embd=20480 \
                 --n_head=128 \
                 --max_batch_size=128 \
                 --max_input_len=128 \
                 --max_output_len=20 \
                 --dtype float16 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --use_layernorm_plugin \
                 --output_dir=gpt_530b 2>&1 | tee build.log
```

### GPT Variant - SantaCoder

The SantaCoder extends the existing GPT model with multi-query attention mechanism. The following example shows building a 4-GPU engine and running simple prompt to generate the implementation of `hello_world()`.

The main differences in this example are:
1. In model conversion `hf_gpt_convert.py` where extra option `--model santacoder` is required to allow converting checkpoint correctly
2. In engine execution `run.py` where `--tokenizer ./santacoder` needs to be specified to decode the output ids correctly.

```bash
git clone https://huggingface.co/bigcode/santacoder

python3 hf_gpt_convert.py -p 8 --model santacoder -i ./santacoder -o ./c-model/santacoder --tensor-parallelism 4 --storage-type float16

python3 build.py \
    --model_dir ./c-model/santacoder/4-gpu \
    --use_gpt_attention_plugin \
    --enable_context_fmha \
    --use_layernorm_plugin \
    --use_gemm_plugin \
    --parallel_build \
    --output_dir santacoder_outputs_tp4 \
    --world_size 4

mpirun -np 4 python3 run.py --engine_dir santacoder_outputs_tp4 --tokenizer ./santacoder --input_text "def print_hello_world():" --max_output_len 20
```

### GPT Variant - StarCoder

For StarCoder, the steps are similar execpt that `santacoder` is swapped with `starcoder`.

```bash
git clone https://huggingface.co/bigcode/starcoder

python3 hf_gpt_convert.py -p 8 --model starcoder -i ./starcoder -o ./c-model/starcoder --tensor-parallelism 4 --storage-type float16

python3 build.py \
    --model_dir ./c-model/starcoder/4-gpu \
    --use_gpt_attention_plugin \
    --enable_context_fmha \
    --use_layernorm_plugin \
    --use_gemm_plugin \
    --parallel_build \
    --output_dir starcoder_outputs_tp4 \
    --world_size 4

mpirun -np 4 python3 run.py --engine_dir starcoder_outputs_tp4 --tokenizer ./starcoder  --input_text "def print_hello_world():" --max_output_len 20
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for GPT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_gpt_attention_plugin float16`.

### 4. Run


#### Single node, single GPU

To run a TensorRT-LLM GPT model on a single GPU, you can use `python3`:

```bash
# Run the GPT-350M model on a single GPU.
python3 run.py --max_output_len=8
```

#### Single node, multiple GPUs

To run a model using multiple GPUs on a single node, you can use `mpirun` as:

```bash
# Run the GPT-175B model on a single node using multiple GPUs.
mpirun -np 8 python3 run.py --max_output_len=8 --engine_dir=gpt_175b
```

#### Multiple nodes, multiple GPUs using [Slurm](https://slurm.schedmd.com)

To run a model using multiple nodes, you should use a cluster manager like `Slurm`. The following section shows how to configure
TensorRT-LLM to execute on two nodes using Slurm.

We start by preparing an `sbatch` script called `tensorrt_llm_run.sub`. That script contains the following code (you must replace
the `<REPLACE ...>` strings with your own values):

```bash
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J <REPLACE WITH YOUR JOB's NAME>
#SBATCH -A <REPLACE WITH YOUR ACCOUNT's NAME>
#SBATCH -p <REPLACE WITH YOUR PARTITION's NAME>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix \
     --container-image <image> \
     --container-mounts <path>:<path> \
     --container-workdir <path> \
     --output logs/tensorrt_llm_%t.out \
     --error logs/tensorrt_llm_%t.error python3 -u run.py --max_output_len=8 --engine_dir <engine_dir>
```

Then, submit the job using:

```bash
sbatch tensorrt_llm_run.sub
```

You might have to contact your cluster's administrator to help you customize the above script.

## Summarization using the GPT model

The following section describes how to run a TensorRT-LLM GPT model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT model.

As previously explained, the first step is to convert from an HF checkpoint and build the TensorRT engines.

```bash
# Load the GPT2 weights from the HF hub.
pip install -r requirements.txt
rm -rf gpt2 && git clone https://huggingface.co/gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

# Convert the weights to FT format.
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2/fp16 --tensor-parallelism 1 --storage-type float16

# Build the model.
python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --use_layernorm_plugin \
                 --max_batch_size 8 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --output_dir trt_engine/gpt2/fp16/1-gpu/ \
                 --hidden_act gelu
```

The summarization can be done using the [`summarize.py`](./summarize.py) script as follows:

```bash
# Run the summarization task.
python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=gpt2 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=14
```

## SmoothQuant

This section explains how to use SmoothQuant on GPT models with TensorRT-LLM.

### Overview

[SmoothQuant](https://arxiv.org/abs/2211.10438) is a post-training quantization
(PTQ) method to quantize LLM models to INT8 for faster inference. As explained
in the article, SmoothQuant modifies a model to enable INT8 quantization
without significantly altering the accuracy.

#### Model Transformation

A LLM model is made of multiple matrix-multiplication operations (or GEMMs): `Y
= XW` where `X` of shape `[n, k]`, holds the activation (produced at run-time)
and `W`, of shape `[k, m]` are the learned weights. `Y`, of shape `[n, m]`, is
the matrix product of `X` and `W`.

SmoothQuant introduces scaling along the `k` dimension by defining a vector of
strictly positive coefficients `s`. `Y = X diag(s)^{-1} diag(s) W`. We now have
`Y = X'W'` where `X' = X diag(s)^{-1}` and `W' = diag(s) W`. This
transformation is introduced so the quantization behaves better. In *normal*
models, `X` tends to be ill-conditioned: it has mostly small-magnitude
coefficients, but also some outliers that makes quantization difficult.
Conversely, the re-scaled `X'` is better suited for INT8 conversion.

In this example, we only replace Attention's QKV and MLP's FC1 GEMMs to their
Smoothquant'd version since it is sufficient to maintain the accuracy for the
GPT model. During inference, `X'` is computed by fusing the channel-wise
multiplication by `diag(s)^{-1}` with the preceding layernorm's lambda and beta
parameters. `W'` is pre-computed and doesn't need additional modification
during inference.

#### INT8 inference

The INT8 quantization scheme used in TensorRT-LLM theoretically works on any
GPT model. However, Smoothquant'd models tend to produce more accurate results
with reduced precision.

INT8 inference modifies GEMMs `Y = XW` so that both `X` and `W` use INT8. The
matrix-multiplication is sped-up because of smaller weight size and fast matrix
products computation thanks to NVIDIA *Tensor Cores* operating on INT8 inputs.

During inference, X is transformed from its standard floating point (fp)
values: `X_{i8} <- X_{fp} * s_x`. This scaling puts `X` values in the INT8
range: `[-128, 127]`. Similarly, W is scaled, `W_{i8} <- W_{fp} * s_w` but that
operation is done at model export time, no need for subsequent operations at
run-time.

The optimised TensorRT-LLM GEMM implementation for SmoothQuant does the integer
matrix-multiplication `Y_{i32} <- X_{i8} W_{i8}` and rescales the result to its
original range `Y_{fp} <- Y_{i32} * (s_x)^{-1} * (s_w)^{-1}`. Note that
`Y_{i32}` isn't stored in memory, the re-scaling happens in the GEMM's epilogue
and only `Y_{fp}` gets saved.

By default `s_x` and `s_w` are single-value coefficients. This is the
*per-tensor* mode. Values for `s_x` and `s_w` are static, estimated at model
export time.

TensorRT-LLM also supports more elaborate modes:
 - per-channel: `s_w` is a fixed vector of size `[1, m]`. For that,
   TensorRT-LLM loads the adequately scaled version of of `W_{i8}` at model
   construction time.
 - per-token: `s_x` is a vector of size `[n, 1]` determined at run-time, based
   on the per-token (a.k.a. per-row) absolute maximum of `X`.
Users can mix-and-match per-channel and per-token options. Both tend to
increase the accuracy of the model at the cost of a slightly increased latency.

### Usage

#### SmoothQuant a HF model, export weights & scales for TensorRT-LLM

For SmoothQuant, [`hf_gpt_convert.py`](./hf_gpt_convert.py) features a
`--smoothquant, -sq` option. It must be set to a decimal value in `[0, 1]` and
corresponds to the `alpha` parameter in the [SmoothQuant
paper](https://arxiv.org/abs/2211.10438). Setting `-sq` will smooth the model
as explained in [model transformation](#model-transformation) and export the
scaling factors needed for INT8 inference.

Example:
```bash
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2-smooth --smoothquant 0.5 -t float16
```

#### Build TensorRT engine(s)

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode, as explained in [INT8
inference](#int8-inference).

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --model_dir=./c-model/gpt2-smooth/1-gpu \
                 --use_smooth_quant

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --model_dir=./c-model/gpt2-smooth/1-gpu \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel
```

### INT8 KV Cache, export weights & scales for TensorRT-LLM

For int8 kv cache, [`hf_gpt_convert.py`](./hf_gpt_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model as
explained in [model transformation](#model-transformation) and export the
scaling factors needed for INT8 KV cache inference.

Example:

```bash
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --calibrate-kv-cache -t float16
```

#### Build TensorRT engine(s)

[`build.py`](./build.py) add new options for the support of INT8 kv cache for models.
`--int8_kv_cache` forces KV cache to int8. We have to use int8 KV cache with GPT attention plugin
Examples of build invocations:

```bash
# Build model for GPT with int8 kv cache.
python3 build.py --model_dir=./c-model/gpt2/1-gpu \
                 --int8_kv_cache --use_gpt_attention_plugin float16
```

## GPT-Next

NVIDIA has released a GPT-like model with some architectural improvements, that you can find here: [https://huggingface.co/nvidia/GPT-2B-001](https://huggingface.co/nvidia/GPT-2B-001)
This architecture is also supported by TensorRT-LLM

### 1. Download weights from HuggingFace Transformers

```bash
wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo
```

### 2. Convert weights from HF Tranformers to FT format

TensorRT-LLM can convert `.nemo` to generic binary files with [`nemo_ckpt_convert.py`](./nemo_ckpt_convert.py) script. For example:

```bash
python3 nemo_ckpt_convert.py -i GPT-2B-001_bf16_tp1.nemo -o ./c-model/gpt-next-2B --tensor-parallelism 1 --storage-type bfloat16
```

### 3. Build TensorRT engine(s)

```bash
# Build a single-GPU float16 engine using FT weights.
# --use_gpt_attention_plugin must be set
python3 build.py --model_dir=./c-model/gpt-next-2B/1-gpu \
                 --dtype bfloat16 \
                 --use_gpt_attention_plugin

# Build GPT-Next architecture engines using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
python3 build.py --vocab_size=256000 \
                 --n_layer=24 \
                 --n_embd=1024 \
                 --n_head=16 \
                 --max_batch_size=256 \
                 --dtype float16 \
                 --no_bias \
                 --hidden_act swiglu \
                 --rotary_pct 0.5 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir=gpt-next-2B
```

### 4. Run

```bash
# Run the GPT-Next model on a single GPU. Use custom tokenizer.
python3 run.py --max_output_len=8 --vocab_file=./c-model/gpt-next-2B/1-gpu/tokenizer.model
```

## Prompt-tuning

For efficient fine-tuning, the NeMo framework allows you to learn virtual tokens to accomplish a downstream task. For more details, please read the
NeMo documentation [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/prompt_learning.html).

TensorRT-LLM supports inference with those virtual tokens. To enable it, pass the prompt embedding table's maximum size at build time with
`--max_prompt_embedding_table_size N`. For example:
```bash
# Build a GPT-Next model with prompt-tuning enabled
python3 build.py --model_dir=./c-model/gpt-next-8B/1-gpu --use_gpt_attention_plugin --max_prompt_embedding_table_size 100
```

You can now export the learned embedding table with:
```bash
python3 nemo_prompt_convert.py -i email_composition.nemo -o email_composition.npy
```
It'll give you a summary of the different tasks in the table, that you can specify at runtime.

Finally, you can run inference on pre-defined tokens:
```bash
python3 run.py --input_tokens input.csv --prompt_table email_composition.npy --tasks 0 --max_output_len=8 --vocab_file=./c-model/gpt-next-8B/1-gpu/tokenizer.model
```
