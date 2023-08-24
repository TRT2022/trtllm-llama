<!-- <img src="docs/logo.png" align="right" alt="logo" height="180"  /> -->
<img src="assets/trt2023.jpeg" align="center" alt="logo"  />

## 基于TensorRT-LLM的LLaMa模型优化方案 :zap:
### LLaMa: Open and Efficient Foundation Language Models for TensorRT Hackathon 2023 <img src="assets/llama.png" alt="logo"  width=4%/>

[![](https://img.shields.io/badge/Github-TensorRT-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/%E9%98%BF%E9%87%8C%E5%A4%A9%E6%B1%A0-TensorRT%20Hackathon%202023-blue)](https://tianchi.aliyun.com/competition/entrance/532108/introduction?spm=a2c22.12281957.0.0.4c885d9bOexwJc)
[![](https://img.shields.io/badge/NVIDIA-TensorRT%20CookBook%20CN-blue)](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
[![](https://img.shields.io/badge/B%E7%AB%99-GodV%20TensorRT%E6%95%99%E7%A8%8B-blue)](https://www.bilibili.com/video/BV1jj411Z7wG/?spm=a2c22.12281978.0.0.49ed2274CQCrY7)
[![](https://img.shields.io/badge/Github-%20%E5%88%9D%E8%B5%9B%E6%80%BB%E7%BB%93-blue)](https://github.com/TRT2022/ControlNet_TensorRT)
[![](https://img.shields.io/badge/Github-LLaMa-blue)](https://github.com/facebookresearch/llama)


:alien: : **美迪康-北航AI Lab** 

### 0.⏳日志

<div align=center>

|时间点|提交内容|说明|
|-|-|-|
|2023-08-21|和NVIDIA导师团队确定优化方案：基于开源LLM的LLaMa模型推断加速优化|选题|
|2023-08-22|创建Github项目                                              |项目创建|
|2023-08-24|完成送分题作答                                               |送分题 |


☣️复赛调优阶段：2023年8月17日-9月21日
</div>


### 1.总述
---

ToDo


### 2.主要开发工作
---

#### 2.1 开发工作的难点

ToDo

#### 2.2 开发与优化过程

ToDo

### 3.优化效果
---

ToDo

### 4.Bug报告
---

<div align=center>

|:bug: Bug名称|Issue|是否被官方确认|说明|
|-|-|:-:|-|
|InstanceNormalization Plugin |<https://github.com/NVIDIA/TensorRT/issues/3165>|❎|官方暂未确认|


</div>

### 5.送分题答案
---

> 🔏 问题1： 请写出 `./tensorrt_llm_july-release-v1/examples/gpt/README` 里面 `“Single node, single GPU”` 部分如下命令的输出（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium) 

```shell
python3 run.py --max_output_len=8 
```
<details>
<summary>🔑点我查看 问题1 解析</summary>

0. 必要的Python Package安装

```shell
cd ./tensorrt_llm_july-release-v1/examples/gpt
pip3 install requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

1. 下载HuggingFace(HF)模型

```shell
# 下载HF模型
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2

# 更新.bin模型
cd gpt2
rm pytorch_model.bin model.safetensors
wget https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin
cd ..
```
2. 将HF weight转为FT weight

TensorRT-LLM 可以直接加载FastTransformer(FT)格式的模型weight文件，因此需要将HF weight转换为FT weight

```shell
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
```
运行上述代码，log中出现如下图所示结果，说明模型转换完成，并在`tensorrt_llm_july-release-v1/examples/gpt/c-model/gpt2/1-gpu`中存放了生成后的FT weight

<div align=center>
<img src="./assets/section5/p1.png"/>
</div>

3. 构建TensorRT-LLM engine

TensorRT-LLM engine的构建过程使用了FT weight和对应的配置文件（已经在第2步生成）和自定义的Tokenizer。过程中如果不指定模型权重路径，TensorRT-LLM默认随机初始化这些weight生成engine。

```shell
# single GPU float16 使用FT weight生成engine
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin
```
执行上述代码，log中出现如下所示结果，说明模型序列化完成，并将engine保存在`./tensorrt_llm_july-release-v1/examples/gpt/gpt_outputs`中

```
[08/24/2023-07:07:38] [TRT-LLM] [I] Total time of building gpt_float16_tp1_rank0.engine: 00:00:35
[08/24/2023-07:07:38] [TRT-LLM] [I] Config saved to gpt_outputs/config.json.
[08/24/2023-07:07:38] [TRT-LLM] [I] Serializing engine to gpt_outputs/gpt_float16_tp1_rank0.engine...
[08/24/2023-07:07:42] [TRT-LLM] [I] Engine serialized. Total time: 00:00:04
[08/24/2023-07:07:42] [TRT-LLM] [I] Timing cache serialized to gpt_outputs/model.cache
[08/24/2023-07:07:42] [TRT-LLM] [I] Total time of building all 1 engines: 00:00:50
```
⚠️注意： `build.py`支持多GPU并行构建TensorRT-LLM engine,详细的可以参考：`./tensorrt_llm_july-release-v1/examples/gpt/README`

4. Single node,single GPU下测试engine

```shell
# single GPU下运行engine
python3 run.py --max_output_len=8
```
运行上述代码，log中输出结果如下：

```
Input: Born in north-east France, Soyer trained as a
Output:  chef and eventually became a chef at a
```
问题1解析完成。

⚠️注意：`run.py`支持single node multiple GPUs(基于`mpirun`)和multiple nodes,multiple GPUs(基于[Slurm](https://slurm.schedmd.com))，详细的可以参考`./tensorrt_llm_july-release-v1/examples/gpt/README`

</details>


> 🔏 问题2: 请写出 `./tensorrt_llm_july-release-v1/examples/gpt/README` 里面 `“Summarization using the GPT model”` 部分如下命令的rouge 分数（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium)

```shell
python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf  --batch_size1  --test_trt_llm  --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14
```

<details>
<summary>🔑点我查看 问题2 解析</summary>

该问题将描述如何使用TensorRT-LLM运行一个文本摘要任务的GPT-2,这里使用的数据集为[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) ，对应生成的摘要使用[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) score来评价TensorRT-LLM的精度变化，确切的说这里使用了`ROUGE-1` score。

关于评价指标[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))的介绍，我们推荐参考知乎的介绍：

+ [中文文本摘要指标-ROUGE](https://zhuanlan.zhihu.com/p/388720967)
+ [NLP评估指标之ROUGE](https://zhuanlan.zhihu.com/p/504279252)

问题1中已经完成了必要的package的安装，这里直接下载HF模型

1. 下载HF模型文件

```shell
# 下载模型文件
rm -rf gpt2 && git clone https://huggingface.co/gpt2 gpt2

# 更新.bin模型文件
cd gpt2
rm pytorch_model.bin model.safetensors
wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin
cd ..
```

2. 将HF weight转换为FT weight

```shell
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2/fp16 --tensor-parallelism 1 --storage-type float16
```

运行上述代码，log中出现如下图所示结果，说明模型转换完成，并在`tensorrt_llm_july-release-v1/examples/gpt/c-model/gpt2/fp16/1-gpu`中存放了生成后的FT weight

<div align=center>
<img src="./assets/section5/p2.png"/>
</div>

3. 构建TensorRT-LLM engine

```shell
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

执行上述代码，log中出现如下所示结果，说明模型序列化完成，并将engine保存在`./tensorrt_llm_july-release-v1/examples/gpt/trt_engine/gpt2/fp16/1-gpu`中

```
[08/24/2023-07:34:42] [TRT-LLM] [I] Total time of building gpt_float16_tp1_rank0.engine: 00:01:32
[08/24/2023-07:34:42] [TRT-LLM] [I] Config saved to trt_engine/gpt2/fp16/1-gpu/config.json.
[08/24/2023-07:34:42] [TRT-LLM] [I] Serializing engine to trt_engine/gpt2/fp16/1-gpu/gpt_float16_tp1_rank0.engine...
[08/24/2023-07:34:43] [TRT-LLM] [I] Engine serialized. Total time: 00:00:01
[08/24/2023-07:34:43] [TRT-LLM] [I] Timing cache serialized to trt_engine/gpt2/fp16/1-gpu/model.cache
[08/24/2023-07:34:43] [TRT-LLM] [I] Total time of building all 1 engines: 00:01:43
```

4. TensorRT-LLM下测试GPT-2文本摘要任务

```shell
python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=gpt2 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=14
```
执行上述代码，结果如下：

```
[08/24/2023-08:40:54] [TRT-LLM] [I] ---------------------------------------------------------
Downloading builder script: 5.60kB [00:00, 6.22MB/s]
Token indices sequence length is longer than the specified maximum sequence length for this model (1151 > 1024). Running this sequence through the model will result in indexing errors
[08/24/2023-08:41:14] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.6481666564941406 sec)
[08/24/2023-08:41:14] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/24/2023-08:41:14] [TRT-LLM] [I]   rouge1 : 15.361040799540035
[08/24/2023-08:41:14] [TRT-LLM] [I]   rouge2 : 3.854022269668396
[08/24/2023-08:41:14] [TRT-LLM] [I]   rougeL : 12.078455591738333
[08/24/2023-08:41:14] [TRT-LLM] [I]   rougeLsum : 13.547802733617264
[08/24/2023-08:41:14] [TRT-LLM] [I] Hugging Face (total latency: 10.39808702468872 sec)
[08/24/2023-08:41:14] [TRT-LLM] [I] HF beam 0 result
[08/24/2023-08:41:14] [TRT-LLM] [I]   rouge1 : 14.75593024343394
[08/24/2023-08:41:14] [TRT-LLM] [I]   rouge2 : 3.3647470801871733
[08/24/2023-08:41:14] [TRT-LLM] [I]   rougeL : 11.124766996533
[08/24/2023-08:41:14] [TRT-LLM] [I]   rougeLsum : 13.031128048110618
```

<div align=center>
<img src="./assets/section5/p3.png"/>
</div>

问题2解析完成。

⚠️注意：过程中需要下载数据，如果运行过程中中断或报错大多是因为网络原因，请耐心多尝试重复运行几次

</details>


### 6.未来工作
---

ToDo

### 7.😉References
---

1. [TensorRT(Github)](https://github.com/NVIDIA/TensorRT)

2. [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)

3. [NVIDIA TensorRT Hackathon 2023 —— 生成式AI模型优化赛(天池)](https://tianchi.aliyun.com/competition/entrance/532108/introduction?spm=a2c22.12281957.0.0.605a3b74bkLhBT)

4. [TensorRT 8.6 讲座(B站)](https://www.bilibili.com/video/BV1jj411Z7wG/)

5. [Llama(Github)](https://github.com/facebookresearch/llama)

6. [Llama paper (arxiv)](https://arxiv.org/abs/2302.13971)

7. [TensorRT-LLM:大语言模型推理：低精度最佳实践(B站)](https://www.bilibili.com/video/BV1h44y1c72B/?share_source=copy_web&vd_source=db3eecb1b88cc6c7a18eeaf6db1ed114)


