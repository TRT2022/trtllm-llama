'''
  llama v1-7b convert hf模型到ft模型

'''

import argparse
from tqdm import tqdm
import configparser
import numpy as np
from pathlib import Path

import os
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import LlamaTokenizer
import dataclasses

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm

from convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 2
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "llama"
    storage_type: str = "fp16"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=2)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="llama",
            type=str,
            help="llama to convert checkpoints correctly",
            # choices=["gpt2", "santacoder", "starcoder"]
            )
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))



# def smooth_gemm(gemm_weights,
#                 act_scales,
#                 layernorm_weights=None,
#                 layernorm_bias=None,
#                 alpha=0.5,
#                 weight_scales=None):

@torch.no_grad()
def smooth_llama_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    # for name, module in model.named_modules():
    #     print("------------------")
    #     print(name)
    #     print(module)
    #     print("----------------")

        # if not isinstance(module, GPT2Block):
        #     continue
    num_layers = 32
    param_to_weights = lambda param: param.detach().cpu().numpy().astype(np.float16)
    for l in range(num_layers):
        # qkv_proj

        qkv_weights =  torch.from_numpy(np.stack([
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight']),
        ],axis=-1)).cuda()

        qkv_weights = qkv_weights.permute(1,2,0)  # add like int8 k/v cache

        qkv_weights_base_name = f'model.layers.{l}.attention.query_key_value'
        # 这块reshape可能有问题？
        smoother = smooth_gemm(qkv_weights.reshape(qkv_weights.shape[0],qkv_weights.shape[1]*qkv_weights.shape[2]),
                            scales[qkv_weights_base_name]["x"], None,
                            None, alpha)
        scales[qkv_weights_base_name]["x"] = scales[qkv_weights_base_name]["x"] / smoother
        scales[qkv_weights_base_name]["w"] = qkv_weights.abs().max(dim=0)[0]


        # # f'model.layers.{l}.self_attn.q_proj.weight'
        # # if "self_attn" in name:
        # # q
        # q_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'])).cuda()
        # smoother = smooth_gemm(q_weight.T,
        #                     scales[f'model.layers.{l}.self_attn.q_proj']["x"], None,
        #                     None, alpha)
        # scales[f'model.layers.{l}.self_attn.q_proj']["x"] = scales[f'model.layers.{l}.self_attn.q_proj']["x"] / smoother
        # scales[f'model.layers.{l}.self_attn.q_proj']["w"] = q_weight.abs().max(dim=0)[0]
        # #k
        # k_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight'])).cuda()
        # smoother = smooth_gemm(k_weight.T,
        #                     scales[f'model.layers.{l}.self_attn.k_proj']["x"], None,
        #                     None, alpha)
        # scales[f'model.layers.{l}.self_attn.k_proj']["x"] = scales[f'model.layers.{l}.self_attn.k_proj']["x"] / smoother
        # scales[f'model.layers.{l}.self_attn.k_proj']["w"] = k_weight.abs().max(dim=0)[0]

        # #v
        # v_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight'])).cuda()
        # smoother = smooth_gemm(v_weight.T,
        #                     scales[f'model.layers.{l}.self_attn.v_proj']["x"], None,
        #                     None, alpha)
        # scales[f'model.layers.{l}.self_attn.v_proj']["x"] = scales[f'model.layers.{l}.self_attn.v_proj']["x"] / smoother
        # scales[f'model.layers.{l}.self_attn.v_proj']["w"] = v_weight.abs().max(dim=0)[0]
        
        # MLP
        mlp_down_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.down_proj.weight'])).cuda()
        smoother = smooth_gemm(mlp_down_weight,
                               scales[f'model.layers.{l}.mlp.down_proj']["x"], None,
                               None, alpha)
        scales[f'model.layers.{l}.mlp.down_proj']["x"] = scales[f'model.layers.{l}.mlp.down_proj']["x"] / smoother
        scales[f'model.layers.{l}.mlp.down_proj']["w"] = mlp_down_weight.T.abs().max(dim=0)[0]

        mlp_gate_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.gate_proj.weight'])).cuda()
        smoother = smooth_gemm(mlp_gate_weight,
                               scales[f'model.layers.{l}.mlp.gate_proj']["x"], None,
                               None, alpha)
        scales[f'model.layers.{l}.mlp.gate_proj']["x"] = scales[f'model.layers.{l}.mlp.gate_proj']["x"] / smoother
        scales[f'model.layers.{l}.mlp.gate_proj']["w"] = mlp_gate_weight.T.abs().max(dim=0)[0]

        mlp_up_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.up_proj.weight'])).cuda()
        smoother = smooth_gemm(mlp_up_weight,
                               scales[f'model.layers.{l}.mlp.up_proj']["x"], None,
                               None, alpha)
        scales[f'model.layers.{l}.mlp.up_proj']["x"] = scales[f'model.layers.{l}.mlp.up_proj']["x"] / smoother
        scales[f'model.layers.{l}.mlp.up_proj']["w"] = mlp_up_weight.T.abs().max(dim=0)[0]
        

        # atten dense  add by debug 2023-09-16 
        o_weight = torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.o_proj.weight'])).cuda()
        # o_weight_base_name = f'model.layers.{l}.attention.dense.weight'
        smoother = smooth_gemm(o_weight,
                        scales[f'model.layers.{l}.self_attn.o_proj']["x"], None,
                        None, alpha)
        scales[f'model.layers.{l}.self_attn.o_proj']["x"] = scales[f'model.layers.{l}.self_attn.o_proj']["x"] / smoother
        scales[f'model.layers.{l}.self_attn.o_proj']["w"] = o_weight.T.abs().max(dim=0)[0]




# SantaCoder separates Q projection from KV projection
def concat_qkv_weight_bias(q, hf_key, hf_model):
    kv = hf_model.state_dict()[hf_key.replace("q_attn", "kv_attn")]
    return torch.cat([q, kv], dim=-1)


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = ["c_attn", "c_proj", "c_fc"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


# def gpt_to_ft_name(orig_name):
#     global_weights = {
#         "transformer.wpe.weight": "model.wpe",
#         "transformer.wte.weight": "model.wte",
#         "transformer.ln_f.bias": "model.final_layernorm.bias",
#         "transformer.ln_f.weight": "model.final_layernorm.weight",
#         "lm_head.weight": "model.lm_head.weight"
#     }

#     if orig_name in global_weights:
#         return global_weights[orig_name]

#     _, _, layer_id, *weight_name = orig_name.split(".")
#     layer_id = int(layer_id)
#     weight_name = "transformer." + ".".join(weight_name)

#     per_layer_weights = {
#         "transformer.ln_1.bias": "input_layernorm.bias",
#         "transformer.ln_1.weight": "input_layernorm.weight",
#         "transformer.attn.c_attn.bias": "attention.query_key_value.bias",
#         "transformer.attn.c_attn.weight": "attention.query_key_value.weight",
#         "transformer.attn.q_attn.weight": "attention.query.weight",
#         "transformer.attn.q_attn.bias": "attention.query.bias",
#         "transformer.attn.kv_attn.weight": "attention.key_value.weight",
#         "transformer.attn.kv_attn.bias": "attention.key_value.bias",
#         "transformer.attn.c_proj.bias": "attention.dense.bias",
#         "transformer.attn.c_proj.weight": "attention.dense.weight",
#         "transformer.ln_2.bias": "post_attention_layernorm.bias",
#         "transformer.ln_2.weight": "post_attention_layernorm.weight",
#         "transformer.mlp.c_fc.bias": "mlp.dense_h_to_4h.bias",
#         "transformer.mlp.c_fc.weight": "mlp.dense_h_to_4h.weight",
#         "transformer.mlp.c_proj.bias": "mlp.dense_4h_to_h.bias",
#         "transformer.mlp.c_proj.weight": "mlp.dense_4h_to_h.weight",
#     }
#     return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_llama_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = False 
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = LlamaForCausalLM.from_pretrained(args.in_file,
                                                device_map={
                "model": "cuda",
                "lm_head": "cuda"
            },  # Load to CPU memory
            torch_dtype="auto")
    # # load position_embedding from rank 0
    # model = AutoModelForCausalLM.from_pretrained(args.in_file,
    #         device_map={
    #             "model": "cpu",
    #             "lm_head": "cpu"
    #         },  # Load to CPU memory
    #         # torch_dtype="auto",
    #         trust_remote_code=True)
    hf_config = vars(model.config)
    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]



    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, LlamaTokenizer.from_pretrained(args.in_file), dataset)
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, LlamaTokenizer.from_pretrained(args.in_file), dataset)

        act_range = capture_activation_range(
            model, LlamaTokenizer.from_pretrained(args.in_file), dataset)


    config = configparser.ConfigParser()
    config["llama"] = {}
    for key in vars(args):
        config["llama"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["llama"][k] = f"{v}"
    config["llama"]["storage_dtype"] = args.storage_type
    config["llama"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    # global_ft_weights = [
    #     "model.wpe", "model.wte", "model.final_layernorm.bias",
    #     "model.final_layernorm.weight", "model.lm_head.weight"
    # ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    param_to_weights = lambda param: param.detach().cpu().numpy().astype(np.float16)

    for l in range(num_layers):
        x_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"]],axis=-1)
        
        y_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"]],axis=-1)
        w_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"],
            act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"]],axis=-1)

        act_range[f'model.layers.{l}.attention.query_key_value']={'x':x_,'y':y_,'w':w_}

        # atten dense
        x_dense = act_range.get(f'model.layers.{l}.self_attn.o_proj')["x"]
        y_dense = act_range.get(f'model.layers.{l}.self_attn.o_proj')["y"]
        w_dense = act_range.get(f'model.layers.{l}.self_attn.o_proj')["w"]
        act_range[f'model.layers.{l}.attention.dense']={'x':x_dense,'y':y_dense,'w':w_dense}



    # 放在下面
    if args.smoothquant is not None:
        smooth_llama_model(model, act_range, args.smoothquant) 

    for l in range(num_layers):
        print(f"converting layer {l}/{num_layers}")
        # first merge QKV into a single weight
        # concat direct to FT shape: [hidden_size, 3, head_num, head_size]
        # copied from huggingface_gptj_ckpt_convert.py
        
        qkv_weights = np.stack([
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight']),
            param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight']),
        ],axis=-1)

        # qkv_weights = torch.cat([torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight'])),
        #     torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight'])),
        #     torch.from_numpy(param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight']))], dim=0).numpy()

        # qkv_weights = np.concatenate([
        #     param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.q_proj.weight']),
        #     param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.k_proj.weight']),
        #     param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.v_proj.weight']),
        # ],axis=0)

        # print("--------------------------------")
        # print(qkv_weights.shape)

        qkv_weights = np.transpose(qkv_weights, (1,2,0))
        qkv_weights_base_name = f'model.layers.{l}.attention.query_key_value.weight'

        # # 新办法
        # qkvArr = np.empty((hidden_size, 3, head_num, head_size), dtype=np.float16)
        # qArr = param_to_weights(model.state_dict()[f"model.layers.{l}.self_attn.q_proj.weight"])
        # # Hopefully this reshaping is correct... last two dims could also be swapped & need to be transposed
        # qkvArr[:, 0, :, :] = qArr.reshape(hidden_size, head_num, head_size)
        # kArr = param_to_weights(model.state_dict()[f"model.layers.{l}.self_attn.k_proj.weight"])
        # qkvArr[:, 1, :, :] = kArr.reshape(hidden_size, head_num, head_size)
        # vArr =param_to_weights(model.state_dict()[f"model.layers.{l}.self_attn.v_proj.weight"])
        # qkvArr[:, 2, :, :] = vArr.reshape(hidden_size, head_num, head_size)

        # # x_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["x"]],axis=-1) 
        
        # # y_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["y"]],axis=-1) 
        # # w_ = torch.concat([act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"],
        # #     act_range.get(f'model.layers.{l}.self_attn.q_proj')["w"]],axis=-1) 

        # # act_range[f'model.layers.{l}.attention.query_key_value']={'x':x_,'y':y_,'w':w_}

        # # local_dim = model.transformer.h[
        # #     0].attn.embed_dim if multi_query_mode else None
        local_dim = None

        split_and_save_weight(0, saved_dir, infer_tp, qkv_weights_base_name, qkv_weights,
        storage_type, act_range.get(qkv_weights_base_name.replace(".weight", "")), config={
                    "int8_outputs": int8_outputs,
                    "multi_query_mode": multi_query_mode,
                    "local_dim": local_dim
                })
            
        ## attention dense
        o_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.self_attn.o_proj.weight']).T
        o_weight_base_name = f'model.layers.{l}.attention.dense.weight'
        split_and_save_weight(0,saved_dir, infer_tp, o_weight_base_name, o_weight,
        storage_type, act_range.get(o_weight_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })
        # MLP
        mlp_down_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.down_proj.weight']).T
        mlp_down_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        split_and_save_weight(0,saved_dir, infer_tp, mlp_down_base_name, mlp_down_weight,
        storage_type, act_range.get(mlp_down_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })
        mlp_gate_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.gate_proj.weight']).T
        mlp_gate_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        split_and_save_weight(0,saved_dir, infer_tp, mlp_gate_base_name, mlp_gate_weight,
        storage_type, act_range.get(mlp_gate_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })

        mlp_up_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.mlp.up_proj.weight']).T
        mlp_up_base_name = f'model.layers.{l}.mlp.up_proj.weight'
        split_and_save_weight(0,saved_dir, infer_tp, mlp_up_base_name, mlp_up_weight,
        storage_type, act_range.get(mlp_up_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })
        
        # LayerNorm
        input_ln_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.input_layernorm.weight'])
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        split_and_save_weight(0,saved_dir, infer_tp, input_ln_base_name, input_ln_weight,
        storage_type, act_range.get(input_ln_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })

        post_attn_ln_weight = param_to_weights(model.state_dict()[f'model.layers.{l}.post_attention_layernorm.weight'])
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        split_and_save_weight(0,saved_dir, infer_tp, post_attn_ln_base_name, post_attn_ln_weight,
        storage_type, act_range.get(post_attn_ln_base_name.replace(".weight", "")), config={
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": local_dim
                    })
        print(f"done layer {l}")

    # final common weights
    for name, param in model.named_parameters():
        if name == 'model.embed_tokens.weight':
            param.detach().cpu().numpy().astype(np.float16).tofile(os.path.join(saved_dir ,"model.wte.weight.bin"))
        elif name == 'model.norm.weight':
            param.detach().cpu().numpy().astype(np.float16).tofile(os.path.join(saved_dir , "model.final_layernorm.weight.bin"))
        elif name == 'lm_head.weight':
            param.detach().cpu().numpy().astype(np.float16).tofile(os.path.join(saved_dir , "model.lm_head.weight.bin"))
        # elif "self_attn.o_proj.weight" in name:
        #     print("------------------------------------>>>>>>>>")
        #     print(name)
        # else:
        #     print("------------------------------------>>>>>>>>")
        #     print(name)
    
    
            

   

def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_llama_converter(args)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
