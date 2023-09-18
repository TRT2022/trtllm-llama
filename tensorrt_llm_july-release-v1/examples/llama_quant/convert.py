"""
    llamav1-7b的int8 k/v cache 和 smooth quant
"""
import numpy as np
import torch

from tensorrt_llm._utils import torch_to_numpy

def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])

def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    val.tofile(dir / f"model.{key}.{suffix}")


def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """
     This function has two purposes:
      - compute quantized weights, scaled either per-tensor or per-column
      - compute scaling factors

      Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
      CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
      CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

      Here is the list of what we need (T means per-tensor, C per-column):
        - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
        - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
        - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
        - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
          to quant range (int8) (used for CUBLAS) (T, C)

      Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
      but then the model would change depending on the number of GPUs used.

      For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
      as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
    """

    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        raise ValueError(
            f"Multi-query w/ int8 quant has not been supported yet")
    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_c)
    if is_qkv:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)
    # print(weights.shape)
    # print( scale_w_orig_quant_t.shape)
    # print((weights * scale_w_orig_quant_t).shape)
    if is_qkv and not multi_query_mode:
        # int8_weight = to_i8(weights * (scale_w_orig_quant_t.reshape(scale_w_orig_quant_t.shape[0])))
        int8_weight = to_i8(weights * (scale_w_orig_quant_t))

    else:
        int8_weight = to_i8(weights * (scale_w_orig_quant_t))
    if is_qkv and not multi_query_mode:
        int8_col_weight = to_i8(weights * scale_w_orig_quant_c)
    else:
        int8_col_weight = to_i8(weights * scale_w_orig_quant_c)
    return {
        "weight.int8": int8_weight,
        "weight.int8.col": int8_col_weight,
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def write_int8(vals,
               dir,
               base_key,
               split_dim,
               tp_rank,
               split_factor,
               kv_cache_only=False):
    if not kv_cache_only:
        save_split(np.split(vals["weight.int8"], split_factor, axis=split_dim),
                   dir, f"{base_key}.weight.int8", tp_rank, split_factor)
        save_split(
            np.split(vals["weight.int8.col"], split_factor, axis=split_dim),
            dir, f"{base_key}.weight.int8.col", tp_rank, split_factor)

    saved_keys_once = ["scale_y_quant_orig"]
    if not kv_cache_only:
        saved_keys_once += [
            "scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant"
        ]
    # per-column scaling factors are loaded per-gpu for ColumnParallel GEMMs (QKV, FC1)
    if not kv_cache_only:
        if split_dim == -1:
            save_split(
                np.split(vals["scale_w_quant_orig.col"],
                         split_factor,
                         axis=split_dim), dir,
                f"{base_key}.scale_w_quant_orig.col", tp_rank, split_factor)
            save_split(
                np.split(vals["scale_y_accum_quant.col"],
                         split_factor,
                         axis=split_dim), dir,
                f"{base_key}.scale_y_accum_quant.col", tp_rank, split_factor)
        else:
            saved_keys_once += [
                "scale_w_quant_orig.col", "scale_y_accum_quant.col"
            ]

    if tp_rank == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")


def str_to_np_dtype(type_str):
    convert_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
    }
    dtype = convert_dict.get(type_str)
    if dtype is None:
        raise ValueError(f"{type_str} is an invalid storage type")
    return dtype


def split_and_save_weight(tp_rank, saved_dir, split_factor, key, vals,
                          storage_type, act_range, config):
    use_attention_nemo_shape = config.get("use_attention_nemo_shape", False)
    split_gated_activation = config.get("split_gated_activation", False)
    num_attention_heads = config.get("num_attention_heads", 0)
    tp_size = config.get("tp_size", 1)
    int8_outputs = config.get("int8_outputs", None)
    multi_query_mode = config.get("multi_query_mode", False)
    local_dim = config.get("local_dim", None)

    save_int8 = int8_outputs == "all" or int8_outputs == "kv_cache_only"

    # if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
    #     "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
    #     "post_attention_layernorm.bias" in key or "mlp.dense_4h_to_h.bias" in key or \
    #     "final_layernorm.weight" in key or "final_layernorm.bias" in key:

    #     # shared weights, only need to convert the weights of rank 0
    #     if i == 0:
    #         save_val(val, saved_dir, key)

    
    if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
        "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
        "post_attention_layernorm.bias" in key or "mlp.gate_proj.bias" in key or "mlp.up_proj.bias" in key or "mlp.down_proj.bias" in key or\
        "final_layernorm.weight" in key or "final_layernorm.bias" in key:

    # if key.find("input_layernorm.weight") != -1 or key.find("post_attention_layernorm.weight") != -1:
        if tp_rank == 0:
            save_val(vals, saved_dir, key)


    # elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
    elif "attention.dense.weight" in key or "mlp.down_proj.weight" in key:
        split_dim = 0
        cat_dim = 0

        val = np.concatenate(vals, axis=cat_dim)
        # print("===============================111")
        # print(val.shape)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        if act_range is not None and int8_outputs == "all":
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(vals, #vals
                                    act_range,
                                    multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank,
                       split_factor)

    # elif "mlp.dense_h_to_4h.weight" in key:
    elif key.find("mlp.gate_proj.weight") != -1 or key.find("mlp.up_proj.weight") != -1:
        cat_dim = -1
        val = np.concatenate(vals, axis=cat_dim)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)

        if act_range is not None and int8_outputs == "all":
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(vals,
                                    act_range,
                                    multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank,
                       split_factor)

    # elif "mlp.dense_h_to_4h.bias" in key:
    #     split_vals = np.split(val, factor, axis=-1)
    #     save_split(split_vals, saved_dir, key, i, factor)

    # elif "attention.query_key_value.bias" in key:
    #     local_dim = val.shape[-1] // 3

    #     val = val.reshape(3, local_dim)
    #     split_vals = np.split(val, factor, axis=-1)
    #     save_split(split_vals, saved_dir, key, i, factor)

    elif "attention.query_key_value.weight" in key:
        cat_dim = -1
        hidden_dim = vals[0].shape[0]
        if local_dim is None:
            local_dim = vals[0].shape[-1] // 3
        val = vals
        # print("==============================")
        # print(vals.shape)
        # print(val.shape)
        # print(local_dim)
        # out_feature = local_dim + 2 * head_size; assumes local_dim equals to hidden_dim
        # head_size = (val.shape[-1] - local_dim) // 2
        # val = val.reshape(hidden_dim, local_dim + 2 * head_size)
        # w_q, w_kv = np.split(val, [local_dim], axis=-1)
        # w_q_split = np.split(w_q, split_factor, axis=-1)
        # split_vals = [np.concatenate((i, w_kv), axis=-1) for i in w_q_split]
        # save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        
        # save_split(vals, saved_dir, key, tp_rank, split_factor)
        save_val(vals, saved_dir, key, tp_num=None)

        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val,
                                    act_range,
                                    is_qkv=True,
                                    multi_query_mode=False)
            write_int8(vals_i8,
                       saved_dir,
                       base_key,
                       cat_dim,
                       tp_rank,
                       split_factor,
                       kv_cache_only=int8_outputs == "kv_cache_only")
            # print("------------")
            # print(int8_outputs == "kv_cache_only")
    # elif "attention.query_key_value.weight" in key:
    #     cat_dim = -1
    #     hidden_dim = vals[0].shape[0]
    #     if local_dim is None:
    #         local_dim = vals[0].shape[-1] // 3
    #     val = vals
    #     # print("==============================")
    #     # print(vals.shape)
    #     # print(val.shape)
    #     # print(local_dim)
    #     # out_feature = local_dim + 2 * head_size; assumes local_dim equals to hidden_dim
    #     # head_size = (val.shape[-1] - local_dim) // 2
    #     # val = val.reshape(hidden_dim, local_dim + 2 * head_size)
    #     # w_q, w_kv = np.split(val, [local_dim], axis=-1)
    #     # w_q_split = np.split(w_q, split_factor, axis=-1)
    #     # split_vals = [np.concatenate((i, w_kv), axis=-1) for i in w_q_split]
    #     # save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        
    #     # save_split(vals, saved_dir, key, tp_rank, split_factor)

        
    #     # add--------------------------
    #     q_emb = vals.shape[0] // 3
    #     model_emb = vals.shape[1]
    #     vals = vals.reshape(3, q_emb, model_emb)
    #     split_vals = split(vals, split_factor, tp_rank, dim=1)
    #     split_vals = split_vals.reshape(3 * (q_emb // split_factor),
    #                         model_emb)

    #     # split_vals = np.split(vals, split_factor, axis=-1)
    #     # save_val(vals, saved_dir, key, tp_num=None)
    #     # save_split(split_vals, saved_dir, key, tp_rank, split_factor)
    #     save_val(split_vals, saved_dir, key, tp_num=None)

    #     if save_int8:
    #         base_key = key.replace(".weight", "")
    #         vals_i8 = generate_int8(val,
    #                                 act_range,
    #                                 is_qkv=True,
    #                                 multi_query_mode=False)
    #         write_int8(vals_i8,
    #                    saved_dir,
    #                    base_key,
    #                    cat_dim,
    #                    tp_rank,
    #                    split_factor,
    #                    kv_cache_only=int8_outputs == "kv_cache_only")
    #         # print("------------")
    #         # print(int8_outputs == "kv_cache_only")

    else:
        print(f"[WARNING] {key} not handled by converter")
    

   
