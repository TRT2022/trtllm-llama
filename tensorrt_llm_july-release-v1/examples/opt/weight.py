import configparser
import time
from pathlib import Path

import numpy as np

import tensorrt_llm
from tensorrt_llm.models import OPTLMHeadModel


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('gpt', 'n_embd')
    n_head = gpt_config.getint('gpt', 'n_head')
    n_layer = gpt_config.getint('gpt', 'n_layer')
    n_positions = gpt_config.getint('gpt', 'n_positions')
    vocab_size = gpt_config.getint('gpt', 'vocab_size')
    do_layer_norm_before = gpt_config.getboolean('gpt',
                                                 'do_layer_norm_before',
                                                 fallback=True)

    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before


def load_from_ft(tensorrt_llm_gpt: OPTLMHeadModel,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 fp16=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=np_dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    if pe is not None:
        tensorrt_llm_gpt.embedding.position_embedding.weight.value = (pe)
    tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = (fromfile(
        dir_path, 'model.wte.bin', [vocab_size, n_embd]))
    if do_layer_norm_before:
        tensorrt_llm_gpt.ln_f.bias.value = (fromfile(
            dir_path, 'model.final_layernorm.bias.bin'))
        tensorrt_llm_gpt.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                              [vocab_size, n_embd])
    if lm_head_weight is None:
        lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                  [vocab_size, n_embd])
    if vocab_size % tensor_parallel != 0:
        # padding
        vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
        split(lm_head_weight, tensor_parallel, rank))
    for i in range(n_layer):
        tensorrt_llm_gpt.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        tensorrt_llm_gpt.layers[i].input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + str(rank) + '.bin',
            [n_embd, 3 * n_embd // tensor_parallel])
        if t is not None:
            dst = tensorrt_llm_gpt.layers[i].attention.qkv.weight
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.bias.' + str(rank) + '.bin')
        if t is not None:
            dst = tensorrt_llm_gpt.layers[i].attention.qkv.bias
            dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_gpt.layers[i].attention.dense.weight
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.dense.weight.' +
            str(rank) + '.bin', [n_embd // tensor_parallel, n_embd])
        dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_gpt.layers[i].attention.dense.bias
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.dense.bias.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.bias
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' +
            str(rank) + '.bin', [n_embd, 4 * n_embd // tensor_parallel])
        tensorrt_llm_gpt.layers[i].mlp.fc.weight.value = np.ascontiguousarray(
            np.transpose(t, [1, 0]))
        tensorrt_llm_gpt.layers[i].mlp.fc.bias.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_h_to_4h.bias.' +
            str(rank) + '.bin')
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' +
            str(rank) + '.bin', [4 * n_embd // tensor_parallel, n_embd])
        tensorrt_llm_gpt.layers[i].mlp.proj.weight.value = (
            np.ascontiguousarray(np.transpose(t, [1, 0])))
        tensorrt_llm_gpt.layers[i].mlp.proj.bias.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
