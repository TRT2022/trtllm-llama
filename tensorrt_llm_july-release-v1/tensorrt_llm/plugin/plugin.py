import ctypes
from enum import IntEnum
from pathlib import Path

from tensorrt_llm.logger import logger

_TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'


def _load_plugin_lib():
    project_dir = str(Path(__file__).parent.parent.absolute())

    # load tensorrt_llm plugin
    plugin_lib = project_dir + '/libs/libnvinfer_plugin_tensorrt_llm.so'
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Plugin is unavailable')

    handle.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initLibNvInferPlugins.restype = ctypes.c_bool
    assert handle.initLibNvInferPlugins(
        None, _TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))


class ContextFMHAType(IntEnum):
    disabled = 0
    # FP16 I/O, FP16 Accumulation
    enabled = 1
    # FP16 I/O, FP32 Accumulation
    enabled_with_fp32_acc = 2


class PluginConfig(object):

    def __init__(self) -> None:
        self.init()

    def init(self):
        self.bert_attention_plugin = False
        self.gpt_attention_plugin = False
        self.inflight_batching_gpt_attention_plugin = False
        self.identity_plugin = False
        self.gemm_plugin = False
        self.smooth_quant_gemm_plugin = False
        self.layernorm_plugin = False
        self.layernorm_quantization_plugin = False
        self.attention_qk_half_accumulation = False
        self.remove_input_padding = False
        self.context_fmha_type = ContextFMHAType.disabled
        self.weight_only_quant_matmul_plugin = False
        self.nccl_plugin = False
        self.quantize_per_token_plugin = False
        self.quantize_tensor_plugin = False
        self.paged_kv_cache = False
        self.lookup_plugin = False
        self.in_flight_batching = False

    def enable_qk_half_accum(self):
        self.attention_qk_half_accumulation = True
        logger.info(f"Attention BMM1(QK) accumulation type is set to FP16")
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert context_fmha_type in \
            [ContextFMHAType.disabled, ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc]
        self.context_fmha_type = context_fmha_type
        if context_fmha_type == ContextFMHAType.enabled:
            logger.info(f"Context FMHA Enabled")
        elif context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
            logger.info(f"Context FMHA with FP32 Accumulation Enabled")
        elif context_fmha_type == ContextFMHAType.disabled:
            logger.info(f"Context FMHA Disabled")
        return self

    def enable_remove_input_padding(self):
        self.remove_input_padding = True
        logger.info(f"Remove Padding Enabled")
        return self

    def enable_paged_kv_cache(self):
        self.paged_kv_cache = True
        logger.info(f"Paged KV Cache Enabled")
        return self

    def enable_in_flight_batching(self):
        self.in_flight_batching = True
        logger.info(f"In-Flight Batching Enabled")
        return self

    def set_gpt_attention_plugin(self, dtype='float16'):
        self.gpt_attention_plugin = dtype
        return self

    def set_inflight_batching_gpt_attention_plugin(self, dtype='float16'):
        self.inflight_batching_gpt_attention_plugin = dtype
        return self

    def set_bert_attention_plugin(self, dtype='float16'):
        self.bert_attention_plugin = dtype
        return self

    def set_identity_plugin(self, dtype='float16'):
        self.identity_plugin = dtype
        return self

    def set_gemm_plugin(self, dtype='float16'):
        self.gemm_plugin = dtype
        return self

    def set_smooth_quant_gemm_plugin(self, dtype='float16'):
        self.smooth_quant_gemm_plugin = dtype
        return self

    def set_layernorm_plugin(self, dtype='float16'):
        self.layernorm_plugin = dtype
        return self

    def set_layernorm_quantization_plugin(self, dtype='float16'):
        self.layernorm_quantization_plugin = dtype
        return self

    def set_weight_only_quant_matmul_plugin(self, dtype='float16'):
        self.weight_only_quant_matmul_plugin = dtype
        return self

    def set_nccl_plugin(self, dtype='float16'):
        self.nccl_plugin = dtype
        return self

    def set_quantize_per_token_plugin(self):
        self.quantize_per_token_plugin = True
        return self

    def set_quantize_tensor_plugin(self):
        self.quantize_tensor_plugin = True
        return self

    def set_lookup_plugin(self, dtype='float16'):
        self.lookup_plugin = dtype
        return self
