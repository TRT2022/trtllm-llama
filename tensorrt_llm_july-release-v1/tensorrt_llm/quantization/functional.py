from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt

from .._common import default_net, default_trtnet
from .._utils import str_dtype_to_np, str_dtype_to_trt
from ..functional import Tensor, _create_tensor, cast, clip, constant, round
from ..plugin import _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE


def smooth_quant_gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                      scales_b: Tensor, per_token_scaling: bool,
                      per_channel_scaling: bool) -> Tensor:
    if not default_net().plugin_config.smooth_quant_gemm_plugin:
        raise TypeError("Smooth Quant GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'SmoothQuantGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        per_channel_scaling = 1 if per_channel_scaling else 0
        per_channel_scaling = trt.PluginField(
            "has_per_channel_scaling",
            np.array(per_channel_scaling, dtype=np.int32),
            trt.PluginFieldType.INT32)

        per_token_scaling = 1 if per_token_scaling else 0
        per_token_scaling = trt.PluginField(
            "has_per_token_scaling", np.array(per_token_scaling,
                                              dtype=np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.smooth_quant_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection(
            [per_channel_scaling, per_token_scaling, pf_type])
        gemm_plug = plg_creator.create_plugin("sq_gemm", pfc)
        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, scales_a.trt_tensor,
            scales_b.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        layer.get_input(0).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_quant_matmul(input: Tensor, weights: Tensor, scales: Tensor,
                             weightTypeId: int) -> Tensor:
    if not default_net().plugin_config.weight_only_quant_matmul_plugin:
        raise TypeError(
            "Weight Only Qunat MatMul is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'WeightOnlyQuantMatmul', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        weight_type_id = trt.PluginField("weight_type_id",
                                         np.array(weightTypeId, dtype=np.int32),
                                         trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.weight_only_quant_matmul_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type, weight_type_id])
        matmul_plug = plg_creator.create_plugin("woq_matmul", pfc)
        plug_inputs = [input.trt_tensor, weights.trt_tensor, scales.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, matmul_plug)
        return _create_tensor(layer.get_output(0), layer)


def smooth_quant_layer_norm(input: Tensor,
                            normalized_shape: Union[int, Tuple[int]],
                            weight: Optional[Tensor] = None,
                            bias: Optional[Tensor] = None,
                            scale: Optional[Tensor] = None,
                            eps: float = 1e-05,
                            use_diff_of_squares: bool = True,
                            dynamic_act_scaling: bool = False) -> Tensor:
    if not default_net().plugin_config.layernorm_quantization_plugin:
        raise TypeError("Smooth Quant Layer Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'LayernormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)
        use_diff_of_squares = trt.PluginField(
            "use_diff_of_squares",
            np.array([int(use_diff_of_squares)], dtype=np.int32),
            trt.PluginFieldType.INT32)

        dyn_act_scaling = trt.PluginField(
            "dyn_act_scaling", np.array([int(dynamic_act_scaling)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.layernorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection(
            [eps, use_diff_of_squares, dyn_act_scaling, pf_type])
        layernorm_plug = plg_creator.create_plugin("layernorm_quantized", pfc)
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)
        if not dynamic_act_scaling:
            return _create_tensor(layer.get_output(0), layer)

        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)


def quantize(input: Tensor,
             scale_factor: Tensor,
             dtype: str,
             axis: int = -1) -> Tensor:
    layer = default_trtnet().add_quantize(input.trt_tensor,
                                          scale_factor.trt_tensor,
                                          str_dtype_to_trt(dtype))
    layer.axis = axis

    output = _create_tensor(layer.get_output(0), layer)

    layer.get_output(0).dtype = str_dtype_to_trt(dtype)

    return output


def dequantize(input: Tensor, scale_factor: Tensor, axis: int = -1) -> Tensor:
    layer = default_trtnet().add_dequantize(input.trt_tensor,
                                            scale_factor.trt_tensor)
    layer.axis = axis

    layer.precision = input.dtype

    output = _create_tensor(layer.get_output(0), layer)

    return output


def quantize_per_token(x: Tensor) -> Tuple[Tensor]:
    if not default_net().plugin_config.quantize_per_token_plugin:
        if x.dtype != trt.float32:
            x = cast(x, 'float32')
        xmax = x.abs().max(-1, keepdim=True)
        scale = xmax / 127.0
        out = x * 127.0 / xmax
        out = round(out)
        out = clip(out, -128, 127)
        quantized_out = cast(out, 'int8')
        return quantized_out, scale
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QuantizePerToken', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        pfc = trt.PluginFieldCollection([])
        quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin",
                                                  pfc)

        plug_inputs = [x.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)

        quantized = _create_tensor(layer.get_output(0), layer)
        quantized.trt_tensor.dtype = str_dtype_to_trt("int8")
        scales = _create_tensor(layer.get_output(1), layer)
        scales.trt_tensor.dtype = str_dtype_to_trt("float32")

        return quantized, scales


def quantize_tensor(x, scale):
    if not default_net().plugin_config.quantize_tensor_plugin:
        scaled = x * scale
        rounded = round(scaled)
        clipped = clip(rounded, -128, 127)
        quantized = cast(clipped, 'int8')
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QuantizeTensor', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        pfc = trt.PluginFieldCollection([])
        quantize_plug = plg_creator.create_plugin("quantize_tensor_plugin", pfc)

        plug_inputs = [x.trt_tensor, scale.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)

        quantized = _create_tensor(layer.get_output(0), layer)
        quantized.trt_tensor.dtype = str_dtype_to_trt("int8")
    return quantized
