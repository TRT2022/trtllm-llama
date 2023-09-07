import math
from typing import Union

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from .._utils import int32_array
from ..functional import (ACT2FN, RaggedTensor, Tensor, allgather, allreduce,
                          cast, concat, constant, gpt_attention, matmul, shape,
                          slice, softmax, split, where)
from ..layers.attention import AttentionMaskType, PositionEmbeddingType
from ..module import Module
from ..parameter import Parameter
from .functional import (dequantize, quantize, quantize_per_token,
                         quantize_tensor, smooth_quant_gemm,
                         smooth_quant_layer_norm, weight_only_quant_matmul)
from .mode import QuantMode


class Quantize(Module):
    """
        Quantize Layer
        For per-tensor mode, the scaling factor is a scalar.
        For per-channel mode, the scaling factor is a vector.
        """

    def __init__(
        self,
        output_dtype: str = 'int8',
        scaling_factor_dtype: str = 'float32',
        in_channels: int = -1,
        axis=-1,
    ) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=(in_channels, ) if axis != -1 else
                                        (),
                                        dtype=scaling_factor_dtype)
        self.output_dtype = output_dtype
        self.axis = axis

    def forward(self, x):
        return quantize(x, self.scaling_factor.value, self.output_dtype,
                        self.axis)


class QuantizePerToken(Module):
    """
        Quantize Per Token and compute dynamic scales for SmoothQuant
        """

    def forward(self, x):
        return quantize_per_token(x)


class Dequantize(Module):
    """
        Dequantize Layer.
        """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=())
        self.axis = axis

    def forward(self, input):
        return dequantize(input, self.scaling_factor.value, self.axis)


class SmoothQuantLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size

        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )

        weights_dtype = dtype
        # Dirty hack to make it work with SmoothQuant int8 weights
        # reinterpreted as fp32 weights due to the int8 TRT plugin limitation.
        if quant_mode.has_act_and_weight_quant():
            assert self.in_features % 4 == 0
            self.in_features = self.in_features // 4
            weights_dtype = "float32"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)

        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.quant_mode = quant_mode

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            # If we are in SmoothQuant with dynamic activation scaling,
            # input x has to be a tuple of int8 tensor and fp32 scaling factors
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # 1. [dim0, local_dim] -> [dim0 * tp_size, local_dim]
            x = allgather(x, self.tp_group)

            # 2. [dim0 * tp_size, local_dim] -> [dim0, local_dim * tp_size]
            # 2.1 split
            split_size = shape(x, dim=0) / self.tp_size
            ndim = x.ndim()
            starts = [constant(int32_array([0])) for _ in range(ndim)]
            sizes = [shape(x, dim=d) for d in range(ndim)]
            sizes[0] = split_size
            sections = []
            for i in range(self.tp_size):
                starts[0] = split_size * i
                sections.append(slice(x, concat(starts), concat(sizes)))
            # 2.2 concat
            x = concat(sections, dim=1)

        return x


SmoothQuantColumnLinear = SmoothQuantLinear


class SmoothQuantRowLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )
        weights_dtype = dtype
        # Dirty hack to make it work with SmoothQuant int8 weights
        # reinterpreted as fp32 weights due to the int8 TRT plugin limitation.
        if quant_mode.has_act_and_weight_quant():
            assert self.in_features % 4 == 0
            self.in_features = self.in_features // 4
            weights_dtype = "float32"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)
        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode

    def forward(self, x):
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x


class SmoothQuantLayerNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 dtype=None,
                 quant_mode=QuantMode(0)):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant layer norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return smooth_quant_layer_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling())


class WeightOnlyQuantLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode.use_weight_only()):
        super().__init__()
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
            quant_type_size_in_bits = 8
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
            quant_type_size_in_bits = 4
        self.in_features = in_features
        self.out_features = out_features // tp_size
        # we use a fake tensor with data_type = float
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features *
                                           quant_type_size_in_bits / 32)),
                                dtype="float32")

        scale_shape = (self.out_features, )
        self.per_channel_scale = Parameter(shape=scale_shape, dtype=dtype)

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # 1. [dim0, local_dim] -> [dim0 * tp_size, local_dim]
            x = allgather(x, self.tp_group)

            # 2. [dim0 * tp_size, local_dim] -> [dim0, local_dim * tp_size]
            # 2.1 split
            split_size = shape(x, dim=0) / self.tp_size
            ndim = x.ndim()
            starts = [constant(int32_array([0])) for _ in range(ndim)]
            sizes = [shape(x, dim=d) for d in range(ndim)]
            sizes[0] = split_size
            sections = []
            for i in range(self.tp_size):
                starts[0] = split_size * i
                sections.append(slice(x, concat(starts), concat(sizes)))
            # 2.2 concat
            x = concat(sections, dim=1)

        return x


WeightOnlyQuantColumnLinear = WeightOnlyQuantLinear


class WeightOnlyQuantRowLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode.use_weight_only()):
        super().__init__()
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
        self.in_features = in_features // tp_size
        self.out_features = out_features
        #we use a fake tensor with data_type = float
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features / 4 /
                                           self.weight_only_quant_mode)),
                                dtype="float32")
        self.per_channel_scale = Parameter(shape=(self.out_features, ),
                                           dtype=dtype)

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x):
        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x


class SmoothQuantMLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.fc = SmoothQuantColumnLinear(hidden_size,
                                          ffn_hidden_size,
                                          bias=bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          gather_output=False,
                                          quant_mode=quant_mode)

        self.proj = SmoothQuantRowLinear(ffn_hidden_size,
                                         hidden_size,
                                         bias=bias,
                                         dtype=dtype,
                                         tp_group=tp_group,
                                         tp_size=tp_size,
                                         quant_mode=quant_mode)

        self.hidden_act = hidden_act
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                inter = quantize_tensor(inter,
                                        self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter = quantize_per_token(inter)
        output = self.proj(inter)
        return output


class FP8RowLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=dtype)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)
        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x):
        act_cast_out = cast(x, 'float32')

        quantized_out = quantize(act_cast_out,
                                 self.activation_scaling_factor.value, 'fp8')
        dequantized_out = dequantize(quantized_out,
                                     self.activation_scaling_factor.value)

        w_cast_out = cast(self.weight.value, 'float32')

        w_quant_out = quantize(w_cast_out, self.weights_scaling_factor.value,
                               'fp8')
        w_deq_out = dequantize(w_quant_out, self.weights_scaling_factor.value)

        x = matmul(dequantized_out, w_deq_out, transb=True)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x


class FP8Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=dtype)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        act_cast_out = cast(x, 'float32')
        quantized_out = quantize(act_cast_out,
                                 self.activation_scaling_factor.value, 'fp8')
        dequantized_out = dequantize(quantized_out,
                                     self.activation_scaling_factor.value)

        w_cast_out = cast(self.weight.value, 'float32')
        w_quant_out = quantize(w_cast_out, self.weights_scaling_factor.value,
                               'fp8')
        w_deq_out = dequantize(w_quant_out, self.weights_scaling_factor.value)

        x = matmul(dequantized_out, w_deq_out, transb=True)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # 1. [dim0, local_dim] -> [dim0 * tp_size, local_dim]
            x = allgather(x, self.tp_group)

            # 2. [dim0 * tp_size, local_dim] -> [dim0, local_dim * tp_size]
            # 2.1 split
            split_size = shape(x, dim=0) / self.tp_size
            ndim = x.ndim()
            starts = [constant(int32_array([0])) for _ in range(ndim)]
            sizes = [shape(x, dim=d) for d in range(ndim)]
            sizes[0] = split_size
            sections = []
            for i in range(self.tp_size):
                starts[0] = split_size * i
                sections.append(slice(x, concat(starts), concat(sizes)))
            # 2.2 concat
            x = concat(sections, dim=1)

        return x


class FP8MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.fc = FP8Linear(hidden_size,
                            ffn_hidden_size,
                            bias=bias,
                            dtype=dtype,
                            tp_group=tp_group,
                            tp_size=tp_size,
                            gather_output=False)
        self.proj = FP8RowLinear(ffn_hidden_size,
                                 hidden_size,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=tp_group,
                                 tp_size=tp_size)
        self.hidden_act = hidden_act
        self.dtype = dtype

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        output = self.proj(inter)
        return output


class SmoothQuantAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.padding,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 neox_rotary_style=False,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 paged_kv_cache=False,
                 quant_mode=QuantMode(0)):
        super().__init__()

        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = False
        self.paged_kv_cache = paged_kv_cache

        self.rotary_embedding_dim = 0
        self.neox_rotary_style = neox_rotary_style
        if self.position_embedding_type == PositionEmbeddingType.rope:
            # TODO: This branch crashes as default_net() doesn't exist at this point
            #       Once we add RoPE outside plugin, this branch won't be needed
            if not default_net().plugin_config.gpt_attention_plugin:
                raise ValueError(
                    'RoPE is only supported with GPTAttention plugin')
            else:
                self.rotary_embedding_dim = hidden_size // num_attention_heads

        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

        qkv_quant_mode = quant_mode
        if self.quant_mode.has_act_and_weight_quant():
            # We need to hijack quant_mode for QKV because QKV always uses per channel scaling
            qkv_quant_mode = QuantMode.from_description(
                True, True, quant_mode.has_per_token_dynamic_scaling(), True)

        if self.quant_mode.has_int8_kv_cache():
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        self.qkv = SmoothQuantColumnLinear(hidden_size,
                                           hidden_size * 3,
                                           bias=bias,
                                           dtype=dtype,
                                           tp_group=tp_group,
                                           tp_size=tp_size,
                                           gather_output=False,
                                           quant_mode=qkv_quant_mode)

        self.dense = SmoothQuantRowLinear(hidden_size,
                                          hidden_size,
                                          bias=bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          quant_mode=quant_mode)

    def forward(self,
                hidden_states: Union[Tensor, RaggedTensor],
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                input_lengths=None,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                inflight_batching_args=None,
                past_key_value_pointers=None):
        # TODO(nkorobov) add in-flight batching to SmoothQuant
        is_input_ragged_tensor = False
        if isinstance(hidden_states, RaggedTensor):
            assert input_lengths is None
            input_lengths = hidden_states.row_lengths
            max_input_length = hidden_states.max_row_length
            hidden_states = hidden_states.data
            is_input_ragged_tensor = True
        if default_net().plugin_config.smooth_quant_gemm_plugin:
            qkv = self.qkv(hidden_states)
        else:
            raise ValueError("smooth_quant_gemm_plugin is not set")
        if default_net().plugin_config.gpt_attention_plugin:
            assert sequence_length is not None
            assert past_key_value_length is not None
            assert masked_tokens is not None
            assert self.attention_mask_type == AttentionMaskType.causal, \
                'Plugin only support masked MHA.'
            assert input_lengths is not None
            kv_quant_scale = self.kv_orig_quant_scale.value if self.quant_mode.has_int8_kv_cache(
            ) else None
            kv_dequant_scale = self.kv_quant_orig_scale.value if self.quant_mode.has_int8_kv_cache(
            ) else None
            context, past_key_value = gpt_attention(
                qkv,
                past_key_value,
                sequence_length,
                past_key_value_length,
                masked_tokens,
                input_lengths,
                max_input_length,
                cache_indirection,
                self.num_attention_heads,
                self.attention_head_size,
                self.q_scaling,
                self.rotary_embedding_dim,
                self.neox_rotary_style,
                self.multi_block_mode,
                self.multi_query_mode,
                kv_quant_scale,
                kv_dequant_scale,
                self.quant_mode.has_int8_kv_cache(),
                kv_cache_block_pointers=kv_cache_block_pointers)
        else:
            assert self.paged_kv_cache == False

            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.quant_mode.has_int8_kv_cache():
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_dequantization_scale.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)
                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            def merge_caches():
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)
                return past_key_value

            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                key_length = shape(key, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                buffer = constant(
                    np.expand_dims(
                        np.tril(
                            np.ones(
                                (self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                        (0, 1)))
                causal_mask = slice(buffer, starts, sizes)

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                attention_scores = matmul(cast(query, 'float32'),
                                          cast(key, 'float32'))

                if self.attention_mask_type == AttentionMaskType.causal:
                    attention_scores = where(causal_mask, attention_scores,
                                             -10000.0)

                attention_scores = attention_scores / self.norm_factor
                attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

            past_key_value = merge_caches()

            if use_cache and self.quant_mode.has_int8_kv_cache():
                past_key_value = quantize_tensor(
                    past_key_value, self.kv_quantization_scale.value)

        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                context = quantize_tensor(
                    context, self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                context = quantize_per_token(context)

        context = self.dense(context)
        if is_input_ragged_tensor:
            context = RaggedTensor.from_row_lengths(context, input_lengths,
                                                    max_input_length)

        if use_cache:
            return (context, past_key_value)

        return context
