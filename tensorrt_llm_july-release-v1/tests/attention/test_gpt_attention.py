import math
import unittest
from itertools import product

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import GPT2Config, GPTBigCodeConfig, GPTJConfig, LlamaConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import \
    GPTBigCodeAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      _expand_mask,
                                                      _make_causal_mask)

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import GenerationSequence, KVCacheManager


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = list(
            product(['gpt2_attention', 'llama_attention', 'gptj_attention'],
                    [ContextFMHAType.disabled], ['float16', 'bfloat16'], [2],
                    [128], [4], [64], [False], [False], [False], [False],
                    [1, 4], [True, False]))

        # TODO: add more unit tests
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16'], [2], [90, 1024], [4], [32, 64, 128],
                    [False], [False], [False], [False, True], [1], [False]))

        # Test cases for the multi-block MMHA.
        test_cases += list(
            product(['llama_attention'], [
                ContextFMHAType.enabled, ContextFMHAType.enabled_with_fp32_acc
            ], ['float16', 'bfloat16'], [2], [2048], [4], [64], [True], [False],
                    [False], [False], [1, 4], [False]))

        # Test cases for the int8 K/V cache.
        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled],
                    ['float16', 'float32'], [2], [128], [4], [64], [False],
                    [False], [True], [False], [1, 4], [False]))

        # test cases for multi-query attention
        test_cases += list(
            product(
                ['gpt2_attention', 'llama_attention', 'gpt_bigcode_attention'],
                [
                    ContextFMHAType.disabled, ContextFMHAType.enabled,
                    ContextFMHAType.enabled_with_fp32_acc
                ], ['float16', 'bfloat16'], [2], [128], [4], [64], [False],
                [True], [False], [False], [1, 4], [False]))

        test_cases += list(
            product(['gpt2_attention'], [ContextFMHAType.disabled], ['float16'],
                    [2], [128], [4], [64], [False], [True], [False], [False],
                    [1, 4], [False]))

        return test_cases

    def custom_name_func(testcase_func, param_num, param):
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )

    @parameterized.expand(load_test_cases, name_func=custom_name_func)
    def test_gpt_attention(self, attention_type, context_fmha_type, dtype,
                           batch_size, in_len, num_heads, head_size,
                           enable_multi_block_mmha,
                           enable_multi_query_attention, use_int8_kv_cache,
                           enable_remove_input_padding, beam_width,
                           paged_kv_cache):

        # Skip duplicated tests.
        if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc and \
            dtype == 'bfloat16':
            pytest.skip("bfloat16 Context FMHA will always accumulate on FP32, \
                so it has been tested with ContextFMHAType.enabled")

        session = None
        kv_cache_dtype = 'int8' if use_int8_kv_cache else dtype
        if use_int8_kv_cache:
            # Fixing seed to avoid flakiness in tests with quantization
            torch.manual_seed(42)

        if beam_width != 1 and paged_kv_cache:
            pytest.skip(
                "Beam search and paged kv cache are not supported in this test yet"
            )

        tokens_per_block = 16 if paged_kv_cache else -1

        def _construct_execution(session, input_tensor, weight, bias,
                                 past_key_value, pointer_array, sequence_length,
                                 past_key_value_length, masked_tokens,
                                 input_lengths, max_input_sequence_length,
                                 cache_indirection, num_heads, hidden_size,
                                 output, dtype, shape_dict,
                                 enable_multi_query_attention,
                                 kv_int8_quant_scale, kv_int8_dequant_scale):
            head_size = hidden_size // num_heads
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.set_gpt_attention_plugin(dtype)
            net.plugin_config.set_context_fmha(context_fmha_type)
            if enable_remove_input_padding:
                net.plugin_config.enable_remove_input_padding()
            if paged_kv_cache:
                net.plugin_config.enable_paged_kv_cache()

            with tensorrt_llm.net_guard(net):
                x_tensor = Tensor(name='input',
                                  shape=tuple(input_tensor.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                past_key_value_tensor = Tensor(
                    name='past_key_value',
                    shape=tuple(past_key_value.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))
                sequence_length_tensor = Tensor(
                    name='sequence_length',
                    shape=tuple(sequence_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                past_key_value_length_tensor = Tensor(
                    name='past_key_value_length',
                    shape=tuple(past_key_value_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                masked_tokens_tensor = Tensor(
                    name='masked_tokens',
                    shape=tuple(masked_tokens.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                input_lengths_tensor = Tensor(
                    name='input_lengths',
                    shape=tuple(input_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                max_input_sequence_length_tensor = Tensor(
                    name='max_input_sequence_length',
                    shape=tuple(max_input_sequence_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                cache_indirection_tensor = Tensor(
                    name='cache_indirection',
                    shape=tuple(cache_indirection.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                kv_int8_quant_scale_tensor = None
                kv_int8_dequant_scale_tensor = None
                if use_int8_kv_cache:
                    kv_int8_quant_scale_tensor = Tensor(
                        name='kv_int8_quant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                    kv_int8_dequant_scale_tensor = Tensor(
                        name='kv_int8_dequant_scale',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                pointer_array_tensor = None
                if paged_kv_cache:
                    pointer_array_tensor = Tensor(
                        name='kv_cache_block_pointers',
                        shape=tuple(pointer_array.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))

                linear = tensorrt_llm.layers.Linear(hidden_size,
                                                    weight.size()[-1],
                                                    bias=attention_type in [
                                                        'gpt2_attention',
                                                        'llama_attention',
                                                        'gpt_bigcode_attention'
                                                    ])
                linear.weight.value = np.ascontiguousarray(
                    torch_to_numpy(weight.T.cpu()))
                if attention_type in [
                        'gpt2_attention', 'llama_attention',
                        'gpt_bigcode_attention'
                ]:
                    linear.bias.value = torch_to_numpy(bias.cpu())
                qkv = linear(x_tensor)

                rotary_embedding_dim = head_size if attention_type in [
                    'llama_attention', 'gptj_attention'
                ] else 0
                neox_rotary_style = True if attention_type in [
                    'llama_attention'
                ] else False
                outputs = tensorrt_llm.functional.gpt_attention(
                    qkv,
                    past_key_value_tensor,
                    sequence_length_tensor,
                    past_key_value_length_tensor,
                    masked_tokens_tensor,
                    input_lengths_tensor,
                    max_input_sequence_length_tensor,
                    cache_indirection_tensor,
                    num_heads=num_heads,
                    head_size=head_size,
                    q_scaling=1.0,
                    rotary_embedding_dim=rotary_embedding_dim,
                    neox_rotary_style=neox_rotary_style,
                    multi_block_mode=enable_multi_block_mmha,
                    multi_query_mode=enable_multi_query_attention,
                    kv_orig_quant_scale=kv_int8_quant_scale_tensor,
                    kv_quant_orig_scale=kv_int8_dequant_scale_tensor,
                    use_int8_kv_cache=use_int8_kv_cache,
                    kv_cache_block_pointers=pointer_array_tensor)

                net._mark_output(outputs[0],
                                 'output',
                                 dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                net._mark_output(
                    outputs[1],
                    'present_key_value',
                    dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))

            inputs = {
                'input': input_tensor,
                'past_key_value': past_key_value,
                'sequence_length': sequence_length,
                'past_key_value_length': past_key_value_length,
                'masked_tokens': masked_tokens,
                'input_lengths': input_lengths,
                'max_input_sequence_length': max_input_sequence_length,
                'cache_indirection': cache_indirection,
            }
            if use_int8_kv_cache:
                inputs['kv_int8_quant_scale'] = kv_int8_quant_scale
                inputs['kv_int8_dequant_scale'] = kv_int8_dequant_scale

            if paged_kv_cache:
                inputs['kv_cache_block_pointers'] = pointer_array

            outputs = {
                'output': output,
                'present_key_value': past_key_value,
            }

            stream = torch.cuda.current_stream()
            builder_config = builder.create_builder_config(
                name=attention_type, precision=dtype, int8=use_int8_kv_cache)
            if session is None:
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(
                    engine)
            session.run(inputs=inputs,
                        outputs=outputs,
                        stream=stream.cuda_stream)

            torch.cuda.synchronize()
            return session, outputs['output'], outputs['present_key_value']

        hidden_size = num_heads * head_size  # embed dimension
        # If enable_multi_query_attention is true and that GPTBigCodeAttention is tested, use compacted IO shape.
        # If enable_multi_query_attention is true but other attention types are tested, use regular IO shape.
        # This is because GPTBigCodeAttention requires single KV head when multi-query attention is used. Other attention types do not support
        # single KV head natively so we emulate the effect of multi-query attention by repeating KV heads.
        kv_num_heads = 1 if enable_multi_query_attention and attention_type == 'gpt_bigcode_attention' else num_heads
        qkv_hidden_size = hidden_size + 2 * kv_num_heads * head_size
        out_len = 8
        max_seq_len = in_len + 24
        max_blocks_per_seq = math.ceil(max_seq_len / tokens_per_block)
        blocks = math.ceil(
            (batch_size * beam_width * max_seq_len) / tokens_per_block)
        shape_dict = {
            'weight': (hidden_size, qkv_hidden_size),
            'bias': (qkv_hidden_size, ),
            'past_key_value_length': (2, ),
            'sequence_length': (batch_size, ),
            'masked_tokens': (batch_size, max_seq_len),
            'input_lengths': (batch_size, ),
            'max_input_sequence_length': (in_len, ),
            'kv_int8_quant_scale': (1, ),
            'kv_int8_dequant_scale': (1, ),
            'cache_indirection': (batch_size, 1, max_seq_len)
        }
        if paged_kv_cache:
            shape_dict['past_key_value'] = (blocks, 2, kv_num_heads,
                                            tokens_per_block, head_size)
        else:
            shape_dict['past_key_value'] = (batch_size, 2, kv_num_heads,
                                            max_seq_len, head_size)
        shape_dict['present_key_value'] = shape_dict['past_key_value']

        present_key_value = torch.zeros(
            shape_dict['past_key_value'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(kv_cache_dtype),
            device='cuda')
        # Init KV cache block manager
        if paged_kv_cache:
            manager = KVCacheManager([present_key_value],
                                     blocks,
                                     tokens_per_block,
                                     max_blocks_per_seq,
                                     beam_width=beam_width)

            # Add sequences to the manager
            for bi in range(batch_size):
                manager.add_sequence(
                    GenerationSequence(seq_idx=bi, batch_idx=bi), in_len)

        weight = torch.randn(shape_dict['weight'],
                             dtype=str_dtype_to_torch(dtype),
                             device='cuda') * 1e-3
        # FIXME(qijun): test_gpt_attention_llama_attention_False_float16_2_90_4_64_False_False_False_True
        # fails with xavier_uniform_ initialization
        # torch.nn.init.xavier_uniform_(weight)

        bias = torch.randn(shape_dict['bias'],
                           dtype=str_dtype_to_torch(dtype),
                           device='cuda') * 1e-2
        torch_present = None

        kv_int8_dequant_scale = torch.randint(
            1,
            10,
            shape_dict['kv_int8_dequant_scale'],
            dtype=str_dtype_to_torch(kv_cache_dtype),
            device='cuda') * 0.0001
        kv_int8_quant_scale = 1.0 / kv_int8_dequant_scale

        ConfigCls = None
        AttentionCls = None
        if attention_type == 'gpt2_attention':
            ConfigCls = GPT2Config
            AttentionCls = GPT2Attention
        elif attention_type == 'gptj_attention':
            ConfigCls = GPTJConfig
            AttentionCls = GPTJAttention
        elif attention_type == 'llama_attention':
            ConfigCls = LlamaConfig
            AttentionCls = LlamaAttention
        elif attention_type == 'gpt_bigcode_attention':
            ConfigCls = GPTBigCodeConfig
            AttentionCls = GPTBigCodeAttention

        configuration = ConfigCls(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            vocab_size=51200,
            use_cache=True,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            hidden_act='gelu',
            torch_dtype=dtype,
        )
        attention = AttentionCls(configuration).cuda().eval()
        if attention_type == 'gpt2_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.clone().detach(), requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
                                 device='cuda'),
                requires_grad=False)
        elif attention_type == 'llama_attention':
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            q_b, k_b, v_b = torch.tensor_split(bias, 3)
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.q_proj.bias = torch.nn.parameter.Parameter(
                data=q_b.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.bias = torch.nn.parameter.Parameter(
                data=k_b.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.bias = torch.nn.parameter.Parameter(
                data=v_b.contiguous().clone().detach(), requires_grad=False)

            attention.o_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.o_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
                                 device='cuda'),
                requires_grad=False)
        elif attention_type == 'gptj_attention':
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            attention.q_proj.weight = torch.nn.parameter.Parameter(
                data=q_w.contiguous().clone().detach(), requires_grad=False)
            attention.k_proj.weight = torch.nn.parameter.Parameter(
                data=k_w.contiguous().clone().detach(), requires_grad=False)
            attention.v_proj.weight = torch.nn.parameter.Parameter(
                data=v_w.contiguous().clone().detach(), requires_grad=False)

            attention.out_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
        elif attention_type == 'gpt_bigcode_attention':
            attention.c_attn.weight = torch.nn.parameter.Parameter(
                data=weight.transpose(0, 1).clone().detach(),
                requires_grad=False)
            attention.c_attn.bias = torch.nn.parameter.Parameter(
                data=bias.clone().detach(), requires_grad=False)
            attention.c_proj.weight = torch.nn.parameter.Parameter(
                data=torch.eye(hidden_size,
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda'),
                requires_grad=False)
            attention.c_proj.bias = torch.nn.parameter.Parameter(
                data=torch.zeros((hidden_size, ),
                                 dtype=str_dtype_to_torch(dtype),
                                 device='cuda'),
                requires_grad=False)
            attention.layer_idx = 0
        else:
            raise RuntimeError("attention_type not properly set")

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * (in_len // 2)
        ctx_attention_mask = torch.ones((batch_size, in_len),
                                        dtype=torch.int32,
                                        device='cuda')
        for i in range(batch_size):
            ctx_attention_mask[i, input_lengths[i]:in_len] = 0

        masked_tokens = torch.zeros((batch_size, max_seq_len),
                                    dtype=torch.int32,
                                    device='cuda')
        for i in range(batch_size):
            masked_tokens[i, input_lengths[i]:in_len] = 1

        # Setup weights/biases for MQA: key/value shares weights/biases across heads
        if attention_type != 'gpt_bigcode_attention' and enable_multi_query_attention:
            q_w, k_w, v_w = torch.tensor_split(weight, 3, dim=-1)
            q_b, k_b, v_b = torch.tensor_split(bias, 3)
            k_w_head = k_w[:, :head_size]
            v_w_head = k_w[:, :head_size]
            k_w_repeat = k_w_head.repeat(1, num_heads)
            v_w_repeat = v_w_head.repeat(1, num_heads)
            k_b_head = k_b[:head_size]
            v_b_head = v_b[:head_size]
            k_b_repeat = k_b_head.repeat(num_heads)
            v_b_repeat = v_b_head.repeat(num_heads)

            # Setup MQA weights/biases for _construct_execution()
            weight = torch.cat([q_w, k_w_repeat, v_w_repeat], dim=-1)
            bias = torch.cat([q_b, k_b_repeat, v_b_repeat])

            # Plugin will always use compacted MQA format without repeating KV heads
            weight_plugin = torch.cat([q_w, k_w_head, v_w_head], dim=-1)
            bias_plugin = torch.cat([q_b, k_b_head, v_b_head])

            # Setup MQA weights/biases for torch
            if attention_type == 'gpt2_attention':
                attention.c_attn.weight = torch.nn.parameter.Parameter(
                    data=weight.clone().detach(), requires_grad=False)
                attention.c_attn.bias = torch.nn.parameter.Parameter(
                    data=bias.clone().detach(), requires_grad=False)
            elif attention_type == 'llama_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.k_proj.bias = torch.nn.parameter.Parameter(
                    data=k_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.bias = torch.nn.parameter.Parameter(
                    data=v_b_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            elif attention_type == 'gptj_attention':
                attention.k_proj.weight = torch.nn.parameter.Parameter(
                    data=k_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
                attention.v_proj.weight = torch.nn.parameter.Parameter(
                    data=v_w_repeat.contiguous().clone().detach(),
                    requires_grad=False)
            else:
                raise RuntimeError("attention_type not properly set")

        else:  # not enable_multi_query_attention
            weight_plugin = weight
            bias_plugin = bias

        def remove_input_padding(tensor):
            batch_size = tensor.shape[0]
            tmp = []
            for b in range(batch_size):
                tmp.append(tensor[b, :in_len // 2, :])
            return torch.cat(tmp,
                             dim=1).cuda().reshape(1, batch_size * in_len // 2,
                                                   -1)

        cache_indirection = torch.full((
            batch_size,
            beam_width,
            max_seq_len,
        ),
                                       0,
                                       dtype=torch.int32,
                                       device='cuda')

        def verify_kv_cache(torch_present):
            if not use_int8_kv_cache and not enable_multi_query_attention and beam_width == 1:
                if paged_kv_cache:
                    kv_cache_cont = manager.blocks_manager.get_continous_caches(
                        0)
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2)
                else:
                    kv_cache_cont = present_key_value
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2, 3, 4)

                key, value = kv_cache_cont.to(torch.float32).chunk(2)

                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # K cache has shape: [batch_size, max_blocks_per_seq, kv_num_heads, tokens_per_block, head_size]
                    key = key.reshape(batch_size, max_blocks_per_seq,
                                      kv_num_heads, tokens_per_block, head_size)
                    key = key.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, kv_num_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    key = key.reshape(batch_size, kv_num_heads, max_seq_len,
                                      head_size)

                # Note K and V shares the same layout now.
                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # V cache has shape: [batch_size, max_blocks_per_seq, kv_num_heads, tokens_per_block, head_size]
                    value = value.reshape(batch_size, max_blocks_per_seq,
                                          kv_num_heads, tokens_per_block,
                                          head_size)
                    value = value.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, kv_num_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    value = value.reshape(batch_size, kv_num_heads, max_seq_len,
                                          head_size)

                tols = {
                    "float32": 2e-04,
                    "float16": 2e-04,
                    "bfloat16": 2e-01,
                }

                np.testing.assert_allclose(
                    key.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[0].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])
                np.testing.assert_allclose(
                    value.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[1].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])

        for step in range(out_len):
            sequence_length = torch.ones(
                (batch_size, ), dtype=torch.int32,
                device='cuda') * (in_len + max(0, step - 1))
            max_input_length = torch.zeros(
                shape_dict['max_input_sequence_length'],
                dtype=torch.int32,
                device='cuda')

            pointer_array = None
            if paged_kv_cache:
                # Get arrays of pointers to the "pages" of KV values
                pointer_array = manager.get_pointer_arrays()[0]

            if step == 0:
                if paged_kv_cache:
                    # Reassemble pointer array to have KV cache for bs context invokations instead of batch_beam
                    pointer_array = pointer_array[:, 0, :, :]
                    pointer_array = pointer_array.reshape(
                        batch_size, 1, 2, max_blocks_per_seq * 2)

                # Context stage
                shape_dict['input'] = (batch_size, in_len, hidden_size)
                shape_dict['output'] = shape_dict['input']
                past_key_value_length = torch.tensor([step, 1],
                                                     dtype=torch.int32)

                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                # torch execution
                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)

                attention_mask = _expand_mask(ctx_attention_mask,
                                              dtype=str_dtype_to_torch(dtype),
                                              tgt_len=in_len)
                if attention_type == 'gpt2_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        use_cache=True,
                        attention_mask=attention_mask)
                elif attention_type == 'llama_attention':
                    attention_mask = attention_mask + _make_causal_mask(
                        input_tensor.shape[:2],
                        dtype=str_dtype_to_torch(dtype),
                        device='cuda',
                        past_key_values_length=0)
                    torch_output, _, torch_present = attention(
                        input_tensor,
                        past_key_value=None,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gptj_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gpt_bigcode_attention':
                    attention_mask = _expand_mask(
                        ctx_attention_mask,
                        dtype=str_dtype_to_torch(dtype),
                        tgt_len=in_len)
                    # source shape = (b, 1, s_query, s_key)
                    # target shape = (b, s_query, h, s_key)
                    attention_mask = (attention_mask >= 0).permute(
                        [0, 2, 1, 3]).expand(batch_size, in_len, num_heads,
                                             in_len)
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=None,
                        attention_mask=attention_mask,
                        use_cache=True)
                else:
                    raise RuntimeError("attention_type not properly set")

                torch.cuda.synchronize()

                if enable_remove_input_padding:
                    shape_dict['input'] = (1, batch_size * in_len // 2,
                                           hidden_size)
                    input_tensor = remove_input_padding(input_tensor)

                shape_dict['output'] = shape_dict['input']
                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')

                session, output, present_key_value = _construct_execution(
                    session, input_tensor, weight_plugin, bias_plugin,
                    present_key_value, pointer_array, sequence_length,
                    past_key_value_length, masked_tokens, input_lengths,
                    max_input_length, cache_indirection, num_heads, hidden_size,
                    output, dtype, shape_dict, enable_multi_query_attention,
                    kv_int8_quant_scale, kv_int8_dequant_scale)
                del session
                session = None

                if enable_remove_input_padding:
                    torch_output = remove_input_padding(torch_output)
                    np.testing.assert_allclose(
                        output.to(torch.float32).cpu().numpy(),
                        torch_output.to(torch.float32).cpu().numpy(),
                        atol=5e-3)
                else:
                    np.testing.assert_allclose(
                        output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        torch_output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        atol=5e-3)

                verify_kv_cache(torch_present)

            else:
                # Generation stage
                shape_dict['input'] = (batch_size, 1, hidden_size)
                past_key_value_length = torch.tensor([in_len + step - 1, 0],
                                                     dtype=torch.int32)
                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                ctx_attention_mask = torch.cat((ctx_attention_mask,
                                                ctx_attention_mask.new_ones(
                                                    (batch_size, 1))),
                                               dim=-1).contiguous()

                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                attention_mask = _expand_mask(ctx_attention_mask,
                                              dtype=str_dtype_to_torch(dtype),
                                              tgt_len=1)

                # torch execution
                if attention_type == 'gpt2_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        use_cache=True,
                        attention_mask=attention_mask)
                elif attention_type == 'llama_attention':
                    attention_mask = attention_mask + _make_causal_mask(
                        input_tensor.shape[:2],
                        dtype=str_dtype_to_torch(dtype),
                        device='cuda',
                        past_key_values_length=in_len + step - 1)
                    torch_output, _, torch_present = attention(
                        input_tensor,
                        past_key_value=torch_present,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gptj_attention':
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=True)
                elif attention_type == 'gpt_bigcode_attention':
                    # source shape = (b, 1, 1, s_key)
                    # target shape = (b, 1, h, s_key)
                    key_seqlen = in_len + step  # ctx_attention_mask.shape[1]
                    attention_mask = (attention_mask >= 0).permute(
                        [0, 2, 1, 3]).expand(batch_size, 1, num_heads,
                                             key_seqlen)
                    torch_output, torch_present = attention(
                        input_tensor,
                        layer_past=torch_present,
                        use_cache=True,
                        attention_mask=attention_mask)

                def tile_beam_width(tensor: torch.Tensor, num_beams: int):
                    if num_beams == 1:
                        return tensor
                    else:
                        new_shape = np.array(tensor.shape)
                        new_shape[0] = new_shape[0] * num_beams
                        tile_size = np.ones(new_shape.shape, dtype=np.int32)
                        tile_size = np.insert(tile_size, 1, num_beams)
                        new_tensor = torch.unsqueeze(tensor, 1)
                        new_tensor = new_tensor.tile(tile_size.tolist())
                        new_tensor = new_tensor.reshape(new_shape.tolist())
                        return new_tensor

                torch_output = tile_beam_width(torch_output, beam_width)
                torch_output = torch_output.reshape(
                    [batch_size, beam_width, -1])

                torch.cuda.synchronize()

                if step == 1:
                    tiled_input_tensor = tile_beam_width(
                        input_tensor, beam_width)
                    tiled_attention_mask = tile_beam_width(
                        attention_mask, beam_width)
                    tiled_input_lengths = tile_beam_width(
                        input_lengths, beam_width)
                    tiled_masked_tokens = tile_beam_width(
                        masked_tokens, beam_width)
                    tiled_present_key_value = tile_beam_width(
                        present_key_value,
                        beam_width) if not paged_kv_cache else present_key_value
                    tiled_sequence_length = tile_beam_width(
                        sequence_length, beam_width)

                if enable_remove_input_padding:
                    shape_dict['input'] = (1, batch_size, hidden_size)
                    input_tensor = input_tensor.view(shape_dict['input'])

                # TRT LLM execution
                shape_dict['output'] = shape_dict['input']
                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')
                if step == 1:
                    input_tensor = input_tensor.reshape(
                        [batch_size, hidden_size])
                    tiled_input_tensor = tile_beam_width(
                        input_tensor, beam_width)
                    tiled_input_tensor = tiled_input_tensor.reshape(
                        [1, batch_size * beam_width, hidden_size])
                    output = output.reshape([batch_size, hidden_size])
                    tiled_output = tile_beam_width(output, beam_width)
                    tiled_output = tiled_output.reshape(
                        [1, batch_size * beam_width, hidden_size])

                session, tiled_output, present_key_value = _construct_execution(
                    session, tiled_input_tensor, weight_plugin, bias_plugin,
                    tiled_present_key_value, pointer_array,
                    tiled_sequence_length, past_key_value_length,
                    tiled_masked_tokens, tiled_input_lengths, max_input_length,
                    cache_indirection, num_heads, hidden_size, tiled_output,
                    dtype, shape_dict, enable_multi_query_attention,
                    kv_int8_quant_scale, kv_int8_dequant_scale)

                del session
                session = None

                # compare result
                np.testing.assert_allclose(
                    torch.flatten(tiled_output).to(torch.float32).cpu().numpy(),
                    torch.flatten(torch_output).to(torch.float32).cpu().numpy(),
                    atol=2e-3)

            if paged_kv_cache:
                # Iterate to the next step. Increase number of tokens for all unfinished sequences
                # And allocate new blocks if needed
                manager.step([False] * batch_size)


if __name__ == "__main__":
    unittest.main()
