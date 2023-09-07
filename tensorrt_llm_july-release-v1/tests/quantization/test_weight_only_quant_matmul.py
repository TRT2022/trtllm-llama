import unittest

import _utils
import tensorrt as trt
import torch
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import weight_only_quant_matmul


class TestWeightOnlyQuantMatmul(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _unconvert_weights(self, weights, scales, dtype, wTypeId):
        if wTypeId == 1 or wTypeId == 2:
            pass
        else:
            assert (False)
        torch_dtype = _utils.woq_torch_dtype(dtype)
        # Init operands for multiplication in int32
        mat1 = torch.eye(weights.shape[0], dtype=torch_dtype)

        return self._run_matmul_plugin(mat1, weights, scales, dtype, wTypeId)

    def _run_matmul_plugin(self, mat1, processed_torch_weights,
                           torch_weight_scales, dtype, wTypeId):
        # int8/int4 workaround for the plugin weights
        processed_torch_weights = processed_torch_weights.view(
            dtype=torch.float32)

        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        # Allow SQ plugin of dtype type
        net.plugin_config.set_weight_only_quant_matmul_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            # Init TensorRT-LLM tensor for mat1
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
            # Init TensorRT-LLM tensor for weight
            y = Tensor(name='y',
                       shape=processed_torch_weights.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt("float32"))

            # Init TensorRT-LLM tensor for per channel scaling
            scale = Tensor(name='scale',
                           shape=torch_weight_scales.shape,
                           dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
            # Get output tensor for WOQ Matmul
            output = weight_only_quant_matmul(x, y, scale, wTypeId).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm._utils.str_dtype_to_trt(dtype)

        # Build engine consisting of only WOQ Matmul
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=False,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        # Infer engine
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'x': mat1.numpy(),
                    # convert to float32 as workaround
                    'y': processed_torch_weights.view(
                        dtype=torch.float32).numpy(),
                    'scale': torch_weight_scales.numpy()
                })

        return torch.tensor(outputs['output'])

    def _woq_matmul(self, m, n, k, dtype, wTypeId):
        # Init operands for multiplication in int32
        mat1 = _utils.woq_gen_weights(m, k, dtype) * 200.0
        weight = _utils.woq_gen_weights(k, n, dtype)

        ref_torch_weights, processed_torch_weights, torch_weight_scales = _utils.woq_conversion(
            weight, wTypeId)
        if wTypeId == 2:
            ref_torch_weights = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8(
                ref_torch_weights)

        output = self._run_matmul_plugin(mat1, processed_torch_weights,
                                         torch_weight_scales, dtype, wTypeId)

        ref = _utils.woq_gt_matmul(m, mat1, ref_torch_weights,
                                   torch_weight_scales, dtype)

        _utils.woq_assert_colwise_near_eq(ref, output, wTypeId)
        '''
        ref = ref.cpu().flatten()
        diff = abs(ref - output)

        max_diff = diff.max()
        ref_value_of_max_diff = ref[diff == max_diff]
        out_value_of_max_diff = output[diff == max_diff]
        print("###############\nmax diff is {} form {} vs {}\n###############\n\n".format(max_diff, out_value_of_max_diff, ref_value_of_max_diff))
        '''

    @parameterized.expand([
        (1, 1024, 4096, 'float16', 1),
        (128, 6144, 12288, 'float16', 1),  #FP16 * INT8
        (1, 1024, 4096, 'float16', 2),
        (128, 6144, 12288, 'float16', 2)
    ])  #FP16 * INT4
    def test_matmul(self, m, n, k, dtype, wTypeId):
        self._woq_matmul(m, n, k, dtype, wTypeId)

    @parameterized.expand([
        (1024, 4096, 'float16', 1), (4096, 512, 'float16', 1),
        (1024, 4096, 'float16', 2), (4096, 512, 'float16', 2)
    ])
    def test_conversion(self, n, k, dtype, wTypeId):
        weight_ref = _utils.woq_gen_weights(n, k, dtype)
        ref_int, perm_int, scale = _utils.woq_conversion(weight_ref, wTypeId)
        weight_act = self._unconvert_weights(perm_int, scale, dtype, wTypeId)

        _utils.woq_assert_colwise_near_eq(weight_ref, weight_act, wTypeId)

    def test_weight_only_matmul_no_plugin(self):
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            tensorrt_llm.default_trtnet()
            # Get output tensor for SQ gemm
            with self.assertRaisesRegex(
                    TypeError,
                    "Weight Only Qunat MatMul is only supported with plugin"):
                weight_only_quant_matmul(None, None, None, 0)


if __name__ == '__main__':
    unittest.main()
