import unittest

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_flip(self):
        dtype = 'float32'
        x_data = torch.randn(4, 6, 3, 4)
        dims = [-2, 0, -3]
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = tensorrt_llm.functional.flip(x, dims).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.flip(x_data, dims)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])
