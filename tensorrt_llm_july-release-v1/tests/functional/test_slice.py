import unittest

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork, Profile,
                                    TrtRunner)

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', ), ('float16')])
    def test_slice(self, dtype):
        # test data
        x_shape = (1, 256)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        starts_data = torch.tensor([0, 128]).int()
        sizes_data = torch.tensor([1, 1]).int()

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            starts = Tensor(name='starts', shape=(2, ), dtype=trt.int32)

            sizes = Tensor(name='sizes', shape=(2, ), dtype=trt.int32)

            output = tensorrt_llm.functional.slice(x, starts, sizes).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        profiles = [
            Profile().add('starts', (0, 0), (0, 128),
                          (0, 256)).add('sizes', (1, 1), (1, 1), (1, 256))
        ]
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config=CreateConfig(profiles=profiles))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'x': x_data.numpy(),
                    'starts': starts_data.numpy(),
                    'sizes': sizes_data.numpy(),
                })

        # pytorch run
        ref = x_data[0:1, 128:129]

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])
