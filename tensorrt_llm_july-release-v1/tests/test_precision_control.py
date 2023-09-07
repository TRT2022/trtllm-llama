import unittest

import numpy as np
import torch
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import rms_norm


class TestPrecisionControl(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_precision_control(self):
        # test data
        test_shape = [2, 5, 10, 10]
        dtype = 'float32'
        x_data = torch.randn(*test_shape)
        m = LlamaRMSNorm(test_shape[-1])  # LlamaRMSNorm only supports last dim

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            output = rms_norm(x,
                              test_shape[-1],
                              weight=tensorrt_llm.constant(
                                  m.weight.detach().cpu().numpy()))
            output = output.trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(precision_constraints='obey'))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-6)
