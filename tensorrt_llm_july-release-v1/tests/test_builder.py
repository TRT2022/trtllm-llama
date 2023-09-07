import tempfile
import unittest

import tensorrt as trt

import tensorrt_llm


class MyAddModule(tensorrt_llm.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TestBuilder(unittest.TestCase):

    def test_basic_builder_flow(self):
        tensorrt_llm.logger.set_level('verbose')
        builder = tensorrt_llm.Builder()
        builder_config = builder.create_builder_config("test", "test")
        builder_config.trt_builder_config.set_flag(trt.BuilderFlag.REFIT)
        model = MyAddModule()

        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            assert tensorrt_llm.default_net() == network
            assert tensorrt_llm.default_trtnet() == network.trt_network
            x = tensorrt_llm.Tensor(name='x', dtype=trt.float32, shape=[1, 1])
            y = tensorrt_llm.Tensor(name='y', dtype=trt.float32, shape=[1, 1])
            # Prepare
            network.set_named_parameters(model.named_parameters())
            # Forward
            z = model(x, y)
            z.mark_output('z', trt.float32)

        with self.assertRaises(AssertionError):
            tensorrt_llm.default_net()
        engine = builder.build_engine(network, builder_config)
        assert engine is not None
        refit_engine = builder.refit_engine(network, engine)
        assert refit_engine is not None
        builder.save_config(builder_config, tempfile.mktemp())

    def test_top_level_dont_have_functional_apis(self):
        # This did not check all the functional apis, but should already prevent
        # from .functional import * in the __init__.py
        with self.assertRaises(AttributeError):
            x = tensorrt_llm.activation
            x = tensorrt_llm.assertion
            x = tensorrt_llm.einsum
            print(x)  # to avoid the delete of x
        x = tensorrt_llm.functional.activation
        x = tensorrt_llm.functional.assertion
        x = tensorrt_llm.functional.einsum
        print(x)  # to avoid the delete of x


if __name__ == '__main__':
    unittest.main()
