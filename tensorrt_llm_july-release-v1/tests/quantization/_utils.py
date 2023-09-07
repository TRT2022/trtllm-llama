import numpy as np
import torch

import tensorrt_llm


def woq_torch_dtype(dtype):
    if dtype == "float16":
        torch_dtype = torch.half
    else:
        assert (False)
    return torch_dtype


def woq_gen_weights(n, k, dtype):
    torch.manual_seed(0)

    torch_dtype = woq_torch_dtype(dtype)
    # Init operands for multiplication in int32
    shape2 = (n, k)
    weight = torch.rand(shape2, dtype=torch_dtype) * 2 - 1.0
    return weight


def woq_conversion(weight, wTypeId):
    # only support int8 weight only
    if wTypeId == 1:
        torch_wTypeId = torch.int8
    elif wTypeId == 2:
        torch_wTypeId = torch.quint4x2
    else:
        assert (False)
    return torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix(
        weight, torch_wTypeId)


def woq_gt_matmul(m,
                  mat1,
                  ref_torch_weights,
                  torch_weight_scales,
                  dtype,
                  bias=None):
    mat1 = mat1.to(dtype=torch.float)
    ref_torch_weights = ref_torch_weights.cuda().to(dtype=torch.float)
    # Do matmul
    ref = torch.matmul(mat1.cpu(), ref_torch_weights.cpu())

    # Prepare per element scaling
    scaling = torch_weight_scales.expand((m, -1))
    # Scale output and cast to right type
    ref = ref.cuda() * scaling.cuda()

    # Round to the nearest int to match cuda rounding
    if dtype == "int32":
        ref = torch.round(ref)

    # Cast ref to the required output typy
    ref = ref.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

    if bias is not None:
        ref += bias.cuda()

    return ref


def woq_assert_colwise_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    # check each column independently
    if ref.shape[0] > 1:
        for col_idx in range(ref.shape[-1]):
            col = ref[:, col_idx]
            max_val = torch.max(col).item()
            atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
            np.testing.assert_allclose(col.cpu().numpy(),
                                       act[:, col_idx].cpu().numpy(),
                                       atol=atol)
    else:
        max_val = torch.max(ref).item()
        atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   act.cpu().numpy(),
                                   atol=atol)


def gt_matmul_smooth_quant(mat1, mat2, scale_a_, scale_b_, dtype, bias=None):
    # Convert to int32 for PyTorch GT Matmul with accumulation in int32.
    mat1 = mat1.to(dtype=torch.int32)
    # Transpose the second matrix to support the native PyTorch format
    mat2 = mat2.cuda().transpose(0, 1).to(dtype=torch.int32)
    # Do matmul
    ref = torch.matmul(mat1.cpu(), mat2.cpu())

    m = 1
    for ii in range(len(mat1.shape) - 1):
        m *= mat1.shape[ii]
    n = mat2.shape[1]

    # Prepare per element scaling
    scale_a = scale_a_.expand((m, 1))
    scale_b = scale_b_.expand((1, n))
    scaling = torch.matmul(scale_a.cuda(), scale_b.cuda()).reshape(ref.shape)
    # Scale output and cast to right type
    ref = ref.cuda() * scaling.cuda()

    # Round to the nearest int to match cuda rounding
    if dtype == "int32":
        ref = torch.round(ref)

    # Cast ref to the required output type
    ref = ref.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

    if bias is not None:
        ref += bias.cuda()

    return ref


def gt_quantize_per_token(x):
    x = x.to(dtype=torch.float32)
    xmax, _ = x.abs().max(dim=-1, keepdim=True)
    x = (x * 127.0 / xmax).round().clip(-128, 127).to(dtype=torch.int8)
    scale_act = (xmax / 127.0).reshape(-1, 1)
    return x, scale_act
