import math
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt

from ._common import default_net, default_trtnet, precision
from ._utils import (dim_resolve_negative, dim_to_trt_axes, fp32_array,
                     int32_array, np_dtype_to_trt, str_dtype_to_np,
                     str_dtype_to_trt)
from .plugin import _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE


class DimRange(object):

    def __init__(self, shape):
        self.min = []
        self.opt = []
        self.max = []
        for dim in shape:
            if isinstance(dim, (list, tuple)):
                assert len(dim) == 3
                self.min.append(dim[0])
                self.opt.append(dim[1])
                self.max.append(dim[2])
            elif isinstance(dim, int):
                self.min.append(dim)
                self.opt.append(dim)
                self.max.append(dim)
            else:
                raise AttributeError(
                    f'Dimension should be [min, opt, max] (dynamic shape) or int (specific value). Got {type(dim)}'
                )


class Tensor(object):
    '''
    The class to represent dense tensors.

    A dense tensor is named, has a shape and contains typed elements. Each
    dimension of a tensor can either be static or dynamic. Static dimensions
    are known at engine compilation by TensorRT. Dynamic dimensions can take
    values determined at runtime. The tensor can be located on the host (CPU)
    or the device (GPU).
    '''

    def __init__(self,
                 name,
                 dtype,
                 shape,
                 dim_range=None,
                 is_network_input=True,
                 location=trt.TensorLocation.DEVICE):
        '''
        Parameters:
            name : str
                The name of the tensor.

            dtype : tensorrt.DataType
                The type of the elements of the tensor. See the TensorRT
                documentation for list of supported data types.

            shape : tensorrt.Dims
                The dimensions of the tensor. In TensorRT-LLM, tensors can have
                static or dynamic dimensions (it is possible to mix static and
                dynamic dimensions).  A static dimension is known when the
                TensorRT engine is built. A dynamic dimension can be set when
                the engine is executed. Use -1 for dynamic dimensions.

            dim_range : OrderedDict
                An ordered dictionary (the positions of the elements matter)
                that associates a name and a range of values to the dimensions.
                For a static dimension, the range must be limited to a single
                value. For a dynamic dimension, the range is defined by three
                values [min, opt, max] where min and max are, respectively, the
                smallest and largest possible values of that dimension.  The
                opt value is used by TensorRT to optimize the engine for the
                most common case.

            is_network_input : bool
                A boolean indicating if that tensor is an input of the network.
                Inputs must be provided by the user to run the engine.

            location : tensorrt.TensorLocation
                A flag to indicate where the tensor will be located. It can be
                on the host (CPU) or the device (GPU).
        '''
        self.dim_range = []
        if is_network_input:
            if dim_range is not None:
                assert isinstance(dim_range, OrderedDict)
                assert len(dim_range) >= 1
                num_dim_range = len(list(dim_range.items())[0][1])
                assert num_dim_range >= 1
                for i in range(num_dim_range):
                    range_shape = []
                    for dim_name in dim_range.keys():
                        assert isinstance(dim_range[dim_name], (list, tuple))
                        assert len(dim_range[dim_name]) == num_dim_range
                        range_shape.append(dim_range[dim_name][i])
                    self.dim_range.append(DimRange(range_shape))

            default_net()._add_input(self, name, dtype, shape, dim_range)
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.location = location

    @property
    def name(self):
        '''
        The name of the tensor.
        '''
        return self.trt_tensor.name

    @name.setter
    def name(self, name):
        '''
        Set the name of the tensor.
        '''
        if name is not None:
            self.trt_tensor.name = name

    @property
    def dtype(self):
        '''
        The type of the elements in the tensor.
        '''
        return self.trt_tensor.dtype

    @dtype.setter
    def dtype(self, dtype):
        '''
        Set the type of the elements in the tensor.
        '''
        if dtype is not None:
            self.trt_tensor.dtype = dtype

    @property
    def shape(self):
        '''
        The shape of the tensor.
        '''
        return self.size()

    @shape.setter
    def shape(self, shape):
        '''
        Set the shape of the tensor. See __init__.
        '''
        if shape is not None:
            self.trt_tensor.shape = shape

    @property
    def location(self):
        '''
        The physical location of the tensor (on the host or the device).
        '''
        return self.trt_tensor.location

    @location.setter
    def location(self, location):
        '''
        Set the physical location of the tensor (on the host or the device). See __init__.
        '''
        if location is not None:
            self.trt_tensor.location = location

    def mark_output(self, name, dtype):
        '''
        Mark a tensor as a network output.

        When a tensor is marked as an output, its content can be obtained after
        the execution of the TensorRT engine. The user is responsible for
        allocating buffers to store the output tensors when preparing the
        execution of the TensorRT engine.
        '''
        default_net()._mark_output(self, name, dtype)

    def __add__(self, b):
        '''
        See functional.add.
        '''
        return add(self, b)

    def __radd__(self, b):
        '''
        See functional.add.
        '''
        return add(b, self)

    def __sub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(self, b)

    def __rsub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(b, self)

    def __mul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(self, b)

    def __rmul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(b, self)

    def __truediv__(self, b):
        '''
        See functional.div.
        '''
        return div(self, b)

    def __lt__(self, b):
        '''
        See functional.lt.
        '''
        return lt(self, b)

    def __gt__(self, b):
        '''
        See functional.gt.
        '''
        return gt(self, b)

    def __eq__(self, b):
        '''
        See functional.eq.
        '''
        return eq(self, b)

    def __ge__(self, b):
        '''
        Maps to functional.gt or functional.eq.
        '''
        return self.__gt__(b) or self.__eq__(b)

    def __le__(self, b):
        '''
        Maps to functional.lt or functional.eq.
        '''
        return self.__lt__(b) or self.__eq__(b)

    def view(self, shape, zero_is_placeholder=True):
        '''
        See functional.view.
        '''
        return view(self, shape, zero_is_placeholder)

    def permute(self, dims):
        '''
        See functional.permute.
        '''
        return permute(self, dims)

    def transpose(self, dim0, dim1):
        '''
        See functional.transpose.
        '''
        return transpose(self, dim0, dim1)

    def mean(self, dim, keepdim=False):
        '''
        See functional.mean.
        '''
        return mean(self, dim, keepdim)

    def max(self, dim, keepdim=False):
        '''
        See functional.max.
        '''
        return max(self, dim, keepdim)

    def abs(self):
        '''
        See functional.abs.
        '''
        return abs(self)

    def sqrt(self):
        '''
        See functional.sqrt.
        '''
        return sqrt(self)

    def cast(self, dtype):
        '''
        See functional.cast.
        '''
        return cast(self, dtype)

    def size(self, dim=None):
        '''
        Returns the shape of the tensor if the dim parameter is None.
        Otherwise, returns a size of the dimension indicated by dim. The
        behavior is undefined if dim is negative or exceeds the rank of the
        tensor.
        '''
        if dim is None:
            return self.trt_tensor.shape

        return self.trt_tensor.shape[dim]

    def rank(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return len(self.trt_tensor.shape)

    def ndim(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return self.rank()

    def split(self, split_size_or_sections, dim=0):
        '''
        See functional.split.
        '''
        return split(self, split_size_or_sections, dim)

    def is_dynamic(self, dim=None):
        '''
        If the argument 'dim' is None, that function returns a boolean that
        indicates if the tensor contains a dynamic dimension (True) or not
        (False). In that case, the first dimension is excluded (as it usually
        corresponds to the batch size).  If the argument is an integer, that
        functions returns a boolean that indicates if the dimension 'dim' is
        dynamic (True) or not (False).
        '''
        if dim is not None:
            return self.trt_tensor.shape[dim] == -1

        for i, s in enumerate(self.trt_tensor.shape):
            if i != 0 and s == -1:
                return True

        return False


class RaggedTensor:
    '''
    The class to represent a ragged tensor.

    A ragged tensor is a compact representation to pack sequences of different
    lengths without using padding. In more details, a ragged tensor contains a
    tensor of data and a tensor of lengths.

    For example, consider the three sequences [0, 1, 2, 3], [4, 5] and [6, 7,
    8] of lengths 4, 2 and 3, respectively. They can be encoded using a ragged
    tensor composed of a data tensor [0, 1, 2, 3, 4, 5, 6, 7, 8] and a tensor
    of lengths [4, 2, 3].

    A ragged tensor also contains a field that encodes the length of longest
    row/sequence.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters:
            data : Tensor
                The tensor of data.

            row_lenghts : Tensor
                The tensor containing the lengths of each sequence/row.

            max_row_length : int
                The length of the longest sequence/row.
        '''
        self._data = kwargs.get('data', None)
        self._row_lengths = kwargs.get('row_lengths', None)
        self._max_row_length = kwargs.get('max_row_length', None)

    @staticmethod
    def from_row_lengths(data, row_lengths, max_row_length=None):
        '''
        Create a tensor from a tensor of data, a tensor of lengths and the
        length of the longest row (optional argument).
        '''
        return RaggedTensor(data=data,
                            row_lengths=row_lengths,
                            max_row_length=max_row_length)

    @property
    def data(self) -> Tensor:
        '''
        The tensor of data.
        '''
        return self._data

    @property
    def row_lengths(self) -> Tensor:
        '''
        The length of the different rows.
        '''
        return self._row_lengths

    @property
    def max_row_length(self) -> Tensor:
        '''
        The length of the longest row.
        '''
        return self._max_row_length


def _create_tensor(trt_tensor: trt.ITensor,
                   producer: trt.ILayer = None) -> Tensor:
    '''
    A helper function to create a TensorRT-LLM Tensor object that encapsulates
    the connection between the TensorRT tensor (trt.ITensor) and the layer
    (trt.ILayer) that produces it.

    That function is expected to be used as:

        # Insert a new layer in the network using the TensorRT API:
        layer = default_trtnet().add_<some_layer>(...)
        # Extract the first output of that layer and connect it to the layer.
        return _create_tensor(layer.get_output(0), layer)

    That function also sets the precision of the layer/producer to the default
    precision of the network.

    Parameters:
        trt_tensor : trt.ITensor
            The TensorRT tensor to connect to its producer (the layer).

        producer : trt.ILayer = None
            The producer.

    Returns:
        The TensorRT-LLM tensor (functional.Tensor) that encapsulates the
        TensorRT tensor and the layer that produces it. The former is
        accessible through the attribute 'trt_tensor' and the latter using the
        attribute 'producer'.
    '''
    assert trt_tensor is not None
    tensor = Tensor(name=trt_tensor.name,
                    dtype=trt_tensor.dtype,
                    shape=trt_tensor.shape,
                    is_network_input=False)
    tensor.trt_tensor = trt_tensor
    tensor.producer = producer

    # Set the layer name since this is the only
    # centralized location to pass the name from
    # module space to the TRT IR
    default_net()._set_layer_name(producer)
    if default_net().dtype is not None:
        if producer.type not in [
                trt.LayerType.CONSTANT, trt.LayerType.GATHER,
                trt.LayerType.CONCATENATION
        ]:
            producer.precision = default_net().dtype
    assert tensor is not None
    return tensor


def activation(input: Tensor, act_type: trt.ActivationType) -> Tensor:
    '''
    Add an activation function.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        act_type : trt.ActivationType
            The type of the activation (RELU, TANH, SIGMOID, ...).

    The following closures are defined in functional.*:

        relu    for op=trt.ActivationType.RELU
        tanh    for op=trt.ActivationType.TANH
        sigmoid for op=trt.ActivationType.SIGMOID

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor, act_type)
    return _create_tensor(layer.get_output(0), layer)


def clip(input: Tensor, alpha: float, beta: float) -> Tensor:
    '''
    Add a CLIP operation that sets the range to [alpha, beta].

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        alpha : float
            The lower bound of the CLIP function.

        beta : float
            The upper bound of the CLIP function.

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor,
                                            trt.ActivationType.CLIP)
    layer.alpha = alpha
    layer.beta = beta
    return _create_tensor(layer.get_output(0), layer)


relu = partial(activation, act_type=trt.ActivationType.RELU)
tanh = partial(activation, act_type=trt.ActivationType.TANH)
sigmoid = partial(activation, act_type=trt.ActivationType.SIGMOID)


def silu(input: Tensor) -> Tensor:
    '''
    Add a SiLU (`x * sigmoid(x)`) operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return input * sigmoid(input)


def swiglu(input: Tensor) -> Tensor:
    '''
    Add a SwiGLU (`x * SiLU(gate)`) operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies SiLU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    x, gate = chunk(input, 2, dim=-1)
    return silu(gate) * x


def cast(input: Tensor, dtype: Union[str, trt.DataType]):
    '''
    Add a cast operation.

    For an input tensor of type INT8, this function sets the dynamic range of
    the input to [-127, 127] for automatic dequantization. For a cast into
    INT8, that function sets the dynamic range of the output to [-127, 127] for
    automatic quantization.

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dtype : str or trt.DataType
            The data type of the output tensor after the cast. When 'dtype' is
            provided as a string, it must be a name amongst the valid names.
            See _str_to_trt_dtype_dict in _utils.py for a list of supported
            types and type names.

    Returns:
        The tensor produced by the inserted layer.
    '''
    if isinstance(dtype, str):
        cvt_dtype = str_dtype_to_trt(dtype)
    elif isinstance(dtype, trt.DataType):
        cvt_dtype = dtype
    else:
        raise TypeError("%s is not supported" % type(dtype))

    layer = default_trtnet().add_cast(input.trt_tensor, cvt_dtype)
    layer.set_output_type(0, cvt_dtype)
    output = _create_tensor(layer.get_output(0), layer)
    if input.dtype == str_dtype_to_trt('int8'):
        layer.get_input(0).set_dynamic_range(-127, 127)
    if cvt_dtype == str_dtype_to_trt('int8'):
        layer.get_output(0).set_dynamic_range(-127, 127)

    return output


def flip(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Reverses the order of an n-D tensor along given axis in dims.

    That flip operation maps to a TensorRT ISliceLayer. For the dimensions
    listed in dims it copies the elements from the last one to the first one
    (from (N-1) down to 0 with a step of -1). For the dimensions not in 'dims',
    it copies the elements from the first one to the last one (from 0 to N-1
    with a step of 1).

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dims : list or tuple
            The axes to flip. Negative indices are supported.

    Returns:
        The tensor produced by the inserted layer.
    '''
    assert not input.is_dynamic()

    ndim = input.ndim()

    for index, value in enumerate(dims):
        assert -ndim <= value < ndim
        if -ndim <= value < 0:
            dims[index] += ndim

    assert len(dims) == len(set(dims))

    start_values = [
        input.size()[i] - 1 if i in dims else 0 for i in range(ndim)
    ]
    stride_values = [-1 if i in dims else 1 for i in range(ndim)]

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=start_values,
                                       shape=input.size(),
                                       stride=stride_values)

    return _create_tensor(layer.get_output(0), layer)


def interpolate(input: Tensor,
                size: Union[int, List[int]] = None,
                scale_factor: Union[float, List[float]] = None,
                mode: str = 'nearest',
                align_corners: bool = False,
                recompute_scale_factor: bool = False,
                antialias: bool = False) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    input_ndim = input.ndim()

    assert 2 < input_ndim < 6, "Only 3D, 4D and 5D input Tensors supported"
    assert (size is not None) ^ (
        scale_factor
        is not None), "Only one of out_shape or scales should be defined"

    assert mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                    'nearest-exact')

    if mode == 'trilinear' and input_ndim != 5:
        raise ValueError("trilinear only supports 5D tensor")

    if mode == "bilinear" and input_ndim != 4:
        raise ValueError("bilinear only supports 4D tensor")

    if mode == "linear" and input_ndim != 3:
        raise ValueError("linear only supports 3D tensor")

    layer = default_trtnet().add_resize(input.trt_tensor)

    input_shape = input.size()

    updated_shape = []
    if scale_factor:
        scale_len = 1 if isinstance(scale_factor,
                                    (float, int)) else len(scale_factor)
        if scale_len == 1 and isinstance(scale_factor, (float, int)):
            updated_scale = [scale_factor for _ in range(input_ndim - 2)]

        else:
            updated_scale = scale_factor
        updated_shape = [
            int(math.floor(updated_scale[i - 2] *
                           input_shape[i])) if i > 1 else input_shape[i]
            for i in range(input_ndim)
        ]

    else:
        size_len = 1 if isinstance(size, int) else len(size)
        assert size_len == input_ndim - 2
        if size_len == 1 and isinstance(size, int):
            updated_size = [size for _ in range(input_ndim - 2)]
        else:
            updated_size = size

        updated_shape = [
            input_shape[i] if i < 2 else updated_size[i - 2]
            for i in range(input_ndim)
        ]
    layer.shape = updated_shape

    if mode in ['nearest', 'nearest-exact'] or mode is None:
        layer.resize_mode = trt.ResizeMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ['linear', 'bilinear', 'trilinear']:
        layer.resize_mode = trt.ResizeMode.LINEAR
        if align_corners:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        else:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        # TODO(guomingz), need to confirm the align_corners effect on bilinear mode.
        if mode == 'bilinear':
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    elif mode in ['bicubic']:
        layer.resize_mode = trt.ResizeMode.CUBIC

        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    else:
        layer.resize_mode = trt.ResizeMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC

    return _create_tensor(layer.get_output(0), layer)


def matmul(input: Tensor,
           mat2: Tensor,
           transa: bool = False,
           transb: bool = False) -> Tensor:
    '''
    Add a matrix multiplication.

    That operation maps to a tensorrt.IMatrixMultiplyLayer layer. As explained
    in the TensorRT documentation, it computes the inner product between the
    two inputs after applying an optional transposition on the inputs.

    Parameters:
        input : Tensor
            The first tensor (often called A).

        mat2 : Tensor
            The second tensor (often called B).

        transa : bool
            Is the first input transposed? Set to 'True' if you want the first
            input to be transposed, 'False' otherwise.

        transb : bool
            Is the second input transposed? Set to 'True' if you want the
            second input to be transposed, 'False' otherwise.

    Returns:
        The tensor produced by the inserted layer.
    '''
    input, mat2 = broadcast_helper(input, mat2)
    op0 = trt.MatrixOperation.TRANSPOSE if transa \
        else trt.MatrixOperation.NONE
    op1 = trt.MatrixOperation.TRANSPOSE if transb \
        else trt.MatrixOperation.NONE
    layer = default_trtnet().add_matrix_multiply(input.trt_tensor, op0,
                                                 mat2.trt_tensor, op1)

    return _create_tensor(layer.get_output(0), layer)


def constant(ndarray: np.ndarray) -> Tensor:
    '''
    Add a constant layer.

    TensorRT graphs encapsulate constant values in the form of constant layers
    (tensorrt.IConstantLayer). That function creates such a layer from a Numpy
    array of values. After compilation of the network by TensorRT, those
    weights are stored in the serialized TensorRT engine.

    Parameters:
        ndarray : numpy.ndarray
            The array of values (weights) encapsulated by this constant layer.

    Returns:
        The tensor produced by the inserted layer.
    '''
    weights = trt.Weights(np_dtype_to_trt(ndarray.dtype), ndarray.ctypes.data,
                          ndarray.size)
    # Prevent underlying numpy array from going out of scope
    default_net().register_ndarray(ndarray)
    layer = default_trtnet().add_constant(trt.Dims(ndarray.shape), weights)
    layer.set_output_type(0, np_dtype_to_trt(ndarray.dtype))
    return _create_tensor(layer.get_output(0), layer)


# TODO(qijun): TensorRT uses sizes of the output dimensions.
# DL framework uses ends usually. Will change it to ends.
def slice(input: Tensor, starts: Union[Tensor, Sequence[int]],
          sizes: Union[Tensor, Sequence[int]]) -> Tensor:
    '''
    Add an operation to extract a slice from a tensor.

    As described in the TensorRT documentation of the ISliceLayer, the slice
    layer has two variants: Static and dynamic.

    For static slicing, this function takes the starts and sizes values in the
    different dimensions to slice at layer creation time via a sequence of
    integers. For dynamic slicing, it accepts starts and sizes as
    tensorrt.ITensor`s.

    The slice layer selects for each dimension a start location from within the
    input tensor, and copies elements to the output tensor using a stride of 1
    across the input tensor. Start and size tensors must be 1-D int32 shape
    tensors if not specified as a sequence of integers.

    As an example, on input = [[0, 2, 4], [1, 3, 5]], the call to

        slice(input, start=[1, 0], size=[1, 2])

    will produce the tensor [[1, 3]] as output. The slice operator when
    executed by TensorRT will copy one row (because size[0] == 1) starting from
    the 2nd row (because start[0] == 1) and two columns (size[1] == 2) starting
    from the 1st column (because start[1] == 0).

    In pseudo-code the behaviour of that operation can be described as follows
    for a 2D tensor (and easily be extended to more dimensions):

        output = Tensor(shape=sizes)
        for ii in range(sizes[0]):
            for jj in range(sizes[1]):
                output[ii][jj] = input[starts[0]+ii][starts[1]+jj]

    Note that it is common in deep-learning frameworks to use ranges
    [start:end] for similar operations. It can be emulated by setting the sizes
    argument such that in each dimension [start:start+size] == [start:end] i.e.
    size = end-start.

    TensorRT supports different slice modes but that function restricts that
    choice to `mode == tensorrt.SliceMode.STRICT_BOUNDS`.

    Parameters:
        input : Tensor
            The input tensor on which the slicing is performed.

        starts : Union[Tensor, Sequence[int]]
            The starting points, in the input tensor, and each dimension.

        sizes : Union[Tensor, Sequence[int]]
            The number of elements in each dimension of the sliced tensor (output).

    Returns:
        The tensor produced by the slice layer.
    '''
    input_ndim = input.ndim()

    trt_starts = starts
    if isinstance(starts, Tensor):
        trt_starts = [0 for _ in range(input_ndim)]  # unused dummy value

    trt_sizes = sizes
    if isinstance(sizes, Tensor):
        trt_sizes = [1 for _ in range(input_ndim)]  # unused dummy value

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=trt_starts,
                                       shape=trt_sizes,
                                       stride=[1 for _ in range(input_ndim)])

    if isinstance(starts, Tensor):
        layer.set_input(1, starts.trt_tensor)

    if isinstance(sizes, Tensor):
        layer.set_input(2, sizes.trt_tensor)

    return _create_tensor(layer.get_output(0), layer)


# TODO(qijun): support step.
def arange(start: Union[Tensor, int], end: Union[Tensor, int],
           dtype: str) -> Tensor:
    '''
    Add an operation to fill a 1D tensor.

    The tensor is filled with the values between start and end with a step of 1
    between the different elements. In pseudo-code, it corresponds to a tensor
    populated with the values:

        output = Tensor([dtype(ii) for ii in range(start, end, 1)])

    For example, a call to arange(3, 6, 'int32') will add an operation to the
    TensorRT graph that will produce [3, 4, 5] when executed. The call to
    arange(2, 5, 'float32') will add a layer to generate [2.0, 3.0, 4.0].

    This operation is implemented using a tensorrt.IFillLayer in
    trt.FillOperation.LINSPACE mode.

    Parameters:
        start : Union[Tensor, int]
            The starting point of the range.

        end : Union[Tensor, int]
            The end point of the range.

        dtype : str
            The type of the elements. See _str_to_trt_dtype_dict in _utils.py
            for a list of supported types and type names.

    Returns:
        The tensor produced by the fill layer. It is a 1D tensor containing
        `end-start` elements of type `dtype`.
    '''
    if isinstance(start, int):
        step = 1
        assert isinstance(end, int)
        assert isinstance(step, int)

        num = len(range(start, end, step))

        layer = default_trtnet().add_fill([num], trt.FillOperation.LINSPACE)
        layer.set_output_type(0, str_dtype_to_trt(dtype))
        layer.set_alpha(start)
        layer.set_beta(step)
        return _create_tensor(layer.get_output(0), layer)
    elif isinstance(start, Tensor):
        step = constant(int32_array([1]))
        assert isinstance(end, Tensor)
        assert isinstance(step, Tensor)

        num = end - start
        num = num.view([1])

        layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE)
        layer.set_input(0, num.trt_tensor)  # rank = 1
        layer.set_input(1, start.trt_tensor)  # rank = 0
        layer.set_input(2, step.trt_tensor)  # rank = 1
        return _create_tensor(layer.get_output(0), layer)
    else:
        raise TypeError("%s is not supported" % type(start))


def expand(input: Tensor, expand_shape: Tensor) -> Tensor:
    '''
    Add an operation to expand a tensor.

    The operation expands the input tensor in the singleton dimensions to the
    size indicated by the corresponding dimension in the `expand_shape` tensor.
    In other words, given an input tensor with dimensions of size 1, those
    dimensions will be expanded to the size in `expand_shape`.

    For example, a tensor of shape [4, 3, 1, 3] will be expanded to a tensor of
    shape [4, 3, 2, 3] by the layer created using expand(input, [4, 3, 2, 3]).

    The expansion may either replicate the values or be mapped to a view with a
    stride of 0 in the expanded dimensions. For example, for a tensor [[3, 2]] of
    shape [1, 2],

        expand([[3, 2]], [2, 2])

    can be used to expand the input to [[3, 2], [3, 2]].

    This operation is implemented using a tensorrt.ISliceLayer. The current
    implementation does not verify that non singleton dimensions are not
    shrinked. In other words, for an input of shape [4, 1, 2],

        expand(input, [3, 2, 2])

    will produce a tensor of shape [3, 2, 2]. That behaviour is subject to
    change in the future.

    Parameters:
        input : Tensor
            The input tensor.

        expand_shape : Tensor
            The new shape of the expanded tensor.

    Returns:
        The tensor produced by the expand layer.
    '''
    ndim = input.rank()
    layer = default_trtnet().add_slice(
        input.trt_tensor,
        start=[0 for _ in range(ndim)],
        shape=[1 for _ in range(ndim)],  # unused dummy value
        stride=[1 for _ in range(ndim)]  # unused dummy value
    )

    # The stride is either:
    #   0 for dimensions of size 1 (i.e. shape(input, i) - 1 == 1 - 1 == 0) or,
    #   1 for dimensions of size > 1 since minimum(value >= 1, 1) == 1.
    stride_tensor = concat(
        [minimum((shape(input, i) - 1), 1) for i in range(ndim)])

    layer.set_input(2, expand_shape.trt_tensor)
    layer.set_input(3, stride_tensor.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def einsum(einsum_eq: str, inputs: Sequence[Tensor]) -> Tensor:
    '''
    Add an Einsum operation.

    That operation maps to tensorrt.IEinsumLayer. As explained in the TensorRT
    documentation, this layer implements a summation over the elements of the
    inputs along dimensions specified by the equation parameter, based on the
    Einstein summation convention. The layer can have one or more inputs of
    rank >= 0.  All the inputs must be of same data type. This layer supports
    all TensorRT data types except bool. There is one output tensor of the same
    type as the input tensors. The shape of output tensor is determined by the
    equation.

    The equation specifies ASCII lower-case letters for each dimension in the
    inputs in the same order as the dimensions, separated by comma for each
    input. The dimensions labeled with the same subscript must match or be
    broadcastable. Repeated subscript labels in one input take the diagonal.
    Repeating a label across multiple inputs means that those axes will be
    multiplied. Omitting a label from the output means values along those axes
    will be summed. In implicit mode, the indices which appear once in the
    expression will be part of the output in increasing alphabetical order. In
    explicit mode, the output can be controlled by specifying output subscript
    labels by adding an arrow (‘->’) followed by subscripts for the output. For
    example, “ij,jk->ik” is equivalent to “ij,jk”. Ellipsis (‘…’) can be used
    in place of subscripts to broadcast the dimensions. See the TensorRT
    Developer Guide for more details on equation syntax.

    Many common operations can be expressed using the Einsum equation. For
    example:
        Matrix Transpose: ij->ji
        Sum: ij-> Matrix-Matrix
        Multiplication: ik,kj->ij
        Dot Product: i,i->
        Matrix-Vector Multiplication: ik,k->i
        Batch Matrix Multiplication: ijk,ikl->ijl
        Batch Diagonal: …ii->…i

    Note that TensorRT does not support ellipsis or diagonal operations so,
    neither, does TensorRT-LLM.

    Parameters:
        einsum_eq : str
            The Einsum equation.

        inputs: Sequence[Tensor]
            The sequence of inputs consumed by the Einsum operation.

    Returns:
        The tensor produced by the Einsum operation.
    '''
    layer = default_trtnet().add_einsum([i.trt_tensor for i in inputs],
                                        einsum_eq)
    return _create_tensor(layer.get_output(0), layer)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Add an operation to permute the dimensions of a tensor.

    The dimensions of the input tensor are permutted according to the sequence
    of dimensions in 'dims'. That operation maps to tensorrt.IShuffleLayer where
    the second transposition is described by the indices in 'dims'.

    Given a tensor of rank N, the result of the permutation is a tensor of rank
    N in which the i-th input dimension maps to the dims[i]-th dimension.

    For example, permute(input, [1, 0]) will transpose a 2D tensor by permuting
    the rows and columns.

    Parameters:
        input : Tensor
            The input tensor to permute.

        dims : Sequence[int]
            The description of the permutation.

    Returns:
        The tensor produced by the permutation layer.
    '''
    dims = dim_resolve_negative(tuple(dims), input.ndim())
    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.second_transpose = dims
    return _create_tensor(layer.get_output(0), layer)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    '''
    Add an operation to transpose two dimensions of a tensor.

    That operation produces a tensor in which the dimensions 'dim0' and 'dim1'
    are permuted. The other dimensions, if the rank of the tensor is greater
    than 2, remain untouched.

    That function is a helper built on the 'functional.permute' function.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        dim0 : int
            The first dimension to transpose.

        dim1 : int
            The second dimension to transpose.

    Returns:
        The tensor produced by the permutation layer.
    '''
    permutation = list(range(input.ndim()))
    permutation[dim0] = dim1
    permutation[dim1] = dim0

    return permute(input, permutation)


def view(input: Tensor,
         shape: Union[Tensor, Sequence[int]],
         zero_is_placeholder: bool = True) -> Tensor:
    '''
    Add an operation to create a view of a tensor.

    That operation adds a tensorrt.IShuffleLayer to the network. If the 'shape'
    parameter is a Tensor, that view is dynamic. Otherwise, it is a static
    view.

    Note that TensorRT limits the number of inferred dimensions to 1. It means
    that the shape sequence or tensor cannot contain more than one -1. This
    function enforces that constraint and will assert if it is not respected.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        shape : Union[Tensor, Sequence[int]]
            The shape of the new tensor.

        zero_is_placeholder : bool
            When that parameter is True, the 0s in 'shape' are replaced by the
            sizes of the corresponding dimensions from the 'input'. Otherwise,
            the dimensions corresponding to 0s are shrinked.

    Returns:
        The tensor produced by the view/shuffle layer.
    '''

    # TensorRT demands that at most one dimension is permitted to be specified as -1
    def assert_no_more_than_one_inferred_dim(list):
        inferred_dim_list = [i for i in list if i == -1]
        assert len(inferred_dim_list) <= 1

    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.zero_is_placeholder = zero_is_placeholder
    if isinstance(shape, Tensor):
        assert_no_more_than_one_inferred_dim(shape.shape)
        layer.set_input(1, shape.trt_tensor)
    elif isinstance(shape, (list, tuple)):
        assert_no_more_than_one_inferred_dim(shape)
        layer.reshape_dims = tuple(shape)
    else:
        raise TypeError("%s is not supported" % type(shape))
    return _create_tensor(layer.get_output(0), layer)


def expand_dims(input: Tensor, dim: Union[int, Sequence[int]]) -> Tensor:
    '''
    Add an operation to expand the tensor shape with singleton dimensions.

    That function adds a tensorrt.IShuffleLayer to the network. Given an 'input'
    of rank N and a sequence of M dimensions, the output tensor produced by
    this operation (when executed by TensorRT) will have a rank of N+M. Singleton
    dimensions will be inserted at the different positions in 'dim'.

    The pseudo-code for that operation is:

        new_shape, ii = [], 0
        for jj in range(input.rank() + len(dim)):
            new_shape.append(1 if jj in dims else input.shape[ii++])

    For example, for a tensor of shape [3, 4, 1, 5]

        expand_dims(input, [0, 2])

    will produce a tensor of shape [1, 3, 1, 4, 1, 5].

    Parameters:
        input : Tensor
            The input tensor to expand.

        dim : Union[int, Sequence[int]]
            The positions in the output tensor where to insert singleton
            dimensions.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(dim, int):
        dim = (dim, )

    out_ndim = len(dim) + input.ndim()

    input_shape = shape(input)
    out_shapes = []
    j = 0
    for i in range(out_ndim):
        if i in dim:
            out_shapes.append(1)
        else:
            out_shapes.append(gather(input_shape, 0, j))
            j = j + 1

    out_shape = concat(out_shapes)

    return view(input, out_shape)


def unsqueeze(input: Tensor, axis: int):
    '''
    Add an operation to insert a singleton dimension to a tensor.

    That functions creates an operation that insert a singleton dimension
    (dimension of size 1) at position 'dim' in the output tensor. It works with
    negative values for the 'axis'.

    For example, for a tensor 'input' of shape [4, 4]:

        unsqueeze(input,  0) will produce an output of shape [1, 4, 4],
        unsqueeze(input,  1) will produce an output of shape [4, 1, 4],
        unsqueeze(input, -1) will produce an output of shape [4, 4, 1],
        unsqueeze(input, -2) will produce an output of shape [4, 1, 4],

    Parameters:
        input : Tensor
            The input tensor to expand with a singleton dimension.

        axis : int
            The index of the singleton dimension in the output tensor.

    Returns:
        The tensor produced by the layer.
    '''
    if axis < 0:
        axis = axis + input.ndim() + 1

    return expand_dims(input, axis)


def expand_dims_like(left: Union[Tensor, int, float], right: Tensor) -> Tensor:
    '''
    Add an operation to expand the first tensor to the same rank as the second
    tensor.

    That function takes a first tensor. It also accepts an integer or a float,
    in which case it creates a constant tensor from it. In both cases, the rank
    of that first tensor is compared to the rank of the second tensor. If they
    are of the same rank, the first tensor is returned. Otherwise, the first
    tensor is expanded on the left to match the rank of the second tensor.

    Note that the shapes do not have to match, only the rank is considered in
    that function.

    For example, for a pair of tensors of shapes [3, 4] and [4, 3, 2], the
    first tensor will be expanded to a tensor of rank 3 and shape [1, 3, 4].

    Parameters:
        left : Union[Tensor, int, float]
            The first tensor to expand. When a scalar value is provided as a
            parameter, that function first creates a tensor before expanding it
            (if needed).

        right : Tensor
            The reference tensor to match.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(left, int):
        left = constant(int32_array([left]))
    elif isinstance(left, float):
        left = constant(fp32_array([left]))

    left_ndim = left.ndim()
    right_ndim = right.ndim()
    if right_ndim > left_ndim:
        new_ndim = list(range(right_ndim - left_ndim))
        return expand_dims(left, new_ndim)
    return left


# If dim is None, return a 1-D TensorRT-LLM tensor of the size
# If dim is not None, return a 0-D TensorRT-LLM tensor of the dimension size
def shape(input: Tensor, dim: Optional[int] = None) -> Tensor:
    '''
    Add an operation to create a shape tensor.

    The shape tensor can either be the shape of the input tensor when the
    parameter dim is None or a scalar (tensor of rank 0) that corresponds to
    the size of dim-th dimension.

    Parameters:
        input : Tensor
            The input tensor from which we want to extract the shape or the
            size in one dimension.

        dim : Optional[int]
            The dimension from which to extract the size. If it is None, the
            entire shape of the input tensor is returned.

    Returns:
        A tensor that contains the shape of the input tensor (if 'dim' is None)
        or the size in the dimension 'dim' of the input tensor. If 'dim' is
        'None', that tensor has the same rank as the input tensor, otherwise
        its rank is 0.
    '''
    layer = default_trtnet().add_shape(input.trt_tensor)
    res = _create_tensor(layer.get_output(0), layer)

    if dim is None:
        return res

    return gather(res, dim=0, indices=dim).view([])


def gather(input: Tensor, dim: int, indices: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to gather elements from a tensor.

    That function implements the GatherElements operator from the ONNX
    specification as described in

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements

    The input and indices arguments must have the same rank >= 1. The operation
    will produce a tensor with the same shape as the indices tensor. The axis
    is the dimension to gather on.

    As shown in the ONNX description, for a 3D tensor, the output is:

        out[i][j][k] = input[indices[i][j][k]][j][k] if axis = 0,
        out[i][j][k] = input[i][indices[i][j][k]][k] if axis = 1,
        out[i][j][k] = input[i][j][indices[i][j][k]] if axis = 2.

    For example,

        gather([[4, 2], [5, 3]], 0, [[1, 0], [0, 1]])

    will produce [[5, 2], [4, 3]].

        gather([[1, 2, 3], [4, 5, 6], 1, [[1], [0]])

    will produce [[2], [4]]. See the ONNX documentation for more examples.

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to gather elements from.

        dim : int
            The dimension to gather on.

        indices : Union[Tensor, int]
            The positions in the 'dim' dimension to gather from.

    Returns:
        The tensor containing the gathered elements. It has the same shape as
        the indices tensor.
    '''
    if isinstance(indices, int):
        indices = constant(int32_array([indices]))

    # The input and indices tensors must have the same rank.
    assert input.rank() == indices.rank()

    layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                           indices.trt_tensor,
                                           mode=trt.GatherMode.ELEMENT)

    if dim < 0:
        dim = input.ndim() + dim
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def select(input: Tensor, dim: int, index: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to select a slice of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    index-th slice of elements in the dimension 'dim' to create a new tensor.
    The output tensor has a shape in which the input dimension 'dim' is
    removed.

    The 'index' can either be an integer or a 1D tensor containing a single
    element.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        select(input, 0, 1)

    will create a tensor of shape [3] that contains the [2, 1, 2].

    Regarding the shape of the output tensor, the dimension 'dim' is removed.
    It means that for a tensor of shape [4, 2, 6, 3],

        select(input, 2, 4)

    will select the 5th slice (index == 4) from the 3rd dimension (dim == 2)
    and return a tensor of shape [4, 2, 3] (i.e. the 3rd dimension is removed).

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Union[Tensor, int]
            The index of the slice in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slice.
    '''
    if isinstance(index, int):
        index = constant(int32_array([index]))
    assert index.rank() == 1 and index.size(
        0) == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor, dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    '''
    Add an operation to select slices of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    slices of elements in the dimension 'dim' at the indices listed in 'index'
    to create a new tensor.  The output tensor has the same rank as the input
    tensor.

    The 'index' is a tensor of rank 1.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        index_select(input, 0, [0, 1])

    will create a tensor of shape [3, 2] that contains the [[4, 2, 5], [2, 1, 2]].

    Regarding the shape of the output tensor, the dimension 'dim' has the same
    size as the 'index' tensor. It means that for a tensor of shape [4, 2, 6, 3],

        index_select(input, 2, [1, 4])

    will select the 2nd and 5th slices (index == 1 or 4) from the 3rd dimension
    (dim == 2) and return a tensor of shape [4, 2, 2, 3] (i.e. the 3rd
    dimension is shrinked to 2).

    Note that this operation can also be used to expand a tensor in the 'dim'
    dimension, for example, on input [[0, 1], [2, 3]],

        index_select(input, 1, [0, 0, 0])

    will produce a tensor of shape [2, 3] containing [[0, 0, 0], [2, 2, 2]].

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Tensor
            The indices of the slices in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slices.
    '''
    assert index.rank() == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))
        else:
            new_shape.append(shape(index, 0))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor, dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


def concat(inputs: Sequence[Union[Tensor, int]], dim: int = 0) -> Tensor:
    '''
    Add an operation to concatenate tensors.

    The function creates an operation that concatenates the tensors from the
    sequence 'inputs'. The concatenation is done along the dimension 'dim'.

    All the tensors in 'inputs' must have the same shape expect for the
    dimension 'dim'.

        for ii in range(inputs[0].rank()):
            assert (ii == dim) or all(inp.shape[ii] == inputs[0].shape[ii] for inp in inputs)

    The shape of the output tensor is defined as:

        for ii in range(inputs[0].rank()):
            # Same size as all the inputs in dimension ii != dim.
            output.shape[ii] = inputs[0].shape[ii]

            # Sum of the sizes in the different inputs in dimension 'dim'.
            if ii == dim:
                for jj in range(1, len(inputs)):
                    output.shape[ii] += inputs[jj].shape[ii]

    For example, given a sequence of two 2D tensors [[0, 1], [2, 3]] and
    [[4, 5], [6, 7]] both of shape [2, 2],

        concat(inputs, 0)

    will produce [[[0, 1], [2, 3]], [[4, 5], [6, 7]]] of shape [4, 2] and

        concat(inputs, 1)

    will produce [[0, 1, 4, 5], [2, 3, 6, 7]] of shape [2, 4].

    Parameters:
        inputs : Sequence[Union[Tensor, int]]
            The sequence of tensors to concatenate. For integers, that function
            creates constant tensors.

        dim : int
            The dimension in which the concatenation is performed.

    Returns:
        A tensor that contains the concatenation of the tensors.
    '''
    tmp = []
    for i in inputs:
        if isinstance(i, int):
            tmp.append(constant(int32_array([i])))
        elif i.rank() == 0:
            tmp.append(i.view([1]))
        else:
            tmp.append(i)

    layer = default_trtnet().add_concatenation([i.trt_tensor for i in tmp])
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    '''
    Add an operation to compute softmax on a tensor.

    That operation computes the softmax on the input tensor in the dimension
    'dim' if specified. Otherwise, it is applied on the last dimension.

    It inserts a ISoftmaxLayer to the TensorRT graph.

    Parameters:
        input : Tensor
            The input tensor on which to apply softmax.

        dim : Optional[int]
            The dimension used to apply softmax.

    Returns:
        The output tensor of the softmax layer.
    '''
    if dim is None:
        dim = input.ndim() - 1
    if dim < 0:
        dim = input.ndim() + dim
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_softmax(input.trt_tensor)
    layer.axes = axes

    return _create_tensor(layer.get_output(0), layer)


def _lookup_plugin(input: Tensor,
                   weight: Tensor,
                   size: int = 0,
                   if_tensor_parallel: bool = False) -> Tensor:
    '''
    Add an operation to perform lookup in a tensor.

    That operation performs the lookup needed by embedding layers. Given a
    'weight' tensor of shape [rows, cols], it produces a tensor of shape
    [inputs.size(0), cols] where the ith row corresponds to the input[i] row in
    the weight tensor.

    It inserts a IPluginV2Layer.

    Parameters:
        input : Tensor
            The input tensor the contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        size : int
            The number fo rows in the weight tensor when tensor parallelism is
            enabled.

        if_tensor_parallel : bool
            Is the tensor parallelism is enabled.

    Returns:
        The output tensor of the lookup layer.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Lookup', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    size = trt.PluginField("size", np.array(size, dtype=np.int32),
                           trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.lookup_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    if_tensor_parallel = trt.PluginField(
        "if_tensor_parallel", np.array(int(if_tensor_parallel), dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([size, pf_type, if_tensor_parallel])
    lookup_plug = plg_creator.create_plugin("lookup", pfc)
    plug_inputs = [input.trt_tensor, weight.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, lookup_plug)
    return _create_tensor(layer.get_output(0), layer)


def embedding(input: Tensor,
              weight: Tensor,
              tp_size=1,
              tp_group=None) -> Tensor:
    '''
    Add an operation to perform embedding lookup.

    That operation performs the embedding lookup. The 'input' tensor contains
    the identifiers of the rows of 'weight' to gather.

    When 'tp_size' is greater than 1 and the 'tp_group' is defined, this
    embedding lookup is distributed accross multiple GPUs. In the current
    implementation, each GPU stores a subset of the rows of the embedding table
    (that number of rows per GPU is given by weights.shape[0] and the offset to
    the 1st row stored on the GPU is given by rank * weights.shape[0]). Each
    parallel rank will query all the indices and set 0s for the weights that
    are not stored on the associated GPU. To compute the final result, a
    parallel all-reduce operation is added to the TensorRT graph. That lookup
    is implemented using a plugin.

    When the default_net().plugin_config.lookup_plugin is set to True, the
    operation is implemented using a plugin (without the all-reduce operation).
    Otherwise, this operation is implemented using the standard IGatherLayer in
    TensorRT.

    Parameters:
        input : Tensor
            The input tensor the contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        tp_size : int
            The number of GPUs collaborating to perform that embedding.

        tg_group : Optional[List[int]]
            The group of world ranks participating in the all-reduce when
            tp_size > 1.

    Returns:
        The tensor produced by the embedding lookup layer.
    '''
    vocab_size_part = weight.shape[0]

    if default_net().plugin_config.lookup_plugin:
        if tp_size > 1 and tp_group is not None:
            x = _lookup_plugin(input,
                               weight,
                               vocab_size_part,
                               if_tensor_parallel=True)
            x = allreduce(x, tp_group)
        else:
            x = _lookup_plugin(input,
                               weight,
                               vocab_size_part,
                               if_tensor_parallel=False)
    else:
        layer = default_trtnet().add_gather(weight.trt_tensor, input.trt_tensor,
                                            0)
        x = _create_tensor(layer.get_output(0), layer)
    return x


def constant_to_tensor_(input: Union[Tensor, int, float]) -> Tensor:
    if isinstance(input, int):
        return constant(int32_array([input]))
    elif isinstance(input, float):
        return constant(fp32_array([input]))
    return input


def broadcast_helper(left: Union[Tensor, int, float],
                     right: Union[Tensor, int, float]) -> Tuple[Tensor, Tensor]:
    '''
    Helper function to perform a broadcast.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

    Returns:
        A pair of tensors of same rank.
    '''
    left = constant_to_tensor_(left)
    right = constant_to_tensor_(right)

    if left.rank() == right.rank():
        return (left, right)

    if left.rank() < right.rank():
        left = expand_dims_like(left, right)
        return (left, right)

    if left.rank() > right.rank():
        right = expand_dims_like(right, left)
        return (left, right)


def elementwise_binary(left: Union[Tensor, int,
                                   float], right: Union[Tensor, int, float],
                       op: trt.ElementWiseOperation) -> Tensor:
    '''
    Add an elementwise operation with two inputs.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one. Then, it performs the
    elementwise operation 'op'.

    The following closures are defined in functional.*:

        add     for op=trt.ElementWiseOperation.SUM
        sub     for op=trt.ElementWiseOperation.SUB
        mul     for op=trt.ElementWiseOperation.PROD
        div     for op=trt.ElementWiseOperation.DIV
        gt      for op=trt.ElementWiseOperation.GREATER
        lt      for op=trt.ElementWiseOperation.LESS
        eq      for op=trt.ElementWiseOperation.EQUAL
        minimum for op=trt.ElementWiseOperation.MIN
        maximum for op=trt.ElementWiseOperation.MAX
        pow     for op=trt.ElementWiseOperation.POW

    It is implemented using the IElementWiseLayer from TensorRT.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

        op : trt.ElementWiseOperation
            The binary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    left, right = broadcast_helper(left, right)
    layer = default_trtnet().add_elementwise(left.trt_tensor, right.trt_tensor,
                                             op)
    return _create_tensor(layer.get_output(0), layer)


add = partial(elementwise_binary, op=trt.ElementWiseOperation.SUM)
sub = partial(elementwise_binary, op=trt.ElementWiseOperation.SUB)
mul = partial(elementwise_binary, op=trt.ElementWiseOperation.PROD)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)
gt = partial(elementwise_binary, op=trt.ElementWiseOperation.GREATER)
lt = partial(elementwise_binary, op=trt.ElementWiseOperation.LESS)
eq = partial(elementwise_binary, op=trt.ElementWiseOperation.EQUAL)
minimum = partial(elementwise_binary, op=trt.ElementWiseOperation.MIN)
maximum = partial(elementwise_binary, op=trt.ElementWiseOperation.MAX)
pow = partial(elementwise_binary, op=trt.ElementWiseOperation.POW)


def where(condition: Union[Tensor, int, float], left: Union[Tensor, int, float],
          right: Union[Tensor, int, float]) -> Tensor:
    '''
    Add a where (aka select or if-then-else) operation.

    Assuming the three input parameters have the same shape, that function creates
    the operation to compute a tensor of the same shape such that:

        for ii in range(mul(condition.shape)):
            output[ii] = left[ii] if condition[ii] else right[ii]

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one. Then, it performs the
    selection.

    It is implemented using the ISelectLayer from TensorRT.

    Parameters:
        left : Union[Tensor, int, float]
            The condition. If that input is an integer or a float, the function
            creates a constant tensor.

        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

        op : trt.ElementWiseOperation
            The binary operation to perform.

    Returns:
        The tensor produced by this select operation.
    '''
    # Convert to tensors.
    condition = constant_to_tensor_(condition)
    left = constant_to_tensor_(left)
    right = constant_to_tensor_(right)

    # Find the tensor with the largest rank of the three.
    largest = condition
    if largest.rank() < left.rank():
        largest = left
    if largest.rank() < right.rank():
        largest = right

    # Expand the tensors to match the largest one.
    if condition is not largest:
        condition = expand_dims_like(condition, largest)
    if left is not largest:
        left = expand_dims_like(left, largest)
    if right is not largest:
        right = expand_dims_like(right, largest)

    # Insert the operation.
    layer = default_trtnet().add_select(condition.trt_tensor, left.trt_tensor,
                                        right.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def unary(input: Tensor, op: trt.UnaryOperation) -> Tensor:
    '''
    Add an elementwise operation on a single input.

    The following closures are defined in functional.*:

        round   for op=trt.UnaryOperation.ROUND
        sqrt    for op=trt.UnaryOperation.SQRT
        exp     for op=trt.UnaryOperation.EXP
        sin     for op=trt.UnaryOperation.SIN
        cos     for op=trt.UnaryOperation.COS
        abs     for op=trt.UnaryOperation.ABS

    It is implemented using the IUnaryLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        op : trt.UnaryOperation
            The unary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    layer = default_trtnet().add_unary(input.trt_tensor, op)
    return _create_tensor(layer.get_output(0), layer)


round = partial(unary, op=trt.UnaryOperation.ROUND)
sqrt = partial(unary, op=trt.UnaryOperation.SQRT)
exp = partial(unary, op=trt.UnaryOperation.EXP)
sin = partial(unary, op=trt.UnaryOperation.SIN)
cos = partial(unary, op=trt.UnaryOperation.COS)
abs = partial(unary, op=trt.UnaryOperation.ABS)


def mean(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the mean along a dimension.

    Computes the mean along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        trt.ReduceOperation.AVG,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


def max(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the max along a dimension.

    Computes the max along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        trt.ReduceOperation.MAX,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


def identity(input: Tensor) -> Tensor:
    '''
    Add an identity operation.

    TODO: Document why it can be done using a plugin!!!

    Parameters:
        input : Tensor
            The input tensor.

    Returns:
        The tensor produced by this identity operation.
    '''
    if not default_net().plugin_config.identity_plugin:
        layer = default_trtnet().add_identity(input.trt_tensor)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Identity', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None
        pfc = trt.PluginFieldCollection()
        id_plug = plg_creator.create_plugin("identity", pfc)
        plug_inputs = [input.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, id_plug)
    return _create_tensor(layer.get_output(0), layer)


def argmax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an argmax operation.

    As explained in the ONNX documentation,

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#argmax

    that function creates a layer computing the indices of the max elements of
    the input tensor's element along the provided dim. The resulting tensor
    has the same rank as the input if keepdims is True. If keepdims is False,
    then the resulting tensor has the reduced dimension pruned.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension in which to compute the argmax indices.

        keepdim : bool
            Do we keep the dimension along which the reduction is performed?
            Yes, if set to True, no otherwise.

    Returns:
        The tensor produced by this argmax operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_topk(input.trt_tensor, trt.TopKOperation.MAX,
                                      1, axes)
    output = layer.get_output(1)

    if keepdim:
        return _create_tensor(output, layer)

    a = list(range(len(input.ndim())))
    a.pop(dim)
    indices = constant(int32_array([a]))
    output_shape = shape(output)
    new_shape = gather(output_shape, 0, indices)
    layer = view(output, new_shape)
    return _create_tensor(layer.get_output(0), layer)


def gelu(x: Tensor) -> Tensor:
    '''
    Add a GELU operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return 0.5 * x * (
        tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * pow(x, 3.0))) + 1.0)


def geglu(x: Tensor) -> Tensor:
    '''
    Add a Gated-GELU operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies GELU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    a, b = chunk(x, 2, dim=-1)
    return a * gelu(b)


def group_norm(input: Tensor,
               num_groups: int,
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05):

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic(1)
    num_channels = input.size()[1]

    ndim = input.ndim()
    old_shape = shape(input)
    new_shape = concat([
        input.size(0),
        num_groups,
        num_channels // num_groups,
    ] + [input.size(i) for i in range(2, ndim)])
    x = input.view(new_shape)

    reduce_dim = tuple(range(2, ndim + 1))
    ux = x.mean(reduce_dim, keepdim=True)
    numerator = x - ux
    varx = numerator * numerator
    varx = varx.mean(reduce_dim, keepdim=True)

    denom = varx + eps
    denom = denom.sqrt()
    y = numerator / denom
    y = y.view(old_shape)

    new_shape = concat([num_channels] + [1 for _ in range(2, ndim)])
    if weight is not None:
        y = y * weight.view(new_shape)
    if bias is not None:
        y = y + bias.view(new_shape)

    return y


def softplus(input: Tensor, beta: float, threshold: float) -> Tensor:
    '''
    Add the softplus activation base on PyTorch definition.

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html for a
    description of that function.

    Parameters:
        input : Tensor
            Input TensorRT-LLM Tensor.
        beta : float
            The parameter for softplus computation.
        threshold : float
            The threshold for reverting to the linear function when input * beta > threashold

    Returns:
        The output tensor created by that layer.
    '''
    sf_layer = default_trtnet().add_activation(input.trt_tensor,
                                               trt.ActivationType.SOFTPLUS)
    sf_layer.alpha = 1 / beta
    sf_layer.beta = beta

    prod_tensor = input * beta
    result = prod_tensor > threshold

    return where(result, input, _create_tensor(sf_layer.get_output(0),
                                               sf_layer))


def outer(input: Tensor, vec2: Tensor) -> Tensor:
    '''
    Add an operation to compute the outer product between two tensors.

    That operation creates an Einsum node.

    Parameters:
        input : Tensor
            The first input tensor.

        vec2 : Tensor
            The second input tensor.

    Returns:
        The output tensor produced by this layer.
    '''
    return einsum('i,j->ij', [input, vec2])


def avg_pool2d(input: Tensor,
               kernel_size: Tuple[int],
               stride: Optional[Tuple[int]] = None,
               padding: Optional[Tuple[int]] = (0, 0),
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> Tensor:

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()
    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    layer = default_trtnet().add_pooling(input.trt_tensor,
                                         trt.PoolingType.AVERAGE, kernel_size)
    if stride is None:
        layer.stride = kernel_size
    else:
        layer.stride = stride

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv2d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1) -> Tensor:

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[0]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_convolution_nd(input.trt_tensor, noutput,
                                                kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation = dilation
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv_transpose2d(input: Tensor,
                     weight: Tensor,
                     bias: Optional[Tensor] = None,
                     stride: Tuple[int, int] = (1, 1),
                     padding: Tuple[int, int] = (0, 0),
                     output_padding: Tuple[int, int] = (0, 0),
                     dilation: Tuple[int, int] = (1, 1),
                     groups: int = 1) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[1]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_deconvolution_nd(input.trt_tensor, noutput,
                                                  kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def split(tensor: Tensor,
          split_size_or_sections: Union[int, Sequence[int]],
          dim: int = 0) -> Sequence[Tensor]:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by slicing it along the dimension 'dim'. If 'split_size_or_sections'
    is an integer, the tensor is split into 'input.shape[dim] /
    split_size_or_sections' slices. If 'split_size_or_sections' is a list of
    sizes, the tensor is split into 'len(split_size_or_sections)' slices and
    the size of the ith slice is given by 'split_size_or_sections[i]'.

    There are several constraints with the current implementation:

        - The input tensor must be static (no dynamic dimension),
        - If 'split_size_or_sections' is an integer, the number of elements in
          the 'dim' dimension of the input must be a multiple of
          'split_size_or_sections': 'input.shape[dim] % split_size_or_sections == 0'.
        - If 'split_size_or_sections' is a sequence, the sum of the elements in
          'split_size_or_sections' must be equal to the size in the dimension
          'dim': 'input.shape[dim] == sum(ii for ii in split_size_or_sections)'.

    That operation is implemented using a 'slice' operation for each output
    slice.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        split_size_or_sections : Union[int, Sequence[int]]
            If it is an integer, it encodes the size of each slice. Otherwise,
            if it is a sequence, it is the size of each slice.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    starts = [constant(int32_array([0])) for _ in range(ndim)]
    sizes = [shape(tensor, i) for i in range(ndim)]

    if isinstance(split_size_or_sections, int):
        # TODO(kaiyu): support non-divisible cases
        assert dim_value % split_size_or_sections == 0
        num_sections = dim_value // split_size_or_sections
        sizes[dim] = constant(int32_array([split_size_or_sections]))

        outputs = []
        for i in range(num_sections):
            starts[dim] = constant(int32_array([split_size_or_sections * i]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs
    else:
        total_size = 0
        for i in split_size_or_sections:
            total_size += i
        assert dim_value == total_size
        num_sections = len(split_size_or_sections)

        outputs = []
        for i in range(num_sections):
            if i > 0:
                starts[dim] = starts[dim] + sizes[dim]
            sizes[dim] = constant(int32_array([split_size_or_sections[i]]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tensor:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by chunking it along the dimension 'dim'. It produces 'chunks'
    sub-tensors.

    That operation is only defined for static tensors (no dynamic dimension)
    and the size of the tensor in the dimension 'dim' must be a multiple of
    'chunks': 'input.shape[dim] % chunks == 0'.

    It maps to 'split' with 'split_size = input.shape[dim] / chunks'.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        chunks : int
            The number of slices to split the input tensor into.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    assert dim_value % chunks == 0

    return split(tensor, dim_value // chunks, dim)


def allreduce(tensor: Tensor, group: List[int]) -> Tensor:
    '''
    Add an operation that performs a collective all-reduce.

    Let's define 'world_size' as the length of the 'group' list. That functions
    creates a layer to compute the sum of 'world_size' tensors distributed
    amongst the 'world_size' participating ranks (one GPU per rank).

    The list 'group' contains the identifiers of the ranks participating into
    the collective operation.

    The tensors in the different ranks must be 1D tensors (or views) and the output
    tensor will have that same shape. The output tensor will be replicated on
    the 'world_size' ranks.

    That operation is implemented using a plugin that wraps the NCCL all-reduce
    collective operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        group : List[int]
            The ranks participating into the all-reduce operation.

    Returns:
        The tensor produced by that layer.
    '''
    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'AllReduce', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert allreduce_plg_creator is not None

    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([group, pf_type])
    ar_plug = allreduce_plg_creator.create_plugin("allreduce", pfc)
    plug_inputs = [tensor.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, ar_plug)
    return _create_tensor(layer.get_output(0), layer)


def allgather(tensor: Tensor, group: List[int]) -> Tensor:
    '''
    Add an operation that performs a collective all-gather.

    Let's define 'world_size' as the length of the 'group' list. That functions
    creates a layer to gather 'world_size' tensors distributed
    amongst the 'world_size' participating ranks (one GPU per rank).

    The list 'group' contains the identifiers of the ranks participating into
    the collective operation.

    The tensors in the different ranks must be 1D tensors (or views) and the
    output tensor will have that same shape.

    Given the 'section_size = input.shape[0] / world_size', each rank
    contributes a section of its input tensor that correspond to
    'rank*section_size:(rank+1)*section_size'.

    That operation is implemented using a plugin that wraps the NCCL all-gather
    collective operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        group : List[int]
            The ranks participating into the all-gather operation.

    Returns:
        The tensor produced by that layer.
    '''
    allgather_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'AllGather', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert allgather_plg_creator is not None

    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([group, pf_type])
    allgather = allgather_plg_creator.create_plugin("allgather", pfc)
    plug_inputs = [tensor.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, allgather)
    return _create_tensor(layer.get_output(0), layer)


def send(tensor: Tensor, tgt: int) -> Tensor:
    '''
    Add an operation that performs a send from a rank to another.

    The send operation sends a tensor from one rank to another. If a rank 'i'
    sends a tensor to a rank 'j', the rank 'j' must have a corresponding 'recv'
    operation from rank 'i'. See 'recv'.

    That operation is implemented using a plugin that wraps the NCCL send
    point-to-point operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        tgt : int
            The rank that receives the tensor.

    Returns:
        The tensor produced by that layer.
    '''
    send_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Send', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert send_plg_creator is not None

    tgt = trt.PluginField("tgtRank", np.array(tgt, dtype=np.int32),
                          trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([tgt, pf_type])
    send_plug = send_plg_creator.create_plugin("send", pfc)
    plug_inputs = [tensor.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, send_plug)
    return _create_tensor(layer.get_output(0), layer)


def recv(tensor: Tensor, src: int) -> Tensor:
    '''
    Add an operation that performs a recv to a rank from another.

    The recv operation receives a tensor from on a rank from another. If a rank 'i'
    receives a tensor from a rank 'j', the rank 'j' must have a corresponding 'send'
    operation to rank 'j'. See 'send'.

    That operation is implemented using a plugin that wraps the NCCL recv
    point-to-point operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        src : int
            The rank that sends the tensor to.

    Returns:
        The tensor produced by that layer.
    '''
    recv_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Recv', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert recv_plg_creator is not None

    src = trt.PluginField("srcRank", np.array(src, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([src, pf_type])
    recv_plug = recv_plg_creator.create_plugin("recv", pfc)
    plug_inputs = [tensor.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, recv_plug)
    return _create_tensor(layer.get_output(0), layer)


def bert_attention(tensor: Tensor, input_lengths: Tensor, num_heads: int,
                   head_size: int, q_scaling: float) -> Tuple[Tensor]:
    '''
    Add an operation that performs the multi-head attention in BERT.

    The multihead-attention (MHA) is the sequence of a batched matmul, a
    softmax and a batched matmul as described in
    https://arxiv.org/abs/1706.03762. That function adds an operation that
    performs those computations using a single GPU kernel.

    The input tensor contains the Q, K and V elements. It is a 2D tensor and
    its shape is '[sum_of_tokens, 3*hidden_dim]' where the 'sum_of_tokens' is
    the sum of the sequence lengths in the batch.

    In MHA, the output of the Q*K^T product is scaled by a constant value that
    is computed as:

        1.f / (q_scaling * sqrt(head_size)).

    That 'q_scaling' constant is the last argument of that function.

    That layer is implemented using a plugin (see bertAttentionPlugin).

    Parameters:
        tensor : Tensor
            The QKV input tensor.

        input_lengths : Tensor
            The length of each sequence. It is a 1D tensor of size 'batch_size'.

        num_heads : int
            The number of heads.

        head_size : int
            The size of each head.

        q_scaling : float
            The factor to compute the scaling factor to scale the output of the
            'Q*K^T' product.

    Returns:
        The tensor produced by that layer.
    '''
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'BertAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None

    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size", np.array(head_size,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling",
                                np.array(q_scaling, dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
    enable_qk_half_accum = trt.PluginField(
        "enable_qk_half_accum",
        np.array(np.int8(
            default_net().plugin_config.attention_qk_half_accumulation),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    p_dtype = default_net().plugin_config.bert_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([
        nheads, head_size, q_scaling, enable_qk_half_accum, context_fmha_type,
        pf_type
    ])

    attn_plug = attn_plg_creator.create_plugin("padding_attn", pfc)
    plug_inputs = [tensor, input_lengths]
    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    assert layer.num_outputs == 1, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected 1"
    output = _create_tensor(layer.get_output(0), layer)
    assert output is not None
    return output


def gpt_attention(
    tensor: Tensor,
    past_key_value: Tensor,
    sequence_length: Tensor,
    past_key_value_length: Tensor,
    masked_tokens: Tensor,
    input_lengths: Tensor,
    max_input_length: Tensor,
    cache_indirection: Tensor,
    num_heads: int,
    head_size: int,
    q_scaling: float,
    rotary_embedding_dim: int,
    neox_rotary_style: bool,
    multi_block_mode: bool,
    multi_query_mode: bool,
    kv_orig_quant_scale: Tensor,
    kv_quant_orig_scale: Tensor,
    use_int8_kv_cache: bool,
    use_fp8_kv_cache: bool = False,
    mask_type: int = 1,
    kv_cache_block_pointers: Tensor = None,
    host_input_lengths: Tensor = None,  # for in-flight batching
    host_request_types: Tensor = None  # for in-flight batching
) -> Tuple[Tensor]:
    '''
    Add an operation that performs the multi-head attention in GPT-like models.

    The signature of the function will change in the future release - we are in
    the process of simplifying the API. The current version is still
    work-in-progress! The following API is provided with hints regarding the
    arguments that are likely to be removed or merged with others in the future
    release.

    See docs/gpt_attention.md for the documentation of that function.

    Parameters:
        qkv: Tensor
            The input QKV tensor. Its shape is [batch_beam_size, max_seqlen, 3
            * hidden_dim] in padded mode and [1, num_tokens, 3 * hidden_dim] in
            packed mode. See QKV Input in docs/gpt_attention.md.

        past_key_value: Tensor
            The tensor that stores KV cache data. Its shape is
            [max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head]
            in contiguous mode and
            [max_blocks, 2, num_heads, num_tokens_per_block, hidden_dim_per_head]
            in paged mode. See KV Cache in doc/functional.py,

        sequence_lengths: Tensor
            The tensor that stores the length of each sequence. Its shape is
            [batch_size]. See QKV Input in doc/functional.py,

        past_key_value_length: Tensor
            An INT32 tensor of shape [2]. (**to be removed?**),

        masked_tokens: Tensor
            (**to be removed?**)

        input_lengths: Tensor
            The tensor that stores the length of each input sequence. Its shape
            is [batch_size]. See QKV Input in doc/functional.py,

        max_input_length: Tensor
            The length of the longest input sequence. See QKV Input in
            doc/functional.py,

        cache_indirection: Tensor
            The tensor to reconstruct the paths when using beam-search. Its
            shape is [batch_size, beam_width, max_seqlen]. See Beam-Search in
            doc/functional.py,

        num_heads: int
            The number of heads,

        hidden_size_per_head: int
            The hidden size per head,

        q_scaling: float
            The value used to compute the scaling factor applied to the output
            of the Q*K^T product. See Scaling Factors in doc/functional.py,

        rotary_embedding_dim: int
            The dimension to compute RoPE. Use 0 to disable RoPE.

        neox_rotary_style: bool
            Do we use GPT-NeoX RoPE or GPT-J RoPE?

        multi_block_mode: bool
            Do we enable multi-block for the masked MHA. See Generation Phase
            in doc/functional.py,

        multi_query_mode: bool
            Do we MQA instead of MHA?

        kv_orig_quant_scale: Tensor
            The tensor to store the scaling factor for quantization to INT8/FP8
            in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache in
            doc/functional.py,

        kv_quant_orig_scale: Tensor
            The tensor to store the scaling factor for dequantization from
            INT8/FP8 in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache
            in doc/functional.py,

        use_int8_kv_cache: bool
            Do we enable the INT8 KV cache?

        use_fp8_kv_cache: bool = False
            Do we enable the FP8 KV cache? (**to be merged with use_int8_kv_cache**),

        mask_type: int = 1
            The type of mask:
                * tensorrt_llm.layers.AttentionMaskType.padding for BERT,
                * tensorrt_llm.layers.AttentionMaskType.causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.bidirectional for ChatGLM,

        kv_cache_block_pointers:
            The tensor of block pointers for the KV cache. Its shape is
            [max_batch_size, max_beam_width, 2, max_blocks_per_sequence * 2]
            See KV cache section in doc/functional.py,

        host_input_lengths: Tensor = None
            (**to be removed?**),

        host_request_types: Tensor = None
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in doc/functional.py,

    Returns:
        The tensor produced by that layer.
    '''
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'GPTAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None
    assert head_size in [32, 48, 64, 80, 96, 128, 144, 160, 192, 224, 256]

    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size", np.array(head_size,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)
    unidirectional = trt.PluginField("unidirectional",
                                     np.array(1, dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling",
                                np.array(q_scaling, dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
    rotary_embedding_dim = trt.PluginField(
        "rotary_embedding_dim", np.array(rotary_embedding_dim, dtype=np.int32),
        trt.PluginFieldType.INT32)
    int8_neox_rotary_style = 1 if neox_rotary_style else 0
    neox_rotary_style = trt.PluginField(
        "neox_rotary_style", np.array(int8_neox_rotary_style, dtype=np.int8),
        trt.PluginFieldType.INT8)
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    p_dtype = default_net().plugin_config.gpt_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    mask_type = trt.PluginField("mask_type", np.array([int(mask_type)],
                                                      np.int32),
                                trt.PluginFieldType.INT32)
    multi_block_mode = trt.PluginField(
        "multi_block_mode", np.array(np.int8(multi_block_mode), dtype=np.int8),
        trt.PluginFieldType.INT8)
    multi_query_mode = trt.PluginField(
        "multi_query_mode", np.array(np.int8(multi_query_mode), dtype=np.int8),
        trt.PluginFieldType.INT8)
    int8_kv_cache = trt.PluginField("int8_kv_cache",
                                    np.array(use_int8_kv_cache, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    fp8_kv_cache = trt.PluginField("fp8_kv_cache",
                                   np.array(use_fp8_kv_cache, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    paged_kv_cache = trt.PluginField(
        "paged_kv_cache",
        np.array(default_net().plugin_config.paged_kv_cache, dtype=np.int32),
        trt.PluginFieldType.INT32)
    in_flight_batching = trt.PluginField(
        "in_flight_batching",
        np.array(default_net().plugin_config.in_flight_batching,
                 dtype=np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([
        nheads, head_size, unidirectional, q_scaling, rotary_embedding_dim,
        neox_rotary_style, context_fmha_type, multi_block_mode,
        multi_query_mode, int8_kv_cache, fp8_kv_cache, remove_input_padding,
        mask_type, paged_kv_cache, pf_type, in_flight_batching
    ])

    attn_plug = attn_plg_creator.create_plugin("causal_attn", pfc)
    plug_inputs = [
        tensor,
        past_key_value,
        sequence_length,
        past_key_value_length,
        masked_tokens,
        input_lengths,
        max_input_length,
        cache_indirection,
    ]
    if use_int8_kv_cache or use_fp8_kv_cache:
        plug_inputs += [kv_orig_quant_scale, kv_quant_orig_scale]

    if default_net().plugin_config.paged_kv_cache:
        plug_inputs += [kv_cache_block_pointers]

    if default_net().plugin_config.in_flight_batching:
        plug_inputs += [host_input_lengths, host_request_types]

    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    assert layer.num_outputs == 2, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected 2"
    output = _create_tensor(layer.get_output(0), layer)
    present_key_value = _create_tensor(layer.get_output(1), layer)
    if use_int8_kv_cache:
        # past key value
        layer.get_input(1).set_dynamic_range(-127, 127)
        # present key value
        layer.get_output(1).set_dynamic_range(-127, 127)

    assert output is not None
    assert present_key_value is not None
    return output, present_key_value


# gpt attention supporting mixed context/generation steps for in-flight batching
#
# past_key_pointers/past_value_pointers/cache_indirection_pointers: INT32 host
# tensors of shape [num_req, 2]. Each row (two int32) is a pointer (in little
# endian) to device memory.
#
# host_past_key_value_length/host_input_lengths: INT32 host tensor of shape [num_req].
#
# The content of host tensors must be ready before calling enqueue(), and usage
# of them is done when enqueue() returns. So they are not associated with the
# stream.
def inflight_batching_gpt_attention(
        # [1, total_num_tokens, sum_qkv_hidden_size]
        tensor: Tensor,
        # [beam_width, 2, local_num_k_heads, max_seq_len, head_size] or
        # [blocks, 2, local_num_k_heads, tokens_per_block, head_size]
        past_key_value: Tensor,
        # host int32 [num_req]. When using beam search, a whole beam with
        # multiple candidate sequences counts as one request
        host_beam_widths: Tensor,
        # host int32 [num_req].
        host_input_lengths: Tensor,
        # device int32 [num_seq]. When using beam search, each candidate in a
        # beam counts as a sequence.
        input_lengths: Tensor,
        # host int32 [num_req, 2]. Each row (two int32) is a pointer (in little
        # endian) to device memory. Each device memory buffer is of shape [2,
        # beam_width, local_num_k_heads, max_seq_len, head_size]
        past_key_value_pointers: Tensor,
        # host int32 [num_req]
        host_past_key_value_lengths: Tensor,
        # host int32 [num_req, 2]. Each row (two int32) is a pointer (in little
        # endian) to device memory. Each device memory buffer is of shape
        # [beam_width, max_seq_len]
        cache_indirection_pointers: Tensor,
        # host int32 [num_req]
        host_req_cache_max_seq_lengths: Tensor,
        num_heads: int,
        head_size: int,
        q_scaling: float,
        rotary_embedding_dim: int,
        neox_rotary_style: bool,
        multi_block_mode: bool,
        multi_query_mode: bool,
        kv_orig_quant_scale: Tensor,
        kv_quant_orig_scale: Tensor,
        use_int8_kv_cache: bool,
        max_input_len: int,
        max_beam_width: int,
        # host int32 [num_req, 2]. Each row (two int32) is a pointer (in little
        # endian) to device memory. Each device memory buffer is of shape [1,
        # 2, max_blocks_per_seq]
        pointers_to_kv_cache_block_pointers: Tensor,
        # [num_req, 2, beam_width, max_blocks_per_seq]
        kv_cache_block_pointers: Tensor,
        mask_type: int = 1,
        use_fp8_kv_cache: bool = False) -> Tuple[Tensor]:
    '''
    That function is deprecated - do not use!!!!
    '''

    # enable the assertion below when trt is updated to allow host tensors for
    # plugin layers
    #
    # for t in [host_beam_widths, host_input_lengths, past_key_pointers,
    # past_value_pointers, host_past_key_value_lengths,
    # cache_indirection_pointers, host_req_cache_max_seq_lengths]:
    #
    #     assert t.trt_tensor.location == trt.TensorLocation.HOST
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'IBGPTAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None
    assert head_size in [32, 48, 64, 80, 96, 128, 144, 160, 192, 224, 256]

    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size", np.array(head_size,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)
    unidirectional = trt.PluginField("unidirectional",
                                     np.array(1, dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling",
                                np.array(q_scaling, dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
    rotary_embedding_dim = trt.PluginField(
        "rotary_embedding_dim", np.array(rotary_embedding_dim, dtype=np.int32),
        trt.PluginFieldType.INT32)
    int8_neox_rotary_style = 1 if neox_rotary_style else 0
    neox_rotary_style = trt.PluginField(
        "neox_rotary_style", np.array(int8_neox_rotary_style, dtype=np.int8),
        trt.PluginFieldType.INT8)
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    assert default_net().plugin_config.remove_input_padding
    p_dtype = default_net().plugin_config.inflight_batching_gpt_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    multi_block_mode = trt.PluginField(
        "multi_block_mode", np.array(np.int8(multi_block_mode), dtype=np.int8),
        trt.PluginFieldType.INT8)
    multi_query_mode = trt.PluginField(
        "multi_query_mode", np.array(np.int8(multi_query_mode), dtype=np.int8),
        trt.PluginFieldType.INT8)
    int8_kv_cache = trt.PluginField("int8_kv_cache",
                                    np.array(use_int8_kv_cache, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    fp8_kv_cache = trt.PluginField("fp8_kv_cache",
                                   np.array(use_fp8_kv_cache, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    mask_type = trt.PluginField("mask_type", np.array([int(mask_type)],
                                                      np.int32),
                                trt.PluginFieldType.INT32)
    max_input_len = trt.PluginField(
        "max_input_len", np.array(np.int32(max_input_len), dtype=np.int32),
        trt.PluginFieldType.INT32)
    max_beam_width = trt.PluginField(
        "max_beam_width", np.array(np.int32(max_beam_width), dtype=np.int32),
        trt.PluginFieldType.INT32)
    paged_kv_cache = trt.PluginField(
        "paged_kv_cache",
        np.array(default_net().plugin_config.paged_kv_cache, dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([
        nheads, head_size, unidirectional, q_scaling, rotary_embedding_dim,
        neox_rotary_style, context_fmha_type, multi_block_mode,
        multi_query_mode, int8_kv_cache, fp8_kv_cache, mask_type, pf_type,
        max_input_len, max_beam_width, paged_kv_cache
    ])

    attn_plug = attn_plg_creator.create_plugin("causal_attn", pfc)
    plug_inputs = [
        tensor, past_key_value, host_beam_widths, host_input_lengths,
        input_lengths, past_key_value_pointers, host_past_key_value_lengths,
        cache_indirection_pointers, host_req_cache_max_seq_lengths
    ]
    if use_int8_kv_cache or use_fp8_kv_cache:
        plug_inputs += [kv_orig_quant_scale, kv_quant_orig_scale]
    if default_net().plugin_config.paged_kv_cache:
        plug_inputs += [
            pointers_to_kv_cache_block_pointers, kv_cache_block_pointers
        ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    assert layer.num_outputs == 2, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected 2"
    output = _create_tensor(layer.get_output(0), layer)
    present_key_value = _create_tensor(layer.get_output(1), layer)
    if use_int8_kv_cache:
        # past key value
        layer.get_input(1).set_dynamic_range(-127, 127)
        # present key value
        layer.get_output(1).set_dynamic_range(-127, 127)

    assert output is not None
    assert present_key_value is not None
    return output, present_key_value


def assertion(condition: Tensor, message: str = '') -> None:
    default_trtnet().add_assertion(condition.trt_tensor, message)


def layer_norm(input: Tensor,
               normalized_shape: Union[int, Tuple[int]],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05,
               use_diff_of_squares: bool = True) -> Tensor:
    '''
    Add a layer-norm operation on a tensor.

    That operation applies the layer-normalization to its input tensor. In its
    simplest form, for large language models, the 'normalized_shape' should be
    set to the hidden dimension of the activation tensor. Otherwise, it is the
    shape of the normalized fraction of the tensor (starting from the
    right-most dimension).

    The 'weight' tensor corresponds to 'gamma' in the layer-norm formula and
    'bias' is 'beta'. The 'eps' value is added to the variance before computing
    the squared-root.

    This implementation (when using the plugin) supports an additional flag to
    enable/disable the use of a difference of squares ('Var = Mean(X^2) -
    Mean(X)^2').

    Parameters:
        input : Tensor
            The tensor to normalize.

        normalized_shape : Union[int, Tuple[int]]
            The shape of the sub-tensor that is normalized. Use 'hidden_dim' to
            normalize the inner-most dimension of an activation tensor in LLMs.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        bias : Optional[Tensor] = None
            The 'beta' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.

        use_diff_of_squares : bool
            Does the plugin use the difference of squares to compute the
            variance?

    Returns:
        The output tensor of that operation.
    '''
    if not default_net().plugin_config.layernorm_plugin:
        input, weight = broadcast_helper(input, weight)
        input, bias = broadcast_helper(input, bias)
        if isinstance(normalized_shape, int):  # FIXME(kaiyu): better way?
            axis = input.ndim() - 1
        else:
            axis = input.ndim() - len(normalized_shape)
        axes_mask = 0
        for i in range(axis, input.ndim()):
            axes_mask |= 1 << i
        layer = default_trtnet().add_normalization(input.trt_tensor,
                                                   weight.trt_tensor,
                                                   bias.trt_tensor, axes_mask)
        layer.epsilon = eps
        return _create_tensor(layer.get_output(0), layer)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Layernorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)
        use_diff_of_squares = trt.PluginField(
            "use_diff_of_squares",
            np.array([int(use_diff_of_squares)], dtype=np.int32),
            trt.PluginFieldType.INT32)
        p_dtype = default_net().plugin_config.layernorm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([eps, use_diff_of_squares, pf_type])
        layernorm_plug = plg_creator.create_plugin("layernorm", pfc)

        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [input.trt_tensor, weight.trt_tensor, bias.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        return _create_tensor(layer.get_output(0), layer)


def rms_norm(input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             weight: Optional[Tensor] = None,
             eps: float = 1e-06) -> Tensor:
    '''
    Add a RMS norm operation on a tensor.

    TODO: Document!
    '''
    normalized_shape = [normalized_shape] if isinstance(
        normalized_shape, int) else normalized_shape

    dim = tuple([-i - 1 for i in range(len(normalized_shape))])

    with precision("float32"):
        varx = pow(input, 2.0)
        varx = varx.mean(dim, keepdim=True)
        denom = varx + eps
        denom = denom.sqrt()
        y = input / denom

    if weight is not None:
        y = y * weight

    return y


def generate_alibi_slopes(num_heads: int) -> Tensor:
    '''
    Compute the ALiBi slopes as described in https://arxiv.org/abs/2211.05100.

    Parameters:
        num_heads : int
            The number of heads.

    Returns:
        A constant tensor that contains the ALiBi slopes.
    '''
    closest_power_of_2 = 2**np.floor(np.log2(num_heads))
    base = np.array(2**(-(2**-(np.log2(closest_power_of_2) - 3))),
                    dtype=np.float32)
    powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
    slopes = np.power(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = np.array(2**(-(2**-(np.log2(2 * closest_power_of_2) - 3))),
                              dtype=np.float32)
        num_remaining_heads = min(closest_power_of_2,
                                  num_heads - closest_power_of_2)
        extra_powers = np.arange(1,
                                 1 + 2 * num_remaining_heads,
                                 2,
                                 dtype=np.int32)
        slopes = np.concatenate(
            [slopes, np.power(extra_base, extra_powers)], axis=0)

    slopes = slopes.astype(np.float32)
    slopes = constant(slopes.reshape(1, num_heads, 1, 1))
    return slopes


def generate_alibi_biases(slopes: Tensor, key_length: Tensor) -> Tensor:
    '''
    Compute the ALiBi biases as described in https://arxiv.org/abs/2211.05100.

    The ALiBi biases are added to the result of the Q*K^T product in the
    multihead-attention block.

    Parameters:
        slopes : Tensor
            The slopes.

        key_length : Tensor
            The size of the K vector per head.

    Returns:
        A constant tensor that contains the ALiBi biases.
    '''
    # We don't need to care about the batch size or query length since we can just broadcast
    # across the batch and query dimensions

    trt_0 = constant(int32_array(0))
    arange_shape = concat([1, 1, 1, key_length])

    arange_tensor = arange(trt_0, key_length, "float32").view(arange_shape)
    arange_tensor = cast(arange_tensor, "float32")
    return slopes * arange_tensor


def expand_mask(mask: Tensor, tgt_len: Optional[Tensor] = None) -> Tensor:
    '''
    Expand an attention mask.

    That function adds the sequence of operations to expand from a tensor of
    shape '[batch_size, src_seq_len]' to a tensor of shape
    '[batch_size, 1, tgt_seq_len, src_seq_len]'. It can be used to create the
    mask applied to the Q*K^T product before the softmax operation in the
    multihead-attention block.

    Parameters:
        mask : Tensor
            The input mask

        tgt_len : Optional[Tensor]
            The dimension of the 3rd dimension in the output tensor. If None,
            the 2nd dimension of the input is used.

    Returns:
        The tensor created by that sequence of operations.
    '''
    bsz = shape(mask, 0)
    src_len = shape(mask, 1)
    tgt_len = tgt_len if tgt_len is not None else src_len

    mask = mask.view(concat([bsz, 1, 1, src_len]))

    mask = expand(mask, concat([bsz, 1, tgt_len, src_len]))
    mask = where(mask == 0, float('-inf'), (1 - mask).cast('float32'))
    return mask


def gather_last_token_logits(hidden_states: Tensor, last_token_ids: Tensor,
                             remove_input_padding: bool) -> Tensor:
    '''
    Extract the logits that correspond to the last token from the hidden states.

    That function adds the operations to extract the logits of the last tokens
    in a batch of sequences.

    Depending on whether 'remove_input_padding' is 'True' or 'False', that
    function assumes inputs of different shapes.

    When 'remove_input_padding' is 'True', the 'hidden_states' tensor is
    assumed to be packed. It has a shape '[num_tokens, hidden_dim]' where
    'num_tokens' is the sum of the lengths of the sequences in the batch and
    'hidden_dim' is the hidden dimension. The 'last_tokens_ids' is a 1D tensor
    that encodes the inclusive prefix-sums of the lengths of the sequences in
    the batch.

    When 'remove_input_padding' is 'False', the 'hidden_states' tensor is
    assumed to be padded. It has a shape '[batch_size, max_seqlen, hidden_dim]'
    where 'max_seqlen' is the length of the longest sequence in the batch and
    'hidden_dim' is the hidden dimension.  The 'last_token_ids' is a 1D tensor
    that encodes the length of each sequence in the batch.

    In both cases, that function produces a tensor of shape '[batch_size,
    hidden_size]' where the row at index 'i' corresponds to the logits of the
    last token from the 'i'-th sequence.

    Parameters:
        hidden_states : Tensor
            The hidden states

        last_token_ids : Tensor
            The inclusive prefix-sum of the lengths or the lenghts of the
            sequences in the batch.

        remove_input_padding : bool
            Indicate if the hidden_states are packed ('True') or padded
            ('False').

    Returns:
        The tensor created by that sequence of operations.
    '''
    if remove_input_padding:
        hidden_states = index_select(hidden_states, 1,
                                     last_token_ids - 1)  # [1, seq_len, hidden]

        hidden_states = hidden_states.view(
            concat([shape(last_token_ids, 0),
                    shape(hidden_states, 2)]))
    else:
        # only calculate logits for the last token
        # [batch_size, seqlen, hidden_size] -> [batch_size, hidden_size]
        last_token_ids = last_token_ids.view(
            concat([shape(last_token_ids, 0), 1, 1]))
        last_token_ids = expand(
            last_token_ids,
            concat([shape(last_token_ids, 0), 1,
                    shape(hidden_states, 2)]))
        last_token_ids = last_token_ids - 1
        hidden_states = gather(
            hidden_states, dim=1, indices=last_token_ids).view(
                concat([shape(hidden_states, 0),
                        shape(hidden_states, 2)]))
    return hidden_states


ACT2FN = {
    'relu': relu,
    'tanh': tanh,
    'gelu': gelu,
    'gelu_new': gelu,
    'gelu_fast': gelu,
    'geglu': geglu,
    'silu': silu,
    'softplus': softplus,
    'swiglu': swiglu,
    'fast-swiglu': swiglu,
}

GATED_ACT_2_ACT = {
    'swiglu': 'silu',
    'fast-swiglu': 'silu',
    'geglu': 'gelu',
}


def is_gated_activation(activation):
    '''
    Is a given activation function gated?

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        True if the function is gated, False otherwise.
    '''
    assert activation in ACT2FN
    return activation in GATED_ACT_2_ACT


def non_gated_version(activation):
    '''
    Given an activation function, get the non-gated version.

    If the activation function is non-gated, it returns the same activation
    function name.

    For example, that function returns 'silu' for 'swiglu' and 'relu' for
    'relu'.

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        The name of the non-gated activation function.
    '''
    if is_gated_activation(activation):
        return GATED_ACT_2_ACT[activation]
    return activation
