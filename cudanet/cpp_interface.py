'''
Created on Jun 29, 2015

@author: tim
'''
import ctypes as ct
import ctypes.util
from ctypes import pythonapi

MAX_ONES = 1024*256

cudanet_lib_path = ct.util.find_library('cconv2_cudanet')
if cudanet_lib_path is None:
    raise OSError("Problems locating libcudanet shared library")
_cudanet = ct.cdll.LoadLibrary(cudanet_lib_path)


_cudanet.get_last_cuda_error.restype = ct.c_char_p
_cudanet.cublas_init.restype = ct.c_int
_cudanet.cublas_shutdown.restype = ct.c_int
_cudanet.cuda_set_device.restype = ct.c_int

_cudanet.init_empty.restype = ct.c_int
# _cudanet.reshape.restype = ct.c_int
_cudanet.copy_to_host.restype = ct.c_int
_cudanet.copy_from.restype = ct.c_int
_cudanet.set_host_mat.restype = ct.c_int
_cudanet.allocate_device_memory = ct.c_int
_cudanet.copy_to_device.restype = ct.c_int
_cudanet.copy_to_device_buffer.restype = ct.c_int
_cudanet.copy_on_device.restype = ct.c_int
_cudanet.free_device_memory.restype = ct.c_int
_cudanet.add_elementwise.restype = ct.c_int
_cudanet.add_scalar.restype = ct.c_int
_cudanet.add_mult.restype = ct.c_int
_cudanet.add_vector.restype = ct.c_int
_cudanet.mat_vector_op.restype = ct.c_int
_cudanet.assign_scalar.restype = ct.c_int
_cudanet.subtract_elementwise.restype = ct.c_int
_cudanet.divide_elementwise.restype = ct.c_int
_cudanet.mult_elementwise.restype = ct.c_int
_cudanet.mult_by_scalar.restype = ct.c_int
_cudanet.sign.restype = ct.c_int
_cudanet.apply_fill.restype = ct.c_int
_cudanet.apply_identity.restype = ct.c_int
_cudanet.apply_logistic.restype = ct.c_int
_cudanet.apply_logistic_grad.restype = ct.c_int
_cudanet.apply_rectified_linear.restype = ct.c_int
_cudanet.apply_rectified_linear_grad.restype = ct.c_int
_cudanet.apply_tanh.restype = ct.c_int
_cudanet.apply_soft_threshold.restype = ct.c_int
_cudanet.apply_dropout.restype = ct.c_int
_cudanet.apply_abs.restype = ct.c_int
_cudanet.apply_log_1_plus_exp.restype = ct.c_int
_cudanet.apply_gamma.restype = ct.c_int
_cudanet.apply_lgamma.restype = ct.c_int
_cudanet.apply_log.restype = ct.c_int
_cudanet.apply_clip_range.restype = ct.c_int
_cudanet.apply_exp.restype = ct.c_int
_cudanet.apply_sqrt.restype = ct.c_int
_cudanet.apply_pow.restype = ct.c_int
_cudanet.apply_pow_matrix.restype = ct.c_int
_cudanet.reciprocal.restype = ct.c_int
_cudanet.convolution.restype = ct.c_int
_cudanet.print_devmat.restype = ct.c_int
_cudanet.get_col_slice_view.restype = ct.c_int
_cudanet.get_col_slice_copy.restype = ct.c_int
_cudanet.set_col_slice.restype = ct.c_int
_cudanet.get_row_slice_view.restype = ct.c_int
_cudanet.get_row_slice_copy.restype = ct.c_int
_cudanet.set_row_slice.restype = ct.c_int
_cudanet.assign_col_slice.restype = ct.c_int
_cudanet.assign_row_slice.restype = ct.c_int

_cudanet.euclid_norm.restype = ct.c_float
_cudanet.manhattan_norm.restype = ct.c_float
_cudanet.vdot.restype = ct.c_float
_cudanet.dot.restype = ct.c_int

_cudanet.less_than.restype = ct.c_int
_cudanet.less_than_scalar.restype = ct.c_int
_cudanet.greater_than.restype = ct.c_int
_cudanet.greater_than_scalar.restype = ct.c_int
_cudanet.greater_equal.restype = ct.c_int
_cudanet.greater_equal_scalar.restype = ct.c_int
_cudanet.less_equal.restype = ct.c_int
_cudanet.less_equal_scalar.restype = ct.c_int
_cudanet.equals.restype = ct.c_int
_cudanet.equals_scalar.restype = ct.c_int
_cudanet.minimum.restype = ct.c_int
_cudanet.minimum_scalar.restype = ct.c_int
_cudanet.maximum.restype = ct.c_int
_cudanet.maximum_scalar.restype = ct.c_int
_cudanet.reshape.restype = ct.c_int
_cudanet.add_col_vec.restype = ct.c_int
_cudanet.add_col_mult.restype = ct.c_int
_cudanet.add_row_vec.restype = ct.c_int
_cudanet.mult_by_col_vec.restype = ct.c_int
_cudanet.mult_by_row_vec.restype = ct.c_int
_cudanet.divide_by_col_vec.restype = ct.c_int
_cudanet.divide_by_row_vec.restype = ct.c_int
_cudanet.max_by_axis.restype = ct.c_int
_cudanet.min_by_axis.restype = ct.c_int
_cudanet.sum.restype = ct.c_int
_cudanet.sumsq.restype = ct.c_int
_cudanet.mean.restype = ct.c_int
_cudanet.convolution_back_errors.restype = ct.c_int
_cudanet.convolution_back_weights.restype = ct.c_int
_cudanet.copy_transpose.restype = ct.c_int

_cudanet.max_pool.restype = ct.c_int
_cudanet.max_pool_undo.restype = ct.c_int
_cudanet.avg_pool.restype = ct.c_int
_cudanet.avg_pool_undo.restype = ct.c_int
_cudanet.l2_pool.restype = ct.c_int
_cudanet.l2_pool_undo.restype = ct.c_int
_cudanet.unpool_forward.restype = ct.c_int
_cudanet.unpool_backward.restype = ct.c_int

_cudanet.adadelta_update.restype = ct.c_int
_cudanet.xcov.restype = ct.c_int
_cudanet.mean_norm.restype = ct.c_int
_cudanet.crossmap_response_norm.restype = ct.c_int
_cudanet.crossmap_response_norm_undo.restype = ct.c_int
_cudanet.local_contrast_norm.restype = ct.c_int
_cudanet.local_contrast_norm_undo.restype = ct.c_int
_cudanet.get_gpu_pointer.restype = ct.c_ulong
_cudanet.get_device_id.restype = ct.c_int
_cudanet.set_device_id.restype = None
_cudanet.get_peer_access.restype = ct.c_int
_cudanet.get_data_device_id.restype = ct.c_int
_cudanet.randomize_gaussian.restype = ct.c_int
_cudanet.randomize_uniform.restype = ct.c_int
_cudanet.randomize_binary.restype = ct.c_int
_cudanet.add_noise_gaussian.restype = ct.c_int
_cudanet.add_noise_uniform.restype = ct.c_int
_cudanet.randomize_uniform_thresh.restype = ct.c_int
_cudanet.init_random.restype = None
_cudanet.init_random_no_seed.restype = None
_cudanet.destroy_random.restype = None
_cudanet.sync_stream.restype = None
_cudanet.softmax.restype = ct.c_int
_cudanet.softmax_grad.restype = ct.c_int
_cudanet.crossent_cost.restype = ct.c_int
_cudanet.crossent_cost_grad.restype = ct.c_int
_cudanet.get_gpu_pythonbuf.restype = ct.py_object
_cudanet.multi_ranked_error.restype = ct.c_int

_cudanet.argsort.restype = ct.c_int


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc



class CUDANetException(Exception):
    pass

def get_last_cuda_error():
    return str(_cudanet.get_last_cuda_error())

def sync_stream():
    """
    Sets the current deviceid context
    """
    _cudanet.sync_stream()

def set_device_id(d):
    """
    Sets the current deviceid context
    """
    _cudanet.set_device_id(ct.c_int(d))

def get_device_id():
    """
    Returns the current deviceid context
    """
    return _cudanet.get_device_id()

def get_num_devices():
    """
    Returns the current deviceid context
    """

    err_code = ct.c_int(0)
    numdev = _cudanet.get_num_devices(ct.byref(err_code))
    if (err_code):
        generate_exception(err_code)
    return numdev


def get_peer_access(src, dest):
    """
    Returns whether deviceid src to deviceid dest access available
    """
    return _cudanet.set_peer_access(ct.c_int(src), ct.c_int(dest))

def generate_exception(err_code):
    """
    Return a CUDANetException object based on the error code err_code.
    """
    if err_code == -1:
        return CUDANetException("Incompatible matrix dimensions.")
    elif err_code == -2:
        return CUDANetException("CUBLAS error.")
    elif err_code == -3:
        return CUDANetException("CUDA error: " + get_last_cuda_error())
    elif err_code == -4:
        return CUDANetException("Operation not supported on views.")
    elif err_code == -5:
        return CUDANetException("Operation not supported on transposed matrices.")
    elif err_code == -6:
        return CUDANetException("Invalid value")
    elif err_code == -7:
        return CUDANetException("Incompatible transposedness.")
    elif err_code == -8:
        return CUDANetException("Matrix is not in device memory.")
    elif err_code == -9:
        return CUDANetException("Operation not supported.")
    elif err_code == -10:
        return CUDANetException("Convolutional dimensions incorrect")
    elif err_code == -11:
        return CUDANetException("Convolution Number of filters must be multiple of 16.")
    elif err_code == -12:
        return CUDANetException("Invalid axis type")
    elif err_code == -13:
        return CUDANetException("Randomizer not initialized")

class NVMat(ct.Structure):
    pass
class HostMat(ct.Structure):
    pass

class _PY_BUFFER(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("smalltable", ctypes.c_ssize_t * 2),
        ("internal", ctypes.c_void_p)
    ]

class cudanetmat(ct.Structure):
    _fields_ = [('data_host', ct.POINTER(HostMat)),
                ('data_device', ct.POINTER(NVMat)),
                ('on_device', ct.c_int),
                ('on_host', ct.c_int),
                ('size', ct.c_int * 2),
                ('is_trans', ct.c_int),
                ('owns_data', ct.c_int)]

class rnd_struct(ct.Structure):
    _fields_ = [('dev_rnd_mults', ct.POINTER(ct.c_uint)), 
                ('dev_rnd_words', ct.POINTER(ct.c_longlong))]


class Transposedarray(object):
    def __init__(self, mat):
        self.mat = cudanetmat()
        ct.memmove(ct.pointer(self.mat), ct.pointer(mat), ct.sizeof(self.mat))
        self.mat.is_trans = 1
        self.p_mat = ct.pointer(self.mat)
        
class lib(object):
    _cudanet = _cudanet

