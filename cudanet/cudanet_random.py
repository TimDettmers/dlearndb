'''
Created on Jun 27, 2015

@author: tim
'''
from cpp_interface import *
import cudanet as gpu

_cudanet = lib.cudanet
_cudanet.init_random_no_seed()

def rand(d0, d1):    
    A = gpu.empty((d0,d1))
    randomize_uniform(A)
    return A    

def randn(d0, d1):    
    A = gpu.empty((d0,d1))
    randomize_gaussian(A,0,1)
    return A    

def normal(mean, std, shape):
    A = gpu.empty(shape)
    randomize_gaussian(A,mean,std)
    return A    


def cudanet_init_random(seed = None):
    """
    Initialize and seed the random number generator.
    """
    if not seed:
        _cudanet.init_random_no_seed()
    else:
        _cudanet.init_random(ct.c_ulonglong(seed))        
        
def cudanet_destroy_random():
    """
    Destroy the random number generator.
    """
    _cudanet.destroy_random()

def randomize_gaussian(A, mean, stdev):
    """
    Fill in matrix with random values according to gaussian distribution with mean
    and stdev
    """
    err_code = _cudanet.randomize_gaussian(A.p_mat, ct.c_float(mean), ct.c_float(stdev));
    if err_code:
        raise generate_exception(err_code)

def randomize_uniform(A):
    """
    Fill in matrix with random values according to uniform distribution
    between 0 and 1
    """
    err_code = _cudanet.randomize_uniform(A.p_mat);
    if err_code:
        raise generate_exception(err_code)

def randomize_binary(A):
    """
    Fill in matrix with random values of {0,1} according to mask on uniform
    distribution
    """
    err_code = _cudanet.randomize_binary(A.p_mat);
    if err_code:
        raise generate_exception(err_code)
    
    