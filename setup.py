#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import sys


class CMakeBuild(build_ext):
    """
    Runs cmake and make install instead of a traditional C/C++ extension build.
    """
    def run(self):
        build_dir = os.path.dirname(os.path.realpath(__file__))
        for cmd, target in [("cmake", ""), ("make -j -C", "install")]:
            if os.system("%s %s %s" % (cmd, build_dir, target)) != 0:
                print("ERROR: Failed to run %s" % cmd)
                sys.exit(1)
        #copy to cuda lib dir/ the cuda lib dir will be in path, so cudanet should automatically work
        os.system("cp %s/cudanet/libcconv2_cudanet.so /usr/local/cuda/lib64" % (build_dir))


cudanet = Extension('cudanet.libcudanet', sources = [],
                    runtime_library_dirs=['cudanet'])
install_requires = ['numpy', ]
test_requires = ['nose', ]

setup(name="cudanet",
      version="0.2.8",
      description="Provides a set of cudamat like functions using cuda-convnet2 kernels",
      ext_modules = [cudanet],
      packages=['cudanet','cudanet.layers'],
      author="Alex Khrizevsky, Nervanasys, and Tim Dettmers",
      author_email="tim.dettmers@gmail.com",
      url="https://code.google.com/p/cuda-convnet2/",
      install_requires=install_requires,
      tests_require=test_requires,
      cmdclass={'build_ext': CMakeBuild},
)
