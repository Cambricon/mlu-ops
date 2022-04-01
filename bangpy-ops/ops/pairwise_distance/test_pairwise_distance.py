# Copyright (C) [2021] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=missing-docstring, invalid-name, too-many-locals
"""A multi-platform code link example test for BANGPy TCP."""
from cmath import pi
from traceback import print_tb
import numpy as np
import pytest

import bangpy
import bangpy as bp
from bangpy import tcp
from bangpy.common import utils, load_op_by_type
from bangpy.platform.bang_config import ALIGN_LENGTH, TARGET
from bangpy.tcp.runtime import TaskType
from pairwise_distance import DTYPES, KERNEL_NAME, TARGET_LIST

@pytest.mark.parametrize(
    "shape", 
    [        
        ((1666, 128, 8), (1666, 128, 8))
    ],
)

@pytest.mark.parametrize(
    "dtype", DTYPES,
)

@pytest.mark.parametrize(
    "p", [2.],
)

@pytest.mark.parametrize(
    "eps", [0,],
)

@pytest.mark.parametrize(
    "keepdim", [False],
)



def test_pairwise_distance(target, shape, p, eps, keepdim, dtype): 
    if target not in TARGET_LIST:
        return

    class pwsdst_processor:
        _dtype = None
        _p = 1
        _eps = 0.000001
        _keepdim = False
        _shape1 = None
        _shape2 = None
        _dev = None

        _ori_input1 = None
        _ori_input2 = None

        _mlu_input1 = None
        _mlu_input2 = None

        _mlu_output = None
        _output_len = -1

        _pd_height = -1
        _pd_width = -1
        _pd_len = -1

        def init(self, shapes, dtype, p, eps, keepdim):
            self._dtype = dtype
            self._p = p
            self._eps = eps
            self._keepdim = keepdim
            if len(shape[0]) > len(shape[1]):
                self._shape1 = shape[0]
                self._shape2 = shape[1]
            else:
                self._shape1 = shape[1]
                self._shape2 = shape[0]

            self._dev = bp.device(0)


        def create_origin_intput(self):
            shape1 = np.array(self._shape1).astype('int32')
            shape2 = np.array(self._shape2).astype('int32')
            self._ori_input1 = np.random.uniform(low=-10, high=10, size=shape1).astype(self._dtype.as_numpy_dtype)
            self._ori_input2 = np.random.uniform(low=-10, high=10, size=shape2).astype(self._dtype.as_numpy_dtype)

        def create_mlu_input(self):
            reshp_x = self._ori_input1.flatten()

            self._mlu_input1 = bp.Array(reshp_x, self._dev)

            reshp_y = self._ori_input2.flatten()
            self._mlu_input2 = bp.Array(reshp_y, self._dev)

        def create_output(self):
            self._output_len = self.get_total_size(self._shape1) // self._shape1[1]
            output_buffer = np.zeros(self._output_len, dtype=self._dtype.as_numpy_dtype)
            self._mlu_output = bp.Array(output_buffer, self._dev)

        def get_total_size(self, shp):
            size = 1
            for s in shp:
                size *= s
            return size

        def create_pd_paras(self):
            self._pd_len = self._shape1[1]
            self._pd_width = 1
            for i in range(2, len(self._shape1)):
                self._pd_width *= self._shape1[i]

            self._pd_height = self._shape1[1] * self._shape1[0]



    ins = pwsdst_processor()
    ins.init(shape, dtype, p, eps, keepdim)    
    ins.create_origin_intput()
    ins.create_output()
    ins.create_pd_paras()
    ins.create_mlu_input()
    
    func = load_op_by_type(KERNEL_NAME, dtype.name)
    func(ins._mlu_input1, ins._mlu_input2, ins.get_total_size(ins._shape1), ins.get_total_size(ins._shape2),
         ins._pd_len, ins._pd_height, ins._pd_width, ins._output_len
         , ins._mlu_output)

    '''bangpy.assert_allclose(
        output_dev.numpy(), data_out.astype(dtype.as_numpy_dtype), rtol=0.1, atol=0.1
    )'''
   