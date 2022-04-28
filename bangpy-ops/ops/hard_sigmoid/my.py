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
import bangpy
from bangpy import tcp
from bangpy.platform.bang_config import TARGET
import numpy as np
from bangpy.tcp.runtime import TaskType
from hard_sigmoid import Hard_sigmoid
import sys
sys.path.append("..")
from add.add import Add
import csv



target = "mlu290"
dev = bangpy.device(0)

def test():
    dtypes = [bangpy.float32]
    # shapes = [(2**26)]
    shapes = [(1,1,1,1,1,1,128,2088960),(1,1,1,1,1,1,256,2088960),(1,1,1,1,1,1,512,2088960),(1,1,1,1,1,1,1024,2088960)]
    task_nums = list(range(4,65,4))

    for dtype in dtypes:
        print(f"dtype: {dtype}")
        data = [[r'shape\task_num']+task_nums]
        for shape in shapes:
            print(f"shape: {shape}")
            data_in = np.random.uniform(low=-10, high=10, size=shape).astype(dtype.as_numpy_dtype)
            data_out = data_in.astype(dtype.as_numpy_dtype).flatten()
            # hard_sigmoid function start
            # ...
            # hard_sigmoid function end

            # reshape
            data_out=data_out.reshape(shape)
           
            # set I/O data
            data_in_dev = bangpy.Array(data_in, dev)
            data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
            row = [shape]
            for task_num in task_nums:
                print(f'task number: {task_num}')
                f=Hard_sigmoid(dtype, target, task_num).compute_body()
                f(data_in_dev,data_out_dev)
                evaluator = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
                t = evaluator(data_in_dev,data_out_dev).mean * 1e3
                row.append(f'{t:.2f}')
                print( "Hardware time : %f ms" % t)
            data.append(row)
        with open(f'compute_nopipe_{dtype.name}.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(data)

def single_test():
    dtype = bangpy.float32
    shape = (1,1,1,1,1,1,128,2088960)
    task_num = 64
    f=Hard_sigmoid(dtype, target, task_num).compute_body()
    print(f.get_source())
    print(f"dtype: {dtype}")
    print(f"shape: {shape}")
    print(f'task number: {task_num}')
    data_in = np.random.uniform(low=-10, high=10, size=shape).astype(dtype.as_numpy_dtype)
    data_out = data_in.astype(dtype.as_numpy_dtype).flatten()
    
    # hard_sigmoid function start
    # ...
    # hard_sigmoid function end

    # reshape
    data_out=data_out.reshape(shape)

    # set I/O data
    data_in_dev = bangpy.Array(data_in, dev)
    data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
    f(data_in_dev,data_out_dev)
    evaluator = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
    t = evaluator(data_in_dev, data_out_dev).mean * 1e3
    print( "Hardware time : %f ms" % t)
    #f(data_in0_dev, data_in1_dev, data_out_dev)
    #print("data_in0")
    #print(data_in0)
    #print("data_in1")
    #print(data_in1)
    #print("data_out")
    #print(data_out)
    #print("data_out_dev")
    #print(data_out_dev.numpy())
    #bangpy.assert_allclose(data_out_dev.numpy(), data_out)

# single_test()
test()
