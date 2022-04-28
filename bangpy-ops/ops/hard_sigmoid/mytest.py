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



target = "mlu290"
task_num = 64
dev = bangpy.device(0)
for dtype in [bangpy.float32]:
    print(f"dtype: {dtype}")
    
    for shape in [(1,1,1,1,1,1,128,2088960)]:

        data_in = np.random.uniform(low=-5, high=5, size=shape).astype(dtype.as_numpy_dtype)
        data_out = data_in.astype(dtype.as_numpy_dtype).flatten()
        # hard_sigmoid activate function
        # for i in range(len(data_out)):
        #     if data_out[i]<=-3:
        #         data_out[i]=0
        #     elif data_out[i]>=3:
        #         data_out[i]=1
        #     else:
        #         data_out[i]=data_out[i]/6+1/2
    
        data_out=data_out.reshape(shape)#Recovering the shape of the tensor

        # set I/O data
        data_in_dev = bangpy.Array(data_in.astype(dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)

        # testing
        f=Hard_sigmoid(dtype, target, task_num).compute_body()
        f(data_in_dev,data_out_dev)
        # bangpy.assert_allclose(data_out_dev.numpy(), data_out.astype(dtype.as_numpy_dtype))

        # hardware time
        evaluator = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
        print( "Hardware time : %f ms" % (evaluator(data_in_dev, data_out_dev).mean * 1e3))
        evaluator1 = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
        print( "Hardware time : %f ms" % (evaluator1(data_in_dev, data_out_dev).mean * 1e3))
        evaluator2 = f.time_evaluator(number=10, repeat=1, min_repeat_ms=0)
        print( "Hardware time : %f ms" % (evaluator2(data_in_dev, data_out_dev).mean * 1e3))

        # BANGC code
        dev_module = f.module.imported_modules[0]
        print("--------------MLU code----------------")
        print(dev_module.get_source())
