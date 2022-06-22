import numpy as np
import pytest

import bangpy
from bangpy.common import load_op_by_type
from cross import DTYPES, KERNEL_NAME, TARGET_LIST

@pytest.mark.parametrize(
    "shape,dim",
    [
    ((1, 1, 1, 1, 2, 3, 4, 5),5),
    ((2,1,2,1,2,2,2,3),7),
    ((2,1,2,1,2,2,3,3),6),
    ((3,2,2,1,1,1,1,1),0),
    ((2,2,2,3,3,4,4,4),4),
    ((1,2,2,2,3,128,1,1),4),
    ((1,2,2,2,3,128,1,1),-4),
    ((1024,2,2,3,3,4,4,4),4),
    ((1,1024,2,4,3,2,3,1024),4),
    ((2,1024,4,4,3,2,3,1024),4),
    ((1,1024,2,4,3,2,3,1024),6),
    ((1,1024,2,4,3,2,8192,2),4),
    ((1,3,3,4,3,2,8192,2),1),
    ((3,3,3,3,3,3,3,8192),6),
    ((1,2,2,2,3,128,1,1),1),  #不合法的输入样例
    ((1,2,2,2,3,128,1,1),-9), #不合法的输入样例
    # ((2,1024,2,4,3,2,8192,2),4),    
        #step>buffer长度的情况，分支2，当group达到这个量级就会报错
        #原因暂时不明，详见设计文档优化记录和测试文档总结分析部分
    ]
)
@pytest.mark.parametrize(
    "dtype", DTYPES,
)

# @pytest.mark.repeat(1000)
def test_cross(target, shape, dim, dtype):
    if target not in TARGET_LIST:
        return
    data_in0 = np.random.uniform(low=-1, high=1, size=shape)
    data_in1 = np.random.uniform(low=-1, high=1, size=shape)

    if -8 <= dim <= 7 and shape[dim] == 3:
        if dim < 0:
            dim = dim + 8

        #以下计算等价于torch.cross(data_in0,data_in1,dim),只不过数据类型一个是tensor一个是numpy array
        axes = list(np.arange(dim))
        axes += list(np.arange(dim+1,len(shape)))
        axes.append(dim)
        axes2 = list(np.arange(dim))
        axes2.append(len(shape)-1)
        axes2 += list(np.arange(dim,len(shape)-1))

        data0 = data_in0.transpose(axes).astype(dtype.as_numpy_dtype)
        data1 = data_in1.transpose(axes).astype(dtype.as_numpy_dtype)
        dataout = np.cross(data0,data1)
        data_out = dataout.transpose(axes2).astype(dtype.as_numpy_dtype)


        dev = bangpy.device(0)

        # set I/O datas

        data_in0_dev = bangpy.Array(data_in0.astype(dtype.as_numpy_dtype), dev)
        data_in1_dev = bangpy.Array(data_in1.astype(dtype.as_numpy_dtype), dev)
        data_out_dev = bangpy.Array(np.zeros(data_out.shape, dtype.as_numpy_dtype), dev)
        dimshape = bangpy.Array(np.array(shape).astype(bangpy.int32.as_numpy_dtype),dev)

        f1 = load_op_by_type(KERNEL_NAME, dtype.name)
        f1(data_in0_dev, data_in1_dev, dimshape, dim, data_out_dev)

        #diff3测试
        data_out = data_out.flatten()
        data_out_dev = data_out_dev.numpy().flatten()
        diff = np.abs(data_out - data_out_dev)
        data_out = np.abs(data_out)
        maxdiff3 = 0
        if dtype == bangpy.float16:
            th = 1e-4
        elif dtype == bangpy.float32:
            th = 1e-6
        for i,data in enumerate(data_out):
            if data > th:
                diff3 = diff[i] / data
            else:
                diff3 = diff[i]
            if diff3 > maxdiff3:
                maxdiff3 = diff3
        assert maxdiff3 == 0

        # bangpy.assert_allclose(data_out_dev.numpy(), data_out)

    else :
        print("shape or dim are illegal! shape:", end =" ")
        print(shape, end = " ")
        print("dim:", end = " ")
        print(dim)
