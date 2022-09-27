算子描述符
============

某些算子具备一些固有属性，如Convolution算子在计算时需要的pad、stride等信息。Cambricon CNNL使用算子描述符（Descriptor）来描述算子的固有属性，并提供相关接口创建、设置和销毁算子描述符，从而简化算子计算接口中参数的数量。对于使用了算子描述符的算子，描述符中包含的参数需要通过设置描述符传入算子计算接口，而不是通过张量来设置。用户需要在调用算子计算接口前完成算子描述符的设置。

需要使用算子描述符的算子如下：

- Activation
- Attention
- Convolution
- Customized Active
- Grep
- Gru
- OpTensor
- Pooling
- Reduce
- Reorg
- Transpose
- Trigon

使用方法
---------------------------

算子描述符被存放在该算子的结构体中，通常命名为 ``cnnlXXXDescriptor_t``，其中 ``XXX`` 为算子名。例如Gru算子的算子描述符信息被存放在 ``cnnlGruDescriptor_t`` 结构体中。

.. attention::
   | Pooling算子根据计算维度，需要调用不同的接口来设置算子描述符。如果是2维Pooling计算，调用 ``cnnlSetPooling2dDescriptor()`` 接口， 如果是3维Pooling计算，调用 ``cnnlSetPoolingNdDescriptor()`` 接口。但是，创建和销毁算子描述符与维度无关，均调用 ``cnnlCreatePoolingDescriptor()`` 和 ``cnnlDestroyPoolingDescriptor()`` 接口。

执行下面步骤设置算子描述符：

1. 调用 ``cnnlCreateXXXDescriptor()`` 接口创建一个描述符。如调用 ``cnnlCreateGruDescriptor()`` 接口创建Gru算子的算子描述符。
2. 调用 ``cnnlSetXXXDescriptor()`` 接口为算子描述符中每一个算子的固有属性赋值，如调用 ``cnnlSetGruDescriptor()`` 为Gru的 ``algo``、``is_bidirectional``、``num_layer`` 赋值。
3. 调用算子计算接口 ``cnnlXXX()`` 时，如 ``cnnlGru``，将算子描述符作为参数传递给接口。
4. 调用 ``cnnlDestroyXXXDescriptor()`` 接口，如 ``cnnlDestroyGruDescriptor()`` 销毁该描述符。

对于不使用算子描述符的算子，按照 ``cnnl.h`` 中暴露的接口进行调用即可。
