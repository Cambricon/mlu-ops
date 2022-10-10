.. _programming_guide:

编程指南
=================
对于一个网络模型，用户通过调用常用框架，如PyTorch、TensorFlow等或自定义框架，解析模型并生成Cambricon BANGC OPS算子计算所需要的输入参数和数据。根据输入参数和数据，主机端调用Cambricon BANGC OPS算子接口，完成算子的高性能计算。

由于Cambricon BANGC OPS算子的计算是在MLU设备端完成，Cambricon BANGC OPS通过句柄与计算时使用的MLU设备和队列绑定。MLU设备端的设备初始化、计算时输入输出数据的内存分配、队列的创建等都是通过主机端调用CNRT接口完成。

MLU设备端完成计算后，计算结果通过调用CNRT接口返回到主机端CPU来进行下一步任务。

Cambricon BANGC OPS编程中，每个算子都需要绑定句柄。句柄主要用于保存当前运行环境的上下文、计算设备信息和队列信息。不同的算子可以处在不同的队列中，各个队列可以并行执行，而队列内部顺序执行。更多详情，查看 :ref:`句柄`。

部分Cambricon BANGC OPS算子计算时需要额外工作空间用于算子优化。Cambricon BANGC OPS内部不会申请工作空间，用户需要调用工作空间相关接口申请计算所需要的工作空间，并将工作空间大小传入算子接口。详情请查看 :ref:`workspace`。

此外，由于CPU和MLU是异步执行任务，用户需要调用 ``cnrtQueueSync()`` 接口同步队列中执行的任务。

.. _单算子编程指南:

单算子编程指南
-------------------

如果运行单个Cambricon BANGC OPS算子，执行以下操作：

1. 调用 ``cnrtGetDevice()`` 获取设备对应的设备号。
#. 调用 ``cnrtSetDevice()`` 绑定当前线程所使用的设备。
#. 调用 ``mluOpCreate()`` 创建一个Cambricon BANGC OPS句柄。句柄将与当前线程所使用的设备绑定。
#. 调用 ``cnrtQueueCreate()`` 创建一个计算队列。
#. 调用 ``mluOpSetQueue()`` 将队列和Cambricon BANGC OPS句柄绑定。
#. 如果接口需要设置 ``mluOpTensorDescriptor_t``，调用 ``mluOpCreateTensorDescriptor()`` 创建张量描述符，并调用 ``mluOpSetTensorDescriptor()`` 设置算子输入和输出描述信息。具体信息包括数据类型、形状信息、维度顺序等。
#. 如果接口需要设置算子描述符 ``mluOpXXXDescriptor_t``，调用 ``mluOpCreateXXXDescriptor()`` 创建算子描述符，并调用 ``mluOpSetXXXDescriptor()`` 为该算子添加描述。``XXX`` 需要替换为算子名称。
#. 如果接口需要申请额外工作空间，即需要设置 ``workspace`` 和 ``workspace_size``，调用 ``mluOpXXXGetWorkspaceSize()`` 推导该算子需要的最小的临时空间大小。``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtMalloc()`` 开辟算子需要的输入、输出和临时空间。
#. 调用 ``cnrtMemcpy()`` 将输入数据拷贝到设备端。拷贝到设备端的数据必须与之前设置的张量描述符信息保持一致。
#. 调用算子的API接口 ``mluOpXXX()``，传入Cambricon BANGC OPS句柄信息和接口所需所有参数。``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtQueueSync()`` 同步CPU和MLU端。
#. 调用 ``cnrtMemcpy()`` 将输出从设备端拷贝回主机端。
#. 调用 ``cnrtFree()`` 释放设备端的空间。
#. 如果执行了第7步，调用 ``mluOpDestroyTensorDescriptor()`` 释放描述信息。
#. 如果执行了第8步，调用 ``mluOpDestroyXXXDescriptor()`` 释放资源。``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtQueueDestroy()`` 释放队列信息。
#. 调用 ``mluOpDestroy()`` 释放Cambricon BANGC OPS句柄。

有关接口详情，请参见《Cambricon BANGC OPS Developer Guide》。

多算子搭建网络
----------------------------

使用Cambricon BANGC OPS搭建一个多算子的网络，可以考虑如下方法，达到空间复用的效果。操作步骤如下：

1. 执行 单算子编程指南_ 的前六步完成初始化等操作。
#. 为网络每一层准备张量描述符 ``mluOpTensorDescriptor_t``，调用 ``mluOpCreateTensorDescriptor()`` 创建张量描述符，并调用 ``mluOpSetTensorDescriptor()`` 设置算子输入和输出描述信息。具体信息包括数据类型、形状信息、维度顺序等。
#. 为网络里面含有算子描述符的层创建算子 ``mluOpXXXDescriptor_t``，调用 ``mluOpCreateXXXDescriptor()`` 创建算子描述符，并调用 ``mluOpSetXXXDescriptor()`` 为该算子添加描述。其中 ``XXX`` 需要替换为算子名称。
#. 为网络里面所有需要的工作空间的层设置 ``workspace`` 和 ``workspace_size``，调用 ``mluOpXXXGetWorkspaceSize()`` 推导该算子需要的最小的临时空间大小，其中 ``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtMemcpy()`` 将输入层的数据拷贝到设备端。
#. 按照网络层的顺序调用 ``mluOpXXX()``，每一层绑定上面申请的句柄，保证它们处于同一计算队列中。``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtQueueSync()`` 同步CPU端和设备端。
#. 拷回数据、释放句柄和设备端空间。执行 单算子编程指南_ 的13-18步。

有关接口详情，请参见《Cambricon BANGC OPS Developer Guide》。

