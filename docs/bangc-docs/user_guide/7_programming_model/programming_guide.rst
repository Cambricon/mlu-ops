.. _programming_guide:

编程指南
=================

Cambricon CNNL编程中，每个算子都需要绑定句柄。句柄主要用于保存当前运行环境的上下文、计算设备信息和队列信息。不同的算子可以处在不同的队列中，各个队列可以并行执行，而队列内部顺序执行。更多详情，查看 :ref:`句柄`。

部分Cambricon CNNL算子计算时需要额外内存空间workspace用于算子优化。Cambricon CNNL内部不会申请workspace，用户需要调用worksapce相关接口申请计算所需要的workspace，并将workspace大小传入算子接口。详情请查看 :ref:`workspace`。

此外，由于CPU和MLU是异步执行任务，用户需要调用 ``cnrtQueueSync()`` 接口同步队列中执行的任务。

.. _单算子编程指南:

单算子编程指南
-------------------

如果运行单个Cambricon CNNL算子，执行下面操作：

1. 调用 ``cnrtGetDevice()`` 获取设备对应的设备号。
2. 调用 ``cnrtSetDevice()`` 绑定当前线程所使用的设备。
3. 调用 ``cnnlCreate()`` 创建一个Cambricon CNNL句柄。句柄将与当前线程所使用的设备绑定。
4. 调用 ``cnrtQueueCreate()`` 创建一个计算队列。
5. 调用 ``cnnlSetQueue()`` 将队列和Cambricon CNNL句柄绑定。
6. 如果接口需要设置 ``cnnlTensorDescriptor_t``，调用 ``cnnlCreateTensorDescriptor()`` 创建tensor descriptor，并调用 ``cnnlSetTensorDescriptor()`` 设置算子输入和输出描述信息。具体信息包括数据类型、形状信息、维度顺序等。
7. 如果接口需要设置算子描述符 ``cnnlXXXDescriptor_t``，调用 ``cnnlCreateXXXDescriptor()`` 创建算子描述符，并调用 ``cnnlSetXXXDescriptor()`` 为该算子添加描述。``XXX`` 需要替换为算子名称。
8. 如果接口需要申请额外workspace，即需要设置 ``workspace`` 和 ``workspace_size``，调用 ``cnnlXXXGetWorkspaceSize()`` 推导该算子需要的最小的临时空间大小。``XXX`` 需要替换为算子名称。
9. 调用 ``cnrtMalloc()`` 开辟算子需要的输入、输出和临时空间。
10. 调用 ``cnrtMemcpy()`` 将输入数据拷贝到设备端。拷贝到设备端的数据必须与之前设置的tensor descriptor信息保持一致。
11. 调用算子的API接口 ``cnnlXXX()``，传入Cambricon CNNL句柄信息。``XXX`` 需要替换为算子名称。
12. 调用 ``cnrtQueueSync()`` 同步CPU和MLU端。
13. 调用 ``cnrtMemcpy()`` 将输出从设备端拷贝回主机端。
14. 调用 ``cnrtFree()`` 释放设备端的空间。
15. 如果执行了第7步，调用 ``cnnlDestroyTensorDescriptor()`` 释放描述信息。
16. 如果执行了第8步，调用 ``cnnlDestroyXXXDescriptor()`` 释放资源。``XXX`` 需要替换为算子名称。
17. 调用 ``cnrtQueueDestroy()`` 释放队列信息。
18. 调用 ``cnnlDestroy()`` 释放Cambricon CNNL句柄。

有关接口详情，请查看《Cambricon CNNL Developer Guide》。

多算子搭建网络
----------------------------

使用Cambricon CNNL搭建一个多算子的网络，可以考虑如下方法，达到空间复用的效果。操作步骤如下：

1. 执行 单算子编程指南_ 的1-6步完成初始化等操作。
2. 为网络的每一层准备张量描述符 ``cnnlTensorDescriptor_t``，调用 ``cnnlCreateTensorDescriptor()`` 创建tensor descriptor，并调用 ``cnnlSetTensorDescriptor()`` 设置算子输入和输出描述信息。具体信息包括数据类型、形状信息、维度顺序等。
3. 为网络里面含有算子描述符的层创建算子 ``cnnlXXXDescriptor_t``，调用 ``cnnlCreateXXXDescriptor()`` 创建算子描述符，并调用 ``cnnlSetXXXDescriptor()`` 为该算子添加描述。其中 ``XXX`` 需要替换为算子名称。
4. 为网络里面所有需要的workspace的层设置 ``workspace`` 和 ``workspace_size``，调用 ``cnnlXXXGetWorkspaceSize()`` 推导该算子需要的最小的临时空间大小，其中 ``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtMemcpy()`` 将输入层的数据拷贝到设备端。
#. 按照网络层的顺序调用 ``cnnlXXX()``，每一层绑定上面申请的句柄，保证他们处于同一计算队列中。``XXX`` 需要替换为算子名称。
#. 调用 ``cnrtQueueSync()`` 同步CPU端和设备端。
#. 拷回数据、释放句柄和设备端空间。执行 单算子编程指南_ 的14-19步。

有关接口详情，请查看《Cambricon CNNL Developer Guide》。

