.. _workspace:

工作空间
=========

Cambricon BANGC OPS库中部分算子除了输入和输出数据的存放需要内存空间外，还需要额外的内存空间用于算子计算的优化。这部分额外的内存空间被称为工作空间。

需要工作空间的算子在调用算子接口前，要申请工作空间。Cambricon BANGC OPS为每个需要使用工作空间的算子提供了一个获取工作空间大小的接口，用户通过调用该接口获取算子需要使用的工作空间大小（单位为字节）。接口名为 ``mluOpGetXXXWorkspaceSize()``，其中 ``XXX`` 为算子名。例如，不规则四边形非极大值抑制算子（mluOpPolyNms）获取工作空间大小的接口为 ``mluOpGetPolyNmsWorkspaceSize()`` 将申请的工作空间地址和工作空间大小传给算子计算接口的 ``workspace`` 和 ``workspace_size`` 参数。最后，用户需要调用 ``cnrtFree()`` 释放工作空间。

参考《Cambricon BANGC OPS Developer Guide》中各算子的相关章节了解各算子是否需要额外申请工作空间。如果算子接口中包含 ``workspace`` 和 ``workspace_size`` 参数，则说明该算子需要申请工作空间，否则该算子不需要申请工作空间。

如果网络中有多个算子需要工作空间，用户可以通过调用 ``mluOpGetXXXWorkSpaceSize()`` 分别获取各算子所需的工作空间大小，然后只申请这些算子所需最大工作空间大小，而无需为每个算子都申请工作空间。在这种情况下，workspace_size 并不是确切的工作空间指针指向的内存块的大小，而是指算子需要的工作空间的大小。该方法可以充分复用MLU设备上的内存空间，并且减少多个算子因多次申请和释放工作空间所带来的时间开销。但是为了确保工作空间读写安全性，该方法只限于所有算子严格按照顺序执行，并且这些算子的执行在同一个队列内，而不是在多个队列中并行运行。顺序执行可以确保在同一时间仅有一个任务访问这一块工作空间，而多队列并行运行时可能会造成多个并行任务同时访问这一块工作空间，从而造成多个任务互相踩踏内存，并且CNRT会返回 "mlu unfinished error"或算子计算结果出错。

使用方法
-------------------

执行下面步骤，为一个算子申请工作空间：

1. 调用 ``mluOpCreateTensorDescriptor()`` 和 ``mluOpSetTensorDescriptor()`` 创建张量描述符，并设置算子所需参数。

#. 调用 ``mluOpGetXXXWorkspaceSize()`` 获取到该算子所需要的工作空间大小，其中 ``XXX`` 需替换为算子名。

#. 根据获取的工作空间大小设置 ``size``，调用 ``cnrtMalloc()`` 申请工作空间空间。

#. 调用 ``mluOpXXX()`` ，将张量描述符、输入、输出和工作空间等参数传入该接口来执行算子的计算任务，其中 ``XXX`` 需替换为算子名。

#. 调用 ``cnrtMemcpy()`` 将 MLU 输出结果从设备端拷贝回主机端。

#. 调用 ``mluOpDestroyTensorDescriptor()`` 释放张量描述符。

#. 调用 ``cnrtFree()`` 释放工作空间空间。

相关CNRT接口详情，请参考《Cambricon CNRT Developer Guide》。

示例
------------------

以不规则四边形非极大值抑制（mluOpPolyNms）算子为例展示工作空间的使用，示例中忽略张量描述符、数据内存申请和释放以及同步队列中执行任务的过程。

::

   size_t workspace_size; 

   // 获取mluOpPolyNms算子所需工作空间大小。
   mluOpGetPolyNmsWorkspaceSize(handle, input_tensor_desc, &workspace_size);
   
   void *workspace = NULL;

   // 为 workspace 分配内存。
   cnrtMalloc(&workspace, workspace_size);
   
   // 完成 mluOpPolyNms 计算任务，其中workspace_ptr为工作空间地址，workspace_size为工作空间大小。
   mluOpPolyNms(handle, input_desc, input_ptr, iou_threshold, workspace_ptr, workspace_size, output_desc, output_ptr, output_size_ptr);
   
   // 释放工作空间内存资源。
   cnrtFree(workspace);

