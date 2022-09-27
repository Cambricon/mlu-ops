.. _extraInput:

额外输入
=========

Cambricon CNNL 库中的部分算子除了算子语义上的输入和输出外，还需要一个额外的输入内存空间（extraInput）用于算子的性能优化。extraInput 和正常的输入具有相同的特点。用户在使用带有 extraInput 参数的算子接口时，需要将MLU 设备端的 extraInput 指针传入算子接口中。Cambricon CNNL 会提供相关的接口和操作机制来对 extraInput 指向的内存进行赋值，用户只需要调用相关接口完成extraInput内存的数据赋值即可。

对于使用到 extraInput 的算子，需要在 CPU端和 MLU设备端分别申请一块 extraInput 内存空间。通过 ``cnnlGetXXXExtraInputSize()`` 接口获得，其中 ``XXX`` 为算子名。此外，还需通过 ``cnnlInitXXXExtraInput`` 接口对 CPU 端申请的 extraInput 内存进行数据初始化。然而  ``cnnlInitXXXExtraInput`` 接口的最终目的是将初始化的数据拷贝到 MLU设备端，供算子计算时使用这些数据进行计算优化。所以在完成 CPU端extraInput 内存的初始化之后，还需要调用 ``cnrtMemcpy`` 或其他相关的拷贝接口，将 CPU 侧 extraInput 内存上的数据拷贝到 MLU 侧的 extraInput 内存上，然后将 extraInput 的 MLU 指针传给算子接口。

该内存的生命周期和算子的普通输入保持一致。在每次算子接口调用前，都需要执行 MLU设备端的 extraInput 的内存申请。另外每次算子接口调用前，CPU端都需要执行 extraInput 内存申请、初始化以及向MLU 设备端的拷贝动作，因为该内存大小和实际数据与算子参数如 ``tensor_descriptor`` 相关，并且用户需要确保 CPU 的内存不要存在多线程踩踏行为。

extraInput 与 workspace 不同之处在于，用户无需关心 workspace 中的内存数据，也无需对该内存数据进行任何操作，只需要在 MLU 端申请合适大小的 workspace 内存并传给算子。而 extraInput 作为一个额外的 “输入”， 需要用户 分配和初始化CPU 端的内存以及从 CPU 拷贝数据到 MLU 端。

参考《Cambricon CNNL Developer Guide》中各算子的相关章节了解各算子是否需要额外申请 extraInput。如果算子接口中包含 ``extraInput`` 和 ``extraInput_size`` 参数，则说明该算子需要申请 extraInput，否则该算子不需要申请 extraInput。

使用方法
-------------------

执行下面步骤完成带有 extraInput 参数的算子计算：

1. 调用 ``cnnlCreateTensorDescriptor()`` 和 ``cnnlSetTensorDescriptor()`` 创建 tensor descriptor，并设置算子所需参数。

2. 调用 ``cnnlGetXXXExtraInputSize()`` 获取到该算子所需要的 extraInput 大小 ``extraInput_size``，其中 ``XXX`` 为算子名。这里获取到的 ``extraInput_size`` 是主机端和 MLU 端的内存大小。

#. 根据获取的 ``extraInput_size``，调用接口在 CPU 端申请 extraInput 内存 ``extraInput_cpu``，例如 ``malloc`` 。在 MLU 端调用接口申请 MLU 侧的 extraInput 内存 ``extraInput_mlu``，例如 ``cnrtMalloc`` 。

#. 调用 ``cnnlInitXXXExtraInput`` 接口，在 CPU 端初始化 ``extraInput_cpu`` 内存。

#. 调用如 ``cnrtMemcpy`` 或其他内存拷贝接口，将 CPU 端已经初始化完的内存拷贝到 MLU 端。也就是将 ``extraInput_cpu`` 中的数据拷贝到 ``extraInput_mlu``。

#. 调用 ``cnnlXXX()`` ，将tensor descriptor、输入、输出和 extraInput_MLU 等参数传入该接口来执行算子的计算任务，其中 ``XXX`` 为算子名。

#. 调用 ``cnnlDestroyTensorDescriptor()`` 释放tensor descriptor。

#. 释放其余申请的相关资源。

如果使用CNRT相关接口分配内存、初始化内存和拷贝内存数据，请参考《寒武纪CNRT开发者手册》。

示例
------------------

以 Reorg 算子为例展示 extraInput 的使用，示例中忽略Reorg tensor descriptor、数据内存申请和释放以及同步队列中执行任务的过程。

::

	size_t extraInput_size;
	cnnlGetReorgExtraInputSize(handle, reorg_desc, x_desc, y_desc, &extraInput_size); // 获取 Reorg 算子所需的 extraInput 大小。

	void *extraInput_cpu = NULL;
	void *extraInput_mlu = NULL;
	extraInput_cpu = malloc(extraInput_size); // 为 extraInput 申请 CPU 内存
	cnrtMalloc(&extraInput_mlu, extraInput_size); // 为 extraInput 申请 MLU 内存

	cnnlInitReorgExtraInput(handle, reorg_desc, x_desc, y_desc, extraInput_cpu); // 在 CPU 端初始化 extraInput 内存
	cnrtMemcpy(extraInput_MLU, extraInput_cpu, extraInput_size, CNRT_MEM_TRANS_DIR_HOST2DEV); // 将 CPU 端已初始化的内存数据拷贝至 MLU 端
	cnnlReorg_v2(handle, reorg_desc, x_desc, x, extraInput_mlu, extraInput_size, y_desc, y, workspace, workspace_size);  // 调用算子接口，传入 MLU 端 extraInput 指针及其他相关参数

	// 释放相关资源
	free(extraInput_cpu);
	cnrtFree(extraInput_mlu); 

