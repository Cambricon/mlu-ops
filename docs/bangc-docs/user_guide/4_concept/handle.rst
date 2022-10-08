.. _句柄:

句柄
=================

MLU设备资源不能被直接使用，Cambricon BANGC OPS算子计算时通过句柄来维护MLU设备信息和队列信息等Cambricon BANGC OPS 库运行环境的上下文。因此，需要在使用 Cambricon BANGC OPS 库时，创建句柄并将句柄绑定到使用的MLU设备和队列上。

用户需要先调用 ``mluOpCreate()`` 接口在初始化 Cambricon BANGC OPS时创建一个句柄。创建的句柄随后会作为参数传递给算子相关接口完成数据计算。当Cambricon BANGC OPS库使用结束时，用户需要调用 ``mluOpDestroy()`` 接口释放与 Cambricon BANGC OPS 库有关的资源。

在同一线程中，``mluOpCreate()`` 和 ``mluOpDestroy()`` 接口之间的调用只会使用与句柄绑定的MLU设备。如果想要使用多个MLU设备，用户可以创建多个句柄来完成。通过使用句柄用户可以灵活地在多线程、多设备、多队列等多种场景下，完成相应的计算任务。例如，用户可以在多个线程中调用 ``cnrtSetDevice()`` 接口来设置使用不同的MLU设备，同时调用 ``mluOpCreate()`` 接口创建多个句柄绑定到不同的MLU设备上，再调用 Cambricon BANGC OPS算子计算接口传入句柄， 从而实现在不同线程中，算子的计算任务运行在不同的设备上。

绑定句柄和MLU设备
-------------------

绑定句柄和MLU设备是在句柄创建时完成。Cambricon BANGC OPS 将句柄创建时当前线程使用的设备绑定到句柄中。如果用户没有设置当前线程使用的MLU设备，则默认绑定MLU设备0。

创建句柄和绑定MLU设备的步骤如下：

1. 调用 ``cnrtGetDevice()`` 获取指定MLU设备对应的设备号。
#. 调用 ``cnrtSetDevice()`` 设置当前线程使用的MLU设备。
#. 调用 ``mluOpCreate()`` 创建句柄，将创建的句柄与当前线程所使用的设备绑定。

如果需要切换算子运行的MLU设备，可以重复调用以上步骤来重新设置设备并且创建一个新的句柄。相关CNRT接口说明，请参考《Cambricon CNRT Developer Guide》。

绑定句柄和队列
----------------

Cambricon BANGC OPS 中所有算子计算任务都是通过队列来完成。因此，句柄除了在创建时会绑定设备外，还需要绑定一个队列，使Cambricon BANGC OPS 算子的计算任务能够在指定队列上执行。一旦队列与句柄绑定，Cambricon BANGC OPS 算子的计算任务都是基于句柄中绑定的队列来下发。用户可以通过 ``mluOpSetQueue()`` 来绑定队列和句柄，通过 ``mluOpGetQueue()`` 来获取绑定到句柄上的队列。有关队列详情，请参考《寒武纪CNRT用户手册》。

执行下面步骤将句柄与队列绑定：

1. 调用 ``cnrtQueueCreate()`` 创建一个队列。

#. 调用 ``mluOpCreate()`` 创建一个句柄。

#. 将队列绑定到句柄。调用 ``mluOpSetQueue()`` 将创建好的队列绑定到已有的句柄中。此接口可用来切换句柄中绑定的队列。

#. 使用队列下发任务。 调用 ``mluOpXXX()`` ，将句柄作为参数传入该接口，其中XXX为算子名，如 ``mluOpAbs``。Cambricon BANGC OPS 在任务下发时，会将任务下发至句柄中绑定的队列上。任务下发完成后接口便会异步返回。

#. 调用 ``mluOpGetQueue()`` 获取句柄中已经绑定好的队列。

#. 调用 ``cnrtQueueSync()`` 同步队列。该接口会阻塞队列直到队列中所有任务均完成。

#. 调用 ``cnrtQueueDestroy()`` 销毁队列所占的资源。

此外，用户需要在 Cambricon BANGC OPS 程序运行最后调用 ``mluOpDestroy()`` 接口释放 Cambricon BANGC OPS 运行上下文资源。

句柄使用示例
-------------

下面以abs算子为例说明句柄的使用。示例中忽略abs运算中张量描述符及存放输入输出数据内存的创建和释放过程，仅展示与句柄相关的设备和队列的操作。

::

	int dev;
	cnrtGetDevice(&dev); // 获取设备0对应的设备号。
	cnrtSetDevice(dev); // 设置当前线程下需要使用的设备(设备0)。

	mluOpHandle_t handle;
	mluOpCreate(&handle); // 创建句柄, 此时创建的 handle 会与设备0绑定。

	cnrtQueue_t q0;
	cnrtQueueCreate(&q0); // 创建队列。
	mluOpSetQueue(handle, q0); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

	mluOpAbs(handle, input_desc, input_ptr, output_desc, output_ptr); // 传入已设置好的 handle，此时abs计算任务会在设备0上的 queue 0上运行。

	cnrtQueue_t q1;
	mluOpGetQueue(handle, &q1); // 如果上下文没有保存队列信息，可以通过 mluOpGetQueue 从 handle 中获取队列，此时获取的 q1 与 q0 相等。
	cnrtQueueSync(q1); // cnrtQueueSync(q0)
	
	mluOpDestroy(handle); // 释放 Cambricon BANGC OPS 运行上下文资源。
	cnrtQueueDestroy(q0); // 释放队列资源。
	cnrtQueueDestroy(q1); // 释放队列资源。


