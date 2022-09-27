.. _FAQ:

FAQ
============

**问题1：Cambricon CNNL是什么？**
Cambricon CNNL（寒武纪人工智能计算库）是一个基于寒武纪MLU并针对人工智能网络的应用场景提供了高度优化常用算子的高性能计算库。具体可见 :ref:`概述` 。

**问题2：Cambricon CNNL在寒武纪软件栈中处于什么位置？**

参见 :ref:`CNNL在寒武纪软件栈中如何工作` 一节。

**问题3：Cambricon CNNL运行依赖哪些组件，版本号的对应关系是什么？**

Cambricon CNNL运行时依赖寒武纪CNToolkit，对应的版本依赖参见《Cambricon CNNL版本说明书》中“依赖版本说明”一章。对应的CNToolkit版本依赖的驱动版本参见《寒武纪CNToolkit安装升级使用手册》中“组件依赖关系”一节。

**问题4：Cambricon CNNL量化公式是什么？什么是量化参数？怎么求量化参数？**

量化参数通常指position、scale、offset参数，相关量化公式参见 :ref:`量化` 。

**问题5：Cambricon CNNL如何把量化过程放到算子内部处理？**

部分算子支持量化融合功能，这类算子支持浮点数据类型输入，并根据用户传入的量化参数，在内部完成浮点到定点的转数功能。支持的算子列表，参看 :ref:`支持量化融合卷积运算的CNNL算子接口`，具体使用使用说明，参看 :ref:`量化融合接口` 。

**问题6：Cambricon CNNL与Cambricon CNNL Extra是什么关系，版本之间有什么依赖？**

Cambricon CNNL Extra作为Cambricon CNNL的扩展库，提供了深度人工智能网络应用场景中的高度优化融合算子，同时也为用户开放了添加自定义算子的入口。Cambricon CNNL Extra依赖Cambricon CNNL，具体版本依赖关系，参见《寒武纪人工智能计算扩展库版本说明书》中的“依赖版本说明”一章。

**问题7：Cambricon CNNL与CNML是什么关系？同时使用时是否会有冲突？**

Cambricon CNNL与CNML的实现原理不同。CNML是JIT模式，基于固定网络结构和规模将整个编译得到高度融合的离线模型，适用于网络结构固定的并且要求极致性能的场景。Cambricon CNNL是AOT模式，指令以单算子为粒度存在于编译好的动态库中，支持规模可变，适用于灵活度高的应用场景。理论上在满足兼容性的硬件平台、驱动版本以及CNToolkit版本上，Cambricon CNNL和CNML可以同时使用。

**问题8：Cambricon CNNL是否可以生成离线模型？**

Cambricon CNNL是AOT模式，指令是提前生成并存放于动态库中，不支持也不需要生成离线模型。

**问题9：Cambricon CNNL算子实现的算法和接口与哪些框架保持一致？**

Cambricon CNNL算子开发参照Pytorch和TensorFlow框架算子算法实现，在功能和接口上尽量同时满足两个框架的需求。具体算子算法公式可以参看 :ref:`算子列表` 一章，具体的算子接口使用说明和算子限制，参看 cnnl.h 或《Cambricon CNNL Developer Guide》。

**问题10：当有新的算子需求而Cambricon CNNL中没有相应算子实现时，应如何处理？**

建议先分析是否可以由多个Cambricon CNNL算子拼接实现，如还不能满足要求，可以考虑使用Cambricon CNNL Extra开发自定义算子，或联系寒武纪AE工程师提出需求。

**问题11：运行过程中出现报错，例如出现cnnlError或MLU unfinished，要怎么处理？**

建议先根据报错信息分析是否设置了不支持的参数，若确认参数无问题，可以参考 :ref:`调试方法` 一章做进一步分析定位。

**问题12：Cambricon CNNL是否支持放置到队列中，用于计算硬件耗时的接口？**

可以使用 ``cnrtNotifier_t``，具体介绍可参考CNToolkit中的《寒武纪运行时库用户手册》中“Notifier”一节。

**问题13：部分算子在half数据类型下精度较差，是否有提高Cambricon CNNL算子精度的方法？**

部分算子支持设置更高位宽的计算数据类型以提高计算精度。例如卷积类算子可以将 ``cnnlConvolutionDescriptor_t`` 中的 ``compute_type`` 设置为float数据类型，激活类算子可以设置 ``cnnlComputationPreference_t`` 来选择高精度或高性能模式，``cnnlReduce`` 支持用户通过设置reduce_desc->tensor_type的数据类型来选择高精度或高性能模式。具体参见 cnnl.h 或《Cambricon CNNL Developer Guide》中各算子接口描述。


