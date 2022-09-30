.. _FAQ:

FAQ
============

**问题1：Cambricon BANGC OPS是什么？**
Cambricon BANGC OPS 是一个基于寒武纪MLU并针对人工智能网络的应用场景提供了高度优化常用算子的高性能计算库。具体可见 :ref:`概述` 。

**问题2：Cambricon BANGC OPS在寒武纪软件栈中处于什么位置？**

参见 :ref:`Cambricon BANGC OPS在寒武纪软件栈中如何工作` 一节。

**问题3：Cambricon BANGC OPS运行依赖哪些组件，版本号的对应关系是什么？**

Cambricon BANGC OPS运行时依赖寒武纪CNToolkit，对应的版本依赖参见《Cambricon BANGC OPS版本说明书》中“依赖版本说明”一章。对应的CNToolkit版本依赖的驱动版本参见《寒武纪CNToolkit安装升级使用手册》中“组件依赖关系”一节。

**问题4：Cambricon BANGC OPS与CNML是什么关系？同时使用时是否会有冲突？**

Cambricon BANGC OPS与CNML的实现原理不同。CNML是JIT模式，基于固定网络结构和规模将整个编译得到高度融合的离线模型，适用于网络结构固定的并且要求极致性能的场景。Cambricon BANGC OPS是AOT模式，指令以单算子为粒度存在于编译好的动态库中，支持规模可变，适用于灵活度高的应用场景。理论上在满足兼容性的硬件平台、驱动版本以及CNToolkit版本上，Cambricon BANGC OPS和CNML可以同时使用。

**问题5：Cambricon BANGC OPS是否可以生成离线模型？**

Cambricon BANGC OPS是AOT模式，指令是提前生成并存放于动态库中，不支持也不需要生成离线模型。

**问题6：当有新的算子需求而Cambricon BANGC OPS中没有相应算子实现时，应如何处理？**

建议先分析是否可以由多个Cambricon BANGC OPS算子拼接实现，如还不能满足要求，可以考虑使用Cambricon BANGC OPS Extra开发自定义算子，或联系寒武纪AE工程师提出需求。

**问题7：运行过程中出现报错，例如出现cnnlError或MLU unfinished，要怎么处理？**

建议先根据报错信息分析是否设置了不支持的参数，若确认参数无问题，可以参考 :ref:`调试方法` 一章做进一步分析定位。

**问题8：Cambricon BANGC OPS是否支持放置到队列中，用于计算硬件耗时的接口？**

可以使用 ``cnrtNotifier_t``，具体介绍可参考CNToolkit中的《寒武纪运行时库用户手册》中“Notifier”一节。

