编程模型介绍
=================

Cambricon BANGC OPS采用异构编程模型，实现不同架构和指令集的混合编程。

异构计算系统通常由通用处理器和协处理器组成，其中通用处理器作为控制设备，通常称为host端（主机端），负责调度。协处理器作为辅助计算设备，即MLU端（设备端），负责专有领域的大规模并行计算。Host端和MLU端协同完成计算任务。

Cambricon BANGC OPS异构编程模型是CPU和MLU的协作编程模型。host端负责调用CNRT接口用来初始化设备、管理设备内存、准备Cambricon BANGC OPS的参数、调用Cambricon BANGC OPS接口以及释放设备资源。MLU端作为协处理器，帮助host端CPU完成人工智能任务，并达到低能耗、高可用的效果。Cambricon BANGC OPS每个算子由Host端CPU发射，在MLU端异步执行。


