.. _概述:

概述
====

Cambricon MLU-OPS™ 是一个基于寒武纪MLU，并针对人工智能网络场景提供高速优化、常用算子的计算库。
同时也为用户提供简洁、高效、通用、灵活并且可扩展的编程接口。

Cambricon MLU-OPS™ 具有以下特点：

- 基于BANG C语言（寒武纪针对MLU硬件开发的编程语言）实现算子开发。
- 编译依赖寒武纪驱动应用程序接口CNDrv、寒武纪运行时库CNRT、寒武纪编译器CNCC和寒武纪汇编器CNAS。
- 运行依赖寒武纪驱动应用程序接口CNDrv，寒武纪运行时库CNRT。


- 支持丰富的算子。

  -  常见的网络算子：

     * Abs、Div、Log、Sqrt、SqrtBackward

  -  领域类算子：

     * ActiveRotatedFilterForward
     * BallQuery
     * BboxOverlaps
     * BoxIouRotated
     * CarafeBackward、CarafeForward
     * DeformRoiPoolBackward、DeformRoiPoolForward
     * GenerateProposalsV2
     * GetIndicePairs
     * IndiceConvolutionBackwardData、IndiceConvolutionBackwardFilter
     * IndiceConvolutionForward
     * MsDeformAttnBackward、MsDeformAttnForward
     * PointsInBoxes、PolyNms
     * PriorBox
     * RoiCropBackward、RoiCropForward
     * RoiAlignBackward、RoiAlignForward
     * RoiAlignRotatedBackward、RoiAlignRotatedForward
     * RoiawarePool3dBackward、RoiawarePool3dForward
     * RotatedFeatureAlignBackward、RotatedFeatureAlignForward
     * PsRoiPoolBackward、PsRoiPoolForward
     * PsaMaskBackward、PsaMaskForward
     * ThreeInterpolateBackward、ThreeInterpolateForward
     * ThreeNNForward
     * VoxelPoolingForward
     * Voxelization
     * YoloBox
     * Nms、NmsRotated


- 设计过程中充分考虑易用性，以通用为基本设计原则，算子支持不同的数据布局、灵活的维度限制以及多样的数据类型。
- 结合寒武纪的硬件架构特点，优化 Cambricon MLU-OPS™ 算子，使算子具有最佳性能，并且尽最大可能减少内存占用。
- 提供包含资源管理的接口，满足用户多线程、多板卡的应用场景。
