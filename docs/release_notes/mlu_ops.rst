模块概述
-------------------
Cambricon MLU-OPS是一个基于寒武纪MLU，并针对深度人工智能网络场景提供高速优化、常用算子的计算库。
同时也为用户提供简洁、高效、通用、灵活并且可扩展的编程接口。

Cambricon MLU-OPS具有以下特点：

  - 基于Cambricon BANG C语言（寒武纪针对MLU硬件开发的编程语言）实现算子开发。
  - 编译依赖寒武纪驱动应用程序接口CNDrv、寒武纪运行时库CNRT、寒武纪编译器CNCC和寒武纪汇编器CNAS。
  - 运行依赖寒武纪驱动应用程序接口CNDrv，寒武纪运行时库CNRT。


依赖版本说明
------------------

.. table:: 依赖版本说明
   :class: longtable
   :widths: 3 3

   +-----------------------------+-----------------------------+
   | Cambricon MLU-OPS 版本      | 依赖组件版本                |
   +=============================+=============================+
   | Cambricon MLU-OPS v1.2.z    | CNToolkit >= v3.8.4         |
   |                             +-----------------------------+
   |                             | CNNL >= v1.23.2             |
   +-----------------------------+-----------------------------+
   | Cambricon MLU-OPS v1.1.z    | CNToolkit >= v3.8.4         |
   |                             +-----------------------------+
   |                             | CNNL >= v1.23.2             |
   +-----------------------------+-----------------------------+
   | Cambricon MLU-OPS v1.0.z    | CNToolkit >= v3.8.4         |
   |                             +-----------------------------+
   |                             | CNNL >= v1.23.2             |
   +-----------------------------+-----------------------------+


支持平台说明
------------------

.. table:: 支持平台说明
   :class: longtable
   :widths: 3 3 3

   +-----------------------------+------------------------+--------------------------------+
   | Cambricon MLU-OPS 版本      | 支持的CPU架构          | 支持的MLU架构                  |
   +=============================+========================+================================+
   | Cambricon MLU-OPS v1.2.z    | x86_64                 | MLU370                         |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon MLU-OPS v1.1.z    | x86_64                 | MLU370                         |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon MLU-OPS v1.0.z    | x86_64                 | MLU370                         |
   +-----------------------------+------------------------+--------------------------------+

v1.2.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 无新增特性。

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下问题：

   * 修复算子 mluOpGenerateProposalsV2 在 nan/inf 场景下的功能问题。
   * 修复算子 mluOpDeformRoiPoolBackward、mluOpRoiAlignRotatedForward、mluOpRoiAlignRotatedBackward 理论计算量不准确的问题。
   * 修复算子性能分析工具的代码问题。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- 无。


v1.1.1
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 无新增特性。

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下问题：

   * 修复性能分析工具处理同名测试用例时引入的功能问题。
   * 修复算子 mluOpAdamW 未分配任务类型引入的算子功能问题。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- 无。


v1.1.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增以下算子接口：

   * ``adam_w``

     + mluOpAdamW
     + mluOpCreateAdamWDescriptor
     + mluOpSetAdamWDescAttr
     + mluOpDestroyAdamWDescriptor

   * ``exec_fft``

     + mluOpExecFFT
     + mluOpCreateFFTPlan
     + mluOpDestroyFFTPlan
     + mluOpSetFFTReserveArea
     + mluOpMakeFFTPlanMany


v1.0.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增以下算子接口：

   * ``dcn``

     + mluOpDCNForward

     + mluOpDCNBackwardWeight

     + mluOpDCNBackwardData

     + mluOpCreateDCNDescriptor

     + mluOpDestroyDCNDescriptor

     + mluOpSetDCNDescriptor

     + mluOpGetDCNBakcwardDataWorkspaceSize

     + mluOpGetDCNForwardWorkspaceSize

     + mluOpGetDCNBackwardWeightWorkspaceSize

- 经过一整个大版本的废弃声明，移除以下算子接口，如需使用功能，请调用CNNL对应接口：

   * ``add_n``

     + mluOpAddN

     + mluOpGetAddNWorkspaceSize

     + mluOpAddN_v2

   * ``batch_matmul_bcast``

     + mluOpGetBatchMatMulBCastWorkspaceSize

     + mluOpGetBatchMatMulHeuristicResult

     + mluOpGetBatchMatMulAlgoHeuristic

     + mluOpBatchMatMulBCastDescCreate

     + mluOpBatchMatMulBCastDescDestroy

     + mluOpSetBatchMatMulBCastDescAttr

     + mluOpGetBatchMatMulBCastDescAttr

     + mluOpBatchMatMulBCastAlgoCreate

     + mluOpBatchMatMulBCastAlgoDestroy

     + mluOpGetQuantizeBatchMatMulBCastAlgorithm

     + mluOpGetQuantizeBatchMatMulBCastWorkspaceSize

     + mluOpQuantizeBatchMatMulBCast

     + mluOpBatchMatMulBCast

     + mluOpBatchMatMulBCast_v2

   * ``copy``

     + mluOpCopy

   * ``concat``

     + mluOpConcat

     + mluOpGetConcatWorkspaceSize

   * ``expand``

     + mluOpExpand

   * ``fill``

     + mluOpFill

     + mluOpFill_v3

   * ``gather_nd``

     + mluOpGatherNd

   * ``matmul``

     + mluOpMatMul

     + mluOpMatMulDescCreate

     + mluOpMatMulDescDestroy

     + mluOpSetMatMulDescAttr

     + mluOpGetMatMulDescAttr

     + mluOpCreateMatMulHeuristicResult

     + mluOpDestroyMatMulHeuristicResult

     + mluOpGetMatMulHeuristicResult

     + mluOpGetMatMulAlgoHeuristic

     + mluOpMatMulAlgoCreate

     + mluOpMatMulAlgoDestroy

     + mluOpGetMatMulWorkspaceSize

     + mluOpMatMul_v2

   * ``nms``

     + mluOpNms

   * ``pad``

     + mluOpPad

   * ``reduce``

     + mluOpReduce

     + mluOpCreateReduceDescriptor

     + mluOpDestroyReduceDescriptor

     + mluOpSetReduceDescriptor

     + mluOpSetReduceDescriptor_v2

     + mluOpGetReduceOpWorkspaceSize

   * ``scatter_nd``

     + mluOpScatterNd

     + mluOpScatterNd_v2

   * ``stride_slice``

     + mluOpStrideSlice

   * ``transform``

     + mluOpTransform

   * ``transpose``

     + mluOpCreateTransposeDescriptor

     + mluOpDestroyTransposeDescriptor

     + mluOpSetTransposeDescriptor

     + mluOpGetTransposeWorkspaceSize

     + mluOpTranspose

     + mluOpTranspose_v2

   * ``unique``

     + mluOpUnique

     + mluOpCreateUniqueDescriptor

     + mluOpDestroyUniqueDescriptor

     + mluOpSetUniqueDescriptor

     + mluOpGetUniqueWorkSpace

     + mluOpUniqueGetOutLen

     + mluOpGetUniqueWorkspaceSize

     + mluOpUnique_v2

- 新增编译前对环境中各个依赖项的版本检查。

- 更新公共组件core/GTest代码。

- 更新MLU-OPS仓库中对环境安装、编译、测试流程的叙述。

- 移除对Ubuntu18.04系统的支持。

- 移除BangPy组件，调整MLU-OPS仓库代码结构。

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下算子问题：

   * ``voxel_pooling_forward``

     + 移除GTest中额外调用的API接口。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- ``roi_align_rotated``

   * mluOpRoiAlignRotatedForward接口在输入feature以及rois元素数量接近2G时出现运行超时。

   * mluOpRoiAlignRotatedBackward接口在输入top_grad以及rois元素数量接近2G时出现运行超时。

- ``carafe``

   * mluOpCarafeForward接口在输入input以及mask元素数量超过2G时出现运行错误。

   * mluOpCarafeBackward接口在输入input、mask以及grad_output元素数量接近2G时出现运行超时。


v0.11.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增底层依赖 CNNL。

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下算子问题：

   * 修复算子 ``yolo_box`` 防呆不完整问题。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- 无。


v0.10.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增以下算子：

   * ``pad``

   * ``concat``

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下算子问题：

   * 修复算子 points_in_boxes 防呆缺失问题。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- 无


v0.9.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增以下算子：

   * ``transform``

   * ``strided_slice``

   * ``sync_batchnorm_stats``

   * ``sync_batchnorm_gather_stats_with_counts``

   * ``sync_batchnorm_elemt``

   * ``sync_batchnorm_backward_reduce``

   * ``sync_batch_norm_backward_elemt``

已修复问题
~~~~~~~~~~~~~~~~~~~~~

- 修复以下算子问题：

   * 修复算子 roiaware_pool3d_forward 文档中公式书写错误、防呆缺失等问题。

   * 修复算子 ms_deform_attn_forward 由拆分错误引入的精度问题。

   * 修复算子 voxel_pooling_forward 由地址越界引入的精度问题。

   * 修复算子 nms_rotated 引入的编译 warnings 问题。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

- 无


v0.8.1
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 无新增特性。

已修复问题
~~~~~~~~~~~~~~~~~~~~~

修复 v0.8.0 中潜在的二进制算子缺陷。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

无。


v0.8.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~

- 新增支持以下算子：

   * ``border_align_backward``

   * ``border_align_forward``

   * ``masked_col2im_forward``

   * ``masked_im2col_forward``

   * ``tin_shift_backward``

   * ``tin_shift_forward``

已修复问题
~~~~~~~~~~~~~~~~~~~~

- 修复以下算子问题：

   * 修复dynamic_point_to_voxel_backward在编译时设置memcheck选项暴露的内存越界问题。

   * 修复roi_crop_forward/backward在mlu_op.h中错误的返回值描述。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~

无。

v0.7.1
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 无新增特性。

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

修复 v0.7.0 中潜在的编译缺陷。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。

v0.7.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 适配 x86_64 架构的 KylinV10 系统编译及测试。
- 新增支持以下算子：

   * ``dynamic_point_to_voxel_backward``

   * ``dynamic_point_to_voxel_forward``

   * ``focal_loss_sigmoid_backward``

   * ``focal_loss_sigmoid_forward``

   * ``mutual_information_backward``

   * ``mutual_information_forward``

v0.6.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 不再支持Debian。
- 新增支持以下算子：

   * ``ms_deform_attn_backward``

   * ``ms_deform_attn_forward``

   * ``nms``

   * ``points_in_boxes``

   * ``roi_align_backward``

   * ``roi_align_forward``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.5.1
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 新增支持以下算子：

   * ``nms_rotated``

   * ``moe_dispatch_backward_data``

   * ``moe_dispatch_backward_gate``

   * ``moe_dispatch_forward``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

- 修复了nms_rotated未对large tensor(2GB)防呆导致的计算错误。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.5.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

-  不再支持MLU290。
-  新增支持以下算子：

   * ``active_rotated_filter_forward``

   * ``add_n``

   * ``bbox_overlaps``

   * ``box_iou_rotated``

   * ``carafe_backward``

   * ``carafe_forward``

   * ``deform_roi_pool_backward``

   * ``deform_roi_pool_forward``

   * ``gather_nd``

   * ``get_indice_pairs``

   * ``indice_convolution_backward_data``

   * ``indice_convolution_backward_filter``

   * ``indice_convolution_forward``

   * ``mat_mul``

   * ``reduce``

   * ``roi_align_rotated_backward``

   * ``roi_align_rotated_forward``

   * ``roiaware_pool3d_backward``

   * ``roiaware_pool3d_forward``

   * ``rotated_feature_align_backward``

   * ``rotated_feature_align_forward``

   * ``scatter_nd``

   * ``three_interpolate_backward``

   * ``three_nn_forward``

   * ``transpose``

   * ``unique``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.4.2
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

-  新增支持以下算子：

   * ``box_iou_rotated``

   * ``nms_rotated``


已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.4.1
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

-  不再支持Ubuntu16.04。
-  不再支持AArch64。

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.4.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

-  编译支持板卡、算子可选。
-  支持MLU算子性能比对功能。
-  新增支持以下算子：

   * ``voxel_pooling_forward``

   * ``voxelization``

   * ``psa_mask_forward``

   * ``psa_mask_backward``

   * ``fill``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.3.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 适配 AArch64 架构的 KylinV10 系统编译及测试。
- 新增支持以下算子：

  * ``three_interpolate_forward``

  * ``ball_qeury``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。


v0.2.0
-----------------

特性变更
~~~~~~~~~~~~~~~~~~~~~~

- 新增以下算子：

  * ``yolo_box``

  * ``generate_proposals_v2``

  * ``prior_box``

已修复问题
~~~~~~~~~~~~~~~~~~~~~~

无。

已知遗留问题
~~~~~~~~~~~~~~~~~~~~~~

无。
