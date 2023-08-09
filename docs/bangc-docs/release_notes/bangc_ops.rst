模块概述
-------------------
Cambricon BANG C OPS是一个基于寒武纪MLU，并针对深度人工智能网络场景提供高速优化、常用算子的计算库。
同时也为用户提供简洁、高效、通用、灵活并且可扩展的编程接口。

Cambricon BANG C OPS具有以下特点：

  - 基于Cambricon BANG C语言（寒武纪针对MLU硬件开发的编程语言）实现算子开发。
  - 编译依赖寒武纪驱动应用程序接口CNDrv、寒武纪运行时库CNRT、寒武纪编译器CNCC和寒武纪汇编器CNAS。
  - 运行依赖寒武纪驱动应用程序接口CNDrv，寒武纪运行时库CNRT。


依赖版本说明
------------------

.. table:: 依赖版本说明
   :class: longtable
   :widths: 3 3

   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS 版本   | 依赖组件版本                |
   +=============================+=============================+
   | Cambricon BANG C OPS v0.8.z | CNToolkit >= v3.5.0         |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.7.z | CNToolkit >= v3.5.0         |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.6.z | CNToolkit >= v3.4.1         |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.5.z | CNToolkit >= v3.3.0         |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.4.z | CNToolkit = v3.0.2          |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.3.z | CNToolkit >= v3.1.2         |
   +-----------------------------+-----------------------------+
   | Cambricon BANG C OPS v0.2.z | CNToolkit >= v3.0.1         |
   +-----------------------------+-----------------------------+


支持平台说明
------------------

.. table:: 支持平台说明
   :class: longtable
   :widths: 3 3 3

   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS 版本   | 支持的CPU架构          | 支持的MLU架构                  |
   +=============================+========================+================================+
   | Cambricon BANG C OPS v0.8.z | x86_64                 | MLU370、MLU590                 |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.7.z | x86_64                 | MLU370、MLU590                 |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.6.z | x86_64                 | MLU370、MLU590                 |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.5.z | x86_64                 | MLU370、MLU590                 |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.4.z | x86_64                 | MLU290、MLU370                 |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.3.z | x86_64                 | MLU270、MLU290、MLU370         |
   |                             +------------------------+--------------------------------+
   |                             | AArch64                | MLU270、MLU290、MLU370         |
   +-----------------------------+------------------------+--------------------------------+
   | Cambricon BANG C OPS v0.2.z | x86_64                 | MLU270、MLU290、MLU370         |
   +-----------------------------+------------------------+--------------------------------+



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
