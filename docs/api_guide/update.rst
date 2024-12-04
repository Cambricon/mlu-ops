Update History
===============

This section lists contents that were made for each product release.

* V1.4.0

  **Date:** November 29, 2024

  **Changes:**

  - Added the following new operations:

      - mluOpLogspace
      - mluOpLgamma

* V1.3.2

  **Date:** October 21, 2024

  **Changes:**

  - ``exec_fft``

    - optimize performance.


* V1.3.1

  **Date:** October 10, 2024

  **Changes:**

  - ``exec_fft``

    - adjust memory-free strategy to avoid duplicated memory-free operation.


* V1.3.0

  **Date:** September 6, 2024

  **Changes:**

  - ``exec_fft``

    - support 2D mode for FFT computation.
    - support tensor stride for 1D FFT computation.
    - optimize performance.
    - update called CNNL APIs version.

  - ``adam_w``

    - add fool-proofing for MLU300 series.

  - Other operations.

    - add fool-proofing if it does not support tensor stride.
    - some bug fixes.

  - Kernel utils

    - support philox random algorithm.


* V1.2.4

  **Date:** August 15, 2024

  **Changes:**

  - None.


* V1.2.3

  **Date:** July 25, 2024

  **Changes:**

  - None.


* V1.2.2

  **Date:** July 25, 2024

  **Changes:**

  - None.


* V1.2.1

  **Date:** June 28, 2024

  **Changes:**

  - None.


* V1.2.0

  **Date:** May 27, 2024

  **Changes:**

  - None.


* V1.1.1

  **Date:** April 12, 2024

  **Changes:**

  - None.


* V1.1.0

  **Date:** March 28, 2024

  **Changes:**

  - Added the following new operations:

    - ``adam_w``

      - mluOpAdamW
      - mluOpCreateAdamWDescriptor
      - mluOpSetAdamWDescAttr
      - mluOpDestroyAdamWDescriptor

    - ``exec_fft``

      - mluOpExecFFT
      - mluOpCreateFFTPlan
      - mluOpDestroyFFTPlan
      - mluOpSetFFTReserveArea
      - mluOpMakeFFTPlanMany


* V1.0.0

  **Date:** February 6, 2024

  **Changes:**

  - Added the following new operations:

    - ``dcn``

      - mluOpDCNForward
      - mluOpDCNBackwardWeight
      - mluOpDCNBackwardData
      - mluOpCreateDCNDescriptor
      - mluOpDestroyDCNDescriptor
      - mluOpSetDCNDescriptor
      - mluOpGetDCNBakcwardDataWorkspaceSize
      - mluOpGetDCNForwardWorkspaceSize
      - mluOpGetDCNBackwardWeightWorkspaceSize

  - Removed the following operations:

    - ``add_n``

      - mluOpAddN
      - mluOpGetAddNWorkspaceSize
      - mluOpAddN_v2

    - ``batch_matmul_bcast``

      - mluOpGetBatchMatMulBCastWorkspaceSize
      - mluOpGetBatchMatMulHeuristicResult
      - mluOpGetBatchMatMulAlgoHeuristic
      - mluOpBatchMatMulBCastDescCreate
      - mluOpBatchMatMulBCastDescDestroy
      - mluOpSetBatchMatMulBCastDescAttr
      - mluOpGetBatchMatMulBCastDescAttr
      - mluOpBatchMatMulBCastAlgoCreate
      - mluOpBatchMatMulBCastAlgoDestroy
      - mluOpGetQuantizeBatchMatMulBCastAlgorithm
      - mluOpGetQuantizeBatchMatMulBCastWorkspaceSize
      - mluOpQuantizeBatchMatMulBCast
      - mluOpBatchMatMulBCast
      - mluOpBatchMatMulBCast_v2

    - ``copy``

      - mluOpCopy

    - ``concat``

      - mluOpConcat
      - mluOpGetConcatWorkspaceSize

    - ``expand``

      - mluOpExpand 

    - ``fill``

      - mluOpFill
      - mluOpFill_v3

    - ``gather_nd``

      - mluOpGatherNd

    - ``matmul``

      - mluOpMatMul
      - mluOpMatMulDescCreate
      - mluOpMatMulDescDestroy
      - mluOpSetMatMulDescAttr
      - mluOpGetMatMulDescAttr
      - mluOpCreateMatMulHeuristicResult
      - mluOpDestroyMatMulHeuristicResult
      - mluOpGetMatMulHeuristicResult
      - mluOpGetMatMulAlgoHeuristic
      - mluOpMatMulAlgoCreate
      - mluOpMatMulAlgoDestroy
      - mluOpGetMatMulWorkspaceSize
      - mluOpMatMul_v2

    - ``nms``

      - mluOpNms

    - ``pad``

      - mluOpPad

    - ``reduce``

      - mluOpReduce
      - mluOpCreateReduceDescriptor
      - mluOpDestroyReduceDescriptor
      - mluOpSetReduceDescriptor
      - mluOpSetReduceDescriptor_v2
      - mluOpGetReduceOpWorkspaceSize

    - ``scatter_nd``

      - mluOpScatterNd
      - mluOpScatterNd_v2

    - ``stride_slice``

      - mluOpStrideSlice

    - ``transform``

      - mluOpTransform

    - ``transpose``

      - mluOpCreateTransposeDescriptor
      - mluOpDestroyTransposeDescriptor
      - mluOpSetTransposeDescriptor
      - mluOpGetTransposeWorkspaceSize
      - mluOpTranspose
      - mluOpTranspose_v2

    - ``unique``

      - mluOpUnique
      - mluOpCreateUniqueDescriptor
      - mluOpDestroyUniqueDescriptor
      - mluOpSetUniqueDescriptor
      - mluOpGetUniqueWorkSpace
      - mluOpUniqueGetOutLen
      - mluOpGetUniqueWorkspaceSize
      - mluOpUnique_v2

  - Removed BangPy APIs.


* V0.11.0

  **Date:** December 15, 2023

  **Changes:**

  - None.

* V0.10.0

  **Date:** November 24, 2023

  **Changes:**

  - Added the following new operations:

    - pad
    - concat

* V0.9.0

  **Date:** October 16, 2023

  **Changes:**

  - Added the following new operations:

    - transform
    - strided_slice
    - sync_batchnorm_stats
    - sync_batchnorm_gather_stats_with_counts
    - sync_batchnorm_elemt
    - sync_batchnorm_backward_reduce
    - sync_batch_norm_backward_elemt

* V0.8.1

  **Date:** August 31, 2023

  **Changes:**

  - None.

* V0.8.0

  **Date:** August 9, 2023

  **Changes:**

  - Added the following new operations:

    - border_align_backward
    - border_align_forward
    - masked_col2im_forward
    - masked_im2col_forward
    - tin_shift_backward
    - tin_shift_forward

* V0.7.1

  **Date:** June 16, 2023

  **Changes:**

  - None.

* V0.7.0

  **Date:** June 2, 2023

  **Changes:**

  - Added the following new operations:

    - dynamic_point_to_voxel_backward
    - dynamic_point_to_voxel_forward
    - focal_loss_sigmoid_backward
    - focal_loss_sigmoid_forward
    - mutual_information_backward
    - mutual_information_forward

* V0.6.0

  **Date:** April 14, 2023

  **Changes:**

  - Added the following new operations:

    - ms_deform_attn_backward
    - ms_deform_attn_forward
    - nms
    - points_in_boxes
    - roi_align_backward
    - roi_align_forward

* V0.5.1

  **Date:** March 20, 2023

  **Changes:**

  - Added the following new operations:

    - nms_rotated
    - moe_dispatch_backward_data
    - moe_dispatch_backward_gate
    - moe_dispatch_forward

* V0.5.0

  **Date:** February 20, 2023

  **Changes:**

  - Added the following new operations:

    - active_rotated_filter_forward
    - add_n
    - bbox_overlaps
    - box_iou_rotated
    - carafe_backward
    - carafe_forward
    - deform_roi_pool_backward
    - deform_roi_pool_forward
    - gather_nd
    - get_indice_pairs
    - indice_convolution_backward_data
    - indice_convolution_backward_filter
    - indice_convolution_forward
    - mat_mul
    - reduce
    - roi_align_rotated_backward
    - roi_align_rotated_forward
    - roiaware_pool3d_backward
    - roiaware_pool3d_forward
    - rotated_feature_align_backward
    - rotated_feature_align_forward
    - scatter_nd
    - three_interpolate_backward
    - three_nn_forward
    - transpose
    - unique

* V0.4.2

  **Date:** March 6, 2023

  **Changes:**

  - Added the following new operations:

    - box_iou_rotated
    - nms_rotated

* V0.4.1

  **Date:** December 20, 2022

  **Changes:**

  - None.

* V0.4.0

  **Date:** December 12, 2022

  **Changes:**

  - Added the following new operations:

    - voxel_pooling_forward
    - voxelization
    - psa_mask_forward
    - psa_mask_backward
    - fill

* V0.3.0

  **Date:** October 20, 2022

  **Changes:**

  - Added the following new operations:

    - three_interpolate_forward
    - ball_query

* V0.2.0

  **Date:** September 20, 2022

  **Changes:**

  - Added the following new operations:

    - yolo_box
    - generate_proposals_v2
    - prior_box

* V0.1.0

  **Date:** August 13, 2022

  **Changes:**

  - Initial release.
