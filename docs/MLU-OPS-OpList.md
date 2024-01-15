# 算子

MLU-OPS 提供了检测、分割等任务中常用的算子，支持自定义算子（领域/长尾类算子）

根据算子在仓库中是否有源码实现分为 MLU Source Op 和 MLU Binary Op

- MLU Source Op: 源码算子，Host及Device侧代码在仓库有源码实现

- MLU Binary Op: 非源码算子，Host及Device侧代码在仓库无源码实现，通过调用底层CNNL算子库实现

**算子相关结构**

MLU Source Op算子结构：　
- 设计文档：`docs/design_docs/xxx/xxx.md`
- 实现：　
  - `device`实现：`kernels/xxx/`
  - `cpu`实现：`test/mlu_op_gtest/pb_gtest/src/zoo/xxx`
- 接口：`mlu_op.h`
- 算子间依赖：`kernel_depends.toml`

MLU Binary Op算子结构：　
- 设计文档：无
- 实现：
  - host调用CNNL算子库实现：`kernels/xxx/`
  - `cpu`实现：`test/mlu_op_gtest/pb_gtest/src/zoo/xxx`
- 接口：`mlu_op.h`
- 算子间依赖：`kernel_depends.toml`

**已支持算子**

| Device                                 | MLU Source Op | MLU Binary Op |
| -------------------------------------- | ------------- | ------------- |
| abs                                    | √             |               |
| active_rotated_filter                  | √             |               |
| add_n                                  |               | √             |
| ball_query                             | √             |               |
| bbox_overlaps                          | √             |               |
| border_align_backward                  | √             |               |
| border_align_forward                   | √             |               |
| box_iou_rotated                        | √             |               |
| carafe_backward                        | √             |               |
| carafe_forward                         | √             |               |
| copy                                   |               | √             |
| deform_roi_pool_backward               | √             |               |
| deform_roi_pool_forward                | √             |               |
| diff_iou_rotated_sort_vertices_forward | √             |               |
| div                                    | √             |               |
| dynamic_point_to_voxel_backward        | √             |               |
| dynamic_point_to_voxel_forward         | √             |               |
| expand                                 |               | √             |
| fill                                   |               | √             |
| focal_loss_sigmoid_backward            | √             |               |
| focal_loss_sigmoid_forward             | √             |               |
| gather_nd                              |               | √             |
| generate_proposals_v2                  | √             |               |
| get_indices_pairs                      | √             |               |
| indice_convolution_backward_data       | √             |               |
| indice_convolution_backward_filter     | √             |               |
| indice_convolution_forward             | √             |               |
| masked_col2im_forward                  | √             |               |
| masked_im2col_forward                  | √             |               |
| matmul                                 |               | √             |
| moe_dispatch_backward_data             | √             |               |
| moe_dispatch_backward_gate             | √             |               |
| moe_dispatch_forward                   | √             |               |
| ms_deform_attn_backward                | √             |               |
| ms_deform_attn_forward                 | √             |               |
| mutual_information_backward            | √             |               |
| mutual_information_backward            | √             |               |
| nms                                    |               | √             |
| nms_rotated                            | √             |               |
| points_in_boxes                        | √             |               |
| poly_nms                               | √             |               |
| prior_box                              | √             |               |
| psamask_backward                       | √             |               |
| psamask_forward                        | √             |               |
| psroipool_backward                     | √             |               |
| psroipool_forward                      | √             |               |
| reduce                                 |               | √             |
| roi_align_backward                     |               | √             |
| roi_align_rotated_backward             | √             |               |
| roi_align_rotated_forward              | √             |               |
| roi_crop_backward                      | √             |               |
| roi_crop_forward                       | √             |               |
| roi_pooling_forward                    |               | √             |
| roialign_forward                       |               | √             |
| roiaware_pool3d_backward               | √             |               |
| roiaware_pool3d_forward                | √             |               |
| roipoint_pool3d                        | √             |               |
| rotated_feature_align_backward         | √             |               |
| rotated_feature_align_forward          | √             |               |
| scatter_nd                             |               | √             |
| sqrt                                   | √             |               |
| three_interpolate_backward             | √             |               |
| three_interpolate_forward              | √             |               |
| three_nn_forward                       | √             |               |
| tin_shift_backward                     | √             |               |
| tin_shift_forward                      | √             |               |
| transpose                              |               | √             |
| unique                                 |               | √             |
| voxel_pooling_forward                  | √             |               |
| voxelization                           | √             |               |
| yolo_box                               | √             |               |
| dcn_backward_data                      |               | √             |