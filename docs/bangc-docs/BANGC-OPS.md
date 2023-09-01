# 算子

MLU-OPS 提供了检测、分割等任务中常用的算子，支持自定义算子（领域/长尾类算子）

- MLU Gtest: 算子的cpu实现，用作算子精度分析

- MLU Common: 常规算子，device端在仓库有源码实现

- MLU Bin: 二进制算子，device端在仓库无源码实现，但可通过头文件调用算子功能

| Device                                 | MLU Gtest | MLU Common | MLU Bin |
| -------------------------------------- | --------- | ---------- | ------- |
| abs                                    | √         | √          |         |
| active_rotated_filter                  | √         | √          | √       |
| add_n                                  | √         |            | √       |
| ball_query                             | √         | √          |         |
| bbox_overlaps                          | √         | √          |         |
| border_align_backward                  | √         | √          |         |
| border_align_forward                   | √         | √          |         |
| box_iou_rotated                        | √         | √          |        |
| carafe_backward                        | √         | √          |         |
| carafe_forward                         | √         | √          |         |
| copy                                   | √         |            | √       |
| deform_roi_pool_backward               | √         | √          |         |
| deform_roi_pool_forward                | √         | √          |         |
| diff_iou_rotated_sort_vertices_forward | √         | √          |         |
| div                                    | √         | √          |         |
| dynamic_point_to_voxel_backward        | √         | √          |         |
| dynamic_point_to_voxel_forward         | √         | √          |         |
| expand                                 | √         |            | √       |
| fill                                   | √         |            | √       |
| fiil_zero                              |           | √          |         |
| focal_loss_sigmoid_backward            | √         | √          |         |
| focal_loss_sigmoid_forward             | √         | √          |         |
| gather_nd                              | √         |            | √       |
| generate_proposals_v2                  | √         | √          |         |
| get_indices_pairs                      | √         | √          |         |
| indice_convolution_backward_data       | √         | √          |         |
| indice_convolution_backward_filter     | √         | √          |         |
| indice_convolution_forward             | √         | √          |         |
| masked_col2im_forward                  | √         | √          |         |
| masked_im2col_forward                  | √         | √          |         |
| matmul                                 | √         |            | √       |
| moe_dispatch_backward_data             | √         | √          |         |
| moe_dispatch_backward_gate             | √         | √          |         |
| moe_dispatch_forward                   | √         | √          |         |
| ms_deform_attn_backward                | √         | √          |         |
| ms_deform_attn_forward                 | √         | √          |         |
| mutual_information_backward            | √         | √          |         |
| mutual_information_backward            | √         | √          |         |
| nms                                    | √         |            | √       |
| nms_rotated                            | √         | √          |         |
| points_in_boxes                        | √         | √          |         |
| poly_nms                               | √         | √          |         |
| prior_box                              | √         | √          |         |
| psamask_backward                       | √         | √          |         |
| psamask_forward                        | √         | √          |         |
| psroipool_backward                     | √         | √          |         |
| psroipool_forward                      | √         | √          |         |
| reduce                                 | √         |            | √       |
| roi_align_backward                     | √         |            | √       |
| roi_align_rotated_backward             | √         | √          |         |
| roi_align_rotated_forward              | √         | √          |         |
| roi_crop_backward                      | √         | √          |         |
| roi_crop_forward                       | √         | √          |         |
| roi_pooling_forward                    | √         |            | √       |
| roialign_forward                       | √         |            | √       |
| roiaware_pool3d_backward               | √         | √          |         |
| roiaware_pool3d_forward                | √         | √          |         |
| roipoint_pool3d                        | √         | √          |         |
| rotated_feature_align_backward         | √         | √          |         |
| rotated_feature_align_forward          | √         | √          |         |
| scatter_nd                             | √         |            | √       |
| sqrt                                   | √         | √          |         |
| three_interpolate_backward             | √         | √          |         |
| three_interpolate_forward              | √         | √          |         |
| three_nn_forward                       | √         | √          |         |
| tin_shift_backward                     | √         | √          |         |
| tin_shift_forward                      | √         | √          |         |
| transpose                              | √         |            | √       |
| unique                                 | √         |            | √       |
| voxel_pooling_forward                  | √         | √          |         |
| voxelization                           | √         | √          |         |
| yolo_box                               | √         | √          |         |