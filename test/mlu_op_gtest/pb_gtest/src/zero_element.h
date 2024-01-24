/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLU_OP_GTEST_SRC_ZERO_ELEMENT_H_
#define TEST_MLU_OP_GTEST_SRC_ZERO_ELEMENT_H_

#include <mlu_op.h>
#include <string>
#include <vector>
std::vector<std::string> white_list = {"advanced_index",
                                       "arange",
                                       "batch_matmul",
                                       "batchnorm_forward",
                                       "carafe_backward",
                                       "carafe_forward",
                                       "conv3dbpfilter",
                                       "Convolution_Backward_Data",
                                       "convbpfilter",
                                       "crop_and_resize",
                                       "crop_and_resize_backward_image",
                                       "ctc_loss",
                                       "dcn_backward_data",
                                       "dcn_backward_weight",
                                       "dcn_forward",
                                       "deconvolution",
                                       "depthwise_conv_backprop_filter",
                                       "dynamic_stitch",
                                       "Fused_Ops",
                                       "gather_nd",
                                       "group_conv_backprop_filter",
                                       "layernorm_forward",
                                       "list_diff",
                                       "matmul_inference",
                                       "multihead_attn",
                                       "psamask_backward",
                                       "psamask_forward",
                                       "quantize_matmul",
                                       "random_normal",
                                       "random_truncated_normal",
                                       "random_uniform",
                                       "random_uniform_int",
                                       "reduce",
                                       "roi_align_backward",
                                       "roi_align_forward",
                                       "roi_pooling_backward",
                                       "roi_pooling_forward",
                                       "select",
                                       "shufflechannel",
                                       "sync_batch_norm_backward_elemt",
                                       "tin_shift",
                                       "tin_shift_backward",
                                       "topk",
                                       "unique",
                                       "weightnorm_backward"};

#endif  // TEST_MLU_OP_GTEST_SRC_ZERO_ELEMENT_H_
