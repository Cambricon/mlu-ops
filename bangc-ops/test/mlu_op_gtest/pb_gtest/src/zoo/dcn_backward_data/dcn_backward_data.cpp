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

#include <string>
#include "dcn_backward_data.h"
#define USE_OPENBLAS 0

#if USE_OPENBLAS
#include <openblas/cblas.h>
#endif

namespace mluoptest {

static inline bool isFixData(mluOpDataType_t type) {
  if (MLUOP_DTYPE_INT8 == type || MLUOP_DTYPE_INT16 == type ||
      MLUOP_DTYPE_INT31 == type) {
    return true;
  }
  return false;
}

void DcnBackwardDataExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_dcn_param()) {
    LOG(ERROR) << "Missing dcn param. ";
  }

  int input_tensor_number = parser_->inputs().size();
  int output_tensor_number = parser_->outputs().size();
  if (!((input_tensor_number == 5 && output_tensor_number == 3) ||
        (input_tensor_number == 4 && output_tensor_number == 2))) {
    // if use mask, mask and grad_mask is valid, input number = 5 and
    // output_number =3. otherwise, mask and grad_mask should be null
    LOG(ERROR)
        << "The number of input tensors and output tensors is mismatched.";
  }
  use_mask_ = input_tensor_number == 5;
  auto dcn_desc_node = parser_->getProtoNode()->dcn_param();
  dimNb_ = dcn_desc_node.dimnb();

  for (int pad_iter = 0; pad_iter < dcn_desc_node.pad_size(); pad_iter++) {
    pad_[pad_iter] = dcn_desc_node.pad(pad_iter);
  }
  for (int iter = 0; iter < dcn_desc_node.stride_size(); iter++) {
    stride_[iter] = dcn_desc_node.stride(iter);
  }
  for (int iter = 0; iter < dcn_desc_node.dilation_size(); iter++) {
    dilation_[iter] = dcn_desc_node.dilation(iter);
  }
  conv_group_ = dcn_desc_node.conv_group();
  deformable_group_ = dcn_desc_node.deformable_group();
  im2col_step_ = dcn_desc_node.im2col_step();
  compute_type_ = cvtProtoDtypeToMluOp(dcn_desc_node.compute_type());

  mluOpDataType_t dtype;
  dtype = cvtProtoDtypeToMluOp(
      parser_->getProtoNode()->input(2 + use_mask_).onchip_dtype());
  weight_oc_dt_ = dtype;
  parser_->input(2 + use_mask_)->oc_dt = MLUOP_DTYPE_INVALID;

  dtype = cvtProtoDtypeToMluOp(
      parser_->getProtoNode()->input(3 + use_mask_).onchip_dtype());
  grad_output_oc_dt_ = dtype;
  parser_->input(3 + use_mask_)->oc_dt = MLUOP_DTYPE_INVALID;
}

int DcnBackwardDataExecutor::getCoefficientOfLT2CT() {
  auto input_dtype =
      cvtProtoDtypeToMluOp(parser_->getProtoNode()->input(0).dtype());
  int lt_compute_force = 0;
  int ct_compute_force = input_dtype == MLUOP_DTYPE_FLOAT ? 32 : 64;
  if (isFixData(grad_output_oc_dt_)) {  // intx peak compute force
    if (grad_output_oc_dt_ == MLUOP_DTYPE_INT16 &&
        weight_oc_dt_ == MLUOP_DTYPE_INT16) {
      lt_compute_force = 2 * 2048;
    } else if (grad_output_oc_dt_ == MLUOP_DTYPE_INT16 &&
               weight_oc_dt_ == MLUOP_DTYPE_INT31) {
      lt_compute_force = 2 * 2048 / 2;
    } else if (grad_output_oc_dt_ == MLUOP_DTYPE_INT31 &&
               weight_oc_dt_ == MLUOP_DTYPE_INT16) {
      lt_compute_force = 2 * 2048 / 2;
    } else {  // int31 x int31
      lt_compute_force = 2 * 2048 / 4;
    }
  } else {  // float/half peak compute force
    if (input_dtype == MLUOP_DTYPE_FLOAT) {
      lt_compute_force = 2 * 1.5 * 1024;
    } else {
      lt_compute_force = 2 * 0.375 * 1024;
    }
  }
  return lt_compute_force / ct_compute_force;
}

void DcnBackwardDataExecutor::workspaceMalloc() {
  VLOG(4) << "DCNBackwardDataExecutor workspaceMalloc";
  bool use_mask_ = parser_->inputs().size() == 5;
  bool use_grad_mask_ = parser_->outputs().size() == 3;
  // input
  input_desc_ = parser_->inputs()[0].tensor;
  offset_desc_ = parser_->inputs()[1].tensor;
  mask_desc_ = use_mask_ ? parser_->inputs()[2].tensor : nullptr;
  weight_desc_ = parser_->inputs()[2 + use_mask_].tensor;
  grad_output_desc_ = parser_->inputs()[3 + use_mask_].tensor;
  // output
  grad_input_desc_ = parser_->outputs()[0].tensor;
  grad_offset_desc_ = parser_->outputs()[1].tensor;
  grad_mask_desc_ = use_grad_mask_ ? parser_->outputs()[2].tensor : nullptr;

  grad_output_desc_->onchip_dtype = grad_output_oc_dt_;
  weight_desc_->onchip_dtype = weight_oc_dt_;

  dcn_desc_ = cpu_runtime_.allocate(mluOpCreateDCNDescriptor,
                                    mluOpDestroyDCNDescriptor);
  MLUOP_CHECK(mluOpSetDCNDescriptor(dcn_desc_, dimNb_, pad_, stride_, dilation_,
                                    deformable_group_, conv_group_,
                                    im2col_step_, compute_type_));
  MLUOP_CHECK(mluOpGetDCNBakcwardDataWorkspaceSize(
      handle_, dcn_desc_, input_desc_, offset_desc_, mask_desc_, weight_desc_,
      grad_output_desc_, grad_input_desc_, grad_offset_desc_, grad_mask_desc_,
      &workspace_size_));
  VLOG(4) << "dcn Backward Data workspace size: " << workspace_size_;

  if (workspace_size_ > 0) {
    dev_workspace_ = mlu_runtime_.allocate(workspace_size_);
    workspace_.push_back(dev_workspace_);
  }
}

void DcnBackwardDataExecutor::workspaceFree() {
  if (workspace_size_ > 0) {
    mlu_runtime_.deallocate(dev_workspace_);
  }
}

void DcnBackwardDataExecutor::compute() {
  VLOG(4) << "DCNBackwardDataExecutor compute";
  bool use_mask_ = parser_->inputs().size() == 5;
  bool use_grad_mask_ = parser_->outputs().size() == 3;
  // input
  input_desc_ = parser_->inputs()[0].tensor;
  offset_desc_ = parser_->inputs()[1].tensor;
  mask_desc_ = use_mask_ ? parser_->inputs()[2].tensor : nullptr;
  weight_desc_ = parser_->inputs()[2 + use_mask_].tensor;
  grad_output_desc_ = parser_->inputs()[3 + use_mask_].tensor;
  // output
  grad_input_desc_ = parser_->outputs()[0].tensor;
  grad_offset_desc_ = parser_->outputs()[1].tensor;
  grad_mask_desc_ = use_grad_mask_ ? parser_->outputs()[2].tensor : nullptr;

  void *dev_input = data_vector_[0].device_ptr;
  void *dev_offset = data_vector_[1].device_ptr;
  void *dev_mask = use_mask_ ? data_vector_[2].device_ptr : nullptr;
  void *dev_weight = data_vector_[2 + use_mask_].device_ptr;
  void *dev_grad_output = data_vector_[3 + use_mask_].device_ptr;
  void *dev_grad_input = data_vector_[4 + use_mask_].device_ptr;
  void *dev_grad_offset = data_vector_[5 + use_mask_].device_ptr;
  void *dev_grad_mask =
      use_mask_ ? data_vector_[6 + use_mask_].device_ptr : nullptr;

  grad_output_desc_->onchip_dtype = grad_output_oc_dt_;
  weight_desc_->onchip_dtype = weight_oc_dt_;

  VLOG(4) << "call mluOpDCNBackwardData()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpDCNBackwardData(
      handle_, dcn_desc_, input_desc_, dev_input, offset_desc_, dev_offset,
      mask_desc_, dev_mask, weight_desc_, dev_weight, grad_output_desc_,
      dev_grad_output, dev_workspace_, workspace_size_, grad_input_desc_,
      dev_grad_input, grad_offset_desc_, dev_grad_offset, grad_mask_desc_,
      dev_grad_mask));
  interface_timer_.stop();
}

void DcnBackwardDataExecutor::cpuCompute() {
  bool use_mask_ = parser_->inputs().size() == 5;
  float *host_input = cpu_fp32_input_[0];
  float *host_offset = cpu_fp32_input_[1];
  float *host_mask = use_mask_ ? cpu_fp32_input_[2] : nullptr;
  float *host_weight = cpu_fp32_input_[2 + use_mask_];
  float *host_grad_output = cpu_fp32_input_[3 + use_mask_];

  float *host_grad_input = cpu_fp32_output_[0];
  float *host_grad_offset = cpu_fp32_output_[1];
  float *host_grad_mask = use_mask_ ? cpu_fp32_output_[2] : nullptr;

  input_desc_ = tensor_desc_[0].tensor;
  offset_desc_ = tensor_desc_[1].tensor;
  mask_desc_ = use_mask_ ? tensor_desc_[2].tensor : nullptr;
  weight_desc_ = tensor_desc_[2 + use_mask_].tensor;
  grad_output_desc_ = tensor_desc_[3 + use_mask_].tensor;
  grad_input_desc_ = tensor_desc_[4 + use_mask_].tensor;
  grad_offset_desc_ = tensor_desc_[5 + use_mask_].tensor;
  grad_mask_desc_ = use_mask_ ? tensor_desc_[6 + use_mask_].tensor : nullptr;

  hostDCNBackwardData(input_desc_, host_input, offset_desc_, host_offset,
                      mask_desc_, host_mask, weight_desc_, host_weight,
                      grad_output_desc_, host_grad_output, grad_input_desc_,
                      host_grad_input, grad_offset_desc_, host_grad_offset,
                      grad_mask_desc_, host_grad_mask);
}

void DcnBackwardDataExecutor::transposeGradOutput(const float *output,
                                                  const int group,
                                                  const int ndhw, const int c,
                                                  float *transpose_output) {
  // [1, im2col * d * h * w, group, c]
  // ->
  // [1, group, im2col * d * h * w, c]
  VLOG(4) << "transposeGradOutput()";
  for (int ndhw_iter = 0; ndhw_iter < ndhw; ndhw_iter++) {
    for (int group_iter = 0; group_iter < group; group_iter++) {
      for (int c_iter = 0; c_iter < c; c_iter++) {
        transpose_output[group_iter * ndhw * c + ndhw_iter * c + c_iter] =
            output[ndhw_iter * group * c + group_iter * c + c_iter];
      }
    }
  }
}

void DcnBackwardDataExecutor::transposeGradCol(const float *dcol,
                                               const int group,
                                               const int middle, const int c,
                                               float *transpose_dcol) {
  VLOG(4) << "transposeGradCol()";
  for (int middle_iter = 0; middle_iter < middle; middle_iter++) {
    for (int group_iter = 0; group_iter < group; group_iter++) {
      for (int c_iter = 0; c_iter < c; c_iter++) {
        transpose_dcol[middle_iter * group * c + group_iter * c + c_iter] =
            dcol[group_iter * middle * c + middle_iter * c + c_iter];
      }
    }
  }
}

void DcnBackwardDataExecutor::hostDCNBackwardData(
    const mluOpTensorDescriptor_t input_desc, float *input,
    const mluOpTensorDescriptor_t offset_desc, float *offset,
    const mluOpTensorDescriptor_t mask_desc, float *mask,
    const mluOpTensorDescriptor_t weight_desc, float *weight,
    const mluOpTensorDescriptor_t grad_output_desc, float *grad_output,
    mluOpTensorDescriptor_t grad_input_desc, float *grad_input,
    mluOpTensorDescriptor_t grad_offset_desc, float *grad_offset,
    mluOpTensorDescriptor_t grad_mask_desc, float *grad_mask) {
  int n = mluOpGetTensordimN(input_desc);
  int di = dimNb_ == 5 ? mluOpGetTensordimD(input_desc) : 1;
  int hi = mluOpGetTensordimH(input_desc);
  int wi = mluOpGetTensordimW(input_desc);

  int d_o = dimNb_ == 5 ? mluOpGetTensordimD(grad_output_desc) : 1;
  int ho = mluOpGetTensordimH(grad_output_desc);
  int wo = mluOpGetTensordimW(grad_output_desc);

  int kd = dimNb_ == 5 ? mluOpGetTensordimD(weight_desc) : 1;
  int kh = mluOpGetTensordimH(weight_desc);
  int kw = mluOpGetTensordimW(weight_desc);

  int ci = mluOpGetTensordimC(weight_desc);
  int co = mluOpGetTensordimN(weight_desc) / conv_group_;
  int c_per_deform_group = ci / deformable_group_;
  int im2col_step = im2col_step_;

  float *dcol = cpu_runtime_.allocate(
      new float[im2col_step * d_o * ho * wo * kd * kh * kw * conv_group_ * ci]);

  for (int iter = 0; iter < n * di * hi * wi * conv_group_ * ci; iter++) {
    grad_input[iter] = 0.0;
  }
  for (int iter = 0;
       iter < n * d_o * ho * wo * kd * kh * kw * deformable_group_ * 2;
       iter++) {
    grad_offset[iter] = 0.0;
  }
  if (grad_mask != nullptr) {
    for (int iter = 0;
         iter < n * d_o * ho * wo * kd * kh * kw * deformable_group_; iter++) {
      grad_mask[iter] = 0.0;
    }
  }
#if USE_OPENBLAS
#else
  auto matmul = [](float *lhs, float *rhs, float *output, bool is_trans_a,
                   bool is_trans_b, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        output[m * N + n] = 0.0f;
        for (int k = 0; k < K; k++) {
          int lhs_idx = m * K + k;
          if (is_trans_a) lhs_idx = k * M + m;
          int rhs_idx = k * N + n;
          if (is_trans_b) rhs_idx = n * K + k;
          output[m * N + n] += lhs[lhs_idx] * rhs[rhs_idx];
        }
      }
    }
  };
#endif

  for (int batch_iter = 0; batch_iter < n / im2col_step; batch_iter++) {
    VLOG(4) << "iter: " << batch_iter << " / " << n / im2col_step << ".";
    float *trans_grad_output;
    if (conv_group_ == 1) {
      trans_grad_output =
          grad_output + batch_iter * im2col_step * d_o * ho * wo * co;
    } else {
      trans_grad_output = cpu_runtime_.allocate(
          new float[conv_group_ * im2col_step * d_o * ho * wo * co]);
      float *grad_output_iter =
          grad_output + batch_iter * im2col_step * d_o * ho * wo * co;
      transposeGradOutput(grad_output_iter, conv_group_,
                          im2col_step * d_o * ho * wo, co, trans_grad_output);
      theory_ops_ += conv_group_ * im2col_step * d_o * ho * wo, co;
    }

    for (int group_iter = 0; group_iter < conv_group_; group_iter++) {
      float *grad_output_addr =
          trans_grad_output + group_iter * im2col_step * ho * wo * co;
      float *weight_addr = weight + group_iter * co * kh * kw * ci;
      float *col_addr =
          dcol + group_iter * im2col_step * d_o * ho * wo * kd * kh * kw * ci;
#if USE_OPENBLAS
      const CBLAS_ORDER Order = CblasRowMajor;
      const CBLAS_TRANSPOSE TransA = CblasNoTrans;
      const CBLAS_TRANSPOSE TransB = CblasNoTrans;
      const float alpha = 1.0;
      const float beta = 0.0;

      cblas_sgemm(Order, TransA, TransB, im2col_step * d_o * ho * wo,
                  kd * kh * kw * ci, co, alpha, grad_output_addr, co,
                  weight_addr, kd * kh * kw * ci, beta, col_addr,
                  kd * kh * kw * ci);
#else
      matmul(grad_output_addr, weight_addr, col_addr, false, false,
             im2col_step * d_o * ho * wo, kd * kh * kw * ci, co);
#endif
      int coeff = getCoefficientOfLT2CT();
      theory_ops_ += 2 * im2col_step * d_o * ho * wo * kd * kh * kw * ci * co /
                     coeff;  // lt2ct
    }

    float *trans_grad_col;
    if (conv_group_ != 1) {
      trans_grad_col = cpu_runtime_.allocate(
          new float[im2col_step * d_o * ho * wo * kh * kw * conv_group_ * ci]);
      transposeGradCol(dcol, conv_group_,
                       im2col_step * d_o * ho * wo * kd * kh * kw, ci,
                       trans_grad_col);
      theory_ops_ +=
          conv_group_ * im2col_step * d_o * ho * wo * kd * kh * kw * ci;
    } else {
      trans_grad_col = dcol;
    }

    if (dimNb_ == 4) {
      int input_deal_once = im2col_step * hi * wi * ci;
      int offset_deal_once =
          im2col_step * ho * wo * deformable_group_ * kh * kw * 2;
      int mask_deal_once = im2col_step * ho * wo * deformable_group_ * kh * kw;
      float *input_addr = input + batch_iter * input_deal_once;
      float *offset_addr = offset + batch_iter * offset_deal_once;
      float *mask_addr =
          mask == nullptr ? nullptr : mask + batch_iter * mask_deal_once;
      float *grad_input_addr = grad_input + batch_iter * input_deal_once;
      float *grad_offset_addr = grad_offset + batch_iter * offset_deal_once;
      float *grad_mask_addr = grad_mask == nullptr
                                  ? nullptr
                                  : grad_mask + batch_iter * mask_deal_once;

      col2img4D(trans_grad_col, input_addr, offset_addr, mask_addr,
                grad_input_addr, grad_offset_addr, grad_mask_addr, im2col_step,
                hi, wi, ci, co, kh, kw, pad_, stride_, dilation_,
                deformable_group_, conv_group_);
      // bilinear(16) + grad_mask(2) + grad_offset(8)
      theory_ops_ += im2col_step * ho * wo * kh * kw * deformable_group_ *
                     c_per_deform_group * 28;
      // grad input (2 * grad bilinear loop (4))
      theory_ops_ += im2col_step * ho * wo * kh * kw * deformable_group_ *
                     c_per_deform_group * 4 * 2;
    } else {  // dimNb == 5
      // TODO(sunhui): reserve for 3D DCN Backward Data
    }
    if (conv_group_ != 1) {
      cpu_runtime_.deallocate(trans_grad_col);
      cpu_runtime_.deallocate(trans_grad_output);
    }
  }  // end batch iteration
  cpu_runtime_.deallocate(dcol);
}

/*
 * grad_col: n, ho, wo, kh, kw, deform_group, ci_per_deform_group
 * input: n, hi, wi, conv_group * ci
 * offset: n, ho, wo, deform_group, kh, kw, 2
 * mask: n, ho, wo, deform_group, kh, kw
 */
void DcnBackwardDataExecutor::col2img4D(float *grad_col, float *input,
                                        float *offset, float *mask,
                                        float *grad_input, float *grad_offset,
                                        float *grad_mask, int n, int hi, int wi,
                                        int ci, int co, int kh, int kw,
                                        int pad[], int stride[], int dilation[],
                                        int deform_group, int conv_group) {
  int pt = pad[0];
  int pb = pad[1];
  int pl = pad[2];
  int pr = pad[3];
  int ho = (hi + pt + pb - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
  int wo = (wi + pl + pr - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
  int ci_per_deform_group = ci * conv_group / deform_group;

  float *cur_top_grad = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *mval = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *bilinear_result =
      cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *valh = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *valw = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *interp_weight_h =
      cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *interp_weight_w =
      cpu_runtime_.allocate(new float[ci_per_deform_group]);

  for (int i_iter = 0; i_iter < n * ho * wo * kh * kw * deform_group;
       i_iter++) {
    int deform_iter = i_iter % deform_group;
    int kw_iter = (i_iter / deform_group) % kw;
    int kh_iter = (i_iter / deform_group / kw) % kh;
    int wo_iter = (i_iter / deform_group / kw / kh) % wo;
    int ho_iter = (i_iter / deform_group / kw / kh / wo) % ho;
    int n_iter = (i_iter / deform_group / kw / kh / wo / ho) % n;

    int offset_offset = n_iter * ho * wo * deform_group * kh * kw * 2 +
                        ho_iter * wo * deform_group * kh * kw * 2 +
                        wo_iter * deform_group * kh * kw * 2 +
                        deform_iter * kh * kw * 2 + kh_iter * kw * 2 +
                        kw_iter * 2;
    float offset_h = offset[offset_offset + 0];
    float offset_w = offset[offset_offset + 1];

    float mask_value = 1.0;
    if (mask != nullptr) {
      int mask_offset = n_iter * ho * wo * deform_group * kh * kw +
                        ho_iter * wo * deform_group * kh * kw +
                        wo_iter * deform_group * kh * kw +
                        deform_iter * kh * kw + kh_iter * kw + kw_iter;
      mask_value = mask[mask_offset];
    }

    float h_im = ho_iter * stride[0] - pt + kh_iter * dilation[0] + offset_h;
    float w_im = wo_iter * stride[1] - pl + kw_iter * dilation[1] + offset_w;

    int col_offset =
        n_iter * ho * wo * kh * kw * deform_group * ci_per_deform_group +
        ho_iter * wo * kh * kw * deform_group * ci_per_deform_group +
        wo_iter * kh * kw * deform_group * ci_per_deform_group +
        kh_iter * kw * deform_group * ci_per_deform_group +
        kw_iter * deform_group * ci_per_deform_group +
        deform_iter * ci_per_deform_group;
    for (int iter = 0; iter < ci_per_deform_group; iter++) {
      cur_top_grad[iter] = grad_col[col_offset + iter];
    }

    int cur_h = int(h_im);
    int cur_w = int(w_im);

    // dcn backward data prospect
    for (int dh_iter = -2; dh_iter < 3; dh_iter++) {
      for (int dw_iter = -2; dw_iter < 3; dw_iter++) {
        if (cur_h + dh_iter >= 0 && cur_h + dh_iter < hi &&
            cur_w + dw_iter >= 0 && cur_w + dw_iter < wi &&
            fabs(h_im - (cur_h + dh_iter)) < 1.0 &&
            fabs(w_im - (cur_w + dw_iter)) < 1.0) {
          float weight = 0.0;
          if (h_im <= -1 || h_im >= hi || w_im <= -1 || w_im >= wi) {
            continue;
          } else {
            int h_low = floor(h_im);
            int w_low = floor(w_im);
            int h_high = h_low + 1;
            int w_high = w_low + 1;
            int tmp_h = cur_h + dh_iter;
            int tmp_w = cur_w + dw_iter;
            if (tmp_h == h_low && tmp_w == w_low) {
              weight = (tmp_h + 1.0 - h_im) * (tmp_w + 1.0 - w_im);
            } else if (tmp_h == h_low && tmp_w == w_high) {
              weight = (tmp_h + 1.0 - h_im) * (w_im + 1.0 - tmp_w);
            } else if (tmp_h == h_high && tmp_w == w_low) {
              weight = (h_im + 1.0 - tmp_h) * (tmp_w + 1.0 - w_im);
            } else if (tmp_h == h_high && tmp_w == w_high) {
              weight = (h_im + 1.0 - tmp_h) * (w_im + 1.0 - tmp_w);
            }
          }
          int grad_img_offset =
              n_iter * hi * wi * deform_group * ci_per_deform_group +
              (cur_h + dh_iter) * wi * deform_group * ci_per_deform_group +
              (cur_w + dw_iter) * deform_group * ci_per_deform_group +
              deform_iter * ci_per_deform_group;
          for (int iter = 0; iter < ci_per_deform_group; iter++) {
            grad_input[grad_img_offset + iter] +=
                weight * cur_top_grad[iter] * mask_value;
          }
        }  // if cur h and cur w is valid
      }    // dw iter
    }      // dh iter
    // dcn backward data end

    // dcn backward mask

    for (int iter = 0; iter < ci_per_deform_group; iter++) {
      mval[iter] = 0.0;
    }

    if (h_im > -1 && h_im < hi && w_im > -1 && w_im < wi) {
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        im2col_bilinear(input, n_iter, h_im, w_im, deform_iter, hi, wi,
                        deform_group, ci_per_deform_group, bilinear_result);
        mval[iter] = cur_top_grad[iter] * bilinear_result[iter];
      }
    } else {
      h_im = -2.0;
      w_im = -2.0;
    }
    if (grad_mask != nullptr) {
      int mask_offset = n_iter * ho * wo * deform_group * kh * kw +
                        ho_iter * wo * deform_group * kh * kw +
                        wo_iter * deform_group * kh * kw +
                        deform_iter * kh * kw + kh_iter * kw + kw_iter;
      float grad_mask_value = 0.0;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        grad_mask_value += mval[iter];
      }
      grad_mask[mask_offset] = grad_mask_value;
    }
    // dcn backward mask end

    // dcn backward offset

    col2img_coordinate(input, n_iter, h_im, w_im, deform_iter, hi, wi,
                       deform_group, ci_per_deform_group, interp_weight_h,
                       interp_weight_w);
    for (int iter = 0; iter < ci_per_deform_group; iter++) {
      valh[iter] = cur_top_grad[iter] * interp_weight_h[iter] * mask_value;
      valw[iter] = cur_top_grad[iter] * interp_weight_w[iter] * mask_value;
    }
    float grad_offset_h = 0.0;
    float grad_offset_w = 0.0;
    for (int iter = 0; iter < ci_per_deform_group; iter++) {
      grad_offset_h += valh[iter];
      grad_offset_w += valw[iter];
    }

    grad_offset[offset_offset + 0] = grad_offset_h;
    grad_offset[offset_offset + 1] = grad_offset_w;
  }  // i iter
}

void DcnBackwardDataExecutor::col2img_coordinate(
    float *input, int n_iter, float h_im, float w_im, int deform_iter, int hi,
    int wi, int deform_group, int ci_per_deform_group, float *weight_h,
    float *weight_w) {
  int h_low = floor(h_im);
  int w_low = floor(w_im);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  float lh = h_im - h_low;
  float lw = w_im - w_low;

  for (int iter = 0; iter < ci_per_deform_group; iter++) {
    weight_h[iter] = 0.0;
    weight_w[iter] = 0.0;
  }
  if (h_im > -1 && h_im < hi && w_im > -1 && w_im < wi) {
    if (h_low >= 0 && w_low >= 0) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_low * wi * deform_group * ci_per_deform_group +
                         w_low * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        weight_h[iter] += -1.0 * (w_high - w_im) * input[input_offset + iter];
        weight_w[iter] += -1.0 * (h_high - h_im) * input[input_offset + iter];
      }
    }
    if (h_low >= 0 && w_high < wi) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_low * wi * deform_group * ci_per_deform_group +
                         w_high * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        weight_h[iter] += -1.0 * (w_im - w_low) * input[input_offset + iter];
        weight_w[iter] += 1.0 * (h_high - h_im) * input[input_offset + iter];
      }
    }
    if (h_high < hi && w_low >= 0) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_high * wi * deform_group * ci_per_deform_group +
                         w_low * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        weight_h[iter] += 1.0 * (w_high - w_im) * input[input_offset + iter];
        weight_w[iter] += -1.0 * (h_im - h_low) * input[input_offset + iter];
      }
    }
    if (h_high < hi && w_high < wi) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_high * wi * deform_group * ci_per_deform_group +
                         w_high * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        weight_h[iter] += 1.0 * (w_im - w_low) * input[input_offset + iter];
        weight_w[iter] += 1.0 * (h_im - h_low) * input[input_offset + iter];
      }
    }
  }
}

void DcnBackwardDataExecutor::im2col_bilinear(
    float *input, int n_iter, float h_im, float w_im, int deform_iter, int hi,
    int wi, int deform_group, int ci_per_deform_group, float *bilinear_result) {
  int h_low = floor(h_im);
  int w_low = floor(w_im);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h_im - h_low;
  float lw = w_im - w_low;
  float hh = 1.0 - lh;
  float hw = 1.0 - lw;

  float *v1 = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *v2 = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *v3 = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  float *v4 = cpu_runtime_.allocate(new float[ci_per_deform_group]);
  for (int iter = 0; iter < ci_per_deform_group; iter++) {
    v1[iter] = 0.0;
    v2[iter] = 0.0;
    v3[iter] = 0.0;
    v4[iter] = 0.0;
    bilinear_result[iter] = 0.0;
  }

  if (h_im > -1.0 && w_im > -1.0 && h_im < hi && w_im < wi) {
    if (h_low >= 0 && w_low >= 0) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_low * wi * deform_group * ci_per_deform_group +
                         w_low * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        v1[iter] = input[input_offset + iter];
      }
    }
    if (h_low >= 0 && w_high <= wi - 1) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_low * wi * deform_group * ci_per_deform_group +
                         w_high * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        v2[iter] = input[input_offset + iter];
      }
    }
    if (h_high <= hi - 1 && w_low >= 0) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_high * wi * deform_group * ci_per_deform_group +
                         w_low * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        v3[iter] = input[input_offset + iter];
      }
    }
    if (h_high <= hi - 1 && w_high <= wi - 1) {
      int input_offset = n_iter * hi * wi * deform_group * ci_per_deform_group +
                         h_high * wi * deform_group * ci_per_deform_group +
                         w_high * deform_group * ci_per_deform_group +
                         deform_iter * ci_per_deform_group;
      for (int iter = 0; iter < ci_per_deform_group; iter++) {
        v4[iter] = input[input_offset + iter];
      }
    }
    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    for (int iter = 0; iter < ci_per_deform_group; iter++) {
      bilinear_result[iter] =
          w1 * v1[iter] + w2 * v2[iter] + w3 * v3[iter] + w4 * v4[iter];
    }
  }
  cpu_runtime_.deallocate(v1);
  cpu_runtime_.deallocate(v2);
  cpu_runtime_.deallocate(v3);
  cpu_runtime_.deallocate(v4);
}

int64_t DcnBackwardDataExecutor::getTheoryOps() {
  if (exe_config_->mlu_only) {
    int n = mluOpGetTensordimN(input_desc_);
    int di = dimNb_ == 5 ? mluOpGetTensordimD(input_desc_) : 1;
    int hi = mluOpGetTensordimH(input_desc_);
    int wi = mluOpGetTensordimW(input_desc_);

    int d_o = dimNb_ == 5 ? mluOpGetTensordimD(grad_output_desc_) : 1;
    int ho = mluOpGetTensordimH(grad_output_desc_);
    int wo = mluOpGetTensordimW(grad_output_desc_);

    int kd = dimNb_ == 5 ? mluOpGetTensordimD(weight_desc_) : 1;
    int kh = mluOpGetTensordimH(weight_desc_);
    int kw = mluOpGetTensordimW(weight_desc_);

    int ci = mluOpGetTensordimC(weight_desc_);
    int co = mluOpGetTensordimN(weight_desc_) / conv_group_;
    int c_per_deform_group = ci / deformable_group_;

    theory_ops_ = 0;
    // if conv group > 1, transpose grad output and the column data
    if (conv_group_ > 1) {
      theory_ops_ += parser_->getOutputDataCount(0);
      theory_ops_ += conv_group_ * n * d_o * ho * wo * kd * kh * kw * ci;
    }
    int coeff = getCoefficientOfLT2CT();
    theory_ops_ +=
        2 * conv_group_ * n * d_o * ho * wo * kd * kh * kw * ci * co / coeff;
    // bilinear(16) + grad_mask(2) + grad_offset(8)
    theory_ops_ +=
        n * ho * wo * kh * kw * deformable_group_ * c_per_deform_group * 28;
    // grad input (2 * grad bilinear loop (4))
    theory_ops_ +=
        n * ho * wo * kh * kw * deformable_group_ * c_per_deform_group * 4 * 2;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}

}  // namespace mluoptest
