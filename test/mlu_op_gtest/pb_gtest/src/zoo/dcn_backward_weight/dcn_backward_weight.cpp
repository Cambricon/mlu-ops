/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "dcn_backward_weight.h"
#include "internal_kernel/transpose_cpu/transpose_cpu.h"

#define USE_OPENBLAS 0

#if USE_OPENBLAS
#include <openblas/cblas.h>
#endif

namespace mluoptest {
// input      :[N,hi,wi,ci]
// offset     :[N,ho,wo,dg*kh*kw*2]
// mask       :[N,ho,wo,dg*kh*kw] // optional
// grad_ouput :[N,ho,wo,co]
// grad_weight:[co,kh,kw,ci/g]
// grad_bias  :[co] // optional
static inline bool isFixData(mluOpDataType_t type) {
  if (MLUOP_DTYPE_INT8 == type || MLUOP_DTYPE_INT16 == type ||
      MLUOP_DTYPE_INT31 == type) {
    return true;
  }
  return false;
}

int DcnBackwardWeightExecutor::getCoefficientOfLT2CT() {
  auto input_dtype =
      cvtProtoDtypeToMluOp(parser_->getProtoNode()->input(0).dtype());
  int lt_compute_force = 0;
  int ct_compute_force = input_dtype == MLUOP_DTYPE_FLOAT ? 32 : 64;
  if (input_dtype == MLUOP_DTYPE_FLOAT) {
    lt_compute_force = 2 * 1.5 * 1024;
  } else {
    lt_compute_force = 2 * 0.375 * 1024;
  }
  return lt_compute_force / ct_compute_force;
}

void DcnBackwardWeightExecutor::paramCheck() {
  if (parser_->getInputNum() != 3 && parser_->getInputNum() != 4) {
    LOG(ERROR) << "DCN_Backward_Weight: tensor input number is wrong.";
  }

  auto dtype =
      cvtProtoDtypeToMluOp(parser_->getProtoNode()->input(0).onchip_dtype());
  input_onchip_dtype = dtype;
  if (isFixData(dtype)) {
    parser_->input(0)->oc_dt = MLUOP_DTYPE_INVALID;
  }

  if (parser_->getInputNum() == 3) {
    dtype =
        cvtProtoDtypeToMluOp(parser_->getProtoNode()->input(2).onchip_dtype());
    grad_output_onchip_dtype = dtype;
    if (isFixData(dtype)) {
      parser_->input(2)->oc_dt = MLUOP_DTYPE_INVALID;
    }
  } else {
    dtype =
        cvtProtoDtypeToMluOp(parser_->getProtoNode()->input(3).onchip_dtype());
    grad_output_onchip_dtype = dtype;
    if (isFixData(dtype)) {
      parser_->input(3)->oc_dt = MLUOP_DTYPE_INVALID;
    }
  }

  if (!parser_->getProtoNode()->has_dcn_param()) {
    LOG(ERROR) << "Missing dcn param. ";
  }

  if (parser_->getOutputNum() != 1 && parser_->getOutputNum() != 2) {
    LOG(ERROR) << "DCN_Backward_Weight tensor output number is wrong.";
  }
  TensorLayout input_order = parser_->getProtoNode()->input(0).layout();
  if (input_order != LAYOUT_NHWC) {
    LOG(ERROR) << "DCN_Backward_Weight input tensor layout should be NHWC.";
  }

  int N = parser_->getProtoNode()->input(0).shape().dims(0);
  int ci = parser_->getProtoNode()->input(0).shape().dims(3);
  int co = parser_->getProtoNode()->output(0).shape().dims(0);

  auto dcn_param = parser_->getProtoNode()->dcn_param();
  dimnb = dcn_param.dimnb();
  for (int i = 0; i < dcn_param.pad_size(); ++i) {
    pad[i] = dcn_param.pad(i);
  }
  for (int i = 0; i < dcn_param.stride_size(); ++i) {
    stride[i] = dcn_param.stride(i);
  }
  for (int i = 0; i < dcn_param.dilation_size(); ++i) {
    dilation[i] = dcn_param.dilation(i);
  }
  if (dcn_param.has_deformable_group()) {
    dg = dcn_param.deformable_group();
  }
  if (dcn_param.has_conv_group()) {
    g = dcn_param.conv_group();
  }
  if (dcn_param.has_im2col_step()) {
    im2col_step = dcn_param.im2col_step();
  }

  if (dimnb != 4) {
    LOG(ERROR) << "[DCN_Backward_Weight]: dimnb should be 4.";
  }

  if (ci % dg) {
    LOG(ERROR) << "[DCN_Backward_Weight]: deformable_group is wrong.";
  }

  if (ci % g) {
    LOG(ERROR) << "[DCN_Backward_Weight]: conv_group is wrong.";
  }

  if (co % g) {
    LOG(ERROR) << "[DCN_Backward_Weight]: conv_group is wrong.";
  }

  if (N % im2col_step) {
    LOG(ERROR) << "[DCN_Backward_Weight]: im2col_step is wrong.";
  }
}

void DcnBackwardWeightExecutor::workspaceMalloc() {
  input_desc = tensor_desc_[0].tensor;
  offset_desc = tensor_desc_[1].tensor;
  mluOpDataType_t compute_type;
  auto dcn_param = parser_->getProtoNode()->dcn_param();
  if (dcn_param.has_compute_type()) {
    compute_type = cvtProtoDtypeToMluOp(dcn_param.compute_type());
  } else {
    compute_type = MLUOP_DTYPE_FLOAT;
  }

  mluOpDCNDescriptor_t dcn_desc = cpu_runtime_.allocate(
      mluOpCreateDCNDescriptor, mluOpDestroyDCNDescriptor);

  MLUOP_CHECK(mluOpSetDCNDescriptor(dcn_desc, dimnb, pad, stride, dilation, dg,
                                    g, im2col_step, compute_type));

  if (parser_->getInputNum() == 3) {
    mask_desc = nullptr;
    grad_output_desc = tensor_desc_[2].tensor;
    grad_weight_desc = tensor_desc_[3].tensor;
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[4].tensor;
  } else {
    mask_desc = tensor_desc_[2].tensor;
    grad_output_desc = tensor_desc_[3].tensor;
    grad_weight_desc = tensor_desc_[4].tensor;
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[5].tensor;
  }

  input_desc->onchip_dtype = input_onchip_dtype;
  grad_output_desc->onchip_dtype = grad_output_onchip_dtype;
  MLUOP_CHECK(mluOpGetDCNBackwardWeightWorkspaceSize(
      handle_, dcn_desc, input_desc, offset_desc, mask_desc, grad_output_desc,
      grad_weight_desc, grad_bias_desc, &workspace_size));

  if (workspace_size != 0) {
    workspace = mlu_runtime_.allocate(workspace_size);
  }

  eva_->setMluWorkspaceSize(workspace_size);
  cpu_runtime_.deallocate(dcn_desc);
}

void DcnBackwardWeightExecutor::workspaceFree() {
  if (workspace != nullptr) {
    mlu_runtime_.deallocate(workspace);
  }
}

void DcnBackwardWeightExecutor::compute() {
  input_desc = tensor_desc_[0].tensor;
  offset_desc = tensor_desc_[1].tensor;
  mluOpDataType_t compute_type;
  auto dcn_param = parser_->getProtoNode()->dcn_param();
  if (dcn_param.has_compute_type()) {
    compute_type = cvtProtoDtypeToMluOp(dcn_param.compute_type());
  } else {
    compute_type = input_desc->dtype;
  }

  mluOpDCNDescriptor_t dcn_desc = cpu_runtime_.allocate(
      mluOpCreateDCNDescriptor, mluOpDestroyDCNDescriptor);

  MLUOP_CHECK(mluOpSetDCNDescriptor(dcn_desc, dimnb, pad, stride, dilation, dg,
                                    g, im2col_step, compute_type));

  input = data_vector_[0].device_ptr;
  offset = data_vector_[1].device_ptr;
  if (parser_->getInputNum() == 3) {
    mask_desc = nullptr;
    mask = nullptr;
    grad_output_desc = tensor_desc_[2].tensor;
    grad_output = data_vector_[2].device_ptr;
    grad_weight_desc = tensor_desc_[3].tensor;
    grad_weight = data_vector_[3].device_ptr;
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[4].tensor;
    grad_bias =
        parser_->getOutputNum() == 1 ? nullptr : data_vector_[4].device_ptr;
  } else {
    mask_desc = tensor_desc_[2].tensor;
    mask = data_vector_[2].device_ptr;
    grad_output_desc = tensor_desc_[3].tensor;
    grad_output = data_vector_[3].device_ptr;
    grad_weight_desc = tensor_desc_[4].tensor;
    grad_weight = data_vector_[4].device_ptr;
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[5].tensor;
    grad_bias =
        parser_->getOutputNum() == 1 ? nullptr : data_vector_[5].device_ptr;
  }

  input_desc->onchip_dtype = input_onchip_dtype;
  grad_output_desc->onchip_dtype = grad_output_onchip_dtype;

  VLOG(4) << "call mluOpDCNBackwardWeight()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpDCNBackwardWeight(
      handle_, dcn_desc, input_desc, input, offset_desc, offset, mask_desc,
      mask, grad_output_desc, grad_output, workspace, workspace_size,
      grad_weight_desc, grad_weight, grad_bias_desc, grad_bias));

  interface_timer_.stop();
  cpu_runtime_.deallocate(dcn_desc);
}

static float bilinear(float *input_ptr, const int &ci_offset, const int &hi,
                      const int &wi, const int &ci, const float &h_in,
                      const float &w_in) {
  int h_low = floor(h_in);
  int w_low = floor(w_in);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h_in - h_low;
  float lw = w_in - w_low;
  float hh = 1 - lh;
  float hw = 1 - lw;

  float v1 = 0, v2 = 0, v3 = 0, v4 = 0;

  if (h_low >= 0 && w_low >= 0) {
    v1 = input_ptr[(h_low * wi + w_low) * ci + ci_offset];
  }

  if (h_low >= 0 && w_high <= wi - 1) {
    v2 = input_ptr[(h_low * wi + w_high) * ci + ci_offset];
  }

  if (h_high <= hi - 1 && w_low >= 0) {
    v3 = input_ptr[(h_high * wi + w_low) * ci + ci_offset];
  }

  if (h_high <= hi - 1 && w_high <= wi - 1) {
    v4 = input_ptr[(h_high * wi + w_high) * ci + ci_offset];
  }

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
  return val;
}

static void im2col(const int &N, const int &im2col_step, const int &dg,
                   const int &hi, const int &wi, const int &ci, const int &ho,
                   const int &wo, const int &co, const int &kh, const int &kw,
                   const int &pt, const int &pb, const int &pl, const int &pr,
                   const int &sh, const int &sw, const int &dh, const int &dw,
                   const float *cpu_input, const float *cpu_offset,
                   const float *cpu_mask, float *buffer) {
  // input      :[N,hi,wi,ci]
  // offset     :[N,ho,wo,dg*kh*kw*2]
  // mask       :[N,ho,wo,dg*kh*kw] // optional
  // grad_ouput :[N,ho,wo,co]
  // grad_weight:[co,kh,kw,ci/g]
  // grad_bias  :[co] // optional
  for (int idx_n = 0; idx_n < im2col_step; ++idx_n) {
    for (int idx_ho = 0; idx_ho < ho; ++idx_ho) {
      for (int idx_wo = 0; idx_wo < wo; ++idx_wo) {
        float *input_ptr = (float *)cpu_input + idx_n * hi * wi * ci;
        float *offset_ptr =
            (float *)cpu_offset +
            ((idx_n * ho + idx_ho) * wo + idx_wo) * dg * kh * kw * 2;
        float *mask_ptr =
            cpu_mask != nullptr
                ? (float *)cpu_mask +
                      ((idx_n * ho + idx_ho) * wo + idx_wo) * dg * kh * kw
                : nullptr;
        float *columns_ptr =
            (float *)buffer +
            ((idx_n * ho + idx_ho) * wo + idx_wo) * kh * kw * ci;
        const int hi_start = idx_ho * sh - pt;
        const int wi_start = idx_wo * sw - pl;
        for (int idx_kh = 0; idx_kh < kh; ++idx_kh) {
          for (int idx_kw = 0; idx_kw < kw; ++idx_kw) {
            for (int idx_dg = 0; idx_dg < dg; ++idx_dg) {
              const int data_offset_h =
                  ((idx_dg * kh + idx_kh) * kw + idx_kw) * 2;
              const int data_offset_w =
                  ((idx_dg * kh + idx_kh) * kw + idx_kw) * 2 + 1;
              const int data_mask = (idx_dg * kh + idx_kh) * kw + idx_kw;
              const float offset_h = offset_ptr[data_offset_h];
              const float offset_w = offset_ptr[data_offset_w];
              const float mask =
                  mask_ptr != nullptr ? mask_ptr[data_mask] : 1.0f;
              const float h_in = hi_start + idx_kh * dh + offset_h;
              const float w_in = wi_start + idx_kw * dw + offset_w;
              if (h_in > -1 && w_in > -1 && h_in < hi && w_in < wi) {
                for (int idx_ci = 0; idx_ci < ci / dg; ++idx_ci) {
                  const int ci_offset = idx_dg * ci / dg + idx_ci;
                  const int columns_offset =
                      (idx_kh * kw + idx_kw) * ci + ci_offset;
                  columns_ptr[columns_offset] =
                      bilinear(input_ptr, ci_offset, hi, wi, ci, h_in, w_in) *
                      mask;
                }
              }
            }
          }
        }
      }
    }
  }
}

void DcnBackwardWeightExecutor::transpose(float *input, float *output,
                                          const int dims[], const int dim_num,
                                          int permute[]) {
  int64_t dim_desc = dim_num;
  std::vector<int> permute_desc;
  if (dim_desc > 8 || dim_desc <= 0) {
    LOG(ERROR) << "dim_desc is " << dim_desc
               << ", it shoule less than 8 and greater than 0";
  }
  { std::vector<int>().swap(permute_desc); }
  for (int i = 0; i < dim_num; i++) {
    permute_desc.push_back(permute[i]);
  }
  mluOpTensorDescriptor_t input_desc, output_desc;
  input_desc = cpu_runtime_.allocate(mluOpCreateTensorDescriptor,
                                     mluOpDestroyTensorDescriptor);
  output_desc = cpu_runtime_.allocate(mluOpCreateTensorDescriptor,
                                      mluOpDestroyTensorDescriptor);

  int dims_trans[4];
  for (int i = 0; i < dim_num; ++i) {
    dims_trans[i] = dims[permute[i]];
  }

  MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc, MLUOP_LAYOUT_ARRAY,
                                       MLUOP_DTYPE_FLOAT, dim_num, dims));
  MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc, MLUOP_LAYOUT_ARRAY,
                                       MLUOP_DTYPE_FLOAT, dim_num, dims_trans));

  MLUOP_CHECK(mluOpTransposeCpu(dim_desc, permute_desc, input_desc, input,
                                output_desc, output));
  cpu_runtime_.deallocate(input_desc);
  cpu_runtime_.deallocate(output_desc);
}

static void BatchMatMul(const int &g, const int &m, const int &k, const int &n,
                        float *input_a, float *input_b, float *output,
                        bool is_transa, bool is_transb) {
  const int batch_size = g;

  GTEST_CHECK(batch_size >= 1);
#if USE_OPENBLAS
  const CBLAS_ORDER Order = CblasRowMajor;
  const CBLAS_TRANSPOSE TransA = is_transa ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = is_transb ? CblasTrans : CblasNoTrans;

  int lda = is_transa ? m : k;
  int ldb = is_transb ? k : n;
  int ldc = n;

  float alpha = 1.0f;
  float beta = 1.0f;
#else
  auto matmul = [](float *lhs, float *rhs, float *output, bool is_trans_a,
                   bool is_trans_b, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        // output[m * N + n] = 0.0f;
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
  for (int i = 0; i < batch_size; ++i) {
#if USE_OPENBLAS
    cblas_sgemm(Order, TransA, TransB, m, n, k, alpha, input_a + i * m * k, lda,
                input_b + i * k * n, ldb, beta, output + i * m * n, ldc);
#else
    matmul(input_a + i * m * k, input_b + i * k * n, output + i * m * n,
           is_transa, is_transb, m, n, k);
#endif
  }
}

static void dealBias(float *cpu_grad_output, float *cpu_grad_bias, const int &N,
                     const int &ho, const int &wo, const int &co) {
  for (int idx_n = 0; idx_n < N; ++idx_n) {
    for (int idx_ho = 0; idx_ho < ho; ++idx_ho) {
      for (int idx_wo = 0; idx_wo < wo; ++idx_wo) {
        for (int idx_co = 0; idx_co < co; ++idx_co) {
          cpu_grad_bias[idx_co] +=
              cpu_grad_output[((idx_n * ho + idx_ho) * wo + idx_wo) * co +
                              idx_co];
        }
      }
    }
  }
}

void DcnBackwardWeightExecutor::computeDCNBackwardWeightCPU(
    const int &dg, const int &g, const int &im2col_step,
    const mluOpTensorDescriptor_t input_desc, const void *cpu_input,
    const mluOpTensorDescriptor_t offset_desc, const void *cpu_offset,
    const mluOpTensorDescriptor_t mask_desc, const void *cpu_mask,
    const mluOpTensorDescriptor_t grad_output_desc, const void *cpu_grad_output,
    const mluOpTensorDescriptor_t grad_weight_desc, void *cpu_grad_weight,
    const mluOpTensorDescriptor_t grad_bias_desc, void *cpu_grad_bias,
    float *buffer, int pad[], int stride[], int dilation[],
    int64_t &theory_ops) {
  const int N = input_desc->dims[0];
  const int hi = input_desc->dims[1];
  const int wi = input_desc->dims[2];
  const int ci = input_desc->dims[3];
  const int ho = offset_desc->dims[1];
  const int wo = offset_desc->dims[2];
  const int co = grad_output_desc->dims[3];
  const int kh = grad_weight_desc->dims[1];
  const int kw = grad_weight_desc->dims[2];
  const int pt = pad[0];
  const int pb = pad[1];
  const int pl = pad[2];
  const int pr = pad[3];
  const int sh = stride[0];
  const int sw = stride[1];
  const int dh = dilation[0];
  const int dw = dilation[1];

  int coeff = getCoefficientOfLT2CT();
  if (g == 1) {
    // buffer: |  columns_a  |
    for (int i = 0; i < N / im2col_step; ++i) {
      float *input_i = (float *)cpu_input + i * im2col_step * hi * wi * ci;
      float *offset_i =
          (float *)cpu_offset + i * im2col_step * ho * wo * dg * kh * kw * 2;
      float *mask_i =
          cpu_mask != nullptr
              ? (float *)cpu_mask + i * im2col_step * ho * wo * dg * kh * kw
              : nullptr;
      float *grad_output_i =
          (float *)cpu_grad_output + i * im2col_step * ho * wo * co;
      // 1.im2col
      memset(buffer, 0, (im2col_step * ho * wo * kh * kw * ci) * sizeof(float));
      im2col(N, im2col_step, dg, hi, wi, ci, ho, wo, co, kh, kw, pt, pb, pl, pr,
             sh, sw, dh, dw, input_i, offset_i, mask_i, (float *)buffer);
      theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci *
                    15;  // bilinear(14) + mask(1)

      float *input_a = grad_output_i;
      float *input_b = buffer;
      const int k = im2col_step * ho * wo;
      const int m = co;
      const int n = kh * kw * ci;
      // 2.BMM
      BatchMatMul(g, m, k, n, input_a, input_b, (float *)cpu_grad_weight, true,
                  false);
      theory_ops += 2 * (int64_t)g * m * k * n / coeff;  // lt2ct
    }
  } else {
    // |  columns_a  |  columns_b  |  grad_output  |
    float *buffer_columns_a = buffer;
    float *buffer_columns_b =
        buffer_columns_a + im2col_step * ho * wo * kh * kw * ci;
    float *buffer_grad_output =
        buffer_columns_b + im2col_step * ho * wo * kh * kw * ci;
    for (int i = 0; i < N / im2col_step; ++i) {
      float *input_i = (float *)cpu_input + i * im2col_step * hi * wi * ci;
      float *offset_i =
          (float *)cpu_offset + i * im2col_step * ho * wo * dg * kh * kw * 2;
      float *mask_i =
          cpu_mask != nullptr
              ? (float *)cpu_mask + i * im2col_step * ho * wo * dg * kh * kw
              : nullptr;
      float *grad_output_i =
          (float *)cpu_grad_output + i * im2col_step * ho * wo * co;
      // 1.im2col
      memset(buffer, 0, (im2col_step * ho * wo * kh * kw * ci) * sizeof(float));
      im2col(N, im2col_step, dg, hi, wi, ci, ho, wo, co, kh, kw, pt, pb, pl, pr,
             sh, sw, dh, dw, input_i, offset_i, mask_i, buffer_columns_a);
      theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci *
                    15;  // bilinear_count + mask

      // 2.split columns [im2col_step*ho*wo*kh*kw,g,
      // ci/g]->[g,im2col_step*ho*wo*kh*kw,ci/g]
      int dims_1[3] = {im2col_step * ho * wo * kh * kw, g, ci / g};
      int permute_1[3] = {1, 0, 2};
      transpose(buffer_columns_a, buffer_columns_b, dims_1, 3, permute_1);
      theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci;

      // 3.transpose grad_output [im2col_step*ho*wo,co]->
      // [g,co/g,im2col_step*ho*wo]
      int dims_2[2] = {im2col_step * ho * wo, co};
      int permute_2[2] = {1, 0};
      transpose(grad_output_i, buffer_grad_output, dims_2, 2, permute_2);
      theory_ops += (int64_t)im2col_step * ho * wo * co;

      float *input_a = buffer_grad_output;
      float *input_b = buffer_columns_b;
      const int k = im2col_step * ho * wo;
      const int m = co / g;
      const int n = kh * kw * ci / g;

      // 4.BMM
      BatchMatMul(g, m, k, n, input_a, input_b, (float *)cpu_grad_weight, false,
                  false);
      theory_ops += 2 * (int64_t)g * m * k * n / coeff;  // lt2ct
    }
  }
  // 5.grad_bias
  if (cpu_grad_bias) {
    dealBias((float *)cpu_grad_output, (float *)cpu_grad_bias, N, ho, wo, co);
    theory_ops += (int64_t)N * ho * wo * co;
  }
}

void DcnBackwardWeightExecutor::cpuCompute() {
  input_desc = tensor_desc_[0].tensor;
  offset_desc = tensor_desc_[1].tensor;
  cpu_input = cpu_fp32_input_[0];
  cpu_offset = cpu_fp32_input_[1];
  if (parser_->getInputNum() == 3) {
    mask_desc = nullptr;
    cpu_mask = nullptr;
    grad_output_desc = tensor_desc_[2].tensor;
    cpu_grad_output = cpu_fp32_input_[2];

    grad_weight_desc = tensor_desc_[3].tensor;
    cpu_grad_weight = cpu_fp32_output_[0];
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[4].tensor;
    cpu_grad_bias =
        parser_->getOutputNum() == 1 ? nullptr : cpu_fp32_output_[1];
  } else {
    mask_desc = tensor_desc_[2].tensor;
    cpu_mask = cpu_fp32_input_[2];
    grad_output_desc = tensor_desc_[3].tensor;
    cpu_grad_output = cpu_fp32_input_[3];
    grad_weight_desc = tensor_desc_[4].tensor;
    cpu_grad_weight = cpu_fp32_output_[0];
    grad_bias_desc =
        parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[5].tensor;
    cpu_grad_bias =
        parser_->getOutputNum() == 1 ? nullptr : cpu_fp32_output_[1];
  }

  const int ho = offset_desc->dims[1];
  const int wo = offset_desc->dims[2];
  const int kh = grad_weight_desc->dims[1];
  const int kw = grad_weight_desc->dims[2];
  const int ci = input_desc->dims[3];
  const int co = grad_output_desc->dims[3];

  size_t cpu_buffer_size = 0;
  if (g == 1) {
    cpu_buffer_size =
        (static_cast<size_t>(im2col_step) * ho * wo * kh * kw * ci) *
        sizeof(float);
  } else {
    cpu_buffer_size = (2lu * im2col_step * ho * wo * kh * kw * ci +
                       im2col_step * ho * wo * co) *
                      sizeof(float);
  }

  float *buffer = nullptr;
  buffer = (float *)cpu_runtime_.allocate(cpu_buffer_size);
  if (buffer == nullptr) {
    LOG(ERROR) << "dcn_backward_weight: allocate buffer failed.";
  }
  if (cpu_grad_weight) {
    memset(cpu_grad_weight, 0, co * kh * kw * ci / g * sizeof(float));
  }
  if (cpu_grad_bias) {
    memset(cpu_grad_bias, 0, co * sizeof(float));
  }
  theory_ops = 0;
  computeDCNBackwardWeightCPU(
      dg, g, im2col_step, input_desc, cpu_input, offset_desc, cpu_offset,
      mask_desc, cpu_mask, grad_output_desc, cpu_grad_output, grad_weight_desc,
      cpu_grad_weight, grad_bias_desc, cpu_grad_bias, buffer, pad, stride,
      dilation, theory_ops);

  cpu_runtime_.deallocate(buffer);
}

int64_t DcnBackwardWeightExecutor::getTheoryOps() {
  if (exe_config_->mlu_only) {
    theory_ops = 0;
    input_desc = tensor_desc_[0].tensor;
    offset_desc = tensor_desc_[1].tensor;
    if (parser_->getInputNum() == 3) {
      grad_output_desc = tensor_desc_[2].tensor;
      grad_weight_desc = tensor_desc_[3].tensor;
      grad_bias_desc =
          parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[4].tensor;
    } else {
      grad_output_desc = tensor_desc_[3].tensor;
      grad_weight_desc = tensor_desc_[4].tensor;
      grad_bias_desc =
          parser_->getOutputNum() == 1 ? nullptr : tensor_desc_[5].tensor;
    }
    const int N = input_desc->dims[0];
    const int hi = input_desc->dims[1];
    const int wi = input_desc->dims[2];
    const int ci = input_desc->dims[3];
    const int ho = offset_desc->dims[1];
    const int wo = offset_desc->dims[2];
    const int co = grad_output_desc->dims[3];
    const int kh = grad_weight_desc->dims[1];
    const int kw = grad_weight_desc->dims[2];
    int coeff = getCoefficientOfLT2CT();
    const int k = im2col_step * ho * wo;
    const int m = co / g;
    const int n = kh * kw * ci / g;
    if (g == 1) {
      for (int i = 0; i < N / im2col_step; ++i) {
        theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci *
                      15;  // bilinear(14) + mask(1)
        theory_ops += 2 * (int64_t)g * m * k * n / coeff;  // lt2ct
      }
    } else {
      for (int i = 0; i < N / im2col_step; ++i) {
        theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci *
                      15;  // bilinear_count + mask
        theory_ops += (int64_t)im2col_step * ho * wo * kh * kw * ci;
        theory_ops += (int64_t)im2col_step * ho * wo * co;
        theory_ops += 2 * (int64_t)g * m * k * n / coeff;  // lt2ct
      }
    }
    if (grad_bias_desc) {
      theory_ops += (int64_t)N * ho * wo * co;
    }
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
