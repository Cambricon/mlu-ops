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
#define USE_OPENBLAS 0
#if USE_OPENBLAS
#include <openblas/cblas.h>
#endif
#include "matmul.h"
/*
 * for alpha beta test
 * alpha will be calculated always.
 * if use_beta == false, beta must be 0.0f. D = alpha * A * B,
 * two inputs and one output.
 * if use_beta == true, D = alpha * A * B + beta * C, three inputs and one output.
 * When beta == 0.0f, beta * C will not be calculated.
 */
namespace mluoptest {
void MatmulExecutor::paramCheck() {
  // set flag_quant_mode_ according to quant_mode param
  QuantizeMode quant_mode =
      parser_->getProtoNode()->matmul_param().quant_mode();
  switch (quant_mode) {
    case QUANTIZE_POSITION:
      flag_quant_mode_ = ONLY_POSITION;
      break;
    case QUANTIZE_POSITION_SCALE:
      flag_quant_mode_ = POSITION_SCALE;
      break;
    case QUANTIZE_POSITION_SCALE_OFFSET:
      flag_quant_mode_ = POS_SCALE_OFFSET;
      break;
    default:
      LOG(ERROR) << "MatmulExecutor: matmul param quant_mode set fail.";
  }
}

#if USE_MLUOP_MATMUL_V2
void MatmulExecutor::workspaceMalloc() {
  /////////// set matmul_desc //////////
  auto matmul_param = parser_->getProtoNode()->matmul_param();
  int32_t is_trans_a = matmul_param.is_trans_a();
  int32_t is_trans_b = matmul_param.is_trans_b();
  int32_t use_tf32 = matmul_param.allow_tf32();
  bool use_beta = matmul_param.use_beta();
  atomics_mode_ = handle_->atomics_mode;
  // overwrite atomics mode
  bool matmul_atomics_allowed = matmul_param.atomics_allowed();
  if (matmul_atomics_allowed) {
    mluOpSetAtomicsMode(handle_, MLUOP_ATOMICS_ALLOWED);
  } else {
    mluOpSetAtomicsMode(handle_, MLUOP_ATOMICS_NOT_ALLOWED);
  }
  matmul_desc_ =
      cpu_runtime_.allocate(mluOpMatMulDescCreate, mluOpMatMulDescDestroy);
  MLUOP_CHECK(
      mluOpSetMatMulDescAttr(matmul_desc_, MLUOP_MATMUL_DESC_TRANSA,
                             &(is_trans_a), sizeof(int32_t)));
  MLUOP_CHECK(
      mluOpSetMatMulDescAttr(matmul_desc_, MLUOP_MATMUL_DESC_TRANSB,
                             &(is_trans_b), sizeof(int32_t)));
  MLUOP_CHECK(
      mluOpSetMatMulDescAttr(matmul_desc_, MLUOP_MATMUL_ALLOW_TF32,
                             &(use_tf32), sizeof(int32_t)));
  MLUOP_CHECK(mluOpSetMatMulDescAttr(matmul_desc_, MLUOP_MATMUL_USE_BETA,
                                     &(use_beta), sizeof(bool)));
  /////////// end matmul_desc ///////////
  auto a_desc_ = tensor_desc_[0].tensor;
  auto b_desc_ = tensor_desc_[1].tensor;
  auto c_desc_ = tensor_desc_.back().tensor;
  auto d_desc_ = tensor_desc_.back().tensor;
  if (use_beta) {
    c_desc_ = tensor_desc_[2].tensor;
  }
  /////////// set matmul algo and heuristic result ///////////
  algo_ = cpu_runtime_.allocate(mluOpMatMulAlgoCreate, mluOpMatMulAlgoDestroy);
  heuristic_result_ =
      cpu_runtime_.allocate(mluOpCreateMatMulHeuristicResult,
                            mluOpDestroyMatMulHeuristicResult);
  // mluOpMatMulPrefer_t prefer;
  // not supported now
  int requested_algo_count = 1, return_algo_count = 0;
  mluOpGetMatMulAlgoHeuristic(handle_, matmul_desc_, a_desc_, b_desc_, c_desc_,
      d_desc_, nullptr /* prefer */, requested_algo_count, &heuristic_result_,
      &return_algo_count);
  mluOpGetMatMulHeuristicResult(heuristic_result_, algo_, &workspace_size_);
  /////////// end matmul algo and heuristic result ///////////
  if (workspace_size_ > 0) {
    VLOG(4) << "MatmulExecutor: Malloc workspace space.";
    workspace_ = (void *)mlu_runtime_.allocate(workspace_size_);
    VLOG(4) << "MatmulExecutor: workspace_size: " << workspace_size_;
  } else {
    VLOG(4) << "MatmulExecutor: Don't need to malloc workspace size.";
  }
  eva_->setMluWorkspaceSize(workspace_size_);
}

void MatmulExecutor::workspaceFree() {
  if (workspace_ != nullptr) {
    VLOG(4) << "MatmulExecutor: Free workspace memory.";
    mlu_runtime_.deallocate(workspace_);
  }
}
#endif

void MatmulExecutor::setQuantizedParam() {
  auto input_num = parser_->getInputNum();
  for (int i = 0; i < input_num; ++i) {
    if (!parser_->getMetaTensor(i).is_null) {
      auto pos = parser_->getMetaTensor(i).position;
      auto scale = parser_->getMetaTensor(i).scale;
      auto offset = parser_->getMetaTensor(i).offset;
      MLUOP_CHECK(
          mluOpSetTensorDescriptorPositionScaleAndOffset(
              tensor_desc_[i].tensor, pos, scale, offset));
    }
  }
}

inline bool isFloatData(mluOpDataType_t type) {
  return (MLUOP_DTYPE_HALF == type || MLUOP_DTYPE_FLOAT == type);
}

void MatmulExecutor::castIn() {
  auto input_num = parser_->getInputNum();
  for (int i = 0; i < input_num; ++i) {
    if (parser_->getMetaTensor(i).total_count == 0) {
      continue;  // skip if ptr is nullptr
    }
    // fp32->X
    auto dtype = parser_->getInputDataType(i);
    auto oc_dt = parser_->getInputOnchipDataType(i);
    auto count = parser_->getInputDataCount(i);
    auto src_data = cpu_fp32_input_[i];
    auto dst_data = data_vector_[i].host_ptr;
    // use quantize param in prototxt
    auto input_node = parser_->getProtoNode()->input(i);
    bool online_quantize =
        !(input_node.has_position() &&
          parser_->input(i)->value_type != VALUE_RANDOM);
    int p;
    int o;
    float s;
    mluOpTensorDescriptor_t a_desc = tensor_desc_[0].tensor;
    mluOpTensorDescriptor_t b_desc = tensor_desc_[1].tensor;
    if (isFloatData(a_desc->dtype) && isFloatData(b_desc->dtype) &&
        (isFloatData(a_desc->onchip_dtype) ||
         a_desc->onchip_dtype == MLUOP_DTYPE_INVALID) &&
        (isFloatData(b_desc->onchip_dtype) ||
         b_desc->onchip_dtype == MLUOP_DTYPE_INVALID)) {
      p = 0;
      o = 0;
      s = 1.0f;
    } else {
      p = input_node.has_position() ? input_node.position() : 0;
      o = input_node.has_offset() ? input_node.offset() : 0;
      s = input_node.has_scale() ? input_node.scale() : 1.0f;
    }    // weight(matrix B) do not need asymmetric quant
    auto quant_mode = i == 1 ? POSITION_SCALE : flag_quant_mode_;
    if (oc_dt == MLUOP_DTYPE_INVALID || oc_dt == dtype) {
      castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, count,
                 quant_mode, &p, &s, &o, true, online_quantize);
      MLUOP_CHECK(
          mluOpSetTensorDescriptorPositionScaleAndOffset(
              tensor_desc_[i].tensor, p, s, o));
    } else {
      // if has onchip_dtype
      // cast fp32 to onchip dtype to get p/s/o and dequantify fp32 data
      // (let cpu input == mlu input)
      // and cast fp32 to offchip dtype then memcpy this to mlu
      castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, count,
                 quant_mode, &p, &s, &o, true, online_quantize);
      // get oc_dt's p/s and set to tensor.
      void *temp =
          cpu_runtime_.allocate(count * mluop::getSizeOfDataType(oc_dt));
      castDataIn(src_data, MLUOP_DTYPE_FLOAT, temp, oc_dt, count, quant_mode,
                 &p, &s, &o, true, online_quantize);
      MLUOP_CHECK(
          mluOpSetTensorDescriptorPositionScaleAndOffset(
              tensor_desc_[i].tensor, p, s, o));
      if (oc_dt == MLUOP_DTYPE_INT16) {
        forceFixToFloat((int16_t *)temp, src_data, count);
      } else if (oc_dt == MLUOP_DTYPE_INT8) {
        forceFixToFloat((int8_t *)temp, src_data, count);
      }
      cpu_runtime_.deallocate(temp);
    }
    MetaTensor *ts = &(parser_->getMetaTensor(i));
    if (!ts->stride.empty()) {
      VLOG(4) << "[WARNING] Executor: " << ts->name
              << " host ptr been strided_out.";
      void *temp = cpu_runtime_.allocate(ts->shape_count * sizeof(float));
      memset(temp, 0x0, ts->shape_count * sizeof(float));
      tensor_stride_in(temp, cpu_fp32_input_[i], getTensorShapeSizeT(ts),
                       getTensorStrideSizeT(ts), sizeof(float));
      cpu_runtime_.deallocate(cpu_fp32_input_[i]);
      cpu_fp32_input_[i] = (float *)temp;
      ts->cpu_ptr = (float *)temp;
    }
#if GTEST_DEBUG_ENABLE
    if (exe_config_->dump_data) {
      saveDataToFile("baseline_strided_" + ts->name,
                     cpu_fp32_input_[i], ts->shape_count);
    }
#endif
  }
}

void MatmulExecutor::compute() {
  VLOG(4) << "MatmulExecutor: compute...";
  float alpha = parser_->getProtoNode()->matmul_param().alpha();
  float beta = parser_->getProtoNode()->matmul_param().beta();
#if USE_MLUOP_MATMUL_V2
  VLOG(4) << "MatmulExecutor: call mluOpMatMul_v2";
  bool use_beta = parser_->getProtoNode()->matmul_param().use_beta();
  VLOG(5) << "MatmulExecutor: use_beta: " << use_beta;
  auto a_desc = tensor_desc_[0].tensor;
  auto b_desc = tensor_desc_[1].tensor;
  auto c_desc = tensor_desc_.back().tensor;
  auto d_desc = tensor_desc_.back().tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_b = data_vector_[1].device_ptr;
  auto dev_c = data_vector_.back().device_ptr;
  auto dev_d = data_vector_.back().device_ptr;
  if (use_beta) {
    c_desc = tensor_desc_[2].tensor;
    dev_c = data_vector_[2].device_ptr;
  }
  interface_timer_.start();
  MLUOP_CHECK(mluOpMatMul_v2(handle_, matmul_desc_, algo_, &alpha, a_desc,
                             dev_a, b_desc, dev_b, &beta, c_desc, dev_c,
                             workspace_, workspace_size_, d_desc, dev_d));
  interface_timer_.stop();
#else
  VLOG(4) << "MatmulExecutor: call mluOpMatMul";
  bool is_trans_a = parser_->getProtoNode()->matmul_param().is_trans_a();
  bool is_trans_b = parser_->getProtoNode()->matmul_param().is_trans_b();
  auto a_desc = tensor_desc_[0].tensor;
  auto b_desc = tensor_desc_[1].tensor;
  auto c_desc = tensor_desc_.back().tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_b = data_vector_[1].device_ptr;
  auto dev_c = data_vector_.back().device_ptr;
  alpha = 1.0;
  beta = 0.0;
  interface_timer_.start();
  MLUOP_CHECK(mluOpMatMul(handle_, is_trans_a, is_trans_b, &alpha, a_desc,
                          dev_a, b_desc, dev_b, &beta, c_desc, dev_c));
  interface_timer_.stop();
#endif
  // reset atomics mode
  mluOpSetAtomicsMode(handle_, atomics_mode_);
  VLOG(4) << "MatmulExecutor: compute end";
}

void MatmulExecutor::cpuCompute() {
  VLOG(4) << "MatmulExecutor: cpuCompute...";
  bool is_trans_a = parser_->getProtoNode()->matmul_param().is_trans_a();
  bool is_trans_b = parser_->getProtoNode()->matmul_param().is_trans_b();
  bool use_beta = parser_->getProtoNode()->matmul_param().use_beta();
  float alpha = parser_->getProtoNode()->matmul_param().alpha();
  float beta = parser_->getProtoNode()->matmul_param().beta();
  int K = is_trans_a ? (int)parser_->getProtoNode()->input(0).shape().dims(0)
                     : (int)parser_->getProtoNode()->input(0).shape().dims(1);
  int M = (int)parser_->getProtoNode()->output(0).shape().dims(0);
  int N = (int)parser_->getProtoNode()->output(0).shape().dims(1);
#if USE_OPENBLAS
  const CBLAS_ORDER Order = CblasRowMajor;
  const CBLAS_TRANSPOSE TransA = is_trans_a ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = is_trans_b ? CblasTrans : CblasNoTrans;
  int lda = is_trans_a ? M : K;
  int ldb = is_trans_b ? K : N;
  int ldc = N;
#endif
#if USE_OPENBLAS
  if (use_beta && beta != 0.0f) {
    float *c_temp = (float *)cpu_runtime_.allocate(M * N * sizeof(float));
    memcpy(c_temp, cpu_fp32_input_[2], M * N * sizeof(float));
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha,
                cpu_fp32_input_[0], lda, cpu_fp32_input_[1], ldb,
                beta, c_temp, ldc);
    memcpy(cpu_fp32_output_[0], c_temp, M * N * sizeof(float));
    cpu_runtime_.deallocate(c_temp);
    c_temp = NULL;
  } else {
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha,
                cpu_fp32_input_[0], lda, cpu_fp32_input_[1], ldb,
                0.0, cpu_fp32_output_[0], ldc);
  }
#else
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float matrix_result = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        int64_t lhs_idx = m * K + k;
        if (is_trans_a)
          lhs_idx = k * M + m;
        int64_t rhs_idx = k * N + n;
        if (is_trans_b)
          rhs_idx = n * K + k;
        matrix_result += cpu_fp32_input_[0][lhs_idx] *
                         cpu_fp32_input_[1][rhs_idx];
      }
      if (use_beta && beta != 0.0f) {
        cpu_fp32_output_[0][m * N + n] =
            alpha * matrix_result + beta * cpu_fp32_input_[2][m * N + n];
      } else {
        cpu_fp32_output_[0][m * N + n] = alpha * matrix_result;
      }
    }
  }
#endif
  VLOG(4) << "MatmulExecutor: cpuCompute end";
}

int64_t MatmulExecutor::getTheoryOps() {
  bool is_trans_a = parser_->getProtoNode()->matmul_param().is_trans_a();
  bool is_trans_b = parser_->getProtoNode()->matmul_param().is_trans_b();
  float alpha = parser_->getProtoNode()->matmul_param().alpha();
  float beta = parser_->getProtoNode()->matmul_param().beta();
  int64_t K = is_trans_a ?
        (int)parser_->getProtoNode()->input(0).shape().dims(0) :
        (int)parser_->getProtoNode()->input(0).shape().dims(1);
  int64_t M = (int)parser_->getProtoNode()->output(0).shape().dims(0);
  int64_t N = (int)parser_->getProtoNode()->output(0).shape().dims(1);
  int64_t theory_ops = M * K * N * 2;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
