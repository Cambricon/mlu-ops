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
#define USE_OPENBLAS 0
#include "batch_matmul_bcast.h"
#if USE_OPENBLAS
#include <openblas/cblas.h>
#endif

namespace mluoptest {

bool BatchMatmulBcastExecutor::canBroadCast(std::vector<int> shape0, std::vector<int> shape1) {
  int64_t ndim = shape1.size();
  int64_t tensor_dim = shape0.size();
  if (tensor_dim == 0)
    return false;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    int64_t size = (dim >= 0) ? shape0[dim] : 1;
    if (shape1[i] == -1)
      shape1[i] = size;
    if (shape1[i] != size)
      if (size != 1)
        return false;
  }
  return true;
}

void BatchMatmulBcastExecutor::paramCheck() {
  // set flag_quant_mode_ according to quant_mode param
  QuantizeMode quant_mode = parser_->getProtoNode()->batch_matmul_bcast_param().quant_mode();
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();

  cast_mode_ = parser_->getProtoNode()->batch_matmul_bcast_param().cast_mode();
  if (cast_mode_ != MLUOP_MATMUL_BYPASS_QUANTIZE) {
    switch (cast_mode_) {
      case MLUOP_MATMUL_OFFLINE_SYMMETRIC_QUANTIZE:
        flag_quant_mode_ = POSITION_SCALE;
        break;
      case MLUOP_MATMUL_OFFLINE_ASYMMETRIC_QUANTIZE:
        flag_quant_mode_ = POS_SCALE_OFFSET;
        break;
      case MLUOP_MATMUL_NO_QUANTIZE:
        flag_quant_mode_ = NO_QUANT;
        break;
      default:
        LOG(ERROR) << "BatchMatmulBcastExecutor: batch_matmul_bcast cast_mode not set. ";
    }
  } else {
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
        LOG(ERROR) << "BatchMatmulBcastExecutor: batch_matmul_bcast param quant_mode not set.";
    }
  }

  if (!parser_->getProtoNode()->has_batch_matmul_bcast_param()) {
    LOG(ERROR) << "Lose batch_matmul_bcast param. ";
  }

  if (beta == 0.0f && parser_->getInputNum() != 2) {
    LOG(ERROR) << "batch_matmul_bcast tensor input number is wrong. "
               << "when value of beta is " << beta << ", two inputs are required";
  }

  if (beta != 0.0f && parser_->getInputNum() != 3) {
    LOG(ERROR) << "batch_matmul_bcast tensor input number is wrong. "
               << "when value of beta is " << beta << ", three inputs are required";
  }

  if (parser_->getInputDimSize(0) < 2) {
    LOG(ERROR) << "batch_matmul_bcast tensor input1 size at lest two dimensions. ";
  }

  if (parser_->getInputDimSize(1) < 2) {
    LOG(ERROR) << "batch_matmul_bcast tensor input2 size at lest two dimensions. ";
  }

  bool is_transa = parser_->getProtoNode()->batch_matmul_bcast_param().is_transa();
  bool is_transb = parser_->getProtoNode()->batch_matmul_bcast_param().is_transb();
  bool use_stride = parser_->getProtoNode()->batch_matmul_bcast_param().use_stride();

  int dim_max_a = parser_->getInputDimSize(0);
  int dim_max_b = parser_->getInputDimSize(1);
  int input1_col = (is_transa && !use_stride)
                       ? parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 2)
                       : parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 1);
  int input2_row = (is_transb && !use_stride)
                       ? parser_->getProtoNode()->input(1).shape().dims(dim_max_b - 1)
                       : parser_->getProtoNode()->input(1).shape().dims(dim_max_b - 2);

  if (input1_col != input2_row) {
    LOG(ERROR) << "batch_matmul_bcast input1 cols is not equal to input2 rows. ";
  }

  flag_input_reuse_ = (beta != 0.0f) ? true : false;  // reuse void *c if need bias
  VLOG(4) << "flag_input_reuse_: " << flag_input_reuse_;
}

void BatchMatmulBcastExecutor::setQuantizedParam() {
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  auto input_num = (beta == 0.0f) ? parser_->getInputNum() : parser_->getInputNum() - 1;
  auto output_num = parser_->getOutputNum();
  auto total_num = input_num + output_num;
  for (int i = 0; i < total_num; ++i) {
    if (!parser_->getMetaTensor(i).is_null) {
      auto pos = parser_->getMetaTensor(i).position;
      auto scale = parser_->getMetaTensor(i).scale;
      auto offset = parser_->getMetaTensor(i).offset;
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleAndOffset(tensor_desc_[i].tensor, pos, scale,
                                                               offset));
    }
  }
}

void BatchMatmulBcastExecutor::castIn() {
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  auto input_num = parser_->getInputNum();
  for (int i = 0; i < input_num; ++i) {
    if (parser_->getMetaTensor(i).total_count == 0) {
      continue;  // skip if ptr is nullptr
    }

    // fp32->X
    auto dtype = parser_->getInputDataType(i);
    auto oc_dt = parser_->getInputOnchipDataType(i);
    auto src_data = cpu_fp32_input_[i];
    auto dst_data = data_vector_[i].host_ptr;

    int p = 0, o = 0;
    float s = 1.0f;

    MetaTensor *ts = &(parser_->getMetaTensor(i));

    if (oc_dt == MLUOP_DTYPE_INVALID || oc_dt == dtype) {
      // no onchip p/s/o
      // just cast data from fp32 to dtype
      // then memcpy this to mlu
      // weight(matrix B) do not need asymmetric quant
      if (i == 1) {
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, ts->total_count, POSITION_SCALE, &p,
                   &s, &o, true);
      } else {
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, ts->total_count, flag_quant_mode_,
                   &p, &s, &o, true);
      }
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleAndOffset(tensor_desc_[i].tensor, p, s, o));
    } else {
      if (i == 1) {
        // if has onchip_dtype
        // cast fp32 to onchip dtype to get p/s/o and dequantify fp32 data (let cpu input == mlu
        // input)
        // and cast fp32 to offchip dtype then memcpy this to mlu
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, ts->total_count, POSITION_SCALE, &p,
                   &s, &o, true);
      } else {
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, dst_data, dtype, ts->total_count, flag_quant_mode_,
                   &p, &s, &o, true);
      }

      // get oc_dt's p/s and set to tensor.
      void *temp = cpu_runtime_.allocate(ts->total_count * mluop::getSizeOfDataType(oc_dt));
      if (i == 1) {
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, temp, oc_dt, ts->total_count, POSITION_SCALE, &p, &s,
                   &o, true);
      } else {
        castDataIn(src_data, MLUOP_DTYPE_FLOAT, temp, oc_dt, ts->total_count, flag_quant_mode_, &p,
                   &s, &o, true);
      }
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleAndOffset(tensor_desc_[i].tensor, p, s, o));
      cpu_runtime_.deallocate(temp);
    }

    if (!ts->stride.empty()) {
      VLOG(4) << "[WARNING] Executor: " << ts->name << " host ptr been strided_out.";
      void *temp = cpu_runtime_.allocate(ts->shape_count * sizeof(float));
      memset(temp, 0x0, ts->shape_count * sizeof(float));
      tensor_stride_in(temp, cpu_fp32_input_[i], getTensorShapeSizeT(ts), getTensorStrideSizeT(ts),
                       sizeof(float));
      cpu_runtime_.deallocate(cpu_fp32_input_[i]);
      cpu_fp32_input_[i] = (float *)temp;
      ts->cpu_ptr = (float *)temp;
    }

#if GTEST_DEBUG_ENABLE
    if (exe_config_->dump_data) {
      saveDataToFile("baseline_strided_" + ts->name, cpu_fp32_input_[i], ts->shape_count);
    }
#endif
  }
}

void BatchMatmulBcastExecutor::castOut() {
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  auto input_num = (beta == 0.0f) ? parser_->getInputNum() : parser_->getInputNum() - 1;
  auto output_num = parser_->getOutputNum();
  auto total_num = input_num + output_num;
  auto dtype = parser_->getOutputDataType(0);
  for (int i = input_num; i < total_num; ++i) {
    // fp32 -> X
    auto dtype = parser_->getMetaTensor(i).dtype;
    auto count = parser_->getMetaTensor(i).total_count;
    auto src_data = parser_->getMetaTensor(i).host_ptr;
    auto dst_data = mlu_fp32_output_[i - input_num];
    if ((parser_->device() != CPU) && (dtype == MLUOP_DTYPE_INT8 || dtype == MLUOP_DTYPE_INT16))
      return;
    castDataOut(src_data, dtype, dst_data, MLUOP_DTYPE_FLOAT, count, flag_quant_mode_, pos_, scale_,
                offset_);
  }
}

void BatchMatmulBcastExecutor::workspaceMalloc() {
  int32_t is_transa = parser_->getProtoNode()->batch_matmul_bcast_param().is_transa();
  int32_t is_transb = parser_->getProtoNode()->batch_matmul_bcast_param().is_transb();
  int32_t use_tf32 = parser_->getProtoNode()->batch_matmul_bcast_param().allow_tf32();
  int32_t use_stride = parser_->getProtoNode()->batch_matmul_bcast_param().use_stride();

  bmm_bcast_desc_ = cpu_runtime_.allocate(mluOpMatMulDescCreate, mluOpMatMulDescDestroy);
  MLUOP_CHECK(mluOpSetMatMulDescAttr(bmm_bcast_desc_, MLUOP_MATMUL_DESC_TRANSA, &(is_transa),
                                   sizeof(int32_t)));
  MLUOP_CHECK(mluOpSetMatMulDescAttr(bmm_bcast_desc_, MLUOP_MATMUL_DESC_TRANSB, &(is_transb),
                                   sizeof(int32_t)));
  MLUOP_CHECK(mluOpSetMatMulDescAttr(bmm_bcast_desc_, MLUOP_MATMUL_CAST_MODE, &(cast_mode_),
                                   sizeof(cast_mode_)));
  MLUOP_CHECK(
      mluOpSetMatMulDescAttr(bmm_bcast_desc_, MLUOP_MATMUL_ALLOW_TF32, &(use_tf32), sizeof(int32_t)));
  MLUOP_CHECK(mluOpSetMatMulDescAttr(bmm_bcast_desc_, MLUOP_MATMUL_USE_STRIDE, &(use_stride),
                                   sizeof(int32_t)));

  auto desc_a = tensor_desc_[0].tensor;
  auto desc_b = tensor_desc_[1].tensor;
  auto desc_c = tensor_desc_[2].tensor;

  if (cast_mode_ != MLUOP_MATMUL_BYPASS_QUANTIZE) {
    switch (cast_mode_) {
      case MLUOP_MATMUL_OFFLINE_SYMMETRIC_QUANTIZE:
        flag_quant_mode_ = POSITION_SCALE;
        break;
      case MLUOP_MATMUL_OFFLINE_ASYMMETRIC_QUANTIZE:
        flag_quant_mode_ = POS_SCALE_OFFSET;
        break;
      case MLUOP_MATMUL_NO_QUANTIZE:
        flag_quant_mode_ = NO_QUANT;
        break;
      default:
        LOG(ERROR) << "BatchMatmulBcastExecutor: batch_matmul_bcast param cast_mode not set.";
    }
    cpuComputeForCastOutput();
  } else {
    cpuComputeForCastOutput();
    if (parser_->device() != CPU) {
      auto c_offset = parser_->getMetaTensor(2).offset;
      if (c_offset != 0) {
        flag_quant_mode_ = POS_SCALE_OFFSET;
      }
    }
  }

  void *workspace = NULL;
  algo_ = cpu_runtime_.allocate(mluOpMatMulAlgoCreate, mluOpMatMulAlgoDestroy);

#if 0
  MLUOP_CHECK(
      mluOpGetBatchMatMulBCastWorkspaceSize(handle_, desc_a, desc_b, desc_c, &workspace_size_));
#else
  heuristic_result_ =
      cpu_runtime_.allocate(mluOpCreateMatMulHeuristicResult, mluOpDestroyMatMulHeuristicResult);
  // mluOpMatMulPrefer_t prefer; // not supported now
  int requested_algo_count = 1, return_algo_count = 0;
  mluOpGetBatchMatMulAlgoHeuristic(handle_, bmm_bcast_desc_, desc_a, desc_b, desc_c,
                                  nullptr /* prefer */, requested_algo_count, &heuristic_result_,
                                  &return_algo_count);
  mluOpGetBatchMatMulHeuristicResult(heuristic_result_, algo_, &workspace_size_);
/////////// end batch_matmul algo and heuristic result ///////////
#endif

  if (workspace_size_ > 0) {
    workspace = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(workspace);

  eva_->setMluWorkspaceSize(workspace_size_);
}

void BatchMatmulBcastExecutor::workspaceFree() {
  if (workspace_[0]) {
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

void BatchMatmulBcastExecutor::cpuComputeForCastOutput() {
  if (exe_config_->mlu_only) {
    return;
  }
  // ----------------------- baselineOutputMalloc begin ------------------------
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  auto input_num = (beta == 0.0f) ? parser_->getInputNum() : parser_->getInputNum() - 1;
  auto output_num = parser_->getOutputNum();
  auto total_num = input_num + output_num;
  // for outputs
  for (int i = input_num; i < total_num; ++i) {
    if (!parser_->getMetaTensor(i).is_null) {
      auto count = parser_->getMetaTensor(i).total_count;
      auto name = parser_->getMetaTensor(i).name;
      parser_->getMetaTensor(i).cpu_ptr =
          (float *)cpu_runtime_.allocate(count * sizeof(float), name);
      memset(parser_->getMetaTensor(i).cpu_ptr, 0, count * sizeof(float));
    }
    cpu_fp32_output_.push_back(parser_->getMetaTensor(i).cpu_ptr);
  }
  // ----------------------- baselineOutputMalloc end --------------------------

  if (parser_->device() != CPU) {
    // set pos and scale for output
    float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
    auto input_num = (beta == 0.0f) ? parser_->getInputNum() : parser_->getInputNum() - 1;
    auto output_num = parser_->getOutputNum();
    auto total_num = input_num + output_num;
    for (int i = input_num; i < total_num; ++i) {
      if (parser_->getMetaTensor(i).is_null == false) {
        auto pos = parser_->getMetaTensor(i).position;
        auto scale = parser_->getMetaTensor(i).scale;
        auto offset = parser_->getMetaTensor(i).offset;
        MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleAndOffset(tensor_desc_[i].tensor, pos, scale,
                                                                 offset));
      }
    }
    setOutputQuantParam();
    return;
  }
}

void BatchMatmulBcastExecutor::setOutputQuantParam() {
  auto dtype = parser_->getOutputDataType(0);
  auto count = parser_->getOutputDataCount(0);
  if (dtype == MLUOP_DTYPE_INT8 || dtype == MLUOP_DTYPE_INT16 || dtype == MLUOP_DTYPE_INT31) {
    getQuantizedParam(cpu_fp32_output_[0], count, dtype, flag_quant_mode_, &pos_, &scale_,
                      &offset_);
    MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleAndOffset(tensor_desc_.back().tensor, pos_,
                                                             scale_, offset_));
  }
}

void BatchMatmulBcastExecutor::compute() {
  VLOG(4) << "BatchMatmulBcastExecutor compute ";
  if (!parser_->getProtoNode()->has_batch_matmul_bcast_param()) {
    LOG(ERROR) << "Lose batch_matmul_bcast param. ";
  }
  int32_t is_transa = parser_->getProtoNode()->batch_matmul_bcast_param().is_transa();
  int32_t is_transb = parser_->getProtoNode()->batch_matmul_bcast_param().is_transb();

  float alpha = parser_->getProtoNode()->batch_matmul_bcast_param().alpha();
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();

  auto tensor_a = tensor_desc_[0].tensor;
  auto tensor_b = tensor_desc_[1].tensor;
  auto tensor_c = tensor_desc_[2].tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_b = data_vector_[1].device_ptr;
  auto dev_c = data_vector_[2].device_ptr;
  VLOG(4) << "call mluOp BatchMatMulBCast()";
  interface_timer_.start();
#if 0
  MLUOP_CHECK(mluOpBatchMatMulBCast(handle_, is_transa, is_transb, tensor_a, dev_a, tensor_b, dev_b,
                                  workspace_[0], workspace_size_, tensor_c, dev_c));
#else
  MLUOP_CHECK(mluOpBatchMatMulBCast_v2(handle_, bmm_bcast_desc_, algo_, &alpha, tensor_a, dev_a,
                                     tensor_b, dev_b, &beta, tensor_c, dev_c, workspace_[0],
                                     workspace_size_));
#endif
  interface_timer_.stop();
}

void BatchMatmulBcastExecutor::setMiscellaneousParam() {
  data_vector_[2].setDramTensorType(DramTensorType::BOTH_INPUT_OUTPUT);
}


void BatchMatmulBcastExecutor::float2double(double *dst, float *src, int64_t num) {
  for (int64_t i = 0; i < num; ++i) {
    dst[i] = (double)src[i];
  }
}

void BatchMatmulBcastExecutor::qdouble2float(float *dst, double *src, double scale, int64_t num) {
  for (int64_t i = 0; i < num; ++i) {
    dst[i] = static_cast<float>(scale * src[i]);
  }
}

void BatchMatmulBcastExecutor::int2double(double *dst,
                                          void *src,
                                          mluOpDataType_t src_type,
                                          int64_t num) {
  if (src_type == MLUOP_DTYPE_INT8) {
    int8_t *csrc = (int8_t *)src;
    for (int64_t i = 0; i < num; ++i) {
      dst[i] = static_cast<double>(csrc[i]);
    }
  } else if (src_type == MLUOP_DTYPE_INT16) {
    float *sdst = static_cast<float *>(cpu_runtime_.allocate(num * sizeof(float)));
    cnrtQuantizedParam_t param;
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCreateQuantizedParam(&param, 0, 1.0f, 0.0f));
    GTEST_CHECK(CNRT_RET_SUCCESS ==
                cnrtCastDataType_V2(src, cnrtShort, sdst, cnrtFloat, num, param, cnrtRounding_rm));
    for (int64_t i = 0; i < num; ++i) {
      dst[i] = static_cast<double>(sdst[i]);
    }
    cpu_runtime_.deallocate(sdst);
    cnrtDestroyQuantizedParam(param);
  }
}

void BatchMatmulBcastExecutor::cpuCompute() {
  float alpha = parser_->getProtoNode()->batch_matmul_bcast_param().alpha();
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  if (beta == 0.0f) {
    assert(parser_->getInputNum() == 2);
  } else {
    assert(parser_->getInputNum() == 3);
  }
  assert(parser_->getOutputNum() == 1);
  bool is_transa = parser_->getProtoNode()->batch_matmul_bcast_param().is_transa();
  bool is_transb = parser_->getProtoNode()->batch_matmul_bcast_param().is_transb();
  bool use_stride = parser_->getProtoNode()->batch_matmul_bcast_param().use_stride();
  is_transa = use_stride ? false : is_transa;
  is_transb = use_stride ? false : is_transb;

  auto desc_a = tensor_desc_[0].tensor;
  auto desc_b = tensor_desc_[1].tensor;
  auto desc_c = tensor_desc_[2].tensor;

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getInputDataCount(1);
  if (count1 == 0 || count2 == 0) {
    return;
  }

  int batch_size = 1;
  for (int i = 0; i < desc_c->dim - 2; ++i) {
    batch_size *= desc_c->dims[i];
  }
  assert(batch_size >= 1);

  int dim_max_a = parser_->getInputDimSize(0);
  int dim_max_c = parser_->getOutputDimSize(0);
  int k = is_transa ? (int)parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 2)
                    : (int)parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 1);

  int m = (int)parser_->getProtoNode()->output(0).shape().dims(dim_max_c - 2);
  int n = (int)parser_->getProtoNode()->output(0).shape().dims(dim_max_c - 1);

  std::vector<int> shape_expand_a, shape_expand_b;

  for (int i = 0; i < desc_c->dim - 2; ++i) {
    shape_expand_a.push_back((int)desc_c->dims[i]);
    shape_expand_b.push_back((int)desc_c->dims[i]);
  }

  for (int i = desc_a->dim - 2; i < desc_a->dim; ++i) {
    shape_expand_a.push_back((int)desc_a->dims[i]);
  }

  for (int i = desc_b->dim - 2; i < desc_b->dim; ++i) {
    shape_expand_b.push_back((int)desc_b->dims[i]);
  }

  float *a_expand = (float *)cpu_runtime_.allocate((int64_t)batch_size * m * k * sizeof(float));
  float *b_expand = (float *)cpu_runtime_.allocate((int64_t)batch_size * n * k * sizeof(float));

  int a_position = desc_a->position;
  int b_position = desc_b->position;
  float a_scale = desc_a->scale;
  float b_scale = desc_b->scale;

  if (desc_a->onchip_dtype == MLUOP_DTYPE_INT31) {
    a_position = 0.0;
    a_scale = 1.0;
  }

  if (desc_b->onchip_dtype == MLUOP_DTYPE_INT31) {
    b_position = 0.0;
    b_scale = 1.0;
  }

  double c_scale = 1.0;

  if (desc_a->onchip_dtype != desc_a->dtype && desc_a->onchip_dtype != MLUOP_DTYPE_INVALID) {
    c_scale = pow(2.0, a_position) / (a_scale);
  }

  if (desc_b->onchip_dtype != desc_b->dtype && desc_b->onchip_dtype != MLUOP_DTYPE_INVALID) {
    c_scale = pow(2.0, b_position) / (b_scale);
  }

  if ((desc_a->onchip_dtype != desc_a->dtype && desc_a->onchip_dtype != MLUOP_DTYPE_INVALID) &&
      (desc_b->onchip_dtype != desc_b->dtype && desc_b->onchip_dtype != MLUOP_DTYPE_INVALID)) {
    c_scale = pow(2.0, a_position + b_position) / (a_scale * b_scale);
  }

  cnrtQuantizedParam_t a_f2i_param;
  cnrtQuantizedParam_t b_f2i_param;
  GTEST_CHECK(CNRT_RET_SUCCESS ==
              cnrtCreateQuantizedParam(&a_f2i_param, a_position, a_scale, 0.0f));
  GTEST_CHECK(CNRT_RET_SUCCESS ==
              cnrtCreateQuantizedParam(&b_f2i_param, b_position, b_scale, 0.0f));

  // bool a_broadcast = canBroadCast(shape_a, shape_expand_a);
  expandComputeCpu(std::vector<int>(desc_a->dims, desc_a->dims + desc_a->dim), shape_expand_a,
                   cpu_fp32_input_[0], a_expand);

  // bool can_broadcast = canBroadCast(shape_a, shape_b);
  expandComputeCpu(std::vector<int>(desc_b->dims, desc_b->dims + desc_b->dim), shape_expand_b,
                   cpu_fp32_input_[1], b_expand);

  void *a_int = cpu_runtime_.allocate((int64_t)batch_size * m * k * 2 * sizeof(char));
  void *b_int = cpu_runtime_.allocate((int64_t)batch_size * k * n * 2 * sizeof(char));

  double *a_double =
      static_cast<double *>(cpu_runtime_.allocate((int64_t)batch_size * m * k * sizeof(double)));
  double *b_double =
      static_cast<double *>(cpu_runtime_.allocate((int64_t)batch_size * k * n * sizeof(double)));
  double *c_tmp_double;
  if (beta != 0.0f) {
    c_tmp_double =
        static_cast<double *>(cpu_runtime_.allocate((int64_t)batch_size * m * n * sizeof(double)));
  }
  double *c_double = static_cast<double *>(cpu_runtime_.allocate((int64_t)m * n * sizeof(double)));

  // cast A 2 double
  if (desc_a->onchip_dtype != desc_a->dtype && desc_a->onchip_dtype != MLUOP_DTYPE_INVALID) {
    if (desc_a->onchip_dtype == MLUOP_DTYPE_INT8) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType_V2(a_expand, cnrtFloat, a_int, cnrtChar,
                                                          (int64_t)batch_size * m * (int64_t)k, a_f2i_param,
                                                          cnrtRounding_rm));
      int2double(a_double, a_int, MLUOP_DTYPE_INT8, (int64_t)batch_size * m * (int64_t)k);
    } else if (desc_a->onchip_dtype == MLUOP_DTYPE_INT16) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType_V2(a_expand, cnrtFloat, a_int, cnrtShort,
                                                          (int64_t)batch_size * m * (int64_t)k, a_f2i_param,
                                                          cnrtRounding_rm));
      int2double(a_double, a_int, MLUOP_DTYPE_INT16, (int64_t)batch_size * m * (int64_t)k);
    } else if (desc_a->onchip_dtype == MLUOP_DTYPE_INT31) {
      for (int64_t i = 0; i < (int64_t)batch_size * m * (int64_t)k; ++i) {
        a_double[i] = static_cast<double>(a_expand[i]);
      }
    }
  } else {
    float2double(a_double, a_expand, (int64_t)batch_size * m * (int64_t)k);
  }

  // cast B 2 double
  if (desc_b->onchip_dtype != desc_b->dtype && desc_b->onchip_dtype != MLUOP_DTYPE_INVALID) {
    if (desc_b->onchip_dtype == MLUOP_DTYPE_INT8) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType_V2(b_expand, cnrtFloat, b_int, cnrtChar,
                                                          (int64_t)batch_size * n * (int64_t)k, b_f2i_param,
                                                          cnrtRounding_rm));
      int2double(b_double, b_int, MLUOP_DTYPE_INT8, (int64_t)batch_size * n * (int64_t)k);
    } else if (desc_b->onchip_dtype == MLUOP_DTYPE_INT16) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType_V2(b_expand, cnrtFloat, b_int, cnrtShort,
                                                          (int64_t)batch_size * n * (int64_t)k, b_f2i_param,
                                                          cnrtRounding_rm));
      int2double(b_double, b_int, MLUOP_DTYPE_INT16, (int64_t)batch_size * n * (int64_t)k);
    } else if (desc_b->onchip_dtype == MLUOP_DTYPE_INT31) {
      for (int64_t i = 0; i < (int64_t)batch_size * n * (int64_t)k; ++i) {
        b_double[i] = static_cast<double>(b_expand[i]);
      }
    }
  } else {
    float2double(b_double, b_expand, (int64_t)batch_size * n * (int64_t)k);
  }

  // C to double
  if (beta != 0.0f) {
    float2double(c_tmp_double, cpu_fp32_input_[2], (int64_t)batch_size * m * (int64_t)n);
  }

#if USE_OPENBLAS
  const CBLAS_ORDER Order = CblasRowMajor;
  const CBLAS_TRANSPOSE TransA = is_transa ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = is_transb ? CblasTrans : CblasNoTrans;

  int lda = is_transa ? m : k;
  int ldb = is_transb ? k : n;
  int ldc = n;
#else

  auto matmul = [](double *lhs, double *rhs, double *bias, double *output, bool is_trans_a,
                   bool is_trans_b, int M, int N, int K, double alpha, double beta, bool use_beta) {
    for (int64_t m = 0; m < M; m++) {
      for (int64_t n = 0; n < N; n++) {
        output[(int64_t)m * N + n] = 0.0f;
        for (int64_t k = 0; k < K; k++) {
          int64_t lhs_idx = (int64_t)m * K + k;
          if (is_trans_a)
            lhs_idx = (int64_t)k * M + m;
          int64_t rhs_idx = (int64_t)k * N + n;
          if (is_trans_b)
            rhs_idx = (int64_t)n * K + k;
          output[(int64_t)m * N + n] += (alpha == 0.0f ? 0.0 : alpha * lhs[lhs_idx] * rhs[rhs_idx]);
        }
        if (use_beta == true)
          output[(int64_t)m * N + n] += beta * bias[(int64_t)m * N + n];
      }
    }
  };
#endif

  bool use_beta = false;
  if (beta != 0.0f) {
    use_beta = true;
  }

  for (int i = 0; i < batch_size; ++i) {
#if USE_OPENBLAS

    float *c_tmp_float = (float *)cpu_runtime_.allocate((int64_t)m * n * sizeof(float));
    if (beta != 0.0f) {
      memcpy(c_tmp_float, cpu_fp32_input_[2] + (int64_t)i * m * (int64_t)n, (int64_t)m * n * sizeof(float));
    }
    cblas_sgemm(Order, TransA, TransB, m, n, k, alpha, cpu_fp32_input_[0] + (int64_t)i * m * (int64_t)k, lda,
                cpu_fp32_input_[1] + (int64_t)i * k * (int64_t)n, ldb, beta, c_tmp_float, ldc);
    memcpy(cpu_fp32_output_[0] + (int64_t)i * m * (int64_t)n, c_tmp_float, (int64_t)m * n * sizeof(float));
    cpu_runtime_.deallocate(c_tmp_float);
    c_tmp_float = NULL;

#else

    matmul(a_double + (int64_t)i * m * (int64_t)k, b_double + (int64_t)i * k * (int64_t)n, c_tmp_double + (int64_t)i * m * (int64_t)n, c_double,
           is_transa, is_transb, m, n, k, (double)alpha, (double)beta, use_beta);
    qdouble2float(cpu_fp32_output_[0] + (int64_t)i * m * (int64_t)n, c_double, c_scale, (int64_t)m * n);

#endif
  }

  cpu_runtime_.deallocate(a_expand);
  cpu_runtime_.deallocate(b_expand);
  cpu_runtime_.deallocate(a_int);
  cpu_runtime_.deallocate(b_int);
  cpu_runtime_.deallocate(a_double);
  cpu_runtime_.deallocate(b_double);
  cpu_runtime_.deallocate(c_double);
  if (beta != 0.0f) {
    cpu_runtime_.deallocate(c_tmp_double);
  }
  a_expand = NULL;
  b_expand = NULL;
  a_int = NULL;
  b_int = NULL;
  a_double = NULL;
  b_double = NULL;
  c_double = NULL;
  c_tmp_double = NULL;

  cnrtDestroyQuantizedParam(a_f2i_param);
  cnrtDestroyQuantizedParam(b_f2i_param);
}

int BatchMatmulBcastExecutor::expandNumAfterFirst(int num) {
  int tmp = 0;
  while (num) {
    num = num >> 1;
    tmp++;
  }
  return tmp - 1;
}

void BatchMatmulBcastExecutor::expandComputeCpu(std::vector<int> shape_a,
                                                std::vector<int> shape_b,
                                                float *input,
                                                float *output) {
  if (shape_a.size() < MLUOP_DIM_MAX) {
    shape_a.insert(shape_a.begin(), MLUOP_DIM_MAX - shape_a.size(), 1);
  }
  if (shape_b.size() < MLUOP_DIM_MAX) {
    shape_b.insert(shape_b.begin(), MLUOP_DIM_MAX - shape_b.size(), 1);
  }

  bool can_broadcast = canBroadCast(shape_a, shape_b);
  assert(can_broadcast == 1);

  uint64_t sizeA = 1;
  uint64_t sizeB = 1;

  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    sizeA = sizeA * shape_a[i];
    sizeB = sizeB * shape_b[i];
  }

#if 0
  if (!can_broadcast) {
    memcpy(output, input, sizeA * sizeof(float));
    return;
  }
#endif

  float *tmp = cpu_runtime_.allocate(new float[sizeB]);
  memcpy(tmp, input, sizeA * sizeof(float));

  int64_t is_first = true;
  int64_t leftSizeA = 1;
  int64_t rightSizeA = 1;
  int64_t leftSizeB = 1;
  int64_t rightSizeB = 1;
  int64_t E = 1;
  int64_t ExpandA = 1;

  int64_t size = MLUOP_DIM_MAX;

  for (int64_t i = size - 1; i >= 0; i--) {
    rightSizeA = rightSizeA * shape_a[i];
    rightSizeB = rightSizeB * shape_b[i];
    leftSizeA = sizeA / rightSizeA;
    leftSizeB = sizeB / rightSizeB;
    if (shape_a[i] != shape_b[i]) {
      E = shape_b[i];
      ExpandA = ExpandA * shape_a[i];
      shape_a[i] = shape_b[i];
      for (int64_t j = 0; j < leftSizeA; j++) {
        int64_t numAfter = expandNumAfterFirst(E);
        memcpy(output + j * rightSizeB, tmp + j * (rightSizeB / E), rightSizeB / E * sizeof(float));
        for (int64_t k = 1; k <= numAfter; k++) {
          memcpy(output + j * rightSizeB + (1 << (k - 1)) * (rightSizeB / E),
                 output + j * rightSizeB, (1 << (k - 1)) * (rightSizeB / E) * sizeof(float));
        }
        int64_t done = 1 << numAfter;
        int64_t rem = E - (1 << numAfter);
        memcpy(output + j * rightSizeB + done * (rightSizeB / E), output + j * rightSizeB,
               rem * (rightSizeB / E) * sizeof(float));
      }
      memcpy(tmp, output, sizeB * sizeof(float));
    }
  }
  memcpy(output, tmp, sizeB * sizeof(float));
  cpu_runtime_.deallocate(tmp);
}

void BatchMatmulBcastExecutor::baselineOutputMalloc() {
  return;
}

void BatchMatmulBcastExecutor::getBaselineOutput() {
  if (parser_->device() == CPU) {
    return;
  }

  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  auto input_num = (beta == 0.0f) ? parser_->getInputNum() : parser_->getInputNum() - 1;
  auto output_num = parser_->getOutputNum();
  auto total_num = input_num + output_num;
  for (int i = input_num; i < total_num; ++i) {
    if (parser_->getMetaTensor(i).is_null == false) {
      auto count = parser_->getMetaTensor(i).total_count;
      auto dtype = parser_->getMetaTensor(i).dtype;
      if (dtype == MLUOP_DTYPE_INT8 || dtype == MLUOP_DTYPE_INT16) {
        continue;
      }
      auto data_size = count * mluop::getSizeOfDataType(dtype);
      void *temp = cpu_runtime_.allocate(data_size);
      // MetaTensor is input + output;
      // cpu_fp32_output is ouptut only, so cpu_fp32_output_id = metatensor_id - input_num
      parser_->getOutputData(i - input_num, temp);
      castDataOut(temp, dtype, cpu_fp32_output_[i - input_num], MLUOP_DTYPE_FLOAT, count, NO_QUANT,
                  pos_, scale_, 0);
      cpu_runtime_.deallocate(temp);
    }
  }
}

int64_t BatchMatmulBcastExecutor::getTheoryIoSize() {
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  size_t total_size = 0;
  for (size_t i = 0; i < 2; ++i) {
    MetaTensor *ts = parser_->input(i);
    total_size += ts->shape_count * ts->sizeof_dtype;
  }

  MetaTensor *ts = parser_->output(0);
  total_size += (beta == 0.0f ? 1 : 2) * ts->shape_count * ts->sizeof_dtype;

  VLOG(4) << "BatchMatmulBcastExecutor: getTheoryIOs: " << total_size << " bytes";
  return total_size;
}

int64_t BatchMatmulBcastExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  float beta = parser_->getProtoNode()->batch_matmul_bcast_param().beta();
  if (beta == 0.0f) {
    assert(parser_->getInputNum() == 2);
  } else {
    assert(parser_->getInputNum() == 3);
  }
  assert(parser_->getOutputNum() == 1);
  bool is_transa = parser_->getProtoNode()->batch_matmul_bcast_param().is_transa();
  bool is_transb = parser_->getProtoNode()->batch_matmul_bcast_param().is_transb();

  auto desc_c = tensor_desc_[2].tensor;

  int batch_size = 1;
  for (int i = 0; i < desc_c->dim - 2; ++i) {
    batch_size *= desc_c->dims[i];
  }
  assert(batch_size >= 1);

  int dim_max_a = parser_->getInputDimSize(0);
  int dim_max_c = parser_->getOutputDimSize(0);
  int k = is_transa ? (int)parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 2)
                    : (int)parser_->getProtoNode()->input(0).shape().dims(dim_max_a - 1);
  int m = (int)parser_->getProtoNode()->output(0).shape().dims(dim_max_c - 2);
  int n = (int)parser_->getProtoNode()->output(0).shape().dims(dim_max_c - 1);

  theory_ops = (int64_t)2 * batch_size * m * n * (int64_t)k;

  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
