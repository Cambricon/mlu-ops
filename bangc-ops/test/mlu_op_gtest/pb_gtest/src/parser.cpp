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
#include <chrono>       // NOLINT
#include <algorithm>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <utility>
#include <functional>

#include <sys/types.h>  // NOLINT
#include <sys/stat.h>   // NOLINT
#include <unistd.h>     // NOLINT
#include <fcntl.h>      // NOLINT

#include "pb_test_tools.h"
#include "parser.h"
#include "zero_element.h"

static void zeroElementCreate(mluoptest::Node *node) {
  std::string tem_name = node->op_name();
  const char *env = std::getenv("MLUOP_GTEST_BUILD_ZERO_ELEMENT");
  if (env == NULL) {
    return;
  }
  std::string env_str = env;
  int env_num = std::stoi(env_str);
  if (env_num == 0) {
    return;
  }
  std::vector<std::string>::iterator iter =
      std::find(white_list.begin(), white_list.end(), tem_name);
  if (iter == white_list.end()) {
    VLOG(4) << "op_name = " << tem_name.c_str();
    for (size_t i = 0; i < node->input_size(); i++) {
      auto *in_shape = node->mutable_input(i)->mutable_shape();
      for (size_t j = 0; j < in_shape->dims_size(); j++) {
        in_shape->set_dims(j, 0);
      }
    }
    for (size_t i = 0; i < node->output_size(); i++) {
      auto *out_shape = node->mutable_output(i)->mutable_shape();
      for (size_t j = 0; j < out_shape->dims_size(); j++) {
        out_shape->set_dims(j, 0);
      }
    }
  }
}

namespace mluoptest {

// env for test negative_scale
__attribute__((__unused__)) bool negative_scale_ =
    getEnv("MLUOP_GTEST_NEGATIVE_SCALE", false);

Parser::~Parser() {
  if (proto_node_ != nullptr) {
    delete proto_node_;
    proto_node_ = nullptr;
  }
}

void Parser::parse(const std::string &file) {
  proto_node_ = new Node;
  setCurPbPath(file);  // set root path of pb/prototxt
  GTEST_CHECK(readMessageFromFile(file, proto_node_),
              "Parser: parse *pb/*prototxt failed.");
  isSupportTF32(proto_node_);

  GTEST_CHECK(proto_node_->has_op_name(),
              "Parser: missing op name in prototxt.");
  zeroElementCreate(proto_node_);
  // 1.get device
  if (proto_node_->has_device()) {
    device_ = proto_node_->device();
  } else if (proto_node_->has_test_param()) {
    if (proto_node_->mutable_test_param()->has_baseline_device()) {
      device_ = proto_node_->mutable_test_param()->baseline_device();
    } else {
      LOG(WARNING)
          << "Parser: missing device in test param, using CPU as default.";
      device_ = Device::CPU;  // default cpu
    }
  } else {
    LOG(WARNING) << "Parser: missing baseline device, using CPU as default.";
    device_ = Device::CPU;  // default cpu
  }

  // 2.get criterion
  if (common_threshold()) {
    criterions_.clear();
    // check if there exists complex output tensor
    bool has_complex_output = false;
    for (auto i = 0; i < proto_node_->output_size(); ++i) {
      mluOpDataType_t dtype =
          cvtProtoDtypeToMluOp(proto_node_->mutable_output(i)->dtype());
      if (dtype == MLUOP_DTYPE_COMPLEX_HALF ||
          dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
        has_complex_output = true;
        break;
      }
    }

    if (proto_node_->has_test_param()) {
      auto test_param = proto_node_->mutable_test_param();
      auto error_func_size = test_param->error_func_size();
      GTEST_CHECK(error_func_size > 0);
      GTEST_CHECK(
          test_param->error_func_size() == test_param->error_threshold_size(),
          "Parser: error_func's number should equal to error_threshold's "
          "number, "
          "now they are not equal.");
      if (has_complex_output) {
        // If error_threshold_imag is set, the number must be the same as
        // error_func.
        GTEST_CHECK((test_param->error_threshold_imag_size() == 0) ||
                        (test_param->error_threshold_imag_size() ==
                         test_param->error_func_size()),
                    "Parser: Nb. of error_threshold_imag should be either 0 or "
                    "equal to the Nb. of error_func.");
      }
      for (auto i = 0; i < error_func_size; ++i) {
        auto func = cvtProtoEvaluationCriterion(test_param->error_func(i));
        auto error_thred = test_param->error_threshold(i);
        auto error_thred_imag = error_thred;
        if (has_complex_output) {
          if (test_param->error_threshold_imag_size() > 0) {
            error_thred_imag = test_param->error_threshold_imag(i);
          } else {
            VLOG(4) << "Parser::criterions: set error_threshold_imag to "
                       "error_threshold.";
          }
          criterions_.insert(std::move(
              Evaluator::Criterion(func, error_thred, error_thred_imag)));
        } else {
          criterions_.insert(
              std::move(Evaluator::Criterion(func, error_thred)));
        }
      }
    } else {
      size_t criterion_size = proto_node_->evaluation_criterion_size();
      GTEST_CHECK(criterion_size > 0);
      auto threshold_size = proto_node_->evaluation_threshold_size();
      auto threshold_imag_size = proto_node_->evaluation_threshold_imag_size();
      GTEST_CHECK(criterion_size == threshold_size,
                  "Parser: evaluation_criterion's number should equal to "
                  "evaluation_threshold's number, now they are not equal.");
      if (has_complex_output) {
        // If threshold_imag is set, the number must be the same as criterions.
        GTEST_CHECK(
            (threshold_imag_size == 0) ||
                (threshold_imag_size == criterion_size),
            "Parser: Nb. of evaluation_threshold_imag should be either 0 or "
            "equal to the Nb. of evaluation_criterion.");
      }
      for (auto i = 0; i < criterion_size; ++i) {
        auto proto_func = proto_node_->evaluation_criterion(i);
        auto func = cvtProtoEvaluationCriterion(proto_func);
        auto eval_thred = proto_node_->evaluation_threshold(i);
        auto eval_thred_imag = eval_thred;
        if (has_complex_output) {
          if (threshold_imag_size > 0) {
            eval_thred_imag = proto_node_->evaluation_threshold_imag(i);
          } else {
            VLOG(4) << "Parser::criterions: set evaluation_threshold_imag to "
                       "evaluation_threshold.";
          }
          criterions_.insert(std::move(
              Evaluator::Criterion(func, eval_thred, eval_thred_imag)));
        } else {
          criterions_.insert(std::move(Evaluator::Criterion(func, eval_thred)));
        }
      }
    }
  }

  // 3. inputs/outputs
  auto parse_tensor = [=](MetaTensor *mt, Tensor *pt) {
    mt->is_null = (pt->id().find("NULL") != std::string::npos) ? true : false;
    if (unlikely(mt->is_null)) {
      VLOG(4) << "WARNING: found tensor is null, skip parsing else data.";
      return;  // if null, don't need parse other info.
    }
    mt->name = pt->id();
    mt->value_type = getValueType(pt);

    // 1.shape set to tensor desc
    GTEST_CHECK(pt->has_shape(), "Parser: missing tensor shape in prototxt.");
    mt->shape.resize(pt->mutable_shape()->dims_size());
    // pt->mutable_shape->set_dims(0,0);
    for (size_t i = 0; i < mt->shape.size(); ++i) {
      mt->shape[i] = pt->mutable_shape()->dims(i);
    }
    mt->stride.resize(pt->mutable_shape()->dim_stride_size());
    for (size_t i = 0; i < mt->stride.size(); ++i) {
      mt->stride[i] = pt->mutable_shape()->dim_stride(i);
    }

    // 2.else info
    mt->dtype = cvtProtoDtypeToMluOp(pt->dtype());
    mt->oc_dt = pt->has_onchip_dtype()
                    ? cvtProtoDtypeToMluOp(pt->onchip_dtype())
                    : MLUOP_DTYPE_INVALID;
    mt->layout = cvtProtoLayoutToMluOp(pt->layout());

    // 3.size to malloc memory. (shape may not equal to size)
    // stride_count include stride, if no stride stride_count == shape_count
    mt->total_count = getTensorStrideCount(pt, mt->value_type);
    // shape_count come from value_f/value_i/value_h/value_ui/value_ul and
    // shape. not include stride
    mt->shape_count = getTensorShapeCount(pt);
    mt->sizeof_dtype = mluop::getSizeOfDataType(mt->dtype);
    mt->size_in_bytes = mt->total_count * mt->sizeof_dtype;

    if (mt->total_count != mt->shape_count) {
      VLOG(4) << "WARNING: Parser: the " << mt->name
              << " strided element count is " << mt->total_count
              << " while shape element count is " << mt->shape_count;
    }

    mt->position = pt->has_position() ? pt->position() : 0;
    mt->scale = pt->has_scale() ? pt->scale() : 1.0f;
    mt->offset = pt->has_offset() ? pt->offset() : 0;

    // 4.check
    checkTensorValid(mt, pt);
  };

  bool is_float = false;
  bool is_group = false;
  bool is_round_half_up = false;
  if (proto_node_->has_handle_param()) {
    if (proto_node_->handle_param().has_round_mode()) {
      if (proto_node_->handle_param().round_mode() ==
          mluoptest::ROUND_HALF_UP) {
        is_round_half_up = true;
      }
    }
  }

  inputs_.resize(proto_node_->input_size());
  for (size_t i = 0; i < proto_node_->input_size(); ++i) {
    parse_tensor(&inputs_[i], proto_node_->mutable_input(i));
    if (negative_scale_) {
      if ((inputs_[i].dtype == MLUOP_DTYPE_HALF &&
           (inputs_[i].oc_dt == MLUOP_DTYPE_HALF ||
            inputs_[i].oc_dt == MLUOP_DTYPE_INVALID)) ||
          (inputs_[i].dtype == MLUOP_DTYPE_FLOAT &&
           (inputs_[i].oc_dt == MLUOP_DTYPE_FLOAT ||
            inputs_[i].oc_dt == MLUOP_DTYPE_INVALID))) {
        is_float = true;
      }
    }
  }
  if (!is_group && !is_float && !is_round_half_up && negative_scale_) {
    for (size_t i = 0; i < proto_node_->input_size(); ++i) {
      // only test symmetric quantify
      if (inputs_[i].offset == 0) {
        if (inputs_[i].dtype != MLUOP_DTYPE_INT16 &&
            inputs_[i].dtype != MLUOP_DTYPE_INT8) {
          inputs_[i].scale = -inputs_[i].scale;
        }
      }
    }
  }

  outputs_.resize(proto_node_->output_size());
  for (size_t i = 0; i < proto_node_->output_size(); ++i) {
    parse_tensor(&outputs_[i], proto_node_->mutable_output(i));
  }

  // get gtest ini like black list of zero_input mode
  getTestInfo();
}

// check if tensor value is equal to shape.
// when found value_i value_f value_h value_ui value_ul, just check value_size
// and shape_size when found random_param, check random param here allow
// shape_size != data size saved in pb cuz shape_size is for create tensor and
// data_size is for malloc space so just print warning.
void Parser::checkTensorValid(MetaTensor *mt, Tensor *pt) {
  int shape_count = 1;
  switch (mt->value_type) {
    case VALUE_F:
    case VALUE_I:
    case VALUE_L:
    case VALUE_H:
    case VALUE_UI:
    case VALUE_UL:
      shape_count = std::accumulate(mt->shape.begin(), mt->shape.end(),
                                    shape_count, std::multiplies<int>());
      GTEST_WARNING(mt->shape_count == shape_count,
                    "Parser: found shape count is not equal to value "
                    "size.(mt->shape_count is value_size)");
      break;
    case VALUE_RANDOM:
      if (!mt->empty()) {
        checkRandomParam(pt);
      }
      break;
    case VALUE_PATH: {
      // if found path(only) in pb, but can't access this path, throw.
      auto cur_pb_path = pb_path_ + pt->path();
      GTEST_CHECK((access(cur_pb_path.c_str(), 4) != -1),
                  "Parser: open path saved in *prototxt failed.");
      break;
    }
    case VALUE_INVALID:
      // check output, if may shape empty, value is empty, and not random param,
      // so don't need check.
      break;
    default: {
      GTEST_CHECK(false,
                  "Parser: got unsupported value type, parse tensor failed.");
    } break;
  }
}

void Parser::checkRandomParam(Tensor *pt) {
  GTEST_CHECK(pt->has_random_data(),
              "Parser: missing random param of tensor saved in *pb");
  GTEST_CHECK(pt->mutable_random_data()->has_distribution(),
              "Parser: missing distribution of random param saved in *pb");

  auto random_data = pt->mutable_random_data();
  if (random_data->distribution() == mluoptest::UNIFORM) {
    GTEST_CHECK(
        random_data->has_upper_bound() || random_data->has_upper_bound_double(),
        "Parser: missing upper bound of UNIFORM random param in *pb");
    GTEST_CHECK(
        random_data->has_lower_bound() || random_data->has_lower_bound_double(),
        "Parser: missing lower bound of UNIFORM random param in *pb");
  } else if (random_data->distribution() == mluoptest::GAUSSIAN) {
    GTEST_CHECK(random_data->has_mu() || random_data->has_mu_double(),
                "Parser: missing mu of UNIFORM random param in *pb");
    GTEST_CHECK(random_data->has_sigma() || random_data->has_sigma_double(),
                "Parser: missing sigma of UNIFORM random param in *pb");
  } else {
    GTEST_CHECK(
        false, "Parser: got unsupported distribution when check tensor valid.");
  }
}

// return value's dtype is according to value type.
// if value type is value_*, return dtype is dtype in proto.
// if value type is random, return dtype is fp64 for double and fp32 otherwise.
void Parser::getInputTensorValue(size_t index, void *data, size_t count) {
  getTensorValue(proto_node_->mutable_input(index), data,
                 inputs_[index].value_type, count);
}

// return value's dtype is according to value type.
// if value type is value_*, return dtype is dtype in proto.
// if value type is random, return dtype is fp64 for double and fp32 otherwise.
void Parser::getOutputTensorValue(size_t index, void *data, size_t count) {
  getTensorValue(proto_node_->mutable_output(index), data,
                 outputs_[index].value_type, count);
}

// Get value from field value_f
// TODO(taokai): abandon value_f, use value_h for float numbers instead.
void Parser::getTensorValueF(const Tensor *pt, void *data, size_t count) {
  switch (pt->dtype()) {
    case DTYPE_COMPLEX_HALF:
    case DTYPE_COMPLEX_FLOAT: {
      GTEST_CHECK(pt->value_f_size() == 2 * count,
                  "Parser: when read value_f, expected element num is not "
                  "equal to real element num.");
    } break;
    default:
      GTEST_CHECK(pt->value_f_size() == count,
                  "Parser: when read value_f, expected element num is not "
                  "equal to real element num.");
  }

  switch (pt->dtype()) {
    // may have precision issue since value_f is fixed to float in protobuf
    case DTYPE_DOUBLE:
      for (int i = 0; i < count; ++i) {
        ((double *)data)[i] = (double)pt->value_f(i);
      }
      break;
    case DTYPE_FLOAT:
      for (int i = 0; i < count; ++i) {
        ((float *)data)[i] = pt->value_f(i);
      }
      break;
    case DTYPE_HALF:
      for (int i = 0; i < count; ++i) {
        ((int16_t *)data)[i] = cvtFloatToHalf(pt->value_f(i));
      }
      break;
    case DTYPE_COMPLEX_HALF:
      for (int i = 0; i < 2 * count; i += 2) {
        ((int16_t *)data)[i] = cvtFloatToHalf(pt->value_f(i));
        ((int16_t *)data)[i + 1] = cvtFloatToHalf(pt->value_f(i + 1));
      }
      break;
    case DTYPE_COMPLEX_FLOAT:
      for (int i = 0; i < 2 * count; i += 2) {
        ((float *)data)[i] = pt->value_f(i);
        ((float *)data)[i + 1] = pt->value_f(i + 1);
      }
      break;
    default:
      GTEST_CHECK(false,
                  "Parser: found unsuppored dtype in value_f, value_f only "
                  "supporte float/half.");
  }
}

// get value from value_i
// no quant intx, saved in value_i
void Parser::getTensorValueI(const Tensor *pt, void *data, size_t count) {
  if (pt->dtype() == DTYPE_COMPLEX_HALF || pt->dtype() == DTYPE_COMPLEX_FLOAT) {
    GTEST_CHECK(pt->value_i_size() == 2 * count,
                "Parser: when read value_i, expected element num is not equal "
                "to real element num.");
  } else {
    GTEST_CHECK(pt->value_i_size() == count,
                "Parser: when read value_i, expected element num is not equal "
                "to real element num.");
  }

  switch (pt->dtype()) {
    case DTYPE_INT8:
      for (int i = 0; i < count; ++i) {
        ((int8_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_UINT8:
      for (int i = 0; i < count; ++i) {
        ((uint8_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_INT16:
      for (int i = 0; i < count; ++i) {
        ((int16_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_UINT16:
      for (int i = 0; i < count; ++i) {
        ((uint16_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_INT32:
      for (int i = 0; i < count; ++i) {
        ((int32_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_INT64:
      for (int i = 0; i < count; ++i) {
        ((int64_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_BOOL:  // parser value_i == BOOL
      for (int i = 0; i < count; ++i) {
        ((int8_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_HALF:
      for (int i = 0; i < count; ++i) {
        ((int16_t *)data)[i] = pt->value_i(i);
      }
      break;
    case DTYPE_FLOAT:
      for (int i = 0; i < count; ++i) {
        int value_i = pt->value_i(i);
        ((float *)data)[i] = *((float *)(&value_i));
      }
      break;
    case DTYPE_COMPLEX_HALF:
      for (int i = 0; i < 2 * count; i += 2) {
        ((int16_t *)data)[i] = pt->value_i(i);
        ((int16_t *)data)[i + 1] = pt->value_i(i + 1);
      }
      break;
    case DTYPE_COMPLEX_FLOAT:
      for (int i = 0; i < 2 * count; i += 2) {
        int value_i = pt->value_i(i);
        int value_i_imag = pt->value_i(i + 1);
        ((float *)data)[i] = *((float *)(&value_i));
        ((float *)data)[i + 1] = *((float *)(&value_i_imag));
      }
      break;
    default:
      GTEST_CHECK(
          false,
          "Parser: found unsuppored dtype in value_i, value_i only support "
          "int8/uint8/int16/uint16/int32/int64/bool.");
  }
}

// get value from value_l
// no quant intx, saved in value_l
void Parser::getTensorValueL(const Tensor *pt, void *data, size_t count) {
  GTEST_CHECK(pt->value_l_size() == count,
              "Parser: when read value_l, expected element num is not equal to "
              "real element num.");
  switch (pt->dtype()) {
    case DTYPE_INT64:
      for (int i = 0; i < count; ++i) {
        ((int64_t *)data)[i] = pt->value_l(i);
      }
      break;
    case DTYPE_INT8:
    case DTYPE_BOOL:  // parser value_l == BOOL
      for (int i = 0; i < count; ++i) {
        ((int8_t *)data)[i] = pt->value_l(i);
      }
      break;
    case DTYPE_INT16:
      for (int i = 0; i < count; ++i) {
        ((int16_t *)data)[i] = pt->value_l(i);
      }
      break;
    case DTYPE_INT32:
      for (int i = 0; i < count; ++i) {
        ((int32_t *)data)[i] = pt->value_l(i);
      }
      break;
    case DTYPE_DOUBLE:
      for (int i = 0; i < count; ++i) {
        int64_t value_l = pt->value_l(i);
        ((double *)data)[i] = *((double *)(&value_l));
      }
      break;
    default:
      GTEST_CHECK(
          false,
          "Parser: found unsuppored dtype in value_l, value_l only support "
          "bool/int8/int16/int32/int64.");
  }
}

// get value from value_ui
// no quant uintx, saved in value_ui
void Parser::getTensorValueUI(const Tensor *pt, void *data, size_t count) {
  GTEST_CHECK(pt->value_ui_size() == count,
              "Parser: when read value_ui, expected element num is not equal "
              "to real element num.");
  switch (pt->dtype()) {
    case DTYPE_UINT8:
      for (int i = 0; i < count; ++i) {
        ((uint8_t *)data)[i] = pt->value_ui(i);
      }
      break;
    case DTYPE_UINT16:
      for (int i = 0; i < count; ++i) {
        ((uint16_t *)data)[i] = pt->value_ui(i);
      }
      break;
    case DTYPE_UINT32:
      for (int i = 0; i < count; ++i) {
        ((uint32_t *)data)[i] = pt->value_ui(i);
      }
      break;
    default:
      GTEST_CHECK(
          false,
          "Parser: found unsuppored dtype in value_ui, value_ui only support "
          "uint8/uint16/uint32.");
  }
}

// get value from value_ul
// no quant uintx, saved in value_ul
void Parser::getTensorValueUL(const Tensor *pt, void *data, size_t count) {
  GTEST_CHECK(pt->value_ul_size() == count,
              "Parser: when read value_ul, expected element num is not equal "
              "to real element num.");
  switch (pt->dtype()) {
    case DTYPE_UINT64:
      for (int i = 0; i < count; ++i) {
        ((uint64_t *)data)[i] = pt->value_ul(i);
      }
      break;
    case DTYPE_UINT8:
      for (int i = 0; i < count; ++i) {
        ((uint8_t *)data)[i] = pt->value_ul(i);
      }
      break;
    case DTYPE_UINT16:
      for (int i = 0; i < count; ++i) {
        ((uint16_t *)data)[i] = pt->value_ul(i);
      }
      break;
    case DTYPE_UINT32:
      for (int i = 0; i < count; ++i) {
        ((uint32_t *)data)[i] = pt->value_ul(i);
      }
      break;
    default:
      GTEST_CHECK(
          false,
          "Parser: found unsuppored dtype in value_ul, value_ul only support "
          "uint8/uint16/uint32/uint64.");
  }
}

inline double str2fp64(const std::string *in_str) {
  uint64_t res = 0x0;
  for (int i = 0; i < in_str->size(); ++i) {
    char byte = in_str->c_str()[i];  // 0~f
    res = res << 4;
    res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';  // 0~15
  }
  return *(double *)(&res);
}
inline float str2fp32(const std::string *in_str) {
  uint32_t res = 0x0;
  for (int i = 0; i < in_str->size(); ++i) {
    char byte = in_str->c_str()[i];  // 0~f
    res = res << 4;
    res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';  // 0~15
  }
  return *(float *)(&res);
}
inline uint16_t str2fp16(const std::string *in_str) {
  uint16_t res = 0x0;
  for (int i = 0; i < in_str->size(); ++i) {
    char byte = in_str->c_str()[i];
    res = res << 4;
    res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';
  }
  return res;
}

// get value by value_h (hex)
// we hope all float value come from value_h to keep precision
void Parser::getTensorValueH(Tensor *pt, void *data, size_t count) {
  switch (pt->dtype()) {
    case DTYPE_COMPLEX_HALF:
    case DTYPE_COMPLEX_FLOAT: {
      // each complex number is composed of real and imaginary parts
      GTEST_CHECK(pt->value_h_size() == 2 * count,
                  "Parser: when read value_h, expected element num is not "
                  "equal to real element num.");
    } break;
    default: {
      GTEST_CHECK(pt->value_h_size() == count,
                  "Parser: when read value_h, expected element num is not "
                  "equal to real element num.");
    }
  }

  switch (pt->dtype()) {
    case DTYPE_HALF:
      for (int i = 0; i < count; ++i) {
        ((uint16_t *)data)[i] = str2fp16(pt->mutable_value_h(i));
      }
      break;
    case DTYPE_FLOAT:
      for (int i = 0; i < count; ++i) {
        ((float *)data)[i] = str2fp32(pt->mutable_value_h(i));
      }
      break;
    case DTYPE_DOUBLE:
      for (int i = 0; i < count; ++i) {
        ((double *)data)[i] = str2fp64(pt->mutable_value_h(i));
      }
      break;
    case DTYPE_COMPLEX_HALF:
      for (int i = 0; i < 2 * count; ++i) {
        ((uint16_t *)data)[i] = str2fp16(pt->mutable_value_h(i));
      }
      break;
    case DTYPE_COMPLEX_FLOAT:
      for (int i = 0; i < 2 * count; ++i) {
        ((float *)data)[i] = str2fp32(pt->mutable_value_h(i));
      }
      break;
    default:
      GTEST_CHECK(false,
                  "Parser: found unsuppored dtype in value_h, value_h only "
                  "supporte float/half.");
  }
}

// get value by random data param
void Parser::getTensorValueRandom(Tensor *pt, void *data, size_t count) {
  if (pt->dtype() == DTYPE_DOUBLE) {
    generateRandomData((double *)data, count, pt->mutable_random_data(),
                       pt->dtype());
  } else if (pt->dtype() == DTYPE_COMPLEX_HALF ||
             pt->dtype() == DTYPE_COMPLEX_FLOAT) {
    generateRandomData((float *)data, 2 * count, pt->mutable_random_data(),
                       pt->dtype());
  } else {
    generateRandomData((float *)data, count, pt->mutable_random_data(),
                       pt->dtype());
  }
}

// get value by random data param
void Parser::getTensorValueByFile(Tensor *pt, void *data, size_t count) {
  auto cur_pb_path = pb_path_ + pt->path();
  size_t tensor_length = count * getTensorSize(pt);
  auto start = std::chrono::steady_clock::now();
  std::ifstream fin(cur_pb_path, std::ios::in | std::ios::binary);
  fin.read((char *)data, tensor_length);
  auto stop = std::chrono::steady_clock::now();
  std::chrono::duration<double> cost_s = stop - start;

  ASSERT_TRUE(fin) << "read data in file failed.";
  VLOG(2) << __func__ << " " << cur_pb_path << ", time cost: " << cost_s.count()
          << " s"
          << ", speed: " << tensor_length / 1024. / 1024. / cost_s.count()
          << " MB/s";
  parsed_file_size += tensor_length;
  parsed_cost_seconds += cost_s.count();
}

// set value in proto to meta_tensor.ptr
// random data(for cpu compute) value is fp32 definitely
// valueh valuef valuei dtype is according dtype in proto
void Parser::getTensorValue(Tensor *pt, void *data, ValueType value_type,
                            size_t count) {
  switch (value_type) {
    case VALUE_H:
      getTensorValueH(pt, data, count);
      break;
    case VALUE_F:
      getTensorValueF(pt, data, count);
      break;
    case VALUE_I:
      getTensorValueI(pt, data, count);
      break;
    case VALUE_L:
      getTensorValueL(pt, data, count);
      break;
    case VALUE_UI:
      getTensorValueUI(pt, data, count);
      break;
    case VALUE_UL:
      getTensorValueUL(pt, data, count);
      break;
    case VALUE_RANDOM:
      getTensorValueRandom(pt, data, count);  // cpu mode dtype fp32 or fp64
      break;
    case VALUE_PATH:
      getTensorValueByFile(pt, data, count);  // cpu mode dtype fp32
      break;
    case VALUE_INVALID:
      GTEST_WARNING(
          false,
          "Parser: trying to get value of tensor, but missing data source.");
      break;
    default:
      GTEST_CHECK(false,
                  "Parser: get tensor data failed, unsupported value type.");
  }
}

std::vector<int> Parser::threshold_use() {
  std::vector<int> res;
  for (int i = 0; i < outputs_.size(); ++i) {
    if (proto_node_->mutable_output(i)->has_threshold_use()) {
      int threshold_use = (int)proto_node_->mutable_output(i)->threshold_use();
      res.push_back(threshold_use);
    } else {
      res.push_back(1);
    }
  }
  return res;
}

bool Parser::check_threshold() {
  std::vector<int> threshold_use = Parser::threshold_use();
  bool res = false;

  if (proto_node_->has_test_param()) {
    if (0 != proto_node_->mutable_test_param()->error_threshold_size()) {
      res = true;
    }
  } else if (0 != proto_node_->evaluation_threshold_size()) {
    res = true;
  } else {
    for (int i = 0; i < outputs_.size(); ++i) {
      Tensor *pt = proto_node_->mutable_output(i);  // pt for proto_tensor
      if (0 == threshold_use[i]) {
        // pass
      } else if (0 != pt->thresholds().evaluation_threshold_size()) {
        res = true;
      } else {
        // pass
      }
    }
  }
  return res;
}

bool Parser::common_threshold() {
  bool res = false;
  if (proto_node_->has_test_param()) {
    if (0 != proto_node_->mutable_test_param()->error_threshold_size()) {
      res = true;
    }
  } else if (0 != proto_node_->evaluation_threshold_size()) {
    res = true;
  } else {
    // pass
  }
  return res;
}

std::set<Evaluator::Criterion> Parser::criterions(
    int index, const std::set<Evaluator::Formula> &criterions_use) {
  std::set<Evaluator::Criterion> res;

  // check if there exists complex output tensor
  bool has_complex_output = false;
  for (auto i = 0; i < proto_node_->output_size(); ++i) {
    mluOpDataType_t dtype =
        cvtProtoDtypeToMluOp(proto_node_->mutable_output(i)->dtype());
    if (dtype == MLUOP_DTYPE_COMPLEX_HALF ||
        dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      has_complex_output = true;
      break;
    }
  }

  if (proto_node_->has_test_param()) {
    auto test_param = proto_node_->mutable_test_param();
    GTEST_CHECK(test_param->error_func_size() != 0);
    GTEST_CHECK(
        test_param->error_func_size() == test_param->error_threshold_size(),
        "Parser: error_func's number should equal to error_threshold's number, "
        "now they are not equal.");
    if (has_complex_output) {
      // If error_threshold_imag is set, the number must be the same as
      // error_func.
      GTEST_CHECK((test_param->error_threshold_imag_size() == 0) ||
                      (test_param->error_threshold_imag_size() ==
                       test_param->error_func_size()),
                  "Parser: Nb. of error_threshold_imag should be either 0 or "
                  "equal to the Nb. of error_func.");
    }
    auto error_func_size = test_param->error_func_size();
    for (auto i = 0; i < error_func_size; ++i) {
      auto func = cvtProtoEvaluationCriterion(test_param->error_func(i));
      auto error_thred = test_param->error_threshold(i);
      auto error_thred_imag = error_thred;
      if (has_complex_output) {
        if (test_param->error_threshold_imag_size() > 0) {
          error_thred_imag = test_param->error_threshold_imag(i);
        } else {
          VLOG(4) << "Parser::criterions: set error_threshold_imag to "
                     "error_threshold.";
        }
        res.insert(std::move(
            Evaluator::Criterion(func, error_thred, error_thred_imag)));
      } else {
        res.insert(std::move(Evaluator::Criterion(func, error_thred)));
      }
    }
  } else {
    size_t criterion_size = proto_node_->evaluation_criterion_size();
    GTEST_CHECK(criterion_size > 0);
    size_t threshold_size = 0;
    size_t threshold_imag_size = 0;
    Tensor *pt = nullptr;
    if (-1 == index) {
      // use common evaluation threshold
      threshold_size = proto_node_->evaluation_threshold_size();
      threshold_imag_size = proto_node_->evaluation_threshold_imag_size();
    } else {
      // use evaluation thresholds specified in each tensor
      pt = proto_node_->mutable_output(index);  // pt for proto_tensor
      threshold_size = pt->thresholds().evaluation_threshold_size();
      threshold_imag_size = pt->thresholds().evaluation_threshold_imag_size();
    }
    GTEST_CHECK(criterion_size == threshold_size,
                "Parser: evaluation_criterion's number should equal to "
                "evaluation_threshold's number, now they are not equal.");
    if (has_complex_output) {
      // If threshold_imag is set, the number must be the same as criterions.
      GTEST_CHECK(
          (threshold_imag_size == 0) || (threshold_imag_size == criterion_size),
          "Parser: Nb. of evaluation_threshold_imag should be either 0 or "
          "equal to the Nb. of evaluation_criterion.");
    }
    for (size_t i = 0; i < criterion_size; ++i) {
      auto proto_func = proto_node_->evaluation_criterion(i);
      auto func = cvtProtoEvaluationCriterion(proto_func);
      if (criterions_use.find(func) == criterions_use.end()) {
        VLOG(4) << "Parser::criterions: skip " << Evaluator::Formula2str(func);
        continue;
      }
      auto eval_thred = (index == -1)
                            ? proto_node_->evaluation_threshold(i)
                            : pt->thresholds().evaluation_threshold(i);
      auto eval_thred_imag = eval_thred;
      if (has_complex_output) {
        if (threshold_imag_size > 0) {
          eval_thred_imag = (index == -1)
                                ? proto_node_->evaluation_threshold_imag(i)
                                : pt->thresholds().evaluation_threshold_imag(i);
        } else {
          VLOG(4) << "Parser::criterions: set evaluation_threshold_imag to "
                     "evaluation_threshold.";
        }
        res.insert(
            std::move(Evaluator::Criterion(func, eval_thred, eval_thred_imag)));
      } else {
        res.insert(std::move(Evaluator::Criterion(func, eval_thred)));
      }
    }
  }
  return res;
}

static inline bool strEndsWith(const std::string &self,
                               const std::string &pattern) {
  if (self.size() < pattern.size()) return false;
  return (self.compare(self.size() - pattern.size(), pattern.size(), pattern) ==
          0);
}

bool Parser::readMessageFromFile(const std::string &filename, Node *proto) {
  struct stat file_stat;
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "File open failed: " << filename << ". Reason: " << errno
               << "-" << strerror(errno);
    return false;
  }
  int ret_stat = fstat(fd, &file_stat);
  if (ret_stat == -1) {
    LOG(ERROR) << "File stat failed: " << filename << ". Reason: " << errno
               << "-" << strerror(errno);
    return false;
  }

  VLOG(4) << "Open " << filename << " (File size: " << file_stat.st_size
          << " Bytes)";

  bool status = false;
  auto start = std::chrono::steady_clock::now();

  // ref ProtoBuf docs, `FileInputStream` is preferred over using an ifstream
  // with `IstreamInputStream`
  google::protobuf::io::FileInputStream input(fd);
  if (strEndsWith(filename, ".pb")) {
    google::protobuf::io::CodedInputStream coded_input(&input);
    coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX - 1);
    status = proto->ParseFromCodedStream(&coded_input);
  } else if (strEndsWith(filename, ".prototxt")) {
    status = google::protobuf::TextFormat::Parse(&input, proto);
  } else {
    LOG(ERROR) << "Unsupported file extension";
    return false;
  }

  close(fd);

  auto stop = std::chrono::steady_clock::now();
  std::chrono::duration<double> cost_s = stop - start;
  VLOG(2) << __func__ << " " << filename << ", time cost: " << cost_s.count()
          << " s"
          << ", speed: " << file_stat.st_size / 1024. / 1024. / cost_s.count()
          << " MB/s";
  parsed_file_size += file_stat.st_size;
  parsed_cost_seconds += cost_s.count();

  return status;
}

void Parser::getTestInfo() {
  std::unordered_map<std::string, std::vector<std::string>> test_info;
  test_info =
      readFileByLine("../../test/mlu_op_gtest/pb_gtest/gtest_config/test_list");
  list_rely_real_data_ = test_info["rely_real_data"];
}

Evaluator::Formula Parser::cvtProtoEvaluationCriterion(int f) {
  return Parser::cvtProtoEvaluationCriterion(
      static_cast<EvaluationCriterion>(f));
}

Evaluator::Formula Parser::cvtProtoEvaluationCriterion(EvaluationCriterion f) {
  switch (f) {
    case MAPE:
    case DIFF1:
      return Evaluator::Formula::DIFF1;
    case DIFF2:
      return Evaluator::Formula::DIFF2;
    case MAXAPE:
    case DIFF3:
      return Evaluator::Formula::DIFF3;
    case DIFF3_2:
      return Evaluator::Formula::DIFF3_2;
    case DIFF4:
      return Evaluator::Formula::DIFF4;
    default:
      LOG(ERROR) << "NOT support this evaluation critertion yet";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
  }
}

ValueType Parser::getValueType(const Tensor *t) {
  // value_h > value_f > value_i > random data > path
  if (t->value_h_size() != 0) {
    return VALUE_H;
  } else if (t->value_f_size() != 0) {
    return VALUE_F;
  } else if (t->value_i_size() != 0) {
    return VALUE_I;
  } else if (t->value_l_size() != 0) {
    return VALUE_L;
  } else if (t->value_ui_size() != 0) {
    return VALUE_UI;
  } else if (t->value_ul_size() != 0) {
    return VALUE_UL;
  } else if (t->has_path()) {
    return VALUE_PATH;
  } else if (t->has_random_data()) {
    return VALUE_RANDOM;
  } else {
    return VALUE_INVALID;
  }
}

// include stride
// compute this by shape(stride + dims)
inline size_t Parser::getTensorShapeCount(Tensor *pt) {
  GTEST_CHECK(pt->has_shape(), "Parser: missing tensor shape in prototxt.");
  return shapeElementCount(pt->mutable_shape());
}

// if have value_x return value_size
// else (random/path) return shape_count
inline size_t Parser::getTensorStrideCount(Tensor *pt, ValueType value_type) {
  if (pt->mutable_shape()->dim_stride_size() > 0) {
    GTEST_CHECK(pt->has_shape(), "Parser: missing tensor shape in prototxt.");
    return shapeStrideCount(pt->mutable_shape());
  }

  if (pt->dtype() == DTYPE_COMPLEX_HALF || pt->dtype() == DTYPE_COMPLEX_FLOAT) {
    int num = 0;
    if (value_type == VALUE_F) {
      num = pt->value_f_size();
      GTEST_CHECK(num % 2 == 0,
                  "Parser: number of value_f should be multiples of 2 for "
                  "complex dtype.");
      return num / 2;
    } else if (value_type == VALUE_H) {
      num = pt->value_h_size();
      GTEST_CHECK(num % 2 == 0,
                  "Parser: number of value_h should be multiples of 2 for "
                  "complex dtype.");
      return num / 2;
    } else if (value_type == VALUE_I) {
      num = pt->value_i_size();
      GTEST_CHECK(num % 2 == 0,
                  "Parser: number of value_i should be multiples of 2 for "
                  "complex dtype.");
      return num / 2;
    }
  }

  switch (value_type) {
    case VALUE_H:
      return pt->value_h_size();
    case VALUE_I:
      return pt->value_i_size();
    case VALUE_L:
      return pt->value_l_size();
    case VALUE_UI:
      return pt->value_ui_size();
    case VALUE_UL:
      return pt->value_ul_size();
    case VALUE_F:
      return pt->value_f_size();
    case VALUE_RANDOM:
    case VALUE_PATH:
    case VALUE_INVALID:
      GTEST_CHECK(pt->has_shape(), "Parser: missing tensor shape in prototxt.");
      return shapeStrideCount(pt->mutable_shape());
    default:
      GTEST_CHECK(false,
                  "Parser: got unsupported value type, parse tensor failed.");
  }
}

// so we can get tensor by "parser_->get("input0").tensor"
MetaTensor &Parser::getMetaTensor(const std::string &name) {
  auto it = find_if(inputs_.begin(), inputs_.end(),
                    [=](const MetaTensor &t) { return t.name == name; });
  if (it != inputs_.end()) {
    return *it;
  } else {
    auto it = find_if(outputs_.begin(), outputs_.end(),
                      [=](const MetaTensor &t) { return t.name == name; });
    if (it != outputs_.end()) {
      return *it;
    }
  }
  LOG(ERROR) << "Miss tensor: " << name << " in prototxt.";
  throw std::invalid_argument(std::string(__FILE__) + " +" +
                              std::to_string(__LINE__));
}

// can get tensor by "parser_->get(2).tensor"
// but this index is the tensor index in *prototxt(input/output together)
MetaTensor &Parser::getMetaTensor(int index) {
  if (index < inputs_.size()) {
    return inputs_.at(index);
  } else {
    return outputs_.at(index - inputs_.size());
  }
}

void Parser::setCurPbPath(const std::string &filename) {
  if (filename.find("/") == std::string::npos) {
    pb_path_ = filename;
  } else {
    auto pos_pb = filename.find_last_of("/");
    pb_path_ = filename.substr(0, pos_pb + 1);
  }
}

void Parser::isSupportTF32(Node *protoNode) {
  // get proto_node_'s description
  const google::protobuf::Descriptor *desc = protoNode->GetDescriptor();
  // get proto_node_'s reflection
  const auto reflection = protoNode->GetReflection();
  std::vector<const google::protobuf::FieldDescriptor *> fields;
  reflection->ListFields(*protoNode, &fields);
  // google::protobuf::FieldDescriptor* OpParamTemp;
  std::string op_param;
  for (auto &f : fields) {
    std::string::size_type pos_param = f->name().find("param");
    std::string::size_type pos_test = f->name().find("test");
    std::string::size_type pos_handle = f->name().find("handle");

    if (pos_param != f->name().npos && pos_test == f->name().npos &&
        pos_handle == f->name().npos) {
      op_param = f->name();
    }
  }

  const google::protobuf::FieldDescriptor *op_param_des =
      desc->FindFieldByName(op_param.c_str());
  if (nullptr == op_param_des) {
    is_support_TF32_ = 0;
    return;
  }

  const google::protobuf::Message &message_param = reflection->GetMessage(
      dynamic_cast<google::protobuf::Message &>(*protoNode), op_param_des);
  // get allow_tf32 etc in  message of message
  const google::protobuf::Descriptor *message_param_new =
      message_param.GetDescriptor();
  const google::protobuf::FieldDescriptor *tf32_field_desc =
      message_param_new->FindFieldByName("allow_tf32");
  is_support_TF32_ = 1;
  if (tf32_field_desc == nullptr) {
    is_support_TF32_ = 0;
  } else {
    const auto tf32_ref = message_param.GetReflection();
    int allow_tf32_val = tf32_ref->GetInt32(message_param, tf32_field_desc);
    if (allow_tf32_val == 0) {
      is_support_TF32_ = 0;
    }
  }
}

size_t Parser::getTensorSize(Tensor *pt) {
#define GET_WIDTH_TENSOR_TYPE(TENSOR_DTYPE, WIDTH) \
  case TENSOR_DTYPE:                               \
    return WIDTH;
  switch (pt->dtype()) {
    GET_WIDTH_TENSOR_TYPE(DTYPE_COMPLEX_FLOAT, 8);
    GET_WIDTH_TENSOR_TYPE(DTYPE_COMPLEX_HALF, 4);
    GET_WIDTH_TENSOR_TYPE(DTYPE_DOUBLE, 8);
    GET_WIDTH_TENSOR_TYPE(DTYPE_INT64, 8);
    GET_WIDTH_TENSOR_TYPE(DTYPE_UINT64, 8);
    GET_WIDTH_TENSOR_TYPE(DTYPE_FLOAT, 4);
    GET_WIDTH_TENSOR_TYPE(DTYPE_INT32, 4);
    GET_WIDTH_TENSOR_TYPE(DTYPE_UINT32, 4);
    GET_WIDTH_TENSOR_TYPE(DTYPE_INT16, 2);
    GET_WIDTH_TENSOR_TYPE(DTYPE_UINT16, 2);
    GET_WIDTH_TENSOR_TYPE(DTYPE_INT8, 1);
    GET_WIDTH_TENSOR_TYPE(DTYPE_UINT8, 1);
    GET_WIDTH_TENSOR_TYPE(DTYPE_HALF, 2);
    GET_WIDTH_TENSOR_TYPE(DTYPE_BOOL, 1);
    default:
      GTEST_CHECK(false, "Parser: Unknown tensor DTYPE.");
  }
#undef GET_WIDTH_TENSOR_TYPE
}

}  // namespace mluoptest
