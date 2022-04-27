/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
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
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <functional>
#include "tools.h"
#include "parser.h"

namespace mluoptest {

  Parser::~Parser() {
    if (proto_node_ != nullptr) {
      delete proto_node_;
      proto_node_ = nullptr;
    }
  }

  void Parser::parse(const std::string &file) {
    proto_node_ = new Node;
    GTEST_CHECK(readMessageFromFile(file, proto_node_),
                "Parser: parse *pb/*prototxt failed.");
    GTEST_CHECK(proto_node_->has_op_name(),
                "Parser: missing op name in prototxt.");

    // 1.get device
    if (proto_node_->has_device()) {
      device_ = proto_node_->device();
    } else if (proto_node_->has_test_param()) {
      if (proto_node_->mutable_test_param()->has_baseline_device()) {
        device_ = proto_node_->mutable_test_param()->baseline_device();
      } else {
        LOG(WARNING)
            << "Parser: missing device in test param, using CPU as default.";
        device_ = Device::CPU; // default cpu
      }
    } else {
      LOG(WARNING) << "Parser: missing baseline device, using CPU as default.";
      device_ = Device::CPU; // default cpu
    }

    // 2.get criterion
    if (common_threshold()) {
      criterions_.clear();
      if (proto_node_->has_test_param()) {
        auto test_param = proto_node_->mutable_test_param();
        GTEST_CHECK(test_param->error_func_size() != 0);
        GTEST_CHECK(test_param->error_func_size() ==
                        test_param->error_threshold_size(),
                    "Parser: error func's number should equal to threshold's "
                    "number, now they are "
                    "not equal.");

        auto num = test_param->error_func_size();
        for (int i = 0; i < num; ++i) {
          auto func = cvtProtoEvaluationCriterion(test_param->error_func(i));
          criterions_.insert(std::move(
              Evaluator::Criterion(func, test_param->error_threshold(i))));
        }
      } else {
        GTEST_CHECK(proto_node_->evaluation_criterion_size() != 0);
        GTEST_CHECK(proto_node_->evaluation_criterion_size() ==
                        proto_node_->evaluation_threshold_size(),
                    "Parser: criterion's number should equal to threshold's "
                    "number, now they are not equal.");

        auto num = proto_node_->evaluation_criterion_size();
        for (int i = 0; i < num; ++i) {
          auto proto_func = proto_node_->evaluation_criterion(i);
          auto func       = cvtProtoEvaluationCriterion(proto_func);
          criterions_.insert(std::move(Evaluator::Criterion(
              func, proto_node_->evaluation_threshold(i))));
        }
      }
    }

    // 3. inputs/outputs
    auto parse_tensor = [=](MetaTensor *mt, Tensor *pt) {
      mt->is_null = (pt->id().find("NULL") != std::string::npos) ? true : false;
      if (unlikely(mt->is_null)) {
        VLOG(4) << "WARNING: found tensor is null, skip parsing else data.";
        return; // if null, don't need parse other info.
      }
      mt->name       = pt->id();
      mt->value_type = getValueType(pt);

      // 1.shape set to tensor desc
      GTEST_CHECK(pt->has_shape(), "Parser: missing tensor shape in prototxt.");
      mt->shape.resize(pt->mutable_shape()->dims_size());
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
      // shape_count come from value_f/value_i/value_h and shape.
      // not include stride
      mt->shape_count   = getTensorShapeCount(pt);
      mt->sizeof_dtype  = getSizeOfDataType(mt->dtype);
      mt->size_in_bytes = mt->total_count * mt->sizeof_dtype;

      if (mt->total_count != mt->shape_count) {
        VLOG(4) << "WARNING: Parser: the " << mt->name
                << " strided element count is " << mt->total_count
                << " while shape element count is " << mt->shape_count;
      }

      mt->position = pt->has_position() ? pt->position() : 0;
      mt->scale    = pt->has_scale() ? pt->scale() : 1.0f;
      mt->offset   = pt->has_offset() ? pt->offset() : 0;

      // 4.check
      checkTensorValid(mt, pt);
    };

    inputs_.resize(proto_node_->input_size());
    for (size_t i = 0; i < proto_node_->input_size(); ++i) {
      parse_tensor(&inputs_[i], proto_node_->mutable_input(i));
    }

    outputs_.resize(proto_node_->output_size());
    for (size_t i = 0; i < proto_node_->output_size(); ++i) {
      parse_tensor(&outputs_[i], proto_node_->mutable_output(i));
    }
  }

  // check if tensor value is equal to shape.
  // when found value_i value_f value_h, just check value_size and shape_size
  // when found random_param, check random param
  // here allow shape_size != data size saved in pb
  // cuz shape_size is for create tensor and data_size is for malloc space
  // so just print warning.
  void Parser::checkTensorValid(MetaTensor *mt, Tensor *pt) {
    int shape_count = 1;
    switch (mt->value_type) {
      case VALUE_F:
      case VALUE_I:
      case VALUE_L:
      case VALUE_H:
        shape_count = std::accumulate(mt->shape.begin(),
                                      mt->shape.end(),
                                      shape_count,
                                      std::multiplies<int>());
        GTEST_WARNING(mt->shape_count == shape_count,
                      "Parser: found shape count is not equal to value "
                      "size.(mt->shape_count is value_size)");
        break;
      case VALUE_RANDOM:
        if (!mt->empty()) {
          checkRandomParam(pt);
        }
        break;
      case VALUE_PATH:
        // if found path(only) in pb, but can't access this path, throw.
        GTEST_CHECK((access(pt->path().c_str(), 4) == -1),
                    "Parser: open path saved in *prototxt failed.");
        break;
      case VALUE_INVALID:
        // check output, if may shape empty, value is empty, and not random
        // param, so don't need check.
        break;
      default:
        GTEST_CHECK(false,
                    "Parser: got unsupported value type, parse tensor failed.");
    }
  }

  void Parser::checkRandomParam(Tensor *pt) {
    GTEST_CHECK(pt->has_random_data(),
                "Parser: missing random param of tensor saved in *pb");
    GTEST_CHECK(pt->mutable_random_data()->has_distribution(),
                "Parser: missing distribution of random param saved in *pb");

    auto random_data = pt->mutable_random_data();
    if (random_data->distribution() == mluoptest::UNIFORM) {
      GTEST_CHECK(random_data->has_seed(),
                  "Parser: missing seed of UNIFORM random param in *pb");
      GTEST_CHECK(random_data->has_upper_bound(),
                  "Parser: missing upper bound of UNIFORM random param in *pb");
      GTEST_CHECK(random_data->has_lower_bound(),
                  "Parser: missing lower bound of UNIFORM random param in *pb");
    } else if (random_data->distribution() == mluoptest::GAUSSIAN) {
      GTEST_CHECK(random_data->has_seed(),
                  "Parser: missing seed of GAUSSIAN random param in *pb");
      GTEST_CHECK(random_data->has_mu(),
                  "Parser: missing mu of UNIFORM random param in *pb");
      GTEST_CHECK(random_data->has_sigma(),
                  "Parser: missing sigma of UNIFORM random param in *pb");
    } else {
      GTEST_CHECK(
          false,
          "Parser: got unsupported distribution when check tensor valid.");
    }
  }

  // return value's dtype is according to value type.
  // if value type is value_*, return dtype is dtype in proto.
  // if value type is random, return dtype is fp32
  void Parser::getInputTensorValue(size_t index, void *data, size_t count) {
    getTensorValue(proto_node_->mutable_input(index),
                   data,
                   inputs_[index].value_type,
                   count);
  }

  // return value's dtype is according to value type.
  // if value type is value_*, return dtype is dtype in proto.
  // if value type is random, return dtype is fp32
  void Parser::getOutputTensorValue(size_t index, void *data, size_t count) {
    getTensorValue(proto_node_->mutable_output(index),
                   data,
                   outputs_[index].value_type,
                   count);
  }

  // get value from field value_f
  // but this way have precision problem
  // we will abandon value_f
  void Parser::getTensorValueF(const Tensor *pt, void *data, size_t count) {
    GTEST_CHECK(pt->value_f_size() == count,
                "Parser: when read value_f, expected element num is not equal "
                "to real element num.");
    switch (pt->dtype()) {
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
      default:
        GTEST_CHECK(false,
                    "Parser: found unsuppored dtype in value_f, value_f only "
                    "supporte float/half.");
    }
  }

  // get value from value_i
  // no quant intx, saved in value_i
  void Parser::getTensorValueI(const Tensor *pt, void *data, size_t count) {
    if (pt->dtype() == DTYPE_INT31 || pt->dtype() == DTYPE_COMPLEX_HALF ||
        pt->dtype() == DTYPE_COMPLEX_FLOAT) {
      GTEST_CHECK(pt->value_i_size() == count * 2,
                  "Parser: when read value_i, expected element num is not "
                  "equal to real element num.");
    } else {
      GTEST_CHECK(pt->value_i_size() == count,
                  "Parser: when read value_i, expected element num is not "
                  "equal to real element num.");
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
      case DTYPE_BOOL: // parser value_i == BOOL
        for (int i = 0; i < count; ++i) {
          ((int8_t *)data)[i] = pt->value_i(i);
        }
        break;
      case DTYPE_INT31: {
        // in generator prototxt, 1*int31 split into 2*int16, so data_num =
        // value_size() / 2
        auto num = count;
        for (int i = 0; i < num; ++i) {
          int16_t I1                 = pt->value_i(i);
          int16_t I2                 = pt->value_i(i + num);
          ((int16_t *)data)[i]       = I2; // low int16
          ((int16_t *)data)[i + num] = I1; // high int16
        }
      } break;
      case DTYPE_HALF:
        for (int i = 0; i < count; ++i) {
          ((int16_t *)data)[i] = pt->value_i(i);
        }
        break;
      case DTYPE_FLOAT:
        for (int i = 0; i < count; ++i) {
          int value_i        = pt->value_i(i);
          ((float *)data)[i] = *((float *)(&value_i));
        }
        break;
      case DTYPE_COMPLEX_HALF:
        for (int i = 0; i < 2 * count; i += 2) {
          ((int16_t *)data)[i]     = pt->value_i(i);
          ((int16_t *)data)[i + 1] = pt->value_i(i + 1);
        }
        break;
      case DTYPE_COMPLEX_FLOAT:
        for (int i = 0; i < 2 * count; i += 2) {
          int value_i            = pt->value_i(i);
          int value_i_imag       = pt->value_i(i + 1);
          ((float *)data)[i]     = *((float *)(&value_i));
          ((float *)data)[i + 1] = *((float *)(&value_i_imag));
        }
        break;
      default:
        GTEST_CHECK(
            false,
            "Parser: found unsuppored dtype in value_i, value_i only support "
            "int8/uint8/int16/int31/int32/int64/bool.");
    }
  }

  // get value from value_l
  // no quant intx, saved in value_l
  void Parser::getTensorValueL(const Tensor *pt, void *data, size_t count) {
    GTEST_CHECK(pt->value_l_size() == count,
                "Parser: when read value_l, expected element num is not equal "
                "to real element num.");
    switch (pt->dtype()) {
      case DTYPE_INT64:
        for (int i = 0; i < count; ++i) {
          ((int64_t *)data)[i] = pt->value_l(i);
        }
        break;
      default:
        GTEST_CHECK(false,
                    "Parser: found unsuppored dtype in value_l, value_l only "
                    "support int64.");
    }
  }

  inline float str2fp32(const std::string *in_str) {
    uint32_t res = 0x0000;
    for (int i = 0; i < in_str->size(); ++i) {
      char byte = in_str->c_str()[i]; // 0~f
      res       = res << 4;
      res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0'; // 0~15
    }
    return *(float *)(&res);
  }

  inline uint16_t str2fp16(const std::string *in_str) {
    uint16_t res = 0x00;
    for (int i = 0; i < in_str->size(); ++i) {
      char byte = in_str->c_str()[i];
      res       = res << 4;
      res |= 0xf & (byte >= 'a') ? byte - 'a' + 10 : byte - '0';
    }
    return res;
  }

  // get value by value_h (hex)
  // we hope all float value come from value_h to keep precision
  void Parser::getTensorValueH(Tensor *pt, void *data, size_t count) {
    GTEST_CHECK(pt->value_h_size() == count,
                "Parser: when read value_h, expected element num is not equal "
                "to real element num.");
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
      default:
        GTEST_CHECK(false,
                    "Parser: found unsuppored dtype in value_h, value_h only "
                    "supporte float/half.");
    }
  }

  // get value by random data param
  void Parser::getTensorValueRandom(Tensor *pt, float *data, size_t count) {
    generateRandomData(data, count, pt->mutable_random_data(), pt->dtype());
  }

  // get value by random data param
  void Parser::getTensorValueByFile(Tensor *pt, float *data, size_t count) {
    readDataFromFile(pt->path(), data, count);
  }

  // set value in proto to meta_tensor.ptr
  // random data(for cpu compute) value is fp32 definitely
  // valueh valuef valuei dtype is according dtype in proto
  void Parser::getTensorValue(Tensor *pt,
                              void *data,
                              ValueType value_type,
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
      case VALUE_RANDOM:
        getTensorValueRandom(pt, (float *)data, count); // cpu mode dtype fp32
        break;
      case VALUE_PATH:
        getTensorValueByFile(pt, (float *)data, count); // cpu mode dtype fp32
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
        int threshold_use =
            (int)proto_node_->mutable_output(i)->threshold_use();
        res.push_back(threshold_use);
      } else {
        res.push_back(1);
      }
    }
    return res;
  }

  bool Parser::check_threshold() {
    std::vector<int> threshold_use = Parser::threshold_use();
    bool res                       = false;

    if (proto_node_->has_test_param()) {
      if (0 != proto_node_->mutable_test_param()->error_threshold_size()) {
        res = true;
      }
    } else if (0 != proto_node_->evaluation_threshold_size()) {
      res = true;
    } else {
      for (int i = 0; i < outputs_.size(); ++i) {
        Tensor *pt = proto_node_->mutable_output(i); // pt for proto_tensor
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

  std::set<Evaluator::Criterion>
      Parser::criterions(int index, std::vector<int> criterions_use) {
    std::set<Evaluator::Criterion> res;

    if (proto_node_->has_test_param()) {
      auto test_param = proto_node_->mutable_test_param();
      GTEST_CHECK(test_param->error_func_size() != 0);
      GTEST_CHECK(test_param->error_func_size() ==
                      test_param->error_threshold_size(),
                  "Parser: error func's number should equal to threshold's "
                  "number, now they are not equal.");

      auto num = test_param->error_func_size();
      for (int i = 0; i < num; ++i) {
        auto func = cvtProtoEvaluationCriterion(test_param->error_func(i));
        res.insert(Evaluator::Criterion(func, test_param->error_threshold(i)));
      }
    } else {
      GTEST_CHECK(proto_node_->evaluation_criterion_size() != 0);

      GTEST_CHECK(criterions_use.size() == 4,
                  "criterions_use_'s size should be 4, now it's not");
      auto num   = proto_node_->evaluation_criterion_size();
      Tensor *pt = proto_node_->mutable_output(0); // pt for proto_tensor
      for (int i = 0; i < num; ++i) {
        if (0 == criterions_use[i]) {
          continue;
        }
        auto proto_func = proto_node_->evaluation_criterion(i);
        auto func       = cvtProtoEvaluationCriterion(proto_func);
        if (-1 == index) {
          GTEST_CHECK(proto_node_->evaluation_criterion_size() ==
                          proto_node_->evaluation_threshold_size(),
                      "Parser: criterion's number should equal to threshold's "
                      "number, now they are not "
                      "equal.");
          res.insert(
              Evaluator::Criterion(func, proto_node_->evaluation_threshold(i)));
        } else {
          Tensor *pt =
              proto_node_->mutable_output(index); // pt for proto_tensor
          GTEST_CHECK(proto_node_->evaluation_criterion_size() ==
                          pt->thresholds().evaluation_threshold_size(),
                      "Parser: criterion's number should equal to threshold's "
                      "number, now they are "
                      "not equal.");
          res.insert(Evaluator::Criterion(
              func, pt->thresholds().evaluation_threshold(i)));
        }
      }
    }
    return res;
  }

  bool Parser::readMessageFromFile(const std::string &filename, Node *proto) {
    std::ifstream fin(filename, std::ios::in);
    if (!fin.is_open()) {
      LOG(ERROR) << "File not found: " << filename;
      fin.close();
      return false;
    }
    VLOG(4) << "Open " << filename;

    bool status = false;
    google::protobuf::io::IstreamInputStream input(&fin);
    if (filename.find(".prototxt") != std::string::npos) {
      status = google::protobuf::TextFormat::Parse(&input, proto);
    } else if (filename.find(".pb") != std::string::npos) {
      google::protobuf::io::CodedInputStream coded_input(&input);
      coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX - 1);
      status = proto->ParseFromCodedStream(&coded_input);
    }
    fin.close();

    return status;
  }

  void Parser::getTestInfo() {
    std::unordered_map<std::string, std::vector<std::string>> test_info;
    test_info = readFileByLine("../../test/mluop_gtest/gtest_config/test_list");
    bl_zeroinput_     = test_info["black_list_zero_input"];
    bl_mlu_only_fast_ = test_info["black_list_mlu_only_fast"];
  }

  Evaluator::Formula
      Parser::cvtProtoEvaluationCriterion(EvaluationCriterion f) {
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
    // special case: int31 = int16 * 2 in pb
    if (pt->dtype() == DTYPE_INT31 && value_type == VALUE_I) {
      return pt->value_i_size() / 2;
    }

    switch (value_type) {
      case VALUE_H:
        return pt->value_h_size();
      case VALUE_I:
        return pt->value_i_size();
      case VALUE_L:
        return pt->value_l_size();
      case VALUE_F:
        return pt->value_f_size();
      case VALUE_RANDOM:
      case VALUE_PATH:
      case VALUE_INVALID:
        GTEST_CHECK(pt->has_shape(),
                    "Parser: missing tensor shape in prototxt.");
        return shapeStrideCount(pt->mutable_shape());
      default:
        GTEST_CHECK(false,
                    "Parser: got unsupported value type, parse tensor failed.");
    }
  }

  // so we can get tensor by "parser_->get("input0").tensor"
  MetaTensor &Parser::getMetaTensor(const std::string &name) {
    auto it = find_if(inputs_.begin(), inputs_.end(), [=](const MetaTensor &t) {
      return t.name == name;
    });
    if (it != inputs_.end()) {
      return *it;
    } else {
      auto it = find_if(outputs_.begin(),
                        outputs_.end(),
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

} // namespace mluoptest
