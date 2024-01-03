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
#ifndef CORE_GEN_CASE_H_
#define CORE_GEN_CASE_H_

#include <vector>
#include <iomanip>
#include <string>
#include <limits>
#include <utility>

#include "mlu_op.h"
#include "core/tensor.h"
#include "core/tool.h"

// macro function for user
#define MLUOP_GEN_CASE_ON (mluop::gen_case::isGenCaseOn())
// #define MLUOP_GEN_CASE_ON_NEW (mluop::gen_case::isGenCaseOn())
#define MLUOP_GEN_CASE_ON_NEW (mluop::gen_case::genCaseModeGet(true) > 0)

#define GEN_CASE_START(op_name) \
  mluop::gen_case::PbNode *node = mluop::gen_case::genCaseStart(op_name)

#define GEN_CASE_DATA(is_input, id, data, data_desc, upper_bound, lower_bound) \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc,            \
                               upper_bound, lower_bound)
// when distribution is "GAUSSIAN", upper_bound is mu, lower_bound is sigma.
#define GEN_CASE_DATA_v2(is_input, id, data, data_desc, upper_bound, \
                         lower_bound, distribution)                  \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc,  \
                               upper_bound, lower_bound, distribution)
#define GEN_CASE_DATA_UNFOLD(is_input, id, data, dim, dims, dtype, layout, \
                             upper_bound, lower_bound)                     \
  mluop::gen_case::genCaseData(node, is_input, id, data, dim, dims, dtype, \
                               layout, upper_bound, lower_bound)
// the same with GEN_CASE_DATA_v2
#define GEN_CASE_DATA_UNFOLD_v2(is_input, id, data, dim, dims, dtype, layout, \
                                upper_bound, lower_bound, distribution)       \
  mluop::gen_case::genCaseData(node, is_input, id, data, dim, dims, dtype,    \
                               layout, upper_bound, lower_bound, distribution)
#define GEN_CASE_DATA_REAL(is_input, id, data, data_desc)                    \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc, 10, -10, \
                               "UNIFORM", true)
#define GEN_CASE_DATA_REAL_V2(is_input, id, data, data_desc, upper_bound, \
                              lower_bound)                                \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc,       \
                               upper_bound, lower_bound, "UNIFORM", true)
#define GEN_CASE_DATA_REAL_UNFOLD(is_input, id, data, dim, dims, dtype,    \
                                  layout)                                  \
  mluop::gen_case::genCaseData(node, is_input, id, data, dim, dims, dtype, \
                               layout, 10, -10, "UNIFORM", true)
// special for RNN
#define GEN_CASE_DATA_RNN(is_input, id, data, data_desc, upper_bound, \
                          lower_bound, have_onchip)                   \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc,   \
                               upper_bound, lower_bound, have_onchip)
#define GEN_CASE_DATA_RNN_v2(is_input, id, data, data_desc, upper_bound, \
                             lower_bound, have_onchip, distribution)     \
  mluop::gen_case::genCaseData(node, is_input, id, data, data_desc,      \
                               upper_bound, lower_bound, have_onchip,    \
                               distribution)

#define GEN_CASE_OP_PARAM_SINGLE_HALF(pos, param_node_name, param_name, value) \
  mluop::gen_case::genCaseOpParam(                                             \
      node, param_name, value,                                                 \
      std::string(param_node_name) + std::string("_param"), MLUOP_DTYPE_HALF)
#define GEN_CASE_OP_PARAM_SINGLE(pos, param_node_name, param_name, value, ...) \
  mluop::gen_case::genCaseOpParam(                                             \
      node, param_name, value,                                                 \
      std::string(param_node_name) + std::string("_param"), ##__VA_ARGS__)
#define GEN_CASE_OP_PARAM_SINGLE_NAME(pos, param_node_name, param_name, value) \
  mluop::gen_case::genCaseOpParam(node, param_name, value, param_node_name)
#define GEN_CASE_OP_PARAM_ARRAY(pos, param_node_name, param_name, value, num) \
  mluop::gen_case::genCaseOpParam(                                            \
      node, param_name, value, num,                                           \
      std::string(param_node_name) + std::string("_param"))
#define GEN_CASE_OP_PARAM_SINGLE_SUB(pos, param_node_name, param_name, value,  \
                                     new_child)                                \
  mluop::gen_case::genCaseOpParamSub(node, param_name, value, param_node_name, \
                                     new_child)
#define GEN_CASE_OP_PARAM_ARRAY_SUB(pos, param_node_name, param_name, value, \
                                    num, new_child)                          \
  mluop::gen_case::genCaseOpParamSub(node, param_name, value, num,           \
                                     param_node_name, new_child)

#define GEN_CASE_HANDLE(handle) mluop::gen_case::genCaseHandle(node, handle)
#define GEN_CASE_HANDLE_PARAM() mluop::gen_case::genCaseHandleParam(node)

#define GEN_CASE_TEST_PARAM(is_diff1, is_diff2, is_diff3, diff1_threshold, \
                            diff2_threshold, diff3_threshold, ...)         \
  mluop::gen_case::genCaseTestParam(node, is_diff1, is_diff2, is_diff3,    \
                                    diff1_threshold, diff2_threshold,      \
                                    diff3_threshold, ##__VA_ARGS__);       \
  node->serialize();                                                       \
  node->reset()

#define GEN_CASE_TEST_PARAM_NEW(is_diff1, is_diff2, is_diff3, diff1_threshold, \
                                diff2_threshold, diff3_threshold, ...)         \
  mluop::gen_case::genCaseTestParam(node, is_diff1, is_diff2, is_diff3,        \
                                    diff1_threshold, diff2_threshold,          \
                                    diff3_threshold, ##__VA_ARGS__);           \
  node->serialize();

#define GEN_CASE_END() mluop::gen_case::genCaseEnd()

namespace mluop {
namespace gen_case {

bool ifNeedTensorStrideProcess(mluOpTensorDescriptor_t desc);
std::string descToString(mluOpTensorDescriptor_t desc, char delimiter);
// param node is structured like tree
struct ParamNode {
  std::vector<std::pair<std::string, std::string>> params;
  std::string name = "";
  std::vector<ParamNode> childs;
};

struct TensorNode {
  bool is_input;
  std::string id;
  const void *device_ptr;
  mluOpTensorDescriptor_t desc;
  bool inner_desc = false;
  std::vector<double> params;
  std::string distribution;
  bool dump_data;
  TensorNode(bool is_input, std::string id, const void *device_data,
             mluOpTensorDescriptor_t desc, bool inner_desc,
             std::vector<double> params, std::string distribution,
             bool dump_data)
      : is_input(is_input),
        id(id),
        device_ptr(device_data),
        desc(desc),
        inner_desc(inner_desc),
        params(params),
        distribution(distribution),
        dump_data(dump_data) {}

  TensorNode(const TensorNode &t) {
    is_input = t.is_input;
    id = t.id;
    device_ptr = t.device_ptr;
    inner_desc = t.inner_desc;
    if (inner_desc) {
      mluOpTensorDescriptor_t desc_;
      mluOpCreateTensorDescriptor(&desc_);
      int tensor_dim;
      mluOpTensorLayout_t layout;
      mluOpDataType_t dtype;
      mluOpGetTensorDescriptor_v2(t.desc, &layout, &dtype, &tensor_dim,
                                  nullptr);
      int64_t *dims = new int64_t[tensor_dim];
      int64_t *strides = new int64_t[tensor_dim];
      mluOpGetTensorDescriptorEx_v2(t.desc, &layout, &dtype, &tensor_dim, dims,
                                    strides);
      mluOpSetTensorDescriptorEx_v2(desc_, layout, dtype, tensor_dim, dims,
                                    strides);
      desc = desc_;
      delete[] dims;
      delete[] strides;
    } else {
      desc = t.desc;
    }
    params = t.params;
    distribution = t.distribution;
    dump_data = t.dump_data;
  }

  ~TensorNode() {
    if (inner_desc) {
      if (desc != nullptr) {
        mluOpDestroyTensorDescriptor(desc);
      }
      desc = nullptr;
    }
  }
};

enum DATASTATE { INPUT, OUTPUT };

class PbNode {
 public:
  std::string op_name;
  std::string op_type;
  std::vector<TensorNode> tensors;
  std::vector<std::string> criterions;
  std::vector<double> thresholds;
  std::vector<double> thresholds_imag;
  std::string file_name;       // pt file name
  std::string case_file_name;  // pt file name with dir
  ParamNode op_param;
  ParamNode handle_param;
  mluOpHandle_t handle;
  PbNode() {}
  ~PbNode() { reset(); }
  void reset() {
    op_name = "";
    op_type = "";
    file_name = "";
    case_file_name = "";
    for (auto &t : tensors) {
      if (t.inner_desc) {
        if (t.desc != nullptr) {
          mluOpDestroyTensorDescriptor(t.desc);
        }
        t.desc = nullptr;
      }
    }
    tensors.clear();
    criterions.clear();
    thresholds.clear();
    thresholds_imag.clear();
    op_param.name = "";
    op_param.params.clear();
    // only support one level children
    op_param.childs.clear();
    handle_param.name = "";
    handle_param.params.clear();
  }
  void setOpNameAndType(std::string op_name, std::string op_type);
  void appendTensor(bool is_input, std::string id, const void *device_data,
                    mluOpTensorDescriptor_t desc, bool inner_desc,
                    std::vector<double> params, std::string distribution,
                    bool dump_data);
  // should specialization for pointer on device
  template <typename paramType>
  inline void appendOpParam(std::string param_name, paramType param_value,
                            std::string param_node_name,
                            mluOpDataType_t dtype) {
    op_param.name = param_node_name;
    if (dtype == MLUOP_DTYPE_HALF) {
      op_param.params.push_back(
          {param_name, std::to_string(castHalfToFloat32(param_value))});
    } else if (std::is_same<paramType, int8_t>::value ||
               std::is_same<paramType, uint8_t>::value) {
      op_param.params.push_back(
          {param_name, std::to_string(static_cast<int32_t>(param_value))});
    } else {
      std::stringstream param_ss;
      param_ss.setf(std::ios::fixed);
      param_ss << std::setprecision(
                      std::numeric_limits<paramType>::max_digits10)
               << param_value;
      op_param.params.push_back({param_name, param_ss.str()});
    }
  }
  // user should control order of children
  template <typename paramType>
  inline void appendOpParamSub(std::string param_name, paramType param_value,
                               std::string param_node_name, bool new_child) {
    if (new_child) {
      op_param.childs.push_back(ParamNode());
    }
    op_param.childs.back().name = param_node_name;
    op_param.childs.back().params.push_back(
        {param_name, std::to_string(param_value)});
  }
  template <typename paramType>
  inline void appendOpParam(std::string param_name, paramType *param_value,
                            int num, std::string param_node_name,
                            mluOpDataType_t dtype) {
    for (int i = 0; i < num; i++) {
      appendOpParam(param_name, param_value[i], param_node_name, dtype);
    }
  }
  template <typename paramType>
  inline void appendOpParamSub(std::string param_name, paramType *param_value,
                               int num, std::string param_node_name,
                               bool new_child) {
    for (int i = 0; i < num; i++) {
      appendOpParamSub(param_name, param_value[i], param_node_name, new_child);
    }
  }
  // helper function for dtype
  inline int dtypeRatio(mluOpDataType_t dtype) {
    switch (dtype) {
      case MLUOP_DTYPE_INT31:
      case MLUOP_DTYPE_COMPLEX_HALF:
      case MLUOP_DTYPE_COMPLEX_FLOAT:
        return 2;
      default:
        return 1;
    }
  }
  bool dtypeFloat(mluOpDataType_t dtype) {
    switch (dtype) {
      case MLUOP_DTYPE_HALF:
      case MLUOP_DTYPE_BFLOAT16:
      case MLUOP_DTYPE_FLOAT:
      case MLUOP_DTYPE_DOUBLE:
      case MLUOP_DTYPE_COMPLEX_HALF:
      case MLUOP_DTYPE_COMPLEX_FLOAT:
        return true;
      default:
        return false;
    }
  }

  inline std::string get_tensor_random_string(int index) {
    std::stringstream random_str;
    random_str.setf(std::ios::fixed);
    random_str << "  random_data: {\n    seed: 233\n";
    random_str << "    distribution: " << tensors[index].distribution << "\n";
    if (tensors[index].distribution == "UNIFORM") {
      random_str << "    upper_bound: " << tensors[index].params[0] << "\n";
      random_str << "    lower_bound: " << tensors[index].params[1]
                 << "\n  }\n";
    } else {
      random_str << "    mu: " << tensors[index].params[0] << "\n";
      random_str << "    sigma: " << tensors[index].params[1] << "\n  }\n";
    }
    return random_str.str();
  }
  inline std::string get_dtype_value_string(mluOpDataType_t dtype) {
    switch (dtype) {
      case MLUOP_DTYPE_HALF:
      case MLUOP_DTYPE_FLOAT:
      case MLUOP_DTYPE_DOUBLE:
      case MLUOP_DTYPE_COMPLEX_HALF:
      case MLUOP_DTYPE_COMPLEX_FLOAT:
        return "  value_f: ";
      case MLUOP_DTYPE_INT8:
      case MLUOP_DTYPE_INT16:
      case MLUOP_DTYPE_INT32:
      case MLUOP_DTYPE_BOOL:
      case MLUOP_DTYPE_INT31:
        return "  value_i: ";
      case MLUOP_DTYPE_INT64:
        return "  value_l: ";
      case MLUOP_DTYPE_UINT8:
      case MLUOP_DTYPE_UINT16:
      case MLUOP_DTYPE_UINT32:
      case MLUOP_DTYPE_BFLOAT16:
        return "  value_ui: ";
      case MLUOP_DTYPE_UINT64:
        return "  value_ul: ";
      default:
        return "  value_i: ";
    }
  }
  inline std::string get_data_string(mluOpDataType_t dtype, void *data,
                                     uint64_t offset) {
    switch (dtype) {
      case MLUOP_DTYPE_HALF:
        return std::to_string(castHalfToFloat32(((int16_t *)data)[offset]));
      case MLUOP_DTYPE_BFLOAT16:
        return std::to_string(((uint16_t *)data)[offset]);
      case MLUOP_DTYPE_FLOAT:
        return std::to_string(((float *)data)[offset]);
      case MLUOP_DTYPE_DOUBLE:
        return std::to_string(((double *)data)[offset]);
      case MLUOP_DTYPE_COMPLEX_HALF:
        return std::to_string(castHalfToFloat32(((int16_t *)data)[offset]));
      case MLUOP_DTYPE_COMPLEX_FLOAT:
        return std::to_string(((float *)data)[offset]);
      case MLUOP_DTYPE_INT8:
        return std::to_string(((int8_t *)data)[offset]);
      case MLUOP_DTYPE_INT16:
        return std::to_string(((int16_t *)data)[offset]);
      case MLUOP_DTYPE_INT32:
        return std::to_string(((int32_t *)data)[offset]);
      case MLUOP_DTYPE_BOOL:
        return std::to_string(((int8_t *)data)[offset]);
      case MLUOP_DTYPE_INT31:
        return std::to_string(((int16_t *)data)[offset]);
      case MLUOP_DTYPE_INT64:
        return std::to_string(((int64_t *)data)[offset]);
      case MLUOP_DTYPE_UINT8:
        return std::to_string(((uint8_t *)data)[offset]);
      case MLUOP_DTYPE_UINT16:
        return std::to_string(((uint16_t *)data)[offset]);
      case MLUOP_DTYPE_UINT32:
        return std::to_string(((uint32_t *)data)[offset]);
      case MLUOP_DTYPE_UINT64:
        return std::to_string(((uint64_t *)data)[offset]);
      default:
        return std::to_string(((int8_t *)data)[offset]);
    }
  }
  inline std::string get_data_hex_string(mluOpDataType_t dtype, void *data,
                                         uint64_t offset) {
    std::stringstream s;
    switch (dtype) {
      case MLUOP_DTYPE_HALF:
        s << std::hex << ((uint16_t *)data)[offset];
        break;
      case MLUOP_DTYPE_BFLOAT16:
        s << std::hex << ((uint16_t *)data)[offset];
        break;
      case MLUOP_DTYPE_FLOAT:
        s << std::hex << ((uint32_t *)data)[offset];
        break;
      case MLUOP_DTYPE_DOUBLE:
        s << std::hex << ((uint64_t *)data)[offset];
        break;
      case MLUOP_DTYPE_COMPLEX_HALF:
        s << std::hex << ((uint16_t *)data)[offset];
        break;
      case MLUOP_DTYPE_COMPLEX_FLOAT:
        s << std::hex << ((uint32_t *)data)[offset];
        break;
      default:
        s << std::hex << ((uint32_t *)data)[offset];
        break;
    }
    return "\"" + s.str() + "\"";
  }
  inline uint64_t getTensorSize(int index) {
    int dim;
    mluOpTensorLayout_t layout;
    mluOpDataType_t dtype;
    mluOpGetTensorDescriptor_v2(tensors[index].desc, &layout, &dtype, &dim,
                                nullptr);
    mluOpPointerMode_t pointer_mode;
    mluOpGetTensorDescriptorPointerMode(tensors[index].desc, &pointer_mode);
    int64_t *dims = new int64_t[dim];
    int64_t *strides = new int64_t[dim];
    mluOpGetTensorDescriptorEx_v2(tensors[index].desc, &layout, &dtype, &dim,
                                  dims, strides);
    // if tensor not be set, total_element_num will be 0
    uint64_t count = 1;
    for (int i = 0; i < dim; i++) {
      count *= dims[i];
    }
    // some magic in here
    uint64_t total_num = 1;
    if (count != 1) {
      if (mluop::gen_case::ifNeedTensorStrideProcess(tensors[index].desc)) {
        for (int i = 0; i < dim; i++) {
          if (dims[i] == 0) {
            total_num = 0;
            break;
          }
          total_num += (dims[i] - 1) * strides[i];
        }
      } else {
        total_num = count;
      }
    }
    delete[] dims;
    delete[] strides;
    return total_num;
  }
  inline void *getDeviceData(int index) {
    uint64_t total_num = getTensorSize(index);
    mluOpDataType_t dtype;
    mluOpGetTensorDescriptor(tensors[index].desc, nullptr, &dtype, nullptr,
                             nullptr);
    uint64_t data_size = total_num * mluop::getSizeOfDataType(dtype);
    void *data = malloc(data_size);
    auto memcpy_dir =
        (tensors[index].desc->pointer_mode == MLUOP_POINTER_MODE_HOST
             ? CNRT_MEM_TRANS_DIR_HOST2HOST
             : CNRT_MEM_TRANS_DIR_DEV2HOST);
    if (CNRT_RET_SUCCESS ==
        cnrtMemcpy(data, const_cast<void *>(tensors[index].device_ptr),
                   data_size, memcpy_dir)) {
      return data;
    } else {
      LOG(ERROR) << "[gen_case] Dump data failed! cnrtMemcpy data size is "
                 << data_size << " byte.";
      return nullptr;
    }
  }
  void appendCriterion(std::string criterion, double threshold,
                       double threshold_imag);
  std::string getFileName();
  std::string getFolderName();
  int mkdir();
  void setHandle(mluOpHandle_t handle) { this->handle = handle; }
  void getHandleParam();
  void dumpDataFile(std::string file_name, std::string folder_name, int index,
                    std::ofstream &case_file, enum DATASTATE data_state);
  void dumpOutputFile();
  void dumpToFile(bool valueDump = false);
  void printOnScreen();
  void serialize();
  void debugTensorAddress();
};

template <>
inline void PbNode::appendOpParam<std::string>(std::string param_name,
                                               std::string param_value,
                                               std::string param_node_name,
                                               mluOpDataType_t dtype) {
  op_param.name = param_node_name;
  op_param.params.push_back({param_name, param_value});
}

template <>
inline void PbNode::appendOpParam<const char *>(std::string param_name,
                                                const char *param_value,
                                                std::string param_node_name,
                                                mluOpDataType_t dtype) {
  op_param.name = param_node_name;
  op_param.params.push_back({param_name, std::string(param_value)});
}

template <>
inline void PbNode::appendOpParam<char *>(std::string param_name,
                                          char *param_value,
                                          std::string param_node_name,
                                          mluOpDataType_t dtype) {
  op_param.name = param_node_name;
  op_param.params.push_back({param_name, std::string(param_value)});
}

template <>
inline void PbNode::appendOpParam<const void *>(std::string param_name,
                                                const void *param_value,
                                                std::string param_node_name,
                                                mluOpDataType_t dtype) {
  op_param.name = param_node_name;
  cnrtPointerAttributes_t attr;
  cnrtPointerGetAttributes(&attr, param_value);
  int data_width = mluop::getSizeOfDataType(dtype);
  if (attr.type == cnrtMemTypeDevice) {
    void *data = malloc(data_width);
    if (CNRT_RET_SUCCESS == cnrtMemcpy(data, const_cast<void *>(param_value),
                                       data_width,
                                       CNRT_MEM_TRANS_DIR_DEV2HOST)) {
      op_param.params.push_back({param_name, get_data_string(dtype, data, 0)});
    } else {
      LOG(ERROR) << "[gen_case] dump op param failed, param_name is "
                 << param_name << " param_node_name is " << param_node_name;
    }
    free(data);
  } else {
    op_param.params.push_back(
        {param_name,
         get_data_string(dtype, const_cast<void *>(param_value), 0)});
  }
}

template <>
inline void PbNode::appendOpParamSub<std::string>(std::string param_name,
                                                  std::string param_value,
                                                  std::string param_node_name,
                                                  bool new_child) {
  if (new_child) {
    op_param.childs.push_back(ParamNode());
  }
  op_param.childs.back().name = param_node_name;
  op_param.childs.back().params.push_back({param_name, param_value});
}

template <>
inline void PbNode::appendOpParamSub<char *>(std::string param_name,
                                             char *param_value,
                                             std::string param_node_name,
                                             bool new_child) {
  if (new_child) {
    op_param.childs.push_back(ParamNode());
  }
  op_param.childs.back().name = param_node_name;
  op_param.childs.back().params.push_back(
      {param_name, std::string(param_value)});
}

template <>
inline void PbNode::appendOpParamSub<const char *>(std::string param_name,
                                                   const char *param_value,
                                                   std::string param_node_name,
                                                   bool new_child) {
  if (new_child) {
    op_param.childs.push_back(ParamNode());
  }
  op_param.childs.back().name = param_node_name;
  op_param.childs.back().params.push_back(
      {param_name, std::string(param_value)});
}

bool isGenCaseOn();

// true is used in MLUOP_GEN_CASE_ON, false is for internal use
int genCaseModeGet(bool first);
void genCaseModeRestore();
void genCaseModeSet(int mode);
inline int getOpNameMask(const std::string op_name_, const std::string op_name);

PbNode *genCaseStart(std::string op_name, std::string op_type = "NONE");
void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, mluOpTensorDescriptor_t desc,
                 double param1, double param2,
                 std::string distribution = "UNIFORM", bool dump_data = false);
void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, mluOpSeqDataDescriptor_t desc,
                 double param1, double param2, bool have_onchop,
                 std::string distribution = "UNIFORM", bool dump_data = false);
void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, int64_t *dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2,
                 std::string distribution = "UNIFORM", bool dump_data = false);
void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, const int64_t *dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2,
                 std::string distribution = "UNIFORM", bool dump_data = false);
void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, std::vector<int64_t> dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2,
                 std::string distribution = "UNIFORM", bool dump_data = false);
template <typename paramType>
void genCaseOpParam(PbNode *node, std::string param_name, paramType param_value,
                    std::string param_node_name = "",
                    mluOpDataType_t dtype = MLUOP_DTYPE_FLOAT) {
  node->appendOpParam(param_name, param_value, param_node_name, dtype);
}
template <typename paramType>
void genCaseOpParam(PbNode *node, std::string param_name,
                    paramType *param_value, int num,
                    std::string param_node_name = "",
                    mluOpDataType_t dtype = MLUOP_DTYPE_FLOAT) {
  node->appendOpParam(param_name, param_value, num, param_node_name, dtype);
}
template <typename paramType>
void genCaseOpParamSub(PbNode *node, std::string param_name,
                       paramType param_value, std::string param_node_name = "",
                       bool new_child = false) {
  node->appendOpParamSub(param_name, param_value, param_node_name, new_child);
}
template <typename paramType>
void genCaseOpParamSub(PbNode *node, std::string param_name,
                       paramType param_value, int num,
                       std::string param_node_name = "",
                       bool new_child = false) {
  node->appendOpParamSub(param_name, param_value, num, param_node_name,
                         new_child);
}
void genCaseTestParam(PbNode *node, bool is_diff1, bool is_diff2, bool is_diff3,
                      const float diff1_threshold, const float diff2_threshold,
                      const float diff3_threshold,
                      const float diff1_threshold_imag = -1,
                      const float diff2_threshold_imag = -1,
                      const float diff3_threshold_imag = -1);
void genCaseHandle(PbNode *node, mluOpHandle_t handle);
void genCaseHandleParam(PbNode *node);
void genCaseEnd();
}  // namespace gen_case
}  // namespace mluop
#endif  // CORE_GEN_CASE_H_
