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
#include "core/gen_case.h"

namespace mluop {
namespace gen_case {

// false is for internal use
#define IS_DUMP_DATA (genCaseModeGet(false) == 2)
#define IS_ONLY_SHOW (genCaseModeGet(false) == 3)

// mode_stacks_ is used for eliminate internal prototxt in mluOp interface
__attribute__((__unused__)) std::unordered_map<std::string, std::vector<int>>
    mode_stacks_;
// nodes_ is like mode_stacks, keep nodes_vector for each thread
__attribute__((__unused__)) std::unordered_map<std::string, std::vector<PbNode>>
    nodes_;

// create directory and modify mode_stacks_ should be thread-safe
__attribute__((__unused__)) std::mutex stacks_mutex_;

// details of environment description can be found on Wiki

// Get MLUOP_GEN_CASE from env.
// MLUOP_GEN_CASE can be changed by genCaseModeSet
// MLUOP_GEN_CASE=1: Generate gen_case file without input data
// MLUOP_GEN_CASE=2: Generate gen_case file with input data
// MLUOP_GEN_CASE=3: Print gen_case simple infomation on screen
__attribute__((__unused__)) int gen_case_mode_ =
    mluop::getUintEnvVar("MLUOP_GEN_CASE", 0);

// MLUOP_GEN_CASE_DUMP_INTERNAL control whether dump internal mluOpapi call
__attribute__((__unused__)) bool dump_internal_ =
    mluop::getBoolEnvVar("MLUOP_GEN_CASE_DUMP_INTERNAL", false);

// MLUOP_GEN_CASE_OP_NAME control generating prototxt
// "conv;abs" means only generate prototxt for conv and abs
// "-conv;-abs" means only not generate prototxt for conv and abs
__attribute__((__unused__)) std::string op_name_ =
    mluop::getStringEnvVar("MLUOP_GEN_CASE_OP_NAME", "all");

// MLUOP_GEN_CASE_DUMP_DATA control whether dump input device data in prototxt
// or not 0 : means not dump 1 : means dump readable value, is default value 2 :
// means dump hex value
__attribute__((__unused__)) int dump_data_ =
    mluop::getUintEnvVar("MLUOP_GEN_CASE_DUMP_DATA", 0);

// MLUOP_GEN_CASE_DUMP_DATA_OUTPUT control whether dump output device data in
// prototxt or not 0 : means not dump, is default value 1 : means dump readable
// value 2 : means dump hex value
__attribute__((__unused__)) int dump_data_output_ =
    mluop::getUintEnvVar("MLUOP_GEN_CASE_DUMP_DATA_OUTPUT", 0);

// MLUOP_GEN_CASE_DUMP_DATA_FILE control whether dump data file separately
// 0 : means not dump file
// 1 : means dump file
__attribute__((__unused__)) int dump_data_file_ =
    mluop::getUintEnvVar("MLUOP_GEN_CASE_DUMP_DATA_FILE", 0);

bool isGenCaseOn() { return gen_case_mode_ > 0; }

int genCaseModeGet(bool first) {
  if (gen_case_mode_ > 0) {
    // genCaseModeGet(false) just return gen_case_mode_ for IS_DUMP_DATA
    if (!first) {
      return gen_case_mode_;
    }
    std::lock_guard<std::mutex> guard(stacks_mutex_);
    std::string tid(std::to_string(syscall(SYS_gettid)));
    int mode = dump_internal_ ? gen_case_mode_ : 0;
    auto it = mode_stacks_.find(tid);
    if (it != mode_stacks_.end()) {
      auto &mode_stack = it->second;
      // the top of mode_stack store the gen_case mode for current thread
      if (first) {
        mode_stack.push_back(mode_stack.front());
        mode_stack.front() = mode;
      }
      // during current mluOpapi, gen_case mode is on the bottom
      return mode_stack.back();
    } else {
      std::vector<int> mode_stack(1, gen_case_mode_);
      mode_stack.push_back(mode_stack.front());
      mode_stack.front() = mode;
      mode_stacks_.emplace(tid, mode_stack);
      return gen_case_mode_;
    }
  } else {
    return 0;
  }
}

void genCaseModeRestore() {
  if (gen_case_mode_ > 0) {
    std::lock_guard<std::mutex> guard(stacks_mutex_);
    std::string tid(std::to_string(syscall(SYS_gettid)));
    auto it = mode_stacks_.find(tid);
    if (it != mode_stacks_.end()) {
      auto &mode_stack = it->second;
      // use gen_case mode of current mluOpapi to restore current thread
      mode_stack.front() = mode_stack.back();
      mode_stack.pop_back();
    } else {
      LOG(WARNING) << "[gen_case] GEN_CASE_END not matched, please check.";
    }
  }
}

// should update mode in mode_stacks_
void genCaseModeSet(int mode) {
  if (mode < 0 && mode > 4) {
    mode = 0;
  }
  std::lock_guard<std::mutex> guard(stacks_mutex_);
  for (auto &mode_stacks_it : mode_stacks_) {
    auto &mode_stack = mode_stacks_it.second;
    for (int i = 0; i < mode_stack.size(); i++) {
      mode_stack[i] = mode;
    }
  }
  gen_case_mode_ = mode;
  LOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
}

// TO DO: can use regex and global variable for efficiency
inline int getOpNameMask(const std::string op_name_,
                         const std::string op_name) {
  if (op_name_ == "all") {
    return 1;
  }
  std::unordered_map<std::string, int> op_name_mask;
  // regex may cause invalid pointer
  std::vector<std::string> split;
  std::string tmp;
  for (auto &c : op_name_) {
    if (c == ';') {
      if (!tmp.empty()) {
        split.push_back(tmp);
        tmp = "";
      }
    } else {
      tmp.push_back(c);
    }
  }
  if (!tmp.empty()) {
    split.push_back(tmp);
    tmp = "";
  }

  int specify_op = 0;
  for (auto &name : split) {
    if (name[0] == '-') {
      specify_op = 1;
      op_name_mask.emplace(name.substr(1, name.size() - 1), -1);
    } else if (name[0] == '+') {
      specify_op = 1;
      op_name_mask.emplace(name.substr(1, name.size() - 1), 2);
    } else {
      op_name_mask.emplace(name, 1);
    }
  }
  if (op_name_mask.find(op_name) != op_name_mask.end()) {
    return op_name_mask[op_name] == 1;
  } else {
    return specify_op;
  }
}

PbNode *genCaseStart(std::string op_name) {
  std::lock_guard<std::mutex> guard(stacks_mutex_);
  std::string tid(std::to_string(syscall(SYS_gettid)));
  auto it = nodes_.find(tid);
  if (it != nodes_.end()) {
    auto &nodes_vector = it->second;
    // find empty slot of node
    for (int i = 0; i < nodes_vector.size(); i++) {
      // so after serialization, node should be reset
      if (nodes_vector[i].op_name == "") {
        nodes_vector[i].setOpNameAndType(op_name);
        return &nodes_vector[i];
      }
    }
    // if there is no empty node, should new PbNode
    nodes_vector.push_back(PbNode());
    nodes_vector.back().setOpNameAndType(op_name);
    return &nodes_vector.back();
  } else {
    // first call in this thread, just set a vector of PbNode
    nodes_.emplace(tid, std::vector<PbNode>(1));
    PbNode &node = nodes_[tid].front();
    node.setOpNameAndType(op_name);
    return &node;
  }
}

void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, mluOpTensorDescriptor_t desc,
                 double param1, double param2, std::string distribution,
                 bool dump_data) {
  std::vector<double> params{param1, param2};
  if (desc == nullptr) {
    mluOpTensorDescriptor_t desc_;
    mluOpCreateTensorDescriptor(&desc_);
    std::vector<int> dims{1};
    mluOpSetTensorDescriptor(desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 1,
                             dims.data());
    node->appendTensor(is_input, id, device_data, desc_, true, params,
                       distribution, dump_data);
  } else {
    node->appendTensor(is_input, id, device_data, desc, false, params,
                       distribution, dump_data);
  }
}

void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, int *dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2, std::string distribution,
                 bool dump_data) {
  mluOpTensorDescriptor_t desc;
  mluOpCreateTensorDescriptor(&desc);
  mluOpSetTensorDescriptor(desc, layout, dtype, dim, dims);
  std::vector<double> params{param1, param2};
  node->appendTensor(is_input, id, device_data, desc, true, params,
                     distribution, dump_data);
}

void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, const int *dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2, std::string distribution,
                 bool dump_data) {
  mluOpTensorDescriptor_t desc;
  mluOpCreateTensorDescriptor(&desc);
  mluOpSetTensorDescriptor(desc, layout, dtype, dim, dims);
  std::vector<double> params{param1, param2};
  node->appendTensor(is_input, id, device_data, desc, true, params,
                     distribution, dump_data);
}

void genCaseData(PbNode *node, bool is_input, std::string id,
                 const void *device_data, int dim, std::vector<int> dims,
                 mluOpDataType_t dtype, mluOpTensorLayout_t layout,
                 double param1, double param2, std::string distribution,
                 bool dump_data) {
  mluOpTensorDescriptor_t desc;
  mluOpCreateTensorDescriptor(&desc);
  mluOpSetTensorDescriptor(desc, layout, dtype, dim, dims.data());
  std::vector<double> params{param1, param2};
  node->appendTensor(is_input, id, device_data, desc, true, params,
                     distribution, dump_data);
}

void genCaseTestParam(PbNode *node, bool is_diff1, bool is_diff2, bool is_diff3,
                      const float diff1_threshold, const float diff2_threshold,
                      const float diff3_threshold,
                      const float diff1_threshold_imag,
                      const float diff2_threshold_imag,
                      const float diff3_threshold_imag) {
  if (is_diff1) {
    node->appendCriterion("DIFF1", diff1_threshold, diff1_threshold_imag);
  }
  if (is_diff2) {
    node->appendCriterion("DIFF2", diff2_threshold, diff2_threshold_imag);
  }
  if (is_diff3) {
    node->appendCriterion("DIFF3", diff3_threshold, diff3_threshold_imag);
  }
}

void genCaseHandle(PbNode *node, mluOpHandle_t handle) {
  node->setHandle(handle);
  node->getHandleParam();
}

void genCaseHandleParam(PbNode *node) {
  if (node->handle_param.params.empty()) {
    LOG(ERROR)
        << "[gen_case] HandleParam generation failed, please set handle first!";
  } else {
    node->handle_param.name = "handle_param";
  }
}

void genCaseEnd() {
  // serialize protxt and restore gen case mode
  if (gen_case_mode_ > 0) {
    std::string tid(std::to_string(syscall(SYS_gettid)));
    auto it = mode_stacks_.find(tid);
    auto &mode_stack = it->second;
    if (mode_stack.back() > 0) {
      auto nodes_it = nodes_.find(tid);
      auto &nodes_vector = nodes_it->second;
      // find the last used slot
      int slot_num = 0;
      for (int i = 0; i < nodes_vector.size(); i++) {
        if (nodes_vector[i].op_name != "") {
          slot_num++;
        }
      }
      if (dump_data_output_ != 0) {
        if (dump_data_file_ == 0) {
          nodes_vector[slot_num - 1].serialize();
        } else {
          nodes_vector[slot_num - 1].dumpOutputFile();
        }
      }
      nodes_vector[slot_num - 1].reset();
    }
  }
  genCaseModeRestore();
}

void PbNode::setOpNameAndType(std::string op_name) {
  this->op_name = op_name;
  this->file_name = "";
  this->case_file_name = "";
}

void PbNode::appendTensor(bool is_input, std::string id,
                          const void *device_data, mluOpTensorDescriptor_t desc,
                          bool inner_desc, std::vector<double> params,
                          std::string distribution, bool dump_data) {
  this->tensors.push_back(TensorNode(is_input, id, device_data, desc,
                                     inner_desc, params, distribution,
                                     dump_data));
}

void PbNode::appendCriterion(std::string criterion, double threshold,
                             double threshold_imag) {
  this->criterions.push_back(criterion);
  this->thresholds.push_back(threshold);
  this->thresholds_imag.push_back(threshold_imag);
}

void PbNode::getHandleParam() {
  std::string round_mode_str = "";

  mluOpQuantizeRoundMode_t round_mode_enum;
  mluOpGetQuantizeRoundMode(this->handle, &round_mode_enum);
  switch (round_mode_enum) {
    case 0:
      round_mode_str = "ROUND_TO_EVEN";
      break;
    case 1:
      round_mode_str = "ROUND_HALF_UP";
      break;
    case 2:
      round_mode_str = "ROUND_OFF_ZERO";
      break;
    default:
      LOG(ERROR) << "[gen_case]: unsupportted round mode: " << round_mode_enum;
  }
  this->handle_param.params.push_back({"round_mode", round_mode_str});
}

std::string PbNode::getFileName() {
  // Get current time for file name.
  static platform::EnvTime *env_time = platform::EnvTime::Default();
  uint64_t now_micros = env_time->NowMicros();
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  time_t current_time = time(NULL);
  char char_current_time[64];
  strftime(char_current_time, sizeof(char_current_time), "%Y%m%d_%H_%M_%S_",
           localtime(&current_time));
  std::string string_current_time = char_current_time;
  std::string string_micros_remainder = std::to_string(micros_remainder);
  while (string_micros_remainder.size() < 6) {
    string_micros_remainder = "0" + string_micros_remainder;
  }

  // Get current device index.
  int dev_index = -1;
  cnrtGetDevice(&dev_index);

  // Create file name by op_name and current time.
  std::string file_name = op_name + "_" + string_current_time +
                          string_micros_remainder + "_tid" +
                          std::to_string(syscall(SYS_gettid)) + "_device" +
                          std::to_string(dev_index);
  return file_name;
}

std::string PbNode::getFolderName() {
  // Create folder name by op_name.
  char current_dir[PATH_MAX];
  if (getcwd(current_dir, sizeof(current_dir)) == NULL) {
    LOG(ERROR) << "[gen_case]: get current directory failed! (" << errno << ": "
               << strerror(errno) << ")";
    return "NULL";
  }
  std::string folder_name = current_dir;
  folder_name = folder_name + "/gen_case/" + op_name;
  return folder_name;
}

int PbNode::mkdir() {
  std::string folder_name = getFolderName();
  std::lock_guard<std::mutex> guard(stacks_mutex_);
  int error_number = mkdirRecursive(folder_name.c_str());
  return error_number;
}

void PbNode::serialize(bool isFirst) {
  int state = getOpNameMask(op_name_, op_name);
  if (state != -1) {
    if (state == 1) {
      if (IS_ONLY_SHOW) {
        printOnScreen();
      } else {
        dumpToFile(isFirst, false);
      }
    } else if (state == 2) {
      dumpToFile(isFirst, true);
    }
  }
}

void PbNode::printOnScreen() {
  std::stringstream print_info;
  print_info << "[gen_case] [tid " << std::to_string(syscall(SYS_gettid))
             << "] Show ";
  print_info << "[" << op_name << "] ";
  for (int i = 0; i < tensors.size(); i++) {
    if (tensors[i].is_input) {
      print_info << "input { id: " << tensors[i].id << " ";
    } else {
      print_info << "output { id: " << tensors[i].id << " ";
    }
    print_info << descToString(tensors[i].desc, ' ') << " }";
  }
  LOG(INFO) << print_info.str() << "\n";
}

void PbNode::dumpDataFile(std::string file_name, std::string folder_name,
                          int index, std::ofstream &case_file, bool shouldDump,
                          enum DATASTATE data_state) {
  uint64_t total_num = getTensorSize(index);
  mluOpDataType_t dtype;
  mluOpGetTensorDescriptor(tensors[index].desc, nullptr, &dtype, nullptr,
                           nullptr);
  void *data = getDeviceData(index);
  std::string dataState = data_state == INPUT ? "input" : "output";
  if (data != nullptr) {
    if (dump_data_file_ == 1) {
      std::string tensor_file_suffix =
          file_name + "_data" + std::to_string(index) + "_" + dataState;
      std::string tensor_file_name = folder_name + "/" + tensor_file_suffix;
      if (data_state == INPUT) {
        case_file << "  path: \"" << tensor_file_suffix << "\"\n";
        std::ofstream tensor_file;
        tensor_file.open(tensor_file_name.c_str(), std::ios::binary);
        tensor_file.write(reinterpret_cast<const char *>(data),
                          total_num * mluop::getSizeOfDataType(dtype));
        tensor_file.close();
      } else {
        if (!shouldDump) {
          case_file << "  path: \"" << tensor_file_suffix << "\"\n";
        } else {
          std::ofstream tensor_file;
          tensor_file.open(tensor_file_name.c_str(), std::ios::binary);
          tensor_file.write(reinterpret_cast<const char *>(data),
                            total_num * mluop::getSizeOfDataType(dtype));
          tensor_file.close();
        }
      }
    } else {
      total_num *= dtypeRatio(dtype);
      for (uint64_t j = 0; j < total_num; ++j) {
        if (dump_data_output_ == 2 && dtypeFloat(dtype)) {
          case_file << "  value_h: " << get_data_hex_string(dtype, data, j)
                    << "\n";
        } else {
          case_file << get_dtype_value_string(dtype)
                    << get_data_string(dtype, data, j) << "\n";
        }
      }
    }
    // free resources to avoid memeory leak
    free(data);
  } else {
    case_file << get_tensor_random_string(index);
  }
}

void PbNode::dumpOutputFile() {
  int st = getOpNameMask(op_name_, op_name);
  if (st != 2 && !IS_DUMP_DATA) return;
  int state = getOpNameMask(op_name_, op_name);
  for (int i = 0; i < tensors.size(); i++) {
    if (!tensors[i].is_input) {
      // sync queue to dump output if necessary
      if (dump_data_output_) {
        cnrtQueue_t queue;
        mluOpGetQueue(handle, &queue);
        if (cnrtSuccess != cnrtQueueSync(queue)) {
          LOG(ERROR) << "[gen_case] syncQueue failed!";
        } else {
          // TO DO : should consider malloc failure
          std::string folder_name = getFolderName();
          std::string file_name = this->file_name;
          uint64_t total_num = getTensorSize(i);
          mluOpDataType_t dtype;
          mluOpGetTensorDescriptor(tensors[i].desc, nullptr, &dtype, nullptr,
                                   nullptr);
          void *data = getDeviceData(i);
          std::string dataState = "output";
          if (data != nullptr) {
            std::string tensor_file_suffix =
                file_name + "_data" + std::to_string(i) + "_" + dataState;
            std::string tensor_file_name =
                folder_name + "/" + tensor_file_suffix;
            std::ofstream tensor_file;
            tensor_file.open(tensor_file_name.c_str(), std::ios::binary);
            tensor_file.write(reinterpret_cast<const char *>(data),
                              total_num * mluop::getSizeOfDataType(dtype));
            tensor_file.close();
          }
        }
      }
    }
  }
}

void PbNode::dumpToFile(bool isFirst, bool valueDump) {
  std::string folder_name = getFolderName();
  int error_number = mkdir();
  // use lock to ensure mkdir not conflict
  if (error_number != 0) {
    LOG(ERROR) << "[gen_case]: mkdir folder failed for " << folder_name
               << " ! (" << errno << ": " << strerror(errno) << ")";
    return;
  }
  std::string file_name = "";
  std::string case_file_name = "";
  if (this->file_name == "") {
    file_name = getFileName();
    case_file_name = folder_name + "/" + file_name + ".prototxt";
    this->file_name = file_name;
    this->case_file_name = case_file_name;
    LOG(INFO) << "[gen_case] Generate " + case_file_name;
  } else {
    file_name = this->file_name;
    case_file_name = this->case_file_name;
  }
  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open(case_file_name.c_str(), std::ios::ate | std::ios::out);
    if (case_file) {
      case_file << "op_name: \"" + op_name + "\"\n";
      for (int i = 0; i < tensors.size(); i++) {
        if (tensors[i].is_input) {
          case_file << "input {\n  id: \"" << tensors[i].id << "\"\n";
        } else {
          case_file << "output {\n  id: \"" << tensors[i].id << "\"\n";
        }
        case_file << descToString(tensors[i].desc, '\n');
        if (tensors[i].is_input) {
          // TO DO : can be more elegant
          if (valueDump || IS_DUMP_DATA) {
            if ((tensors[i].dump_data || dump_data_ > 0) &&
                tensors[i].device_ptr != nullptr) {
              // TO DO : should consider malloc failure
              if (isFirst || dump_data_file_ == 0)
                dumpDataFile(file_name, folder_name, i, case_file, true, INPUT);
            } else {
              case_file << get_tensor_random_string(i);
            }
          } else {
            case_file << get_tensor_random_string(i);
          }
        } else {
          // sync queue to dump output if necessary
          if (dump_data_output_) {
            cnrtQueue_t queue;
            mluOpGetQueue(handle, &queue);
            if (cnrtSuccess != cnrtQueueSync(queue)) {
              LOG(ERROR) << "[gen_case] syncQueue failed!";
            } else {
              // TO DO : should consider malloc failure
              dumpDataFile(file_name, folder_name, i, case_file, !isFirst,
                           OUTPUT);
            }
          }
        }
        case_file << "}\n";
      }
      // TO DO : can support child of child
      if (op_param.name != "") {
        case_file << op_param.name << " {\n";
        for (int i = 0; i < op_param.params.size(); i++) {
          case_file << "  " << op_param.params[i].first << ": "
                    << op_param.params[i].second << "\n";
        }
        for (int i = 0; i < op_param.childs.size(); i++) {
          case_file << "  " << op_param.childs[i].name << " {\n";
          for (int j = 0; j < op_param.childs[i].params.size(); j++) {
            case_file << "    " << op_param.childs[i].params[j].first << ": "
                      << op_param.childs[i].params[j].second << "\n";
          }
          case_file << "  }\n";
        }
        case_file << "}\n";
      }
      if (handle_param.name != "") {
        case_file << handle_param.name << " {\n";
        for (int i = 0; i < handle_param.params.size(); i++) {
          case_file << "  " << handle_param.params[i].first << ": "
                    << handle_param.params[i].second << "\n";
        }
        case_file << "}\n";
      }
      case_file << "test_param {\n";
      for (int i = 0; i < criterions.size(); i++) {
        case_file << "  error_func: " << criterions[i] << "\n";
      }
      for (int i = 0; i < criterions.size(); i++) {
        case_file << "  error_threshold: " << thresholds[i] << "\n";
        if (thresholds_imag[i] >= 0) {
          case_file << "  error_threshold_imag: " << thresholds_imag[i] << "\n";
        }
      }
      case_file << "  baseline_device: CPU\n}";
    }
    case_file.close();
  }
}

// Check if tensor need stride process.
// should be same with tensor_stride_process_host.mlu
bool ifNeedTensorStrideProcess(const mluOpTensorDescriptor_t desc) {
  bool needStrideProcess = false;
  int tensor_dim;
  mluOpTensorLayout_t layout;
  mluOpDataType_t dtype;
  mluOpGetTensorDescriptor(desc, &layout, &dtype, &tensor_dim, nullptr);
  int *dims = new int[tensor_dim];
  int *strides = new int[tensor_dim];
  mluOpGetTensorDescriptorEx(desc, &layout, &dtype, &tensor_dim, dims, strides);
  int stride_base = 1;
  for (int i = tensor_dim - 1; i >= 0; i--) {
    if (dims[i] != 1) {
      if (strides[i] == stride_base) {
        stride_base *= dims[i];
      } else {
        needStrideProcess = true;
        break;
      }
    }
  }
  delete[] dims;
  delete[] strides;
  return needStrideProcess;
}

std::string descToString(mluOpTensorDescriptor_t desc, char delimiter) {
  int dim;
  mluOpTensorLayout_t layout;
  mluOpDataType_t dtype;
  mluOpGetTensorDescriptor(desc, &layout, &dtype, &dim, nullptr);
  int *dims = new int[dim];
  int *strides = new int[dim];
  mluOpGetTensorDescriptorEx(desc, &layout, &dtype, &dim, dims, strides);
  mluOpDataType_t onchip_dtype;
  mluOpGetTensorDescriptorOnchipDataType(desc, &onchip_dtype);
  int position, offset;
  float scale;
  mluOpGetTensorDescriptorPositionScaleAndOffset(desc, &position, &scale,
                                                 &offset);
  size_t total_element_num = mluOpGetTensorElementNum(desc);

  std::stringstream tensor_info;
  tensor_info << "  shape: {" << delimiter;
  for (int i = 0; i < dim; i++) {
    tensor_info << "    dims: " << dims[i] << delimiter;
  }
  if (total_element_num != 1) {
    if (mluop::gen_case::ifNeedTensorStrideProcess(desc)) {
      // Write the dim_stride of shape module.
      for (int i = 0; i < dim; i++) {
        tensor_info << "    dim_stride: " << strides[i] << delimiter;
      }
    }
  }
  tensor_info << "  }" << delimiter;
  tensor_info << "  layout: " << getNameOfTensorLayout(layout) << delimiter;
  tensor_info << "  dtype: " << getNameOfDataType(dtype) << delimiter;
  if (onchip_dtype != MLUOP_DTYPE_INVALID) {
    tensor_info << "  onchip_dtype: " << getNameOfDataType(onchip_dtype)
                << delimiter;
  }
  tensor_info << "  position: " << position << delimiter;
  tensor_info << "  scale: " << scale << delimiter;
  tensor_info << "  offset: " << offset << delimiter;
  delete[] dims;
  delete[] strides;
  return tensor_info.str();
}

}  // namespace gen_case
}  // namespace mluop
void mluOpSetGenCaseMode(int mode) { mluop::gen_case::genCaseModeSet(mode); }
