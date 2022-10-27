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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_PARSER_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_PARSER_H_

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "mlu_op_test.pb.h"

namespace mluoptest {

// value source
enum ValueType {
  VALUE_F,
  VALUE_I,
  VALUE_L,
  VALUE_H,
  VALUE_RANDOM,
  VALUE_PATH,
  VALUE_INVALID,
};

// only the tensor info saved in *pb.
struct MetaTensor {
  std::string name = "unknown";
  bool is_null = false;  // null means desc and ptr both null;

  // these size are for memory malloc
  size_t shape_count = 0;  // not include stride.
  // count is 0, means 0 elements and ptr is null
  // but desc may not null.
  size_t total_count = 0;    // include stride_count
  size_t size_in_bytes = 0;  // size_in_bytes = total_count * sizeof(mlu_dtype)
  size_t sizeof_dtype = 0;

  // these for set to tensor desc
  mluOpDataType_t dtype = MLUOP_DTYPE_INVALID;
  mluOpDataType_t oc_dt = MLUOP_DTYPE_INVALID;
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;

  // shape may not equal to total_count.
  // maybe total_count is 0 (for nullptr), but shape is not 0 element
  // for testing api foolproof
  std::vector<int> shape;
  std::vector<int> stride;

  int position = 0;
  float scale = 1.0;
  int offset = 0;

  ValueType value_type = VALUE_INVALID;

  inline bool empty() { return shape_count == 0 || total_count == 0; }
  inline bool null() { return is_null; }

  // it's a not good idea to put *ptr and layout.. together.
  mluOpTensorDescriptor_t tensor = nullptr;
  // bool is_output       = false;
  void *host_ptr = nullptr;
  void *dev_ptr = nullptr;
  void *dev_origin_ptr = nullptr;  // for origin device data
  void *dev_perf_ptr = nullptr;    // for perf test data
  float *cpu_ptr = nullptr;
  void print() {
    std::cout << "-=-=-=-=-=--=-=-=\n";
    std::cout << "tensor :" << name << " \n";
    std::cout << "total count is " << total_count << " \n";
    std::cout << "shape count is " << shape_count << " \n";
    std::cout << "size_in_bytes is " << size_in_bytes << " \n";
    std::cout << "shape : ";
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "stride : ";
    for (size_t i = 0; i < stride.size(); ++i) {
      std::cout << stride[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-=-=-=-=-=--=-=-=\n";
  }
};

class Parser {
 public:
  Parser() {}
  virtual ~Parser();
  void parse(const std::string &file);

  inline const std::vector<MetaTensor> &inputs() { return inputs_; }
  inline const std::vector<MetaTensor> &outputs() { return outputs_; }
  inline MetaTensor *input(size_t index) { return &(inputs_.at(index)); }
  inline MetaTensor *output(size_t index) { return &(outputs_.at(index)); }

  void getInputTensorValue(size_t index, void *data, size_t count);
  void getOutputTensorValue(size_t index, void *data, size_t count);

  // op params
  inline Node *node() { return proto_node_; }
  inline std::string getOpName() { return proto_node_->op_name(); }
  void getTestInfo();

  // else
  inline Device device() { return device_; }
  std::vector<int> threshold_use();
  bool common_threshold();
  bool check_threshold();
  inline std::set<Evaluator::Criterion> criterions() { return criterions_; }
  std::set<Evaluator::Criterion> criterions(int index,
                                            std::vector<int> criterions_use);
  std::vector<std::string> getBlOfZeroInput() { return bl_zeroinput_; }
  std::vector<std::string> getBlOfMluOnlyFast() { return bl_mlu_only_fast_; }

  MetaTensor &getMetaTensor(const std::string &name);
  MetaTensor &getMetaTensor(int index);
  inline int getInputNum() { return inputs_.size(); }
  inline int getOutputNum() { return outputs_.size(); }
  inline bool inputIsNull(int index) { return inputs_.at(index).is_null; }
  inline bool outputIsNull(int index) { return outputs_.at(index).is_null; }
  inline float getInputScale(int index) { return inputs_.at(index).scale; }
  inline float getOutputScale(int index) { return outputs_.at(index).scale; }
  inline int getInputPosition(int index) { return inputs_.at(index).position; }
  inline int getOutputPosition(int index) {
    return outputs_.at(index).position;
  }
  inline int getInputOffset(int index) { return inputs_.at(index).offset; }
  inline int getOutputOffset(int index) { return outputs_.at(index).offset; }
  inline mluOpDataType_t getInputDataType(int index) {
    return inputs_.at(index).dtype;
  }
  inline mluOpDataType_t getOutputDataType(int index) {
    return outputs_.at(index).dtype;
  }
  inline mluOpDataType_t getInputOnchipDataType(int index) {
    return inputs_.at(index).oc_dt;
  }
  inline mluOpDataType_t getOutputOnchipDataType(int index) {
    return outputs_.at(index).oc_dt;
  }
  inline mluOpTensorLayout_t getInputLayout(int index) {
    return inputs_.at(index).layout;
  }
  inline mluOpTensorLayout_t getOutputLayout(int index) {
    return outputs_.at(index).layout;
  }
  inline size_t getInputDataCount(int index) {
    return inputs_.at(index).shape_count;
  }
  inline size_t getOutputDataCount(int index) {
    return outputs_.at(index).shape_count;
  }
  inline int getInputDimSize(int index) {
    return inputs_.at(index).shape.size();
  }
  inline int getOutputDimSize(int index) {
    return outputs_.at(index).shape.size();
  }
  inline int getInputDimStrideSize(int index) {
    return inputs_.at(index).stride.size();
  }
  inline int getOutputDimStrideSize(int index) {
    return outputs_.at(index).stride.size();
  }
  inline void getInputData(int index, void *data) {
    getInputTensorValue(index, data, inputs_[index].total_count);
  }
  inline void getOutputData(int index, void *data) {
    getOutputTensorValue(index, data, outputs_[index].total_count);
  }
  inline void getInputDims(int index, int dim_size, int *dim_array,
                           int *dim_stride = nullptr) {
    auto ts = inputs_.at(index);
    memcpy(dim_array, ts.shape.data(), ts.shape.size() * sizeof(int));
    if (dim_stride) {
      memcpy(dim_stride, ts.stride.data(), ts.stride.size() * sizeof(int));
    }
  }
  inline void getOutputDims(int index, int dim_size, int *dim_array,
                            int *dim_stride = nullptr) {
    auto ts = outputs_.at(index);
    memcpy(dim_array, ts.shape.data(), ts.shape.size() * sizeof(int));
    if (dim_stride) {
      memcpy(dim_stride, ts.stride.data(), ts.stride.size() * sizeof(int));
    }
  }
  inline ValueType getInputValueType(int index) {
    return inputs_.at(index).value_type;
  }
  inline Node *getProtoNode() { return proto_node_; }

 private:
  Node *proto_node_ = nullptr;
  std::vector<MetaTensor> inputs_;
  std::vector<MetaTensor> outputs_;
  std::set<Evaluator::Criterion> criterions_;
  std::string op_name_;
  std::string pb_path_;
  std::vector<std::string> bl_zeroinput_;
  std::vector<std::string> bl_mlu_only_fast_;
  Device device_ = CPU;

  ValueType getValueType(const Tensor *t);
  void getTensorValue(Tensor *pt, void *data, ValueType value_type,
                      size_t count);
  void getTensorValueH(Tensor *pt, void *data, size_t count);
  void getTensorValueF(const Tensor *pt, void *data, size_t count);
  void getTensorValueI(const Tensor *pt, void *data, size_t count);
  void getTensorValueL(const Tensor *pt, void *data, size_t count);
  void getTensorValueRandom(Tensor *pt, float *data, size_t count);
  void getTensorValueByFile(Tensor *pt, float *data, size_t count);

  void checkTensorValid(MetaTensor *mt, Tensor *t);
  void checkRandomParam(Tensor *t);

  // if no stride return shape count
  size_t getTensorStrideCount(Tensor *pt, ValueType type);
  // only return shape count, from value_size or shape
  size_t getTensorShapeCount(Tensor *pt);

  Evaluator::Formula cvtProtoEvaluationCriterion(EvaluationCriterion c);
  bool readMessageFromFile(const std::string &filename, Node *proto);
  size_t getTensorSize(Tensor *pt);
  void setCurPbPath(const std::string &file);
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_PARSER_H_
