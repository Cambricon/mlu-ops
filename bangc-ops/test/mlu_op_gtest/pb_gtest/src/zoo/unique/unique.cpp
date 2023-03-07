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
#include <vector>
#include "unique.h"
namespace mluoptest {
void UniqueExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "unique input number is wrong. ";
  }  // get unique param
  mode_ = (mluOpUniqueSort_t)parser_->getProtoNode()->unique_param().mode();
  dim_ = parser_->getProtoNode()->unique_param().dim();
  return_inverse_ = parser_->getProtoNode()->unique_param().return_inverse();
  return_counts_ = parser_->getProtoNode()->unique_param().return_counts();
  version_ = parser_->getProtoNode()->unique_param().version();
  int output_num = 0;
  if (mode_ == MLUOP_UNSORT_FORWARD) {
    output_num = 3;
  } else if (mode_ == MLUOP_UNSORT_REVERSE || mode_ == MLUOP_SORT_ASCEND) {
    if (return_inverse_ && return_counts_) {
      output_num = 4;
    } else if ((return_inverse_ && !return_counts_) ||
               (!return_inverse_ && return_counts_)) {
      output_num = 3;
    } else if (!return_inverse_ && !return_counts_) {
      output_num = 2;
    }
  } else {
    LOG(ERROR) << "unique unspport this sort mode. ";
  }
  if (parser_->getOutputNum() != output_num) {
    LOG(ERROR) << "unique output number is wrong. ";
  }
}

void UniqueExecutor::compute() {
  VLOG(4) << "UniqueExecutor compute ";
  auto tensor_input = tensor_desc_[0].tensor;
  auto tensor_output = tensor_desc_[2].tensor;
  auto dev_input = data_vector_[0].device_ptr;
  auto dev_output1 = data_vector_[1].device_ptr;  // len_out
  auto dev_output2 = data_vector_[2].device_ptr;  // data_out
  mluOpUniqueDescriptor_t unique_desc =
      cpu_runtime_.allocate(mluOpCreateUniqueDescriptor,
                            mluOpDestroyUniqueDescriptor);
  MLUOP_CHECK(
      mluOpSetUniqueDescriptor(unique_desc, mode_, dim_, return_inverse_,
                               return_counts_));
  if (version_ == 1) {
    VLOG(4) << "call mluOp UniqueGetOutLen";
    MLUOP_CHECK(mluOpUniqueGetOutLen(
        handle_, unique_desc, tensor_input, dev_input, workspace_.at(0),
        (int *)dev_output1));
    int *dev_out_len = (int *)cpu_runtime_.allocate(sizeof(int));
    GTEST_CHECK(cnrtSuccess == cnrtQueueSync(handle_->queue));
    GTEST_CHECK(CNRT_RET_SUCCESS ==
        cnrtMemcpy(dev_out_len, dev_output1, sizeof(int),
                   CNRT_MEM_TRANS_DIR_DEV2HOST));
    VLOG(4) << "call mluOp Unique";
    if (mode_ == MLUOP_UNSORT_FORWARD) {
      auto dev_output3 = data_vector_[3].device_ptr;
      interface_timer_.start();
      MLUOP_CHECK(
          mluOpUnique(handle_, unique_desc, tensor_input, dev_input,
                      *dev_out_len, workspace_.at(0), dev_output2,
                      (int *)dev_output3, NULL));
      interface_timer_.stop();
      data_vector_[1].is_output = true;
      data_vector_[2].is_output = true;
      data_vector_[3].is_output = true;
    } else {
      if (return_inverse_ && return_counts_) {
        auto dev_output3 = data_vector_[3].device_ptr;
        auto dev_output4 = data_vector_[4].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique(handle_, unique_desc, tensor_input, dev_input,
                                *dev_out_len, workspace_.at(0), dev_output2,
                                (int *)dev_output3, (int *)dev_output4));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
        data_vector_[4].is_output = true;
      } else if (return_inverse_ && !return_counts_) {
        auto dev_output3 = data_vector_[3].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique(handle_, unique_desc, tensor_input, dev_input,
                                *dev_out_len, workspace_.at(0), dev_output2,
                                (int *)dev_output3, NULL));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
      } else if (!return_inverse_ && return_counts_) {
        auto dev_output3 = data_vector_[3].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique(handle_, unique_desc, tensor_input, dev_input,
                                *dev_out_len, workspace_.at(0), dev_output2,
                                NULL, (int *)dev_output3));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
      } else {
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique(handle_, unique_desc, tensor_input, dev_input,
                                *dev_out_len, workspace_.at(0), dev_output2,
                                NULL, NULL));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
      }
    }
    cpu_runtime_.deallocate(dev_out_len);
  } else {
    if (mode_ == MLUOP_UNSORT_FORWARD) {
      auto tensor_index = tensor_desc_[3].tensor;
      auto dev_output3 = data_vector_[3].device_ptr;
      interface_timer_.start();
      MLUOP_CHECK(mluOpUnique_v2(handle_, unique_desc, tensor_input, dev_input,
                                 workspace_.at(0), workspace_size_,
                                 (int32_t *)dev_output1, tensor_output,
                                 dev_output2, tensor_index, dev_output3,
                                 NULL, NULL));
      interface_timer_.stop();
      data_vector_[1].is_output = true;
      data_vector_[2].is_output = true;
      data_vector_[3].is_output = true;
    } else {
      if (return_inverse_ && return_counts_) {
        auto tensor_index = tensor_desc_[3].tensor;
        auto tensor_counts = tensor_desc_[4].tensor;
        auto dev_output3 = data_vector_[3].device_ptr;
        auto dev_output4 = data_vector_[4].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique_v2(handle_, unique_desc, tensor_input,
                                   dev_input, workspace_.at(0),
                                   workspace_size_, (int32_t *)dev_output1,
                                   tensor_output, dev_output2, tensor_index,
                                   dev_output3, tensor_counts, dev_output4));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
        data_vector_[4].is_output = true;
      } else if (return_inverse_ && !return_counts_) {
        auto tensor_index = tensor_desc_[3].tensor;
        auto dev_output3 = data_vector_[3].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique_v2(handle_, unique_desc, tensor_input,
                                   dev_input, workspace_.at(0),
                                   workspace_size_, (int32_t *)dev_output1,
                                   tensor_output, dev_output2, tensor_index,
                                   dev_output3, NULL, NULL));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
      } else if (!return_inverse_ && return_counts_) {
        auto tensor_counts = tensor_desc_[3].tensor;
        auto dev_output3 = data_vector_[3].device_ptr;
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique_v2(handle_, unique_desc, tensor_input,
                                   dev_input, workspace_.at(0),
                                   workspace_size_, (int32_t *)dev_output1,
                                   tensor_output, dev_output2, NULL, NULL,
                                   tensor_counts, dev_output3));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
        data_vector_[3].is_output = true;
      } else {
        interface_timer_.start();
        MLUOP_CHECK(mluOpUnique_v2(handle_, unique_desc, tensor_input,
                                   dev_input, workspace_.at(0),
                                   workspace_size_, (int32_t *)dev_output1,
                                   tensor_output, dev_output2, NULL, NULL,
                                   NULL, NULL));
        interface_timer_.stop();
        data_vector_[1].is_output = true;
        data_vector_[2].is_output = true;
      }
    }
  }
}

void UniqueExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  auto tensor_input = tensor_desc_[0].tensor;
  int input_len = parser_->getInputDataCount(0);
  std::vector<int> tag_input(input_len, 1);
  std::vector<int> tag_counts(input_len, 0);
  std::vector<int> tag_index(input_len, 1);
  int out_len = 0;
  if (mode_ == MLUOP_UNSORT_FORWARD) {
    for (int i = 0; i < input_len; i++) {
      if (tag_input[i] > 0) {
        cpu_fp32_output_[2][i] = out_len;
        cpu_fp32_output_[1][out_len] = cpu_fp32_input_[0][i];
        for (int j = i; j < input_len; j++) {
          if (cpu_fp32_input_[0][i] == cpu_fp32_input_[0][j]) {
            tag_input[j] = -i;
            cpu_fp32_output_[2][j] = out_len;
          }
        }
        out_len++;
      }
    }
  } else {
    // uniuqe
    for (int i = input_len - 1; i >= 0; i--) {
      if (tag_input[i] > 0) {
        tag_index[i] = out_len;
        cpu_fp32_output_[1][out_len] = cpu_fp32_input_[0][i];
        for (int j = i; j >= 0; j--) {
          if (cpu_fp32_input_[0][i] == cpu_fp32_input_[0][j]) {
            tag_index[j] = out_len;
            tag_input[j] = -i;
            tag_counts[out_len]++;
          }
        }
        out_len++;
      }
    }
    // sort
    float tmp_output = 0;
    int tmp_counts = 0;
    if (mode_ == MLUOP_SORT_ASCEND) {
      // bubble sort
      for (int i = 0; i < out_len - 1; i++) {
        for (int j = 0; j < out_len - 1 - i; j++) {
          if (cpu_fp32_output_[1][j] > cpu_fp32_output_[1][j + 1]) {
            tmp_output = cpu_fp32_output_[1][j];
            cpu_fp32_output_[1][j] = cpu_fp32_output_[1][j + 1];
            cpu_fp32_output_[1][j + 1] = tmp_output;
            tmp_counts = tag_counts[j];
            tag_counts[j] = tag_counts[j + 1];
            tag_counts[j + 1] = tmp_counts;
          }
        }
      }
    }
    // index
    if (mode_ == MLUOP_SORT_ASCEND && return_inverse_) {
      for (int i = 0; i < input_len; i++) {
        for (int j = 0; j < out_len; j++) {
          if (cpu_fp32_input_[0][i] == cpu_fp32_output_[1][j]) {
            tag_index[i] = j;
          }
        }
      }
    }
    if (return_inverse_ && return_counts_) {
      for (int i = 0; i < input_len; i++) {
        cpu_fp32_output_[2][i] = tag_index[i];
      }
      for (int i = 0; i < out_len; i++) {
        cpu_fp32_output_[3][i] = tag_counts[i];
      }
      parser_->output(3)->total_count = out_len;
    } else if (return_inverse_ && !return_counts_) {
      for (int i = 0; i < input_len; i++) {
        cpu_fp32_output_[2][i] = tag_index[i];
      }
    } else if (return_counts_ && !return_inverse_) {
      for (int i = 0; i < out_len; i++) {
        cpu_fp32_output_[2][i] = tag_counts[i];
      }
      parser_->output(2)->total_count = out_len;
    }
  }
  parser_->output(1)->total_count = out_len;
  cpu_fp32_output_[0][0] = out_len;
  VLOG(4) << "unique length :" << out_len;
}

void UniqueExecutor::workspaceMalloc() {
  auto tensor_input = tensor_desc_[0].tensor;
  workspace_size_ = 0;
  void *tmp = nullptr;
  mluOpUniqueDescriptor_t unique_desc =
      cpu_runtime_.allocate(mluOpCreateUniqueDescriptor,
                            mluOpDestroyUniqueDescriptor);
  MLUOP_CHECK(mluOpSetUniqueDescriptor(unique_desc, mode_, dim_,
                                       return_inverse_, return_counts_));
  // allocate extra space for tmp.
  if (version_ == 1) {
    MLUOP_CHECK(mluOpGetUniqueWorkSpace(handle_, unique_desc, tensor_input,
                                        &workspace_size_));
  } else {
    MLUOP_CHECK(mluOpGetUniqueWorkspaceSize(handle_, unique_desc, tensor_input,
                                            &workspace_size_));
  }
  VLOG(4) << "Malloc workspace space.";
  tmp = mlu_runtime_.allocate(workspace_size_);
  VLOG(4) << "[0] Malloc addr: " << tmp << ", size: " << workspace_size_;
  workspace_.push_back(tmp);
  eva_->setMluWorkspaceSize(workspace_size_);
}

void UniqueExecutor::workspaceFree() {
  if (workspace_.at(0) != nullptr) {
    VLOG(4) << "workspace free: " << workspace_.at(0);
    mlu_runtime_.deallocate(workspace_.at(0));
  }
}

int64_t UniqueExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
