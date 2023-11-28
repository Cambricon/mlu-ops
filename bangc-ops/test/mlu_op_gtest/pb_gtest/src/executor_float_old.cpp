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
#include "executor.h"

// TODO(None): delete this file when all ops use void *
namespace mluoptest {

// cast mlu's output to fp32
// and set them on mlu_fp32_output_
void Executor::castOut() {
  auto data_blocks = getOutputBlocks(true);
  auto output_num = data_blocks.size();
  // output data block num == output num
  // don't mark data block as output casually,
  // it's num must equals to output num in prototxt.
  GTEST_CHECK(output_num == parser_->outputs().size(),
              "Executor: output_num in *pb is not equal to num of tensor that "
              "marked as is_output = true.");

  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(data_blocks[i]->count == 0)) {
      // if input reusing, here should check parser_->input().empty(), check
      // data_blocks[] instead. if not here should check
      // parser_->output().empty()
      continue;
    }

    // fp32 -> X
    void *src_data = data_blocks[i]->host_ptr;
    float *dst_data = mlu_fp32_output_[i];
    mluOpDataType_t cpu_dtype = getCpuDtype(ts->dtype);
    int pos = 0, offset = 0;
    float scale = 1.0f;
    if (parser_->device() == CPU) {
      // get p/s/o by cpu result
      // if quant mode is NOT NO_QUANT, and dtype is int8/int16/int31, return
      // p/s/o else return 0/1/0.
      getQuantizedParam(cpu_fp32_output_[i],  // src data
                        ts->shape_count,      // count
                        ts->dtype,            // dst dtype
                        flag_quant_mode_, &pos, &scale,
                        &offset);  // return p/s/o
    }
    castDataOut(src_data, ts->dtype,  // src data, mlu raw output
                dst_data,
                cpu_dtype,        // dst data, cast mlu output into cpu array
                ts->total_count,  // count
                flag_quant_mode_, pos, scale, offset);  // quant param.
  }
}

// read in output data from *pb
// and set it to cpu_fp32_output_
// ONLY FOR NON-CPU MODE,
// cpu mode will call cpuCompute() to compute output, don't need read from *pb.
// sometimes need cast mlu-dtype to fp32, if dtype in *pb is not fp32.
// use this cpu_fp32_output_ and mlu_fp32_output_ to compute diff.
void Executor::getBaselineOutput() {
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(ts->empty())) {
      continue;
    }
    void *temp = cpu_runtime_.allocate(ts->shape_count * ts->sizeof_dtype);
    VLOG(7) << "total_count: " << ts->total_count << ",\t"
            << "shape_count: " << ts->shape_count;
    parser_->getOutputTensorValue(i, temp, ts->shape_count);
    mluOpDataType_t cpu_dtype = getCpuDtype(ts->dtype);
    castDataOut(temp, ts->dtype, cpu_fp32_output_[i], cpu_dtype,
                ts->shape_count, NO_QUANT);
    cpu_runtime_.deallocate(temp);
  }
}

// malloc for baseline output.
// and it's fp32.
// the output of cpu/gpu will write to this ptr, and then compute diff.
void Executor::baselineOutputMalloc() {
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(ts->empty())) {
      cpu_fp32_output_.push_back(nullptr);
      continue;
    }
    size_t cpu_dtype_size = mluop::getSizeOfDataType(getCpuDtype(ts->dtype));
    ts->cpu_ptr = (float *)cpu_runtime_.allocate(
        ts->shape_count * cpu_dtype_size, ts->name);
    cpu_fp32_output_.push_back(ts->cpu_ptr);
    memset(ts->cpu_ptr, 0x0, ts->shape_count * cpu_dtype_size);
  }
}

// malloc a memory for mlu output
// it's fp32, only for computing diff.
// the output of mlu (no matter what dtype) will cast to fp32, and saved here.
void Executor::mluOutputMalloc() {
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(ts->empty())) {
      mlu_fp32_output_.push_back(nullptr);
      continue;
    }
    size_t cpu_dtype_size = mluop::getSizeOfDataType(getCpuDtype(ts->dtype));
    void *temp =
        cpu_runtime_.allocate(ts->total_count * cpu_dtype_size, ts->name);
    mlu_fp32_output_.push_back((float *)temp);
    memset(temp, 0x0, ts->total_count * cpu_dtype_size);
  }
}

void Executor::strideOutput() {
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (!ts->stride.empty()) {  // TODO(None): 2023-7-13: fix here
      VLOG(4) << "[WARNING] Executor: " << ts->name
              << " cpu ptr been strided_out.";
      size_t cpu_dtype_size = mluop::getSizeOfDataType(getCpuDtype(ts->dtype));
      void *temp = cpu_runtime_.allocate(ts->total_count * cpu_dtype_size);
      if (!flag_input_reuse_) {  // TODO(None): fix after zhaolianshui
                                 // fix is_output
        memset(temp, 0x0, ts->total_count * cpu_dtype_size);
      } else {
        // if input is reused, need init cpu_output by input data
        for (int i = 0; i < data_vector_.size(); i++) {
          // BUG(zhaolianshui): wrong, always get to the first one
          if (data_vector_[i].is_output()) {
            memcpy(temp, cpu_fp32_stride_input_[i],
                   ts->total_count *
                       cpu_dtype_size);  // TODO(None): cpu_stride?
            break;
          }
        }
      }
      tensor_stride_out(temp, cpu_fp32_output_[i], getTensorShapeSizeT(ts),
                        getTensorStrideSizeT(ts), cpu_dtype_size);
      cpu_runtime_.deallocate(cpu_fp32_output_[i]);
      cpu_fp32_output_[i] = (float *)temp;
      ts->cpu_ptr = (float *)temp;
    }
  }
}
}  // namespace mluoptest
