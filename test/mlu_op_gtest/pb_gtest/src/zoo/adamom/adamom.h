/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
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

namespace mluoptest {

class AdamomExecutor : public Executor {
 public:
  AdamomExecutor() {}
  ~AdamomExecutor() {}

  void paramCheck() override;
  void compute() override;
  void setMiscellaneousParam() override;
  void cpuCompute() override;
  void workspaceMalloc() override;
  void workspaceFree() override;
  void *dev_nan_inf_found = nullptr;
  void *dev_lr            = nullptr;
  void *dev_beta1         = nullptr;
  void *dev_beta2         = nullptr;
  void *dev_weight_decay  = nullptr;
  void *dev_epsilon       = nullptr;
  bool dev_ptr_inited = false;
 private:

  void destroy_dev_ptr() {
    if (dev_ptr_inited) {
      mlu_runtime_.deallocate(dev_nan_inf_found);
      mlu_runtime_.deallocate(dev_lr);
      mlu_runtime_.deallocate(dev_beta1);
      mlu_runtime_.deallocate(dev_beta2);
      mlu_runtime_.deallocate(dev_weight_decay);
      mlu_runtime_.deallocate(dev_epsilon);
    }
    dev_nan_inf_found = nullptr;
    dev_lr            = nullptr;
    dev_beta1         = nullptr;
    dev_beta2         = nullptr;
    dev_weight_decay  = nullptr;
    dev_epsilon       = nullptr;
    dev_ptr_inited    = false;
  }

  void init_dev_ptr_by_type(mluOpDataType_t dtype) {
    auto size = sizeof(float);
    if (!dev_ptr_inited) {
      dev_nan_inf_found = mlu_runtime_.allocate(sizeof(bool));
      dev_lr            = mlu_runtime_.allocate(size);
      dev_beta1         = mlu_runtime_.allocate(size);
      dev_beta2         = mlu_runtime_.allocate(size);
      dev_weight_decay  = mlu_runtime_.allocate(size);
      dev_epsilon       = mlu_runtime_.allocate(size);
      dev_ptr_inited    = true;
    }
  }
};

}  // namespace mluoptest
