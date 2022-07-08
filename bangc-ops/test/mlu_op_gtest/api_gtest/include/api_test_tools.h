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
#ifndef TEST_MLU_OP_GTEST_API_GTEST_INCLUDE_TEST_TOOLS_H_
#define TEST_MLU_OP_GTEST_API_GTEST_INCLUDE_TEST_TOOLS_H_

#include <vector>
#include <tuple>
#include <string>

#include "gtest/gtest.h"
#include "core/mlu_op_core.h"
#include "core/context.h"
#include "tools.h"

namespace mluopapitest {
class MLUOpTensorParam {
 public:
  MLUOpTensorParam(mluOpTensorLayout_t layout, mluOpDataType_t dtype,
                   int dim_nb, std::vector<int> dim_size,
                   std::vector<int> dim_stride = {},
                   mluOpDataType_t onchip_dtype = MLUOP_DTYPE_INVALID) {
    layout_ = layout;
    dtype_ = dtype;
    dim_nb_ = dim_nb;
    dim_size_ = dim_size;
    dim_stride_ = dim_stride;
    onchip_dtype_ = onchip_dtype;
  }

  mluOpTensorLayout_t get_layout() { return layout_; }
  mluOpDataType_t get_dtype() { return dtype_; }
  int get_dim_nb() { return dim_nb_; }
  std::vector<int> get_dim_size() { return dim_size_; }
  std::vector<int> get_dim_stride() { return dim_stride_; }
  mluOpDataType_t get_onchip_dtype() { return onchip_dtype_; }

 private:
  mluOpTensorLayout_t layout_;
  mluOpDataType_t dtype_;
  int dim_nb_;
  std::vector<int> dim_size_;
  std::vector<int> dim_stride_;
  mluOpDataType_t onchip_dtype_;
};
}  // mluopapitest

#endif // TEST_MLU_OP_GTEST_API_GTEST_INCLUDE_TEST_TOOLS_H_
