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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
#include "executor.h"

namespace mluoptest {
class PsroipoolBackwardExecutor : public Executor {
 public:
  PsroipoolBackwardExecutor() {}
  ~PsroipoolBackwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  template <typename T1, typename T2>
  void transposeNchwToNhwc(const T1 *in, const T2 dim0,
                           const T2 dim1, const T2 dim2,
                           const T2 dim3, T1 *out);
  template <typename T1, typename T2>                   
  void transposeNhwcToNchw(const T1 *in, const T2 dim0,
                           const T2 dim1, const T2 dim2,
                           const T2 dim3, T1 *out);
 private:
  void initData();
  int batch_size_;
  int height_;
  int width_;
  int channels_;
  int pooled_height_;
  int pooled_width_;
  int output_dim_;
  int rois_num_;
  int rois_offset_;
  float spatial_scale_;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
