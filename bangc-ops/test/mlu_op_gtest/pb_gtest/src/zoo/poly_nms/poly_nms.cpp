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
#include <sys/time.h>
#include "poly_nms.h"
#include "pnms_impl.h"
using namespace PNMS;
namespace mluoptest {
void PolyNmsExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_poly_nms_param()) {
    LOG(ERROR) << "Lose pnms param. ";
  }
  // if (parser_->getInputNum() != 2) {
  //   LOG(ERROR) << "nms tensor input number is wrong. ";
  // }
}

void PolyNmsExecutor::workspaceMalloc() {
  size_t workspace_size = 0;
  auto tensor_box = parser_->getMetaTensor("input1").tensor;
  MLUOP_CHECK(mluGetPolyNmsWorkspaceSize(handle_, tensor_box, &workspace_size));
  VLOG(4) << "Malloc workspace space.";
  void *temp = mlu_runtime_.allocate(workspace_size);
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << workspace_size;
  eva_->setMluWorkspaceSize(workspace_size);

  auto output_tensor = parser_->getMetaTensor("output1").tensor;
  void *output_ptr = parser_->getMetaTensor("output1").dev_origin_ptr;
  size_t output_size = parser_->getMetaTensor("output1").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(output_ptr, 0, output_size));
}

void PolyNmsExecutor::workspaceFree() {
  VLOG(4) << "Free device workspace space.";
  mlu_runtime_.deallocate(workspace_[0]);
}

void PolyNmsExecutor::compute() {
  float iou_threshold = parser_->getProtoNode()->poly_nms_param().iou_threshold();
  VLOG(4) << "[mluPolyNms] iou_threshold=" << iou_threshold;

  // get tensor by name (in prototxt)
  auto tensor_boxes = parser_->getMetaTensor("input1").tensor;
  auto tensor_output = parser_->getMetaTensor("output1").tensor;
  auto boxes_ptr = parser_->getMetaTensor("input1").dev_ptr;
  auto output_ptr = parser_->getMetaTensor("output1").dev_ptr;
  auto result_num = parser_->getMetaTensor("output2").dev_ptr;

  VLOG(4) << "[mluPolyNms] call cnnlGetPnmsWorkspaceSize()";

  size_t workspace_size = 0;
  MLUOP_CHECK(mluGetPolyNmsWorkspaceSize(handle_, tensor_boxes, &workspace_size));

  // struct timeval tstart;
  // struct timeval tend;

  // gettimeofday(&tstart, NULL);
  interface_timer_.start();

  VLOG(4) << "[mluPolyNms] call mluPolyNms()";
  MLUOP_CHECK(mluPolyNms(handle_, tensor_boxes, boxes_ptr, iou_threshold, workspace_[0],
                      workspace_size, tensor_output, output_ptr, result_num));

  // cnrtQueueSync(handle_->queue);
  // VLOG(4) << "[mluPolyNms] parser_->getOutputDimSize(0):"
  //         << ((int32_t*)result_num)[0];

  interface_timer_.stop();
  // gettimeofday(&tend, NULL);
  VLOG(4) << "[mluPolyNms] mluPolyNms end.";

  // uint32_t time_usec = (uint32_t)tend.tv_usec - (uint32_t)tstart.tv_usec;
  // uint32_t time_sec = (uint32_t)tend.tv_sec - (uint32_t)tstart.tv_sec;
}

void PolyNmsExecutor::pnms_detection_cpu(float *output_data,
                                         int &output_box_num,
                                         float *input_data,
                                         int input_box_num,
                                         float iou_thresh) {
  vector<vector<float>> input_vvec;
  for (int i = 0; i < input_box_num; i++) {
    vector<float> tmp_vec;
    for (int j = 0; j < 9; j++) {
      tmp_vec.push_back(input_data[j + i * 9]);
    }
    input_vvec.push_back(tmp_vec);
  }

  vector<int> pnms_ret = pnms_impl(input_vvec, iou_thresh);
  output_box_num = pnms_ret.size();
  for (int i = 0; i < pnms_ret.size(); i++) {
    output_data[i] = pnms_ret[i];
  }
}

void PolyNmsExecutor::cpuCompute() {
  float iou_thresh = parser_->getProtoNode()->poly_nms_param().iou_threshold();
  auto input_box_desc = tensor_desc_[0].tensor;
  int input_boxes_num = input_box_desc->dims[0];

  VLOG(4) << "[mluPolyNms] cpu compute start, input_boxes_num:" << input_boxes_num;
  auto input_boxes = parser_->getMetaTensor(0).cpu_ptr;
  auto output_ptr = parser_->getMetaTensor(1).cpu_ptr;

  int total_output_boxes_num = 0;
  pnms_detection_cpu((float *)output_ptr, total_output_boxes_num, (float *)input_boxes,
                     input_boxes_num, iou_thresh);
  auto output_size = parser_->getMetaTensor(2).cpu_ptr;
  output_size[0] = total_output_boxes_num;
  VLOG(4) << "[mluPolyNms] cpu compute end, total_output_boxes_num:" << total_output_boxes_num;
}

int64_t PolyNmsExecutor::getTheoryOps() {
  // VLOG(4) << "getTheoryOps";
  // int cp_count = 24;//??
  // int theory_ops = parser_->getMetaTensor("input1").tensor->dims[0];
  // theory_ops *= parser_->getMetaTensor("input1").tensor->dims[1];

  // int actual_output_boxes = parser_->getMetaTensor("output1").tensor->dims[0];
  // VLOG(4) << "get the output boxes num:" << actual_output_boxes;
  // cp_count *= actual_output_boxes;
  // theory_ops *= cp_count;
  // VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  // return theory_ops;
}

int64_t PolyNmsExecutor::getTheoryIoSize() {
  // int64_t theory_ios = 0;
  // int cp_count = 24;//??
  // int theory_ops = parser_->getMetaTensor("input1").tensor->dims[0];
  // theory_ops *= parser_->getMetaTensor("input1").tensor->dims[1];

  // int actual_output_boxes = parser_->getMetaTensor("output1").tensor->dims[0];
  // VLOG(4) << "get the output boxes num:" << actual_output_boxes;
  // cp_count *= actual_output_boxes;
  // theory_ops *= cp_count;
  // theory_ios =theory_ops;
  // VLOG(4) << "getTheoryIOs: " << theory_ios << " ops";
  // return theory_ios;
}


}  // namespace mluoptest
