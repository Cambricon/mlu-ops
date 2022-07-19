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
#include <algorithm>
#include <string>
#include <vector>

#include "mlu_op.h"

#include "psroipool_forward.h"

namespace mluoptest {
void PsroipoolForwardExecutor::paramCheck() {
  VLOG(4) << "psroipool_forward param check";
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "psroipool_forward input number is wrong, it should be 2, "
               << "but now is " << parser_->getInputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 2) {
    LOG(ERROR) << "psroipool_forward output number is wrong, it should be 2, "
               << "but now is" << parser_->getOutputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  for (int i = 0; i < parser_->getInputNum(); i++) {
    if (i == 1 && parser_->inputIsNull(i)) {
      LOG(ERROR) << "psroipool_forward input [" << i << "] is nullptr.";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
    }
  }
}

void PsroipoolForwardExecutor::workspaceMalloc() {
  mluOpTensorDescriptor_t output_desc = tensor_desc_[2].tensor;
  size_t workspace_size = 0;
  MLUOP_CHECK(mluOpGetPsRoiPoolWorkspaceSize(handle_, output_desc, &workspace_size));
  VLOG(4) << "Malloc workspace space.";

  void *temp = mlu_runtime_.allocate(workspace_size);
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << workspace_size;
}

void PsroipoolForwardExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "Free device workspace space.";
    GTEST_CHECK(CNRT_RET_SUCCESS == mlu_runtime_.deallocate(workspace_[0]));
    workspace_[0] = nullptr;
  }
}

void PsroipoolForwardExecutor::transposeNchwToNhwc(
    const float *in, const uint32_t dim0, const uint32_t dim1,
    const uint32_t dim2, const uint32_t dim3, float *out) {
  for (int n = 0; n < dim0; n++) {
    for (int c = 0; c < dim1; c++) {
      for (int h = 0; h < dim2; h++) {
        for (int w = 0; w < dim3; w++) {
          out[n * dim1 * dim2 * dim3 + h * dim1 * dim3 + w * dim1 + c] =
              in[n * dim1 * dim2 * dim3 + c * dim2 * dim3 + h * dim3 + w];
        }
      }
    }
  }
}

void PsroipoolForwardExecutor::transposeNhwcToNchw(
    const float *in, const uint32_t dim0, const uint32_t dim1,
    const uint32_t dim2, const uint32_t dim3, float *out) {
  for (int n = 0; n < dim0; n++) {
    for (int h = 0; h < dim1; h++) {
      for (int w = 0; w < dim2; w++) {
        for (int c = 0; c < dim3; c++) {
          out[n * dim1 * dim2 * dim3 + c * dim1 * dim2 + h * dim2 + w] =
              in[n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c];
        }
      }
    }
  }
}

void PsroipoolForwardExecutor::initData() {
  output_dim_ = parser_->getProtoNode()->psroipool_forward_param().output_dim();
  pooled_height_ =
      parser_->getProtoNode()->psroipool_forward_param().pooled_height();
  pooled_width_ =
      parser_->getProtoNode()->psroipool_forward_param().pooled_width();
  spatial_scale_ =
      parser_->getProtoNode()->psroipool_forward_param().spatial_scale();
  group_size_ = parser_->getProtoNode()->psroipool_forward_param().group_size();
  batch_size_ = tensor_desc_[0].tensor->dims[0];
  rois_offset_ = tensor_desc_[1].tensor->dims[1];
}

void PsroipoolForwardExecutor::compute() {
  initData();
  mluOpTensorDescriptor_t input_desc = tensor_desc_[0].tensor;
  mluOpTensorLayout_t input_layout;
  mluOpDataType_t input_dtype;
  int input_dim = 0;
  int input_dims[8] = {0};
  mluOpGetTensorDescriptor(input_desc, &input_layout, &input_dtype,
                           &input_dim, input_dims);
  int channels = input_dims[3];
  int height = input_dims[1];
  int width = input_dims[2];
  mluOpTensorDescriptor_t rois_desc = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t output_desc = tensor_desc_[2].tensor;
  mluOpTensorDescriptor_t mapping_channel_desc = tensor_desc_[3].tensor;
  auto input = data_vector_[0].device_ptr;
  auto rois = data_vector_[1].device_ptr;
  auto output = data_vector_[2].device_ptr;
  auto mapping_channel = data_vector_[3].device_ptr;
  size_t workspace_size = 0;
  MLUOP_CHECK(mluOpGetPsRoiPoolWorkspaceSize(handle_, output_desc, &workspace_size));
  interface_timer_.start();
  MLUOP_CHECK(mluOpPsRoiPoolForward(
      handle_, spatial_scale_, group_size_, input_desc, input,
      rois_desc, rois, workspace_[0], workspace_size,
      output_desc, output, mapping_channel_desc, mapping_channel));
  interface_timer_.stop();
}

void PsroipoolForwardExecutor::cpuCompute() {
  mluOpTensorDescriptor_t input_desc = tensor_desc_[0].tensor;
  mluOpTensorLayout_t input_layout;
  mluOpDataType_t input_dtype;
  int input_dim = 0;
  int input_dims[8] = {0};
  MLUOP_CHECK(mluOpGetTensorDescriptor(input_desc, &input_layout, &input_dtype,
                           &input_dim, input_dims));
  int channels = input_dims[3];
  int height = input_dims[1];
  int width = input_dims[2];
  mluOpTensorDescriptor_t rois_desc = tensor_desc_[1].tensor;
  int desc_dim = 0;
  int desc_dims[8] = {0};
  mluOpTensorLayout_t desc_layout;
  mluOpDataType_t desc_datatype;
  MLUOP_CHECK(mluOpGetTensorDescriptor(rois_desc, &desc_layout, &desc_datatype,
                           &desc_dim, desc_dims));
  auto input = cpu_fp32_input_[0];
  auto output = cpu_fp32_output_[0];
  auto *mapping_channel = cpu_fp32_output_[1];
  int input_count = batch_size_ * height * width * channels;
  // tans input data
  float *input_trans =
      (float *)cpu_runtime_.allocate(new float[input_count]);
  transposeNhwcToNchw(input, batch_size_, height, width, channels,
                      input_trans);
  auto rois = cpu_fp32_input_[1];
  int rois_num = desc_dims[0] * desc_dims[1] / rois_offset_;
  int rois_count = rois_num * rois_offset_;  // cur batch
  // output
  int top_data_count = rois_num * pooled_height_ * pooled_width_ * output_dim_;
  std::vector<float> output_vec(top_data_count, -65504.0);
  float *output_pre = output_vec.data();
  // mapping_channel
  std::vector<float> mapping_channel_vec(top_data_count, -65504.0);
  float *mapping_channel_pre = mapping_channel_vec.data();
  float *rois_trans =
      (float *)cpu_runtime_.allocate(new float[rois_count]);
  memcpy(rois_trans, rois, rois_count * sizeof(float));
  for (int rois_id = 0; rois_id < rois_num; rois_id++) {
    int roi_add = rois_id * 5;
    int rois_batch_ind;
    float roi_start_w, roi_start_h, roi_end_w, roi_end_h;

    rois_batch_ind = rois_trans[roi_add];
    roi_start_w = static_cast<float>(round(rois_trans[roi_add + 1])) *
                  spatial_scale_;
    roi_start_h = static_cast<float>(round(rois_trans[roi_add + 2])) *
                  spatial_scale_;
    roi_end_w = static_cast<float>(round(rois_trans[roi_add + 3]) + 1.) *
                spatial_scale_;
    roi_end_h = static_cast<float>(round(rois_trans[roi_add + 4]) + 1.) *
                spatial_scale_;

    float roi_width = std::max(roi_end_w - roi_start_w, (float)0.1);
    float roi_height = std::max(roi_end_h - roi_start_h, (float)0.1);
    float bin_size_h = (float)roi_height / (float)(pooled_height_);
    float bin_size_w = (float)roi_width / (float)(pooled_width_);

    for (int ctop = 0; ctop < output_dim_; ctop++) {
      for (int ph = 0; ph < pooled_height_; ph++) {
        for (int pw = 0; pw < pooled_width_; pw++) {
          int index = rois_id * output_dim_ * pooled_height_ * pooled_width_ +
                      ctop * pooled_height_ * pooled_width_ +
                      ph * pooled_width_ + pw;
          int hstart = floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
          int wstart = floor(static_cast<float>(pw) * bin_size_w + roi_start_w);
          int hend =
              ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
          int wend =
              ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);

          hstart = std::min(std::max(hstart, 0), height);
          hend = std::min(std::max(hend, 0), height);
          wstart = std::min(std::max(wstart, 0), width);
          wend = std::min(std::max(wend, 0), width);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          int gw = pw;
          int gh = ph;
          int c = (ctop * group_size_ + gh) * group_size_ + gw;
          float out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * width + w;
              out_sum += input_trans[(rois_batch_ind * channels + c) *
                                              height * width +
                                          bottom_index];
            }
          }
          float bin_area = (hend - hstart) * (wend - wstart);
          if (is_empty) {
            output_pre[index] = 0;
          } else {
            output_pre[index] = out_sum / bin_area;
          }
          mapping_channel_pre[index] = c;
        }
      }
    }
  }
  cpu_runtime_.deallocate(input_trans);
  cpu_runtime_.deallocate(rois_trans);
  transposeNchwToNhwc(output_pre, rois_num, output_dim_, pooled_height_,
                      pooled_width_, output);
  transposeNchwToNhwc(mapping_channel_pre, rois_num, output_dim_,
                      pooled_height_, pooled_width_, mapping_channel);
}

int64_t PsroipoolForwardExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }

  int64_t theory_ops = 0;
  mluOpTensorDescriptor_t input_desc = tensor_desc_[0].tensor;
  mluOpTensorLayout_t input_layout;
  mluOpDataType_t input_dtype;
  int input_dim = 0;
  int input_dims[8] = {0};
  MLUOP_CHECK(mluOpGetTensorDescriptor(input_desc, &input_layout, &input_dtype,
                           &input_dim, input_dims));
  int channels = input_dims[3];
  int height = input_dims[1];
  int width = input_dims[2];
  mluOpTensorDescriptor_t rois_desc = tensor_desc_[1].tensor;
  int desc_dim = 0;
  int desc_dims[8] = {0};
  mluOpTensorLayout_t desc_layout;
  mluOpDataType_t desc_datatype;
  MLUOP_CHECK(mluOpGetTensorDescriptor(rois_desc, &desc_layout, &desc_datatype,
                           &desc_dim, desc_dims));
  float *input = cpu_fp32_input_[0];
  float *output = cpu_fp32_output_[0];
  auto *mapping_channel = cpu_fp32_output_[1];
  int input_count = batch_size_ * height * width * channels;
  // tans input data
  float *input_trans =
      (float *)cpu_runtime_.allocate(new float[input_count]);
  transposeNhwcToNchw(input, batch_size_, height, width, channels,
                      input_trans);
  float *rois = cpu_fp32_input_[1];
  int rois_num = desc_dims[0] * desc_dims[1] / rois_offset_;
  int rois_count = rois_num * rois_offset_;  // cur batch

  // output
  int top_data_count = rois_num * pooled_height_ * pooled_width_ * output_dim_;
  std::vector<float> output_vec(top_data_count, -65504.0);
  float *output_pre = output_vec.data();
  // mapping_channel
  std::vector<float> mapping_channel_vec(top_data_count, -65504.0);
  float *mapping_channel_pre = mapping_channel_vec.data();
  float *rois_trans =
      (float *)cpu_runtime_.allocate(new float[rois_count]);
  memcpy(rois_trans, rois, rois_count * sizeof(float));
  for (int rois_id = 0; rois_id < rois_num; rois_id++) {
    int roi_add = rois_id * 5;
    int rois_batch_ind;
    float roi_start_w, roi_start_h, roi_end_w, roi_end_h;

    rois_batch_ind = rois_trans[roi_add];
    roi_start_w = static_cast<float>(rint(rois_trans[roi_add + 1])) *
                  spatial_scale_;
    roi_start_h = static_cast<float>(rint(rois_trans[roi_add + 2])) *
                  spatial_scale_;
    roi_end_w = static_cast<float>(rint(rois_trans[roi_add + 3]) + 1.) *
                spatial_scale_;
    roi_end_h = static_cast<float>(rint(rois_trans[roi_add + 4]) + 1.) *
                spatial_scale_;

    float roi_width = std::max<float>(roi_end_w - roi_start_w, 0.1);
    float roi_height = std::max<float>(roi_end_h - roi_start_h, 0.1);
    float bin_size_h = roi_height / static_cast<float>(pooled_height_);
    float bin_size_w = roi_width / static_cast<float>(pooled_width_);

    for (int ctop = 0; ctop < output_dim_; ctop++) {
      for (int ph = 0; ph < pooled_height_; ph++) {
        for (int pw = 0; pw < pooled_width_; pw++) {
          int index = rois_id * output_dim_ * pooled_height_ * pooled_width_ +
                      ctop * pooled_height_ * pooled_width_ +
                      ph * pooled_width_ + pw;
          int hstart = floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
          int wstart = floor(static_cast<float>(pw) * bin_size_w + roi_start_w);
          int hend =
              ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
          int wend =
              ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);

          hstart = std::min(std::max(hstart, 0), height);
          hend = std::min(std::max(hend, 0), height);
          wstart = std::min(std::max(wstart, 0), width);
          wend = std::min(std::max(wend, 0), width);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          int gw = pw;
          int gh = ph;
          int c = (ctop * group_size_ + gh) * group_size_ + gw;
          float out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              theory_ops += 8;
            }
          }
        }
      }
    }
  }

  cpu_runtime_.deallocate(input_trans);
  cpu_runtime_.deallocate(rois_trans);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
