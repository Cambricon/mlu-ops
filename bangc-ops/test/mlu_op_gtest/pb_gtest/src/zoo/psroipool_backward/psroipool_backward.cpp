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

#include "psroipool_backward.h"

namespace mluoptest {
void PsroipoolBackwardExecutor::paramCheck() {
  VLOG(4) << "psroipool_backward param check";
  if (parser_->getInputNum() != 3) {
    LOG(ERROR) << "psroipool_backward input number is wrong, it should be 3, "
                  "but now is "
               << parser_->getInputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "psroipool_forward output number is wrong, it should be 1, "
                  "but now is"
               << parser_->getOutputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  for (int i = 0; i < parser_->getInputNum(); i++) {
    if ( i == 1 && parser_->inputIsNull(i)) {
      LOG(ERROR) << "psroipool_backward input [" << i << "] is nullptr.";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
    }
  }
}

template <typename T1, typename T2>
void PsroipoolBackwardExecutor::transposeNchwToNhwc(
    const T1 *in, const T2 dim0, const T2 dim1,
    const T2 dim2, const T2 dim3, T1 *out) {
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

template <typename T1, typename T2>
void PsroipoolBackwardExecutor::transposeNhwcToNchw(
    const T1 *in, const T2 dim0, const T2 dim1,
    const T2 dim2, const T2 dim3, T1 *out) {
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

void PsroipoolBackwardExecutor::initData() {
  pooled_height_ =
      parser_->getProtoNode()->psroipool_backward_param().pooled_height();
  pooled_width_ =
      parser_->getProtoNode()->psroipool_backward_param().pooled_width();

  output_dim_ = parser_->getProtoNode()->psroipool_backward_param().output_dim();
  spatial_scale_ =
      parser_->getProtoNode()->psroipool_backward_param().spatial_scale();
  rois_num_ = tensor_desc_[1].tensor->dims[0];
  rois_offset_ = tensor_desc_[1].tensor->dims[1];
  batch_size_ = tensor_desc_[3].tensor->dims[0];
  height_ = tensor_desc_[3].tensor->dims[1];
  width_ = tensor_desc_[3].tensor->dims[2];
  channels_ = tensor_desc_[3].tensor->dims[3];
}

void PsroipoolBackwardExecutor::compute() {
  printf("begin Compute\n");
  initData();
  mluOpTensorDescriptor_t top_grad_desc = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t rois_desc = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t mapping_channel_desc = tensor_desc_[2].tensor;
  mluOpTensorDescriptor_t bottom_grad_desc = tensor_desc_[3].tensor;
  auto top_grad = data_vector_[0].device_ptr;
  auto rois = data_vector_[1].device_ptr;
  auto mapping_channel = data_vector_[2].device_ptr;
  auto bottom_grad = data_vector_[3].device_ptr;
  
  interface_timer_.start();
  MLUOP_CHECK(mluOpPsRoiPoolBackward(
      handle_, pooled_height_, pooled_width_, spatial_scale_, output_dim_,
      top_grad_desc, top_grad, rois_desc, rois, mapping_channel_desc,
      mapping_channel, bottom_grad_desc, bottom_grad));
  
  interface_timer_.stop();
  printf("end cpuCompute\n");
}

void PsroipoolBackwardExecutor::cpuCompute() {
  printf("begin cpuCompute\n");
  int rois_count = rois_num_ * rois_offset_;
  auto top_grad = cpu_fp32_input_[0];
  auto rois = cpu_fp32_input_[1];
  auto mapping_channel = cpu_fp32_input_[2];
  auto bottom_grad = cpu_fp32_output_[0];
  printf("begin trans\n");
  // tans top_grad/mapping_channel
  int top_grad_count = rois_num_ * pooled_height_ * pooled_width_ * output_dim_;
  int mapping_channel_count = top_grad_count;
  float *top_grad_trans = (float *)cpu_runtime_.allocate(new float[top_grad_count]);
  int *mapping_channel_trans = (int *)cpu_runtime_.allocate(new int[mapping_channel_count]);
  transposeNhwcToNchw(top_grad, rois_num_, pooled_height_, pooled_width_, output_dim_,
                      top_grad_trans);
  transposeNhwcToNchw((int *)mapping_channel, rois_num_, pooled_height_, pooled_width_, output_dim_,
                      (int *)mapping_channel_trans);
  int bottom_grad_count = batch_size_ * height_ * width_ * channels_;
  std::vector<float> bottom_grad_vec(top_grad_count, -65504.0);
  float *bottom_grad_pre = bottom_grad_vec.data();
  float *rois_trans =
      (float *)cpu_runtime_.allocate(new float[rois_count]);
  memcpy(rois_trans, rois, rois_count * sizeof(float));
  printf("begin compute\n");
  for (int rois_id = 0; rois_id < rois_num_; rois_id++) {
    printf("begin 1\n");
    int roi_add = rois_id * 5;
    int rois_batch_ind;
    float roi_start_w, roi_start_h, roi_end_w, roi_end_h;
    printf("begin 2\n");
    rois_batch_ind = rois_trans[roi_add];
    roi_start_w = static_cast<float>(round(rois_trans[roi_add + 1])) *
                  spatial_scale_;
    roi_start_h = static_cast<float>(round(rois_trans[roi_add + 2])) *
                  spatial_scale_;
    roi_end_w = static_cast<float>(round(rois_trans[roi_add + 3]) + 1.) *
                spatial_scale_;
    roi_end_h = static_cast<float>(round(rois_trans[roi_add + 4]) + 1.) *
                spatial_scale_;
    printf("begin 3\n");
    float roi_width = std::max(roi_end_w - roi_start_w, (float)0.1);
    float roi_height = std::max(roi_end_h - roi_start_h, (float)0.1);
    float bin_size_h = (float)roi_height / (float)(pooled_height_);
    float bin_size_w = (float)roi_width / (float)(pooled_width_);
    printf("begin 4\n");
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
          hstart = std::min(std::max(hstart, 0), height_);
          hend = std::min(std::max(hend, 0), height_);
          wstart = std::min(std::max(wstart, 0), width_);
          wend = std::min(std::max(wend, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int c = mapping_channel[index];
          int offset = (rois_batch_ind * channels_ + c) * height_ * width_;
          float bin_area = (hend - hstart)*(wend - wstart);
          float diff_val = is_empty ? 0. : top_grad[index] / bin_area;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * width_ + w;
              bottom_grad_pre[offset + bottom_index] += diff_val;
            }
          }
        }
      }
    }
  }

  cpu_runtime_.deallocate(rois_trans);
  cpu_runtime_.deallocate(top_grad_trans);
  cpu_runtime_.deallocate(mapping_channel_trans);
  transposeNchwToNhwc(bottom_grad_pre, batch_size_, channels_, height_, width_, bottom_grad);
  printf("end cpucompute\n");
}

int64_t PsroipoolBackwardExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }

  int64_t theory_ops = 0;
  int rois_count = rois_num_ * rois_offset_;
  auto top_grad = cpu_fp32_input_[0];
  auto input_rois = cpu_fp32_input_[1];
  auto mapping_channel = cpu_fp32_input_[2];
  auto bottom_grad = cpu_fp32_output_[0];

  // tans top_grad/mapping_channel
  int top_grad_count = rois_num_ * pooled_height_ * pooled_width_ * output_dim_;
  int mapping_channel_count = top_grad_count;
  float *top_grad_trans = (float *)cpu_runtime_.allocate(new float[top_grad_count]);
  int *mapping_channel_trans = (int *)cpu_runtime_.allocate(new int[mapping_channel_count]);
  transposeNhwcToNchw(top_grad, rois_num_, pooled_height_, pooled_width_, output_dim_,
                      top_grad_trans);
  transposeNhwcToNchw((int *)mapping_channel, rois_num_, pooled_height_, pooled_width_, output_dim_,
                      (int *)mapping_channel_trans);

  int bottom_grad_count = batch_size_ * height_ * width_ * channels_;
  std::vector<float> bottom_grad_vec(top_grad_count, -65504.0);
  float *bottom_grad_pre = bottom_grad_vec.data();
  float *rois_trans =
      (float *)cpu_runtime_.allocate(new float[rois_count]);
  memcpy(rois_trans, input_rois, rois_count * sizeof(float));
  for (int rois_id = 0; rois_id < rois_num_; rois_id++) {
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

          hstart = std::min(std::max(hstart, 0), height_);
          hend = std::min(std::max(hend, 0), height_);
          wstart = std::min(std::max(wstart, 0), width_);
          wend = std::min(std::max(wend, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          // 
          int c = mapping_channel[index];
          int offset = (rois_batch_ind * channels_ + c) * height_ * width_;
          float bin_area = (hend - hstart)*(wend - wstart);
          float diff_val = is_empty ? 0. : top_grad[index] / bin_area;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              theory_ops += 4;
            }
          }
        }
      }
    }
  }

  cpu_runtime_.deallocate(rois_trans);
  cpu_runtime_.deallocate(top_grad_trans);
  cpu_runtime_.deallocate(mapping_channel_trans);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
