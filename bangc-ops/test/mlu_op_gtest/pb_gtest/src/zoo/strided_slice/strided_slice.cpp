/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "strided_slice.h"

#include <complex>

namespace mluoptest {

void StridedSliceExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_strided_slice_param()) {
    LOG(ERROR) << "Lose strided_slice param. ";
  }
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "strided_slice input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "strided_slice output number is wrong. ";
  }
}

void StridedSliceExecutor::compute() {
  VLOG(4) << "StridedSliceExecutor compute ";
  if (!parser_->getProtoNode()->has_strided_slice_param()) {
    LOG(ERROR) << "Lose strided_slice param. ";
  }

  auto input_desc = tensor_desc_[0].tensor;
  auto output_desc = tensor_desc_[1].tensor;
  auto input = data_vector_[0].device_ptr;
  auto output = data_vector_[1].device_ptr;

  int begin[MLUOP_DIM_MAX];
  int end[MLUOP_DIM_MAX];
  int stride[MLUOP_DIM_MAX];

  for (int i = 0; i < input_desc->dim; i++) {
    begin[i] = parser_->getProtoNode()->strided_slice_param().begin(i);
    end[i] = parser_->getProtoNode()->strided_slice_param().end(i);
    stride[i] = parser_->getProtoNode()->strided_slice_param().stride(i);
  }

  VLOG(4) << "call mluOpStridedSlice()";

  interface_timer_.start();
  MLUOP_CHECK(mluOpStridedSlice(handle_, input_desc, input, begin, end, stride,
                                output_desc, output));
  interface_timer_.stop();
}

void StridedSliceExecutor::cpuCompute() {
  const int kDimNb = tensor_desc_[0].tensor->dim;

  // int out_dim;
  int input_dims[kDimNb];
  int output_dims[kDimNb];
  int begin[kDimNb];
  int end[kDimNb];
  int stride[kDimNb];

  // check parameters
  for (int i = 0; i < kDimNb; i++) {
    input_dims[i] = tensor_desc_[0].tensor->dims[i];
    output_dims[i] = tensor_desc_[1].tensor->dims[i];
    begin[i] = parser_->getProtoNode()->strided_slice_param().begin(i);
    end[i] = parser_->getProtoNode()->strided_slice_param().end(i);
    stride[i] = parser_->getProtoNode()->strided_slice_param().stride(i);

    // assert(stride[i] != 0);

    // unwrap negative index
    if (begin[i] < 0) {
      begin[i] += input_dims[i];
    }
    if (end[i] < 0) {
      end[i] += input_dims[i];
    }

    // deal with out-of-bound indices
    if (stride[i] > 0 && begin[i] < end[i] && end[i] > 0 &&
        begin[i] < input_dims[i]) {
      // case 1: stride > 0, end > 0, begin < end, begin < dim
      if (begin[i] < 0) {
        begin[i] = 0;
      }
      if (end[i] > input_dims[i] - 1) {
        end[i] = input_dims[i];
      }

      // out_dim = (end[i] - begin[i] + stride[i] - 1) / stride[i];
    } else if (stride[i] < 0 && end[i] < begin[i] && begin[i] >= 0 &&
               end[i] < input_dims[i] - 1) {
      // case 2: stride < 0, begin >= 0, end < begin, end < dim-1
      if (begin[i] > input_dims[i] - 1) {
        begin[i] = input_dims[i] - 1;
      }
      if (end[i] < 0) {
        end[i] = -1;  // to include the first element
      }
      // out_dim = (end[i] - begin[i] + stride[i] - 1) / stride[i];
    }
  }

  // get strides of input/output tensors
  int input_strides[kDimNb];
  int output_strides[kDimNb];

  input_strides[kDimNb - 1] = 1;
  output_strides[kDimNb - 1] = 1;
  for (int i = kDimNb - 2; i >= 0; i--) {
    input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
    output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
  }

  // number of output elements
  int output_size = 1;
  for (int i = 0; i < kDimNb; i++) {
    output_size *= output_dims[i];
  }

  // extract each output element
  int input_indices[kDimNb];
  int output_indices[kDimNb];

  for (int output_index = 0; output_index < output_size; output_index++) {
    // get output tensor indices
    int index = output_index;
    for (int idim = kDimNb - 1; idim >= 1; idim--) {
      output_indices[idim] = index % output_dims[idim];
      index /= output_dims[idim];
    }
    output_indices[0] = index;

    // get corresponding input tensor indices
    for (int idim = 0; idim < kDimNb; idim++) {
      input_indices[idim] = begin[idim] + output_indices[idim] * stride[idim];
    }

    // get 1D index into the input tensor
    int input_index = 0;
    for (int idim = 0; idim < kDimNb; idim++) {
      input_index += input_indices[idim] * input_strides[idim];
    }

    // copy values
    if (parser_->input(0)->dtype == MLUOP_DTYPE_DOUBLE) {
      double *host_input = reinterpret_cast<double *>(cpu_fp32_input_[0]);
      double *host_output = reinterpret_cast<double *>(cpu_fp32_output_[0]);
      host_output[output_index] = host_input[input_index];
    } else if (parser_->input(0)->dtype == MLUOP_DTYPE_COMPLEX_HALF ||
               parser_->input(0)->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      std::complex<float> *host_input =
          reinterpret_cast<std::complex<float> *>(cpu_fp32_input_[0]);
      std::complex<float> *host_output =
          reinterpret_cast<std::complex<float> *>(cpu_fp32_output_[0]);
      host_output[output_index] = host_input[input_index];
    } else {
      float *host_input = cpu_fp32_input_[0];
      float *host_output = cpu_fp32_output_[0];
      host_output[output_index] = host_input[input_index];
    }
  }
}

int64_t StridedSliceExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t StridedSliceExecutor::getTheoryIoSize() {
  int64_t theory_ios = 0;
  auto tensor_out = tensor_desc_[1].tensor;

  theory_ios = 2 * tensor_out->total_tensor_size;
  VLOG(4) << "getTheoryIOs: " << theory_ios << " bytes";
  return theory_ios;
}

}  // namespace mluoptest
