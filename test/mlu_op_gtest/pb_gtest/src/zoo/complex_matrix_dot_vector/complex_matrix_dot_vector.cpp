/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include "complex_matrix_dot_vector.h"
namespace mluoptest {

void ComplexMatrixDotVectorExecutor::initData() {
  pad_num_ = parser_->getProtoNode()->complex_matrix_dot_vector_param().pad_num();
  row_major_ = parser_->getProtoNode()->complex_matrix_dot_vector_param().row_major();
  output_type_ = parser_->getProtoNode()->complex_matrix_dot_vector_param().output_type();
  VLOG(4) << pad_num_ << " " << row_major_;
}

void ComplexMatrixDotVectorExecutor::paramCheck() {
  GTEST_CHECK(parser_->outputs().size() == 1,
              "tensor output number is wrong.");
  GTEST_CHECK(parser_->inputs().size() == 2,
              "tensor intput number is wrong.");
}

void ComplexMatrixDotVectorExecutor::compute() {
  VLOG(4) << "ComplexMatrixDotVectorExecutor compute ";
  initData();

  auto vector_desc = tensor_desc_[0].tensor;
  auto vector_input = data_vector_[0].device_ptr;
  auto matrix_desc = tensor_desc_[1].tensor;
  auto matrix_input = data_vector_[1].device_ptr;

  auto output_desc = tensor_desc_[2].tensor;
  auto output = data_vector_[2].device_ptr;

  VLOG(4) << "call mluOpComplexMatrixDotVector()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpComplexMatrixDotVector(handle_, vector_desc, vector_input, matrix_desc, matrix_input, pad_num_, row_major_, output_type_, output_desc, output));
  interface_timer_.stop();
}

inline float real_part(float p_real, float p_imag, float q_real, float q_imag){
  return p_real * q_real - p_imag * q_imag;
}

inline float imag_part(float p_real, float p_imag, float q_real, float q_imag){
  return p_real * q_imag + p_imag * q_real;
}

void ComplexMatrixDotVectorExecutor::cpuCompute() {
  VLOG(4) << "call cpuCompute()";
  auto dtype = parser_->getInputDataType(1);
  std::vector<int64_t> vector_shape = parser_->input(0)->shape;
  std::vector<int64_t> matrix_shape = parser_->input(1)->shape;
  std::vector<int64_t> output_shape = parser_->output(0)->shape;
  bool real_input = false;
  // VLOG(4) << "dtype " << parser_->getOutputDataType(0);
  if (dtype == MLUOP_DTYPE_FLOAT) {
      real_input = true;
      VLOG(4) << "real_input" ;
  }

  int64_t matrix_dim = matrix_shape.size();
  int64_t output_dim = output_shape.size();
  int64_t batch = 1;
  int64_t row_num = 1;
  int64_t col_num = 1;
  if(output_type_ == 0) {
    if(row_major_) {
      if (matrix_dim == 1) {
        col_num = matrix_shape[0];
      } else if(matrix_dim == 2) {
        batch = matrix_shape[0];
        col_num = matrix_shape[1];
      } else if(matrix_dim == 3) {
        batch = matrix_shape[0];
        row_num = matrix_shape[1];
        col_num = matrix_shape[2];
      }
    } else {
      if(matrix_dim == 2) {
        row_num = matrix_shape[0];
        col_num = matrix_shape[1];
      } else if(matrix_dim == 3) {
        batch = matrix_shape[0];
        row_num = matrix_shape[1];
        col_num = matrix_shape[2];
      }
    }
  } else if (output_type_ == 1) {
    if(row_major_) {
      if (matrix_dim == 1) {
        col_num = output_shape[0];
      } else if(matrix_dim == 2) {
        batch = output_shape[0];
        col_num = output_shape[1];
      } else if(matrix_dim == 3) {
        batch = output_shape[0];
        row_num = output_shape[1];
        col_num = output_shape[2];
      }
    } else {
      if(matrix_dim == 2) {
        row_num = output_shape[0];
        col_num = output_shape[1];
      } else if(matrix_dim == 3) {
        batch = output_shape[0];
        row_num = output_shape[1];
        col_num = output_shape[2];
      }
    }
  }

  VLOG(5) << "batch: " << batch;
  VLOG(5) << "row_num: " << row_num;
  VLOG(5) << "col_num: " << col_num;
  VLOG(5) << "row_major: " << row_major_;

  auto total_output = parser_->getOutputDataCount(0);
  for(int k = 0; k < total_output; k++)
  {
      cpu_fp32_output_[0][2 * k] = 0;
      cpu_fp32_output_[0][2 * k + 1] = 0;
  }

  for (int i = 0; i < batch; i++)
    {
      for (int j = 0; j < row_num; j++) 
      {
        for (int k = 0; k < col_num; k++)
        {
          // printf(" %d ", index);
          if(row_major_) {
            if(output_type_ == 0) {
              int in_idx = k + j * col_num + i * col_num * row_num;
              int ou_idx = k + j * pad_num_ + i * col_num * pad_num_;
              if(!real_input) {
                cpu_fp32_output_[0][2 * ou_idx] = real_part(
                    cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                    cpu_fp32_input_[1][2 * in_idx], cpu_fp32_input_[1][2 * in_idx + 1]);
                // cpu_fp32_input_[1][2k]*cpu_fp32_input_[0][2 * index] -
                // cpu_fp32_input_[1][2k+1]*cpu_fp32_input_[0][2 * index+1];
                cpu_fp32_output_[0][2 * ou_idx + 1] = imag_part(
                  cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                  cpu_fp32_input_[1][2 * in_idx], cpu_fp32_input_[1][2 * in_idx + 1]);
              } else {
                cpu_fp32_output_[0][2 * ou_idx] = real_part(
                  cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                  cpu_fp32_input_[1][in_idx], 0);
              // cpu_fp32_input_[1][2k]*cpu_fp32_input_[0][2 * index] -
              // cpu_fp32_input_[1][2k+1]*cpu_fp32_input_[0][2 * index+1];
                cpu_fp32_output_[0][2 * ou_idx + 1] = imag_part(
                  cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                  cpu_fp32_input_[1][in_idx], 0);
              }
            } else if(output_type_ == 1) {
              int in_idx = k + j * pad_num_ + i * col_num * pad_num_;
              int ou_idx = k + j *col_num + i * col_num * row_num;
              // printf("%f %f ", cpu_fp32_input_[0][2 * k],
              //   cpu_fp32_input_[0][2 * k + 1]);
              printf("%f %f ", cpu_fp32_input_[1][2 * in_idx],
                     cpu_fp32_input_[1][2 * in_idx + 1]);
              // if(k >= col_num && k < ) {
              cpu_fp32_output_[0][2 * ou_idx] = real_part(
                  cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                  cpu_fp32_input_[1][2 * in_idx],
                  cpu_fp32_input_[1][2 * in_idx + 1]);
              // cpu_fp32_input_[1][2k]*cpu_fp32_input_[0][2 * index] -
              // cpu_fp32_input_[1][2k+1]*cpu_fp32_input_[0][2 * index+1];
              cpu_fp32_output_[0][2 * ou_idx + 1] = imag_part(
                  cpu_fp32_input_[0][2 * k], cpu_fp32_input_[0][2 * k + 1],
                  cpu_fp32_input_[1][2 * in_idx],
                  cpu_fp32_input_[1][2 * in_idx + 1]);
              // }
            }
          } else{
            printf("test\n");
          //   if(!real_input) {
          //     cpu_fp32_output_[0][2 * index] = real_part(
          //         cpu_fp32_input_[0][2 * j], cpu_fp32_input_[0][2 * j + 1],
          //         cpu_fp32_input_[1][2 * index], cpu_fp32_input_[1][2 * index + 1]);
          //     cpu_fp32_output_[0][2 * index + 1] = imag_part(
          //       cpu_fp32_input_[0][2 * j], cpu_fp32_input_[0][2 * j + 1],
          //       cpu_fp32_input_[1][2 * index], cpu_fp32_input_[1][2 * index + 1]);
          //     // printf("%f %f ", cpu_fp32_output_[0][2 * index], cpu_fp32_output_[0][2 * index + 1]);
          //   } else {
          //     cpu_fp32_output_[0][2 * index] = real_part(
          //       cpu_fp32_input_[0][2 * j], cpu_fp32_input_[0][2 * j + 1],
          //       cpu_fp32_input_[1][index], 0);
          //     cpu_fp32_output_[0][2 * index + 1] = imag_part(
          //       cpu_fp32_input_[0][2 * j], cpu_fp32_input_[0][2 * j + 1],
          //       cpu_fp32_input_[1][index], 0);
          //   }
          }
        }
        // printf("\n");
      }
    }
  // printf("\ncpu_result\n");
  //   for (int i = 0; i < col_num; i++) {
  //       printf("%f %f ", cpu_fp32_output_[0][2 * i],
  //              cpu_fp32_output_[0][2 * i + 1]);
  //   }
  // printf("\n");

  // for (int i = 0; i < length_; i++) {
  //   printf("%f %f ", cpu_fp32_output_[0][2*i], cpu_fp32_output_[0][2*i+1]);
  // }
}

int64_t ComplexMatrixDotVectorExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->output(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
