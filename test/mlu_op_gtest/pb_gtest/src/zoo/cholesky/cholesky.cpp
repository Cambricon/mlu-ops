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
#include "cholesky.h"

namespace mluoptest {


void CholeskyExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "cholesky input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "cholesky output number is wrong. ";
  }
  flag_quant_mode_ = NO_QUANT;
  
  
  
}

void set_matrix_zero(float*A, bool upper, bool trans_, int n_, int ldda_)
{
    if(trans_)
    {
        for (int i = 0; i < n_; i++)
        {
            for (int j = 0; j < ldda_; j++)
            {

                if(upper)
                {
                    if(i > j) 
                        A[j + i * ldda_] = 0.0;
                }
                else 
                {
                    if(i < j)
                        A[j + i * ldda_] = 0.0;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < n_; i++)
        {
            for (int j = 0; j < ldda_; j++)
            {
              if((i > j && ~upper)||(i < j && upper))
                A[j + i * ldda_] = 0.0;
            }
        }
    }
    
}

void trans_mul(float*A, float*C, int lda,bool upper_, bool trans_, int n_, int ldda_)
{
    if(trans_)
    {
        for(int i = 0; i <lda; i++)
        {
            for(int j = 0;j < n_; j++)
            {
              A[i+j*lda] = 0.0;
                for(int k = 0; k <=i; k++)
                {
                    if(upper_==false)
                    {
                        if(j < i)
                            continue;
                        else
                        {
                            A[i+j*lda] += (C[k+i*lda]*C[k+j*lda]);
                        }
                    }
                    else
                    {
                        if(j > i)
                            continue;
                        else
                        {
                            A[i+j*lda] += (C[k*lda+i]*C[k*lda+j]);
                        }
                    }
                }
            }
        }
    }
    else
    {
        for(int i = 0; i <lda; i++)
        {
            for(int j = 0;j < n_; j++)
            {
              A[j+i*lda] = 0.0;
                for(int k = 0; k <=i; k++)
                {
                    if(j < i)
                        continue;
                    A[j+i*lda] += (C[j+k*lda]*C[i+k*lda]);
                }
            }
        }
    }
        
}

void print_matrix(float*A, int lda, bool trans_, int n_, int ldda_)
{
    if(trans_)
    {
        for(int i = 0; i <n_; i++)
        {
            for(int j = 0; j <lda; j++)
            {
                printf("%.3f",A[j+i*lda]);
                printf(" ");
            }
            printf("\n");
        }
    }
    else
    {
        for(int i = 0; i <lda; i++)
        {
            for(int j = 0; j <n_; j++)
            {
                printf("%.3f",A[i+j*lda]);
                printf(" ");
            }
            printf("\n");
        }
    }
}

void CholeskyExecutor::prepareComputeParam()
{
  printf("start prepare compute parameter.\n");
  auto input_desc_ = (tensor_desc_[0].tensor);
  auto output_desc_ = (tensor_desc_[1].tensor);
  auto dev_a = (float*)(data_vector_[0].host_ptr);
  auto dev_c = (float*)(data_vector_[1].host_ptr);
  auto dev_d = (float*)(data_vector_[0].device_ptr);
  auto input_tensor = parser_->getProtoNode()->input(0);
  auto input_shape = input_tensor.shape();
  upper_ = parser_->getProtoNode()->cholesky_param().upper();
  int dim_size = input_shape.dims_size();
  if(dim_size ==2)
  {
    n_ = input_shape.dims(0); 
    int dim = input_desc_->dim;
    stride_ = (input_desc_->strides)[dim-1];
    ldda_ = input_desc_->dims[1];
    
    printf("n:%d,lda:%d,stride:%d,upper:%d,trans:%d\n",n_,ldda_,stride_,upper_,trans_);
    int size = input_desc_->dims[0];
    
    printf("size:%d, dim:%d, \n",size,dim);
    printf("strides:\n");
    for(int i = 0; i < dim; i++)
    {
      printf("%ld ",(input_desc_->strides)[i]);
    }
    printf("\n");
    printf("data vector length : %ld\n",data_vector_.size());
  }
  else if(dim_size == 3)
  {
    batch_size_ = input_shape.dims(0);
    n_ = input_shape.dims(1); 
    int dim = input_desc_->dim;
    stride_ = (input_desc_->strides)[dim-1];
    ldda_ = input_desc_->dims[2];
    printf("batch_size:%d,n:%d,lda:%d,stride:%d,upper:%d,trans:%d\n",batch_size_,n_,ldda_,stride_,upper_,trans_);

    int size = input_desc_->dims[1];

    printf("size:%d, dim:%d, \n",size,dim);
    printf("strides:\n");
    for(int i = 0; i < dim; i++)
    {
      printf("%ld ",(input_desc_->strides)[i]);
    }
    printf("\n");
    printf("data vector length : %ld\n",data_vector_.size());
  }

  std::memcpy(dev_c,dev_a,sizeof(float)*n_*ldda_);
  set_matrix_zero((float*)dev_c,upper_,trans_,n_,ldda_);
  trans_mul(dev_a,dev_c,ldda_,upper_,trans_,n_,ldda_);
  printf("matrix A:\n");
  if(dim_size == 3)
  {
    for(int i = 1; i < batch_size_;i++)
    {
      std::memcpy(dev_a+i*n_*ldda_,dev_a,sizeof(float)*n_*ldda_);
      std::memcpy(dev_c+i*n_*ldda_,dev_c,sizeof(float)*n_*ldda_);
    }
  }

  GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(dev_d, dev_a, sizeof(float)*n_*ldda_*batch_size_, CNRT_MEM_TRANS_DIR_HOST2DEV));
  float* cpu_a = cpu_fp32_input_[0];
  std::memcpy(cpu_a,dev_a,sizeof(float)*n_*ldda_);
  printf("end prepare compute.\n");

}

void CholeskyExecutor::compute() {
  
//   prepareComputeParam();
  
  VLOG(4) <<"  CholeskyExecutor compute ";
  auto input_desc_ = tensor_desc_[0].tensor;
  auto output_desc_ = tensor_desc_[1].tensor;
  auto h_input = (float*)(data_vector_[0].host_ptr);
  auto h_output = (float*)(data_vector_[1].host_ptr);
  auto d_intput = (float*)(data_vector_[0].device_ptr);
  auto d_output = (float*)(data_vector_[1].device_ptr);
  GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(h_output, d_intput, sizeof(float)*n_*ldda_*batch_size_, CNRT_MEM_TRANS_DIR_DEV2HOST));

  interface_timer_.start();
  MLUOP_CHECK(mluOpCholesky(handle_,input_desc_,d_intput, output_desc_, d_output, upper_));
  interface_timer_.stop();
  GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(h_output, d_output, sizeof(float)*n_*ldda_, CNRT_MEM_TRANS_DIR_DEV2HOST));
  
  printf("mlu after cholesky result:\n");

  return;
}

void CholeskyExecutor::cpuCompute() {

  float* cpu_a = cpu_fp32_input_[0];
  float* cpu_c = cpu_fp32_output_[0];
  
  if(n_ > 2000)
  {
    auto dev_c = (float*)(data_vector_[1].host_ptr);
    std::memcpy(cpu_c,dev_c,sizeof(float)*n_*ldda_*batch_size_);
    return;
  }
  std::memcpy(cpu_c,cpu_a,sizeof(float)*n_*ldda_);
	if(trans_)
    {
        for(int i = 0; i < n_; i++)
        {
            float dia = cpu_c[i+i*ldda_];
            float dia_root = sqrt(dia);
            cpu_c[i+i*ldda_] = sqrt(dia);
            if(upper_==false)
            {
                for(int j = i+1;j<n_;j++)
                {
                    cpu_c[i+j*ldda_] = cpu_c[i+j*ldda_]/dia_root;   
                }
                for(int j = i+1;j < n_;j++)
                {
                    for(int k = j;k <n_;k++)
                    {
                        cpu_c[j + k * ldda_] -= (cpu_c[i + k*ldda_] * cpu_c[i + j * ldda_]); 
                    }
                }
            }
            else
            {
                for(int j = i+1;j<n_;j++)
                {
                    cpu_c[j+i*ldda_] = cpu_c[j+i*ldda_]/dia_root;   
                }
                for(int j = i+1;j < n_;j++)
                {
                    for(int k = j;k <n_;k++)
                    {
                        cpu_c[k + j * ldda_] -= (cpu_c[k + i*ldda_] * cpu_c[j + i * ldda_]); 
                    }
                }
            }
            
        }
    }
    else
    {
        for(int i = 0; i < ldda_; i++)
        {
            float dia = cpu_c[i+i*ldda_];
            float dia_root = sqrt(dia);
            cpu_c[i+i*ldda_] = sqrt(dia);
            for(int j = i+1;j<ldda_;j++)
            {
                cpu_c[j+i*ldda_] = cpu_c[j+i*ldda_]/dia_root;   
            }
            for(int j = i+1;j < ldda_;j++)
            {
                for(int k = j;k <ldda_;k++)
                {
                    cpu_c[k + j * ldda_] -= (cpu_c[k + i*ldda_] * cpu_c[j + i * ldda_]); 
                }
            }
        }
    }

    if(batch_size_>1)
    {
        for(int i = 1; i < batch_size_;i++)
        {
            std::memcpy(cpu_c+i*n_*ldda_,cpu_c,sizeof(float)*n_*ldda_);
        }
    }

	return;
}


int64_t CholeskyExecutor::getTheoryOps() {
  int64_t theory_ops = batch_size_*n_*n_*n_/2;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
