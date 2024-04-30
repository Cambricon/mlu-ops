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
#include "sgetrf.h"

namespace mluoptest {

static void luDecomposition(float* matrix, int m, int n) {
    int size = (m < n) ? m : n; // 获取较小的维度作为 LU 分解的大小
    for (int k = 0; k < size; k++) {
        for (int i = k + 1; i < m; i++) {
            matrix[i * n + k] /= matrix[k * n + k];
            for (int j = k + 1; j < n; j++) {
                matrix[i * n + j] -= matrix[i * n + k] * matrix[k * n + j];
            }
        }
    }
}

static void assign_lower_upper(float *A, float *L, float *U, int m, int n) {
    if (m <= 0 || n <= 0) {
        //printf("矩阵维度无效\n");
        return;
    }

    // 将下三角赋值给 L
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) {
                L[i * n + j  ] = A[i * n + j];
            }
            else if(j==i) L[i * n + j ] = 1 ;
        }
    }

    // 将上三角赋值给 U
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j >= i) {
                U[i * n  + j ] = A[i * n + j ];
            }
        }
    }
    // //printf("assigning result A to L and U:\n");
    // printCol(m,n,A);
    // printCol(m,n,L);
    // printCol(n,n,U);
}

// 计算 LU 分解误差

/*
LU(dst)是L*U后反解原矩阵后的结果
res(src)是对原矩阵进行LU分解后的结果，其上三角是U,下三角是L
*/
static void computeLUError(float *LU, float *res, int m,int n)
{
    double error = 0.0;
    double normA = 0.0;
    // float *L[m*n]={0};
    // float *U[n*n]={0};
    std::unique_ptr<float[]> L(new float[m*n]());
    std::unique_ptr<float[]> U(new float[n*n]());
    // printf("111\n");

    assign_lower_upper(res,L.get(),U.get(),m,n);


    // printf("111\n");
    // 计算 A 的范数
    // float matnorm = compute_frobenius_norm(A, m, n);
    // //printf("A matnorm: %f \n",matnorm);

    // 计算 L*U-A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // double sum=LU[i * n + j ];
            // double c=0;
            double temp=0;
            for (int k = 0; k <= i && k < n; k++) {
                // double y=L[i * n + k] * U[k * n + j]-c;
                // double t=sum+y;
                // c=(t-sum)-y;
                // sum=t;
                temp+=L[i * n + k ] * U[k * n + j];
                // LU[i * n + j ] += L[i * n + k ] * U[k * n + j];
            }
            // LU[i * n + j] = sum;
            LU[i * n + j] = temp;
            // if(i==36&&j==4)
            // {
            //     //printf("(36,4) %.0f \n",LU[i  + j * m]);
            // }
            // LU[i  + j * m]-=A[i  + j * m];
        }
    }
    // printf("111\n");
    // //printf("A A-LU:\n");
    // printCol(m,n,A);
    // printCol(m,n,LU);

    // float residual = compute_frobenius_norm(LU, m, n);

    // //printf("residual / (matnorm * n): %.10f/(%.10f*%d) \n",residual,matnorm,n);
    // 归一化误差
    // return LU;
    return;
}


static int actest2(float *src,float *dst,int m,int n)
{
    int flag=0;
    int frst_flag=1;
    double max_error=0.f,frst_error=0.f;
    int max_err_i,max_err_j=-1;
    int frst_err_i,frst_err_j=-1;
    double avg_error=0.f;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            double temp=fabs(src[i*n+j]-dst[i*n+j]);
            if(temp>=max_error)
            {
                if(frst_flag==1&&temp>1)
                {
                    frst_error=temp;
                    frst_err_i=i;
                    frst_err_j=j;
                    frst_flag=0;
                }
                max_error=temp;
                max_err_i=i;
                max_err_j=j;
            }
            avg_error+=temp;
          
        }
    }
    avg_error=avg_error/(m*n);
    printf("max_error at (%d,%d) = %.7f - %.7f = %.7f, avg_error :  %.7f\n",max_err_i,max_err_j,src[max_err_i*n+max_err_j],dst[max_err_i*n+max_err_j],max_error,avg_error);
    if(frst_flag==0)
        printf("first_error at (%d,%d) = %.7f - %.7f = %.7f\n",frst_err_i,frst_err_j,src[frst_err_i*n+frst_err_j],dst[frst_err_i*n+frst_err_j],frst_error);
    flag=max_error<3e-6?0:-1;
    return flag;
}

void SgetrfExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 1,
              "abs tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "abs tensor output number is wrong.");
}



void SgetrfExecutor::compute() {
  VLOG(4) << "AbsExecutor compute ";

  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;

  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpAbs()";
  auto dev_a = data_vector_[0].host_ptr;
  auto dev_b = data_vector_[1].host_ptr;

  auto count = parser_->input(0)->shape_count;
  int row = tensor_x->dims[0] ;
  int col = tensor_x->dims[1] ;
  
  int ipiv[row];
  int info;
  int mode;
  for(int i=0;i<row;i++)
      ipiv[i]=row-i-1;
  interface_timer_.start();
  MLUOP_CHECK(mluOpSgetrf(handle_, tensor_x, dev_x, tensor_y, dev_y,ipiv,&info,mode));
  interface_timer_.stop();

  int m=row;
  int n=col;
  std::unique_ptr<float[]> hA2(new float[m*n]);
  std::unique_ptr<float[]> hA(new float[m*n]);
  std::unique_ptr<float[]> LU(new float[m*n]());
  float *hA_raw=hA.get();
  float *LU_=LU.get();
  cnrtMemcpy(hA_raw,dev_y,m*n*sizeof(float),cnrtMemcpyDevToHost);

  computeLUError(LU_,hA_raw,m,n);
  cnrtMemcpy(hA2.get(),dev_x,m*n*sizeof(float),cnrtMemcpyDevToHost);
//   if(actest2(LU_,hA2.get(),m,n)==0) printf("Passed!\n");
  cnrtMemcpy(dev_y,LU_,m*n*sizeof(float),cnrtMemcpyHostToDev); 
}

void SgetrfExecutor::cpuCompute() {
  auto count = parser_->input(0)->shape_count;
  int row = tensor_desc_[0].tensor->dims[0] ;
  int col = tensor_desc_[0].tensor->dims[1] ;
  printf("CPUinput count row col %lu %d %d\n",count,row,col);
  
  for (int i = 0; i < count; ++i) {
    // if(std::isinf(cpu_fp32_input_[0][i])||std::isnan(cpu_fp32_input_[0][i]))
    // {
    //   printf("%d is inf or nan\n",i);
    // }
    cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
  }
}

int64_t SgetrfExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
