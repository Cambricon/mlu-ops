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
// #include <complex.h>
#include "sgetrf2.h"

namespace mluoptest {

void printMatrix(const float *matrix, int rows, int cols, const char *name) {
  printf("%s矩阵:\n", name);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%.0f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}
void separateComplexMatrix(const float *matrix, float *realMatrix,
                           float *imagMatrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0, k = 0; j < cols; j += 2, k++) {
      float value = matrix[i * cols + j];
      realMatrix[i * (cols / 2) + k] = matrix[i * cols + j];
      imagMatrix[i * (cols / 2) + k] = matrix[i * cols + j + 1];
    }
  }
}
// GEMM function to multiply L and U
void gemm(float *L, float *U, float *C, int m, int k, int n) {
  // Initialize result matrix C to zero
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0.0;
    }
  }

  // Perform matrix multiplication C = L * U
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        C[i * n + j] += L[i * k + p] * U[p * n + j];
      }
    }
<<<<<<< HEAD
    void complexMultiply(const float *a, const float *b, float *result)
    {
<<<<<<< HEAD
        result[0] = a[0] * b[0] - a[1] * b[1]; 
        result[1] = a[0] * b[1] + a[1] * b[0]; 
    }
    void complexAdd(const float *a, const float *b, float *result)
    {
        result[0] = a[0] + b[0]; 
=======
        result[0] = a[0] * b[0] - a[1] * b[1];
        result[1] = a[0] * b[1] + a[1] * b[0];
    }
    void complexAdd(const float *a, const float *b, float *result)
    {
        result[0] = a[0] + b[0];
>>>>>>> fix workspace problem and some bugs
        result[1] = a[1] + b[1];
    }
    void complexMatrixMultiply(const float *A, const float *B, float *C, int m, int k, int n)
    {
        // 初始化结果矩阵 C
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
<<<<<<< HEAD
                C[(i * n + j) * 2] = 0.0f;    
                C[(i * n + j) * 2 + 1] = 0.0f; 
=======
                C[(i * n + j) * 2] = 0.0f;
                C[(i * n + j) * 2 + 1] = 0.0f;
>>>>>>> fix workspace problem and some bugs
            }
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
<<<<<<< HEAD
                float sum[2] = {0.0f, 0.0f}; 
=======
                float sum[2] = {0.0f, 0.0f};
>>>>>>> fix workspace problem and some bugs
                for (int l = 0; l < k; ++l)
                {
                    float product[2];
                    const float *a = &A[(i * k + l) * 2];
                    const float *b = &B[(l * n + j) * 2];
                    complexMultiply(a, b, product);
                    complexAdd(sum, product, sum);
                }
                C[(i * n + j) * 2] = sum[0];
                C[(i * n + j) * 2 + 1] = sum[1];
            }
        }
    }

    static void luDecomposition(float *matrix, int m, int n)
    {
<<<<<<< HEAD
        int size = (m < n) ? m : n; 
=======
        int size = (m < n) ? m : n;
>>>>>>> fix workspace problem and some bugs
        for (int k = 0; k < size; k++)
        {
            for (int i = k + 1; i < m; i++)
            {
                matrix[i * n + k] /= matrix[k * n + k];
                for (int j = k + 1; j < n; j++)
                {
                    matrix[i * n + j] -= matrix[i * n + k] * matrix[k * n + j];
                }
            }
        }
    }

    static void assign_lower_upper(float *A, float *L, float *U, int m, int n)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j < i)
                {
                    L[i * n + j] = A[i * n + j];
                }
                else if (j == i)
                    L[i * n + j] = 1;
            }
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j >= i)
                {
                    U[i * n + j] = A[i * n + j];
                }
            }
        }
<<<<<<< HEAD

=======
>>>>>>> fix workspace problem and some bugs
=======
  }
}
void cgemm(const float *x1, const float *x2, float *y, int r1, int c1, int c2) {
  float real, imag;
  int i, j, k;

  for (i = 0; i < r1; i++) {
    for (j = 0; j < c2; j++) {
      real = 0;
      imag = 0;

      for (k = 0; k < c1; k++) {
        real += (x1[i * 2 * c1 + 2 * k] * x2[k * 2 * c2 + 2 * j] -
                 x1[i * 2 * c1 + 2 * k + 1] * x2[k * 2 * c2 + 2 * j + 1]);
        imag += (x1[i * 2 * c1 + 2 * k] * x2[k * 2 * c2 + 2 * j + 1] +
                 x1[i * 2 * c1 + 2 * k + 1] * x2[k * 2 * c2 + 2 * j]);
      }

      y[i * 2 * c2 + 2 * j] = real;
      y[i * 2 * c2 + 2 * j + 1] = imag;
    }
  }
}
void complexMultiply(const float *a, const float *b, float *result) {
  result[0] = a[0] * b[0] - a[1] * b[1];
  result[1] = a[0] * b[1] + a[1] * b[0];
}
void complexAdd(const float *a, const float *b, float *result) {
  result[0] = a[0] + b[0];
  result[1] = a[1] + b[1];
}
void complexMatrixMultiply(const float *A, const float *B, float *C, int m,
                           int k, int n) {
  // 初始化结果矩阵 C
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[(i * n + j) * 2] = 0.0f;
      C[(i * n + j) * 2 + 1] = 0.0f;
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum[2] = {0.0f, 0.0f};
      for (int l = 0; l < k; ++l) {
        float product[2];
        const float *a = &A[(i * k + l) * 2];
        const float *b = &B[(l * n + j) * 2];
        complexMultiply(a, b, product);
        complexAdd(sum, product, sum);
      }
      C[(i * n + j) * 2] = sum[0];
      C[(i * n + j) * 2 + 1] = sum[1];
    }
  }
}

static void luDecomposition(float *matrix, int m, int n) {
  int size = (m < n) ? m : n;
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
    return;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (j < i) {
        L[i * n + j] = A[i * n + j];
      } else if (j == i) {
        L[i * n + j] = 1;
      }
>>>>>>> [Fix](mluOpSgetrf2): fix some bugs, reset workspace and update docs
    }
  }

<<<<<<< HEAD
    static void computeLUError(float *LU, float *res, int m, int n)
    {
        double error = 0.0;
        double normA = 0.0;
        std::unique_ptr<float[]> L(new float[m * n]());
        std::unique_ptr<float[]> U(new float[n * n]());
        // printf("111\n");

        assign_lower_upper(res, L.get(), U.get(), m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double temp = 0;
                for (int k = 0; k <= i && k < n; k++)
                {
                    temp += L[i * n + k] * U[k * n + j];
<<<<<<< HEAD

                }
                LU[i * n + j] = temp;

            }
        }
  
=======
                }
                LU[i * n + j] = temp;
            }
        }

>>>>>>> fix workspace problem and some bugs
        return;
=======
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (j >= i) {
        U[i * n + j] = A[i * n + j];
      }
>>>>>>> [Fix](mluOpSgetrf2): fix some bugs, reset workspace and update docs
    }
  }
}

static void computeLUError(float *LU, float *res, int m, int n) {
  double error = 0.0;
  double normA = 0.0;
  std::unique_ptr<float[]> L(new float[m * n]());
  std::unique_ptr<float[]> U(new float[n * n]());
  // printf("111\n");

  assign_lower_upper(res, L.get(), U.get(), m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double temp = 0;
      for (int k = 0; k <= i && k < n; k++) {
        temp += L[i * n + k] * U[k * n + j];
      }
      LU[i * n + j] = temp;
    }
  }

  return;
}

void cassign_lower_upper(const float *A, float *L, float *U, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j < i) {
        L[(i * n + j) * 2] = A[(i * n + j) * 2];
        L[(i * n + j) * 2 + 1] = A[(i * n + j) * 2 + 1];
        U[(i * n + j) * 2] = 0.0f;
        U[(i * n + j) * 2 + 1] = 0.0f;
      } else if (j == i) {
        L[(i * n + j) * 2] = 1.0f;
        L[(i * n + j) * 2 + 1] = 0.0f;
        U[(i * n + j) * 2] = A[(i * n + j) * 2];
        U[(i * n + j) * 2 + 1] = A[(i * n + j) * 2 + 1];
      } else {
        L[(i * n + j) * 2] = 0.0f;
        L[(i * n + j) * 2 + 1] = 0.0f;
        U[(i * n + j) * 2] = A[(i * n + j) * 2];
        U[(i * n + j) * 2 + 1] = A[(i * n + j) * 2 + 1];
      }
    }
<<<<<<< HEAD

<<<<<<< HEAD

=======
>>>>>>> fix workspace problem and some bugs
    // Function to swap two rows of a matrix
    void swap_rows(float *matrix, int row1, int row2, int cols)
    {
        for (int i = 0; i < cols; i++)
        {
            float temp = matrix[row1 * cols + i];
            matrix[row1 * cols + i] = matrix[row2 * cols + i];
            matrix[row2 * cols + i] = temp;
=======
  }
}

void ccomputeLUError(float *LU, const float *res, int m, int n) {
  std::unique_ptr<float[]> L(new float[m * n * 2]());
  std::unique_ptr<float[]> U(new float[m * n * 2]());

  cassign_lower_upper(res, L.get(), U.get(), m, n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float temp[2] = {0.0f, 0.0f};
      for (int k = 0; k < n; ++k) {
        if (k <= i) {
          float product[2];
          complexMultiply(&L[(i * n + k) * 2], &U[(k * n + j) * 2], product);
          complexAdd(temp, product, temp);
>>>>>>> [Fix](mluOpSgetrf2): fix some bugs, reset workspace and update docs
        }
      }
      LU[(i * n + j) * 2] = temp[0];
      LU[(i * n + j) * 2 + 1] = temp[1];
    }
  }
}

// Function to swap two rows of a matrix
void swap_rows(float *matrix, int row1, int row2, int cols) {
  for (int i = 0; i < cols; i++) {
    float temp = matrix[row1 * cols + i];
    matrix[row1 * cols + i] = matrix[row2 * cols + i];
    matrix[row2 * cols + i] = temp;
  }
}

// Function to apply row swaps to matrix A using ipiv
void apply_row_swaps(float *A, int m, int n, int *ipiv) {
  for (int i = 0; i < m; i++) {
    while (ipiv[i] != i) {
      swap_rows(A, i, ipiv[i], n);
      int temp = ipiv[ipiv[i]];
      ipiv[ipiv[i]] = ipiv[i];
      ipiv[i] = temp;
    }
  }
}

void Sgetrf2Executor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 1,
              "abs tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "abs tensor output number is wrong.");
}

void Sgetrf2Executor::compute() {
  VLOG(4) << "AbsExecutor compute ";

  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;

  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpAbs()";
  auto dev_a = data_vector_[0].host_ptr;
  auto dev_b = data_vector_[1].host_ptr;

  auto count = parser_->input(0)->shape_count;
  int row, col, batch = 1;
  if (tensor_desc_[0].tensor->dim == 2) {
    row = tensor_desc_[0].tensor->dims[0];
    col = tensor_desc_[0].tensor->dims[1];
  } else if (tensor_desc_[0].tensor->dim == 3) {
    batch = tensor_desc_[0].tensor->dims[0];
    row = tensor_desc_[0].tensor->dims[1];
    col = tensor_desc_[0].tensor->dims[2];
  } else if (tensor_desc_[0].tensor->dim == 4) {
    batch = tensor_desc_[0].tensor->dims[0] * tensor_desc_[0].tensor->dims[1];
    row = tensor_desc_[0].tensor->dims[2];
    col = tensor_desc_[0].tensor->dims[3];
  }
  std::unique_ptr<int[]> ipiv0(new int[batch * row]);
  int *ipiv = ipiv0.get();
  int info;
  int mode = 0;
  mode = parser_->getProtoNode()->sgetrf2_param().mode();
  
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < row; i++) ipiv[i + b * row] = i;
  }

  int m = row;
  int n = col;
  mluOpDataType_t dtype = tensor_x->dtype;

  size_t workspace_size = 0;
  void *workspace;
  MLUOP_CHECK(mluOpGetLUWorkspaceSize(handle_, tensor_x, &workspace_size));

  if (workspace_size > 0) {
    workspace = mlu_runtime_.allocate(workspace_size);
  }
  printf("workspace malloced %lu %ld\n", workspace_size, sizeof(size_t));
  interface_timer_.start();
  MLUOP_CHECK(mluOpSgetrf2(handle_, tensor_x, dev_x, tensor_y, dev_y, workspace,
                           ipiv, &info, mode));
  interface_timer_.stop();

  mlu_runtime_.deallocate(workspace);

  float *typed_dev_x = static_cast<float *>(dev_x);
  float *typed_dev_y = static_cast<float *>(dev_y);
  if (dtype == MLUOP_DTYPE_FLOAT) {
    for (int offset = 0, b = 0; offset < count; offset += m * n, b++) {
      std::unique_ptr<float[]> hA2(new float[m * n]);
      std::unique_ptr<float[]> hA(new float[m * n]);
      std::unique_ptr<float[]> LU(new float[m * n]());
      float *hA_raw = hA.get();
      float *LU_ = LU.get();
      cnrtMemcpy(hA_raw, (void *)(typed_dev_y + offset), m * n * sizeof(float),
                 cnrtMemcpyDevToHost);
      computeLUError(LU_, hA_raw, m, n);
      cnrtMemcpy(hA2.get(), (void *)(typed_dev_x + offset),
                 m * n * sizeof(float), cnrtMemcpyDevToHost);
      if (mode == 1) {
        apply_row_swaps(LU_, m, n, ipiv + b * m);
      }
      cnrtMemcpy((void *)(typed_dev_y + offset), LU_, m * n * sizeof(float),
                 cnrtMemcpyHostToDev);
    }
<<<<<<< HEAD

    void Sgetrf2Executor::compute()
    {
        VLOG(4) << "AbsExecutor compute ";

        auto tensor_x = tensor_desc_[0].tensor;
        auto tensor_y = tensor_desc_[1].tensor;

        auto dev_x = data_vector_[0].device_ptr;
        auto dev_y = data_vector_[1].device_ptr;
        VLOG(4) << "call mluOpAbs()";
        auto dev_a = data_vector_[0].host_ptr;
        auto dev_b = data_vector_[1].host_ptr;

        auto count = parser_->input(0)->shape_count;
        int row, col, batch = 1;
        if (tensor_desc_[0].tensor->dim == 2)
        {
            row = tensor_desc_[0].tensor->dims[0];
            col = tensor_desc_[0].tensor->dims[1];
        }
        else if (tensor_desc_[0].tensor->dim == 3)
        {
            batch = tensor_desc_[0].tensor->dims[0];
            row = tensor_desc_[0].tensor->dims[1];
            col = tensor_desc_[0].tensor->dims[2];
        }
        else if (tensor_desc_[0].tensor->dim == 4)
        {
            batch = tensor_desc_[0].tensor->dims[0] * tensor_desc_[0].tensor->dims[1];
            row = tensor_desc_[0].tensor->dims[2];
            col = tensor_desc_[0].tensor->dims[3];
        }
        std::unique_ptr<int[]> ipiv0(new int[batch * row]);
        int *ipiv = ipiv0.get();
        int info;
        int mode = 1;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < row; i++)
                ipiv[i + b * row] = i;
        }

        int m = row;
        int n = col;
        mluOpDataType_t dtype = tensor_x->dtype;

<<<<<<< HEAD
        interface_timer_.start();
        MLUOP_CHECK(mluOpSgetrf2(handle_, tensor_x, dev_x, tensor_y, dev_y, ipiv, &info, mode));
        interface_timer_.stop();

=======
        int workspace_size = 0;
        void *workspace = NULL;
        MLUOP_CHECK(mluOpGetLUWorkspace(handle_, tensor_x, &workspace_size, &workspace));
        interface_timer_.start();
        MLUOP_CHECK(mluOpSgetrf2(handle_, tensor_x, dev_x, tensor_y, dev_y, workspace, ipiv, &info, mode));
        interface_timer_.stop();

        MLUOP_CHECK(mluOpFreeLUWorkspace(&workspace));

>>>>>>> fix workspace problem and some bugs
        float *typed_dev_x = static_cast<float *>(dev_x);
        float *typed_dev_y = static_cast<float *>(dev_y);
        if (dtype == MLUOP_DTYPE_FLOAT)
        {
            for (int offset = 0, b = 0; offset < count; offset += m * n, b++)
            {
                std::unique_ptr<float[]> hA2(new float[m * n]);
                std::unique_ptr<float[]> hA(new float[m * n]);
                std::unique_ptr<float[]> LU(new float[m * n]());
                float *hA_raw = hA.get();
                float *LU_ = LU.get();
                cnrtMemcpy(hA_raw, (void *)(typed_dev_y + offset), m * n * sizeof(float), cnrtMemcpyDevToHost);
<<<<<<< HEAD
                computeLUError(LU_, hA_raw, m, n); 
                cnrtMemcpy(hA2.get(), (void *)(typed_dev_x + offset), m * n * sizeof(float), cnrtMemcpyDevToHost);
                if (mode == 1)
                {
                    apply_row_swaps(LU_, m, n, ipiv + b * m); 
   
=======
                computeLUError(LU_, hA_raw, m, n);
                cnrtMemcpy(hA2.get(), (void *)(typed_dev_x + offset), m * n * sizeof(float), cnrtMemcpyDevToHost);
                if (mode == 1)
                {
                    apply_row_swaps(LU_, m, n, ipiv + b * m);
>>>>>>> fix workspace problem and some bugs
                }
                cnrtMemcpy((void *)(typed_dev_y + offset), LU_, m * n * sizeof(float), cnrtMemcpyHostToDev);
            }
        }
        else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
        {
            for (int offset = 0, b = 0; offset < batch * m * n * 2; offset += m * n * 2, b++)
            {
                std::unique_ptr<float[]> hA2(new float[m * n * 2]);
                std::unique_ptr<float[]> hA(new float[m * n * 2]);
                std::unique_ptr<float[]> LU(new float[m * n * 2]());
                float *hA_raw = hA.get();
                float *LU_ = LU.get();
                cnrtMemcpy(hA_raw, (void *)(typed_dev_y + offset), m * n * 2 * sizeof(float), cnrtMemcpyDevToHost);
                ccomputeLUError(LU_, hA_raw, m, n);
                cnrtMemcpy(hA2.get(), (void *)(typed_dev_x + offset), m * n * 2 * sizeof(float), cnrtMemcpyDevToHost);
                if (mode == 1)
                {
                    apply_row_swaps(LU_, m, 2 * n, ipiv + b * m); //
                }
                cnrtMemcpy((void *)(typed_dev_y + offset), LU_, m * n * 2 * sizeof(float), cnrtMemcpyHostToDev);
            }
        }
    }

    void Sgetrf2Executor::cpuCompute()
    {
        auto count = parser_->input(0)->shape_count;
        int row, col, batch = 1;
        if (tensor_desc_[0].tensor->dim == 2)
        {
            row = tensor_desc_[0].tensor->dims[0];
            col = tensor_desc_[0].tensor->dims[1];
        }
        else if (tensor_desc_[0].tensor->dim == 3)
        {
            batch = tensor_desc_[0].tensor->dims[0];
            row = tensor_desc_[0].tensor->dims[1];
            col = tensor_desc_[0].tensor->dims[2];
        }
        else if (tensor_desc_[0].tensor->dim == 4)
        {
            batch = tensor_desc_[0].tensor->dims[0] * tensor_desc_[0].tensor->dims[1];
            row = tensor_desc_[0].tensor->dims[2];
            col = tensor_desc_[0].tensor->dims[3];
        }

        int m = row;
        int n = col;
<<<<<<< HEAD
       
=======

>>>>>>> fix workspace problem and some bugs
        if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_FLOAT)
        {
            for (int i = 0; i < count; ++i)
            {
                cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
            }
        }
        else if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
        {
            for (int i = 0; i < m * n * 2 * batch; i++)
            {
                cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
            }
        }
=======
  } else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    for (int offset = 0, b = 0; offset < batch * m * n * 2;
         offset += m * n * 2, b++) {
      std::unique_ptr<float[]> hA2(new float[m * n * 2]);
      std::unique_ptr<float[]> hA(new float[m * n * 2]);
      std::unique_ptr<float[]> LU(new float[m * n * 2]());
      float *hA_raw = hA.get();
      float *LU_ = LU.get();
      cnrtMemcpy(hA_raw, (void *)(typed_dev_y + offset),
                 m * n * 2 * sizeof(float), cnrtMemcpyDevToHost);
      ccomputeLUError(LU_, hA_raw, m, n);
      cnrtMemcpy(hA2.get(), (void *)(typed_dev_x + offset),
                 m * n * 2 * sizeof(float), cnrtMemcpyDevToHost);
      if (mode == 1) {
        apply_row_swaps(LU_, m, 2 * n, ipiv + b * m);  //
      }
      cnrtMemcpy((void *)(typed_dev_y + offset), LU_, m * n * 2 * sizeof(float),
                 cnrtMemcpyHostToDev);
    }
  }
}

void Sgetrf2Executor::cpuCompute() {
  auto count = parser_->input(0)->shape_count;
  int row, col, batch = 1;
  if (tensor_desc_[0].tensor->dim == 2) {
    row = tensor_desc_[0].tensor->dims[0];
    col = tensor_desc_[0].tensor->dims[1];
  } else if (tensor_desc_[0].tensor->dim == 3) {
    batch = tensor_desc_[0].tensor->dims[0];
    row = tensor_desc_[0].tensor->dims[1];
    col = tensor_desc_[0].tensor->dims[2];
  } else if (tensor_desc_[0].tensor->dim == 4) {
    batch = tensor_desc_[0].tensor->dims[0] * tensor_desc_[0].tensor->dims[1];
    row = tensor_desc_[0].tensor->dims[2];
    col = tensor_desc_[0].tensor->dims[3];
  }

  int m = row;
  int n = col;

  if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_FLOAT) {
    for (int i = 0; i < count; ++i) {
      cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
>>>>>>> [Fix](mluOpSgetrf2): fix some bugs, reset workspace and update docs
    }
  } else if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    for (int i = 0; i < m * n * 2 * batch; i++) {
      cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
    }
  }
}

int64_t Sgetrf2Executor::getTheoryOps() {
  int row, col, batch = 1;
  if (tensor_desc_[0].tensor->dim == 2) {
    row = tensor_desc_[0].tensor->dims[0];
    col = tensor_desc_[0].tensor->dims[1];
  } else if (tensor_desc_[0].tensor->dim == 3) {
    batch = tensor_desc_[0].tensor->dims[0];
    row = tensor_desc_[0].tensor->dims[1];
    col = tensor_desc_[0].tensor->dims[2];
  } else if (tensor_desc_[0].tensor->dim == 4) {
    batch = tensor_desc_[0].tensor->dims[0] * tensor_desc_[0].tensor->dims[1];
    row = tensor_desc_[0].tensor->dims[2];
    col = tensor_desc_[0].tensor->dims[3];
  }

  int64_t theory_ops = batch * row * col * MIN(row, col);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
