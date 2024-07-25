#include "cholesky.h"
// calculates the required workspace size for performing the Cholesky
// decomposition on a given matrix or batch of matrices.
mluOpStatus_t MLUOP_WIN_API mluOpGetCholeskyWorkspace(
    mluOpTensorDescriptor_t input_desc, size_t* size, float** workspace) {
  PARAM_CHECK("mluOpCholesky", input_desc != NULL);

  PARAM_CHECK("mluOpCholesky", input_desc->dim == 2 || input_desc->dim == 3);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);

  if (input_desc->dim == 3) {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
  }

  mluOpDataType_t dtype = input_desc->dtype;
  PARAM_CHECK("mluOpCholesky",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

  unsigned long type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));
  long int size_a = 0, lda = 0, size_c = 0, ldc = 0;
  long int batch_size = 1;
  int dim = input_desc->dim;
  if (dim == 2) {
    size_a = input_desc->dims[0];
  } else if (dim == 3) {
    batch_size = input_desc->dims[0];
    size_a = input_desc->dims[1];
  }

  if (dtype == MLUOP_DTYPE_FLOAT) {
    *size = size_a * size_a * sizeof(float) * batch_size * 3;
  } else {
    *size = size_a * size_a * sizeof(float) * 2 * batch_size * 3;
  }
  printf("workspace size:%ul\n", (int)(*size));
  if (*size > 0) {
    CHECK_RETURN("mluOpCholesky", workspace_malloc(*size, workspace));
  }
  return MLUOP_STATUS_SUCCESS;
}

// releases the allocated workspace memory used for Cholesky decomposition
// calculations. It ensures that the workspace pointer is not only valid but
// also points to allocated memory before attempting to free it.
mluOpStatus_t MLUOP_WIN_API mluOpFreeCholeskyWorkspace(float** workspace) {
  PARAM_CHECK("mluOpCholesky", workspace != NULL);
  if (*workspace != NULL) {
    CHECK_RETURN("mluOpCholesky", workspace_free(workspace));
    *workspace = NULL;
  }
  return MLUOP_STATUS_SUCCESS;
}

// performs the necessary operations to compute matrix transformations,
// potentially involving Cholesky decomposition or matrix transposition,
// depending on the input parameters.
mluOpStatus_t MLUOP_WIN_API
calculate_body(mluOpHandle_t handle, int batch_size,
               const mluOpTensorDescriptor_t input_desc, float* d_input,
               const mluOpTensorDescriptor_t output_desc, float* d_output,
               bool upper, float* workspace) {
  mluOpDataType_t dtype = input_desc->dtype;

  int recnb = REC_NB;
  int gbstep = 0;
  int dim = input_desc->dim;
  bool is_row_major = (input_desc->strides)[dim - 1] == 1;

  unsigned long type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));
  int size_a = 0, lda = 0, size_c = 0, ldc = 0;
  if (dim == 2) {
    size_a = input_desc->dims[0];
    lda = input_desc->dims[1];
    size_c = output_desc->dims[0];
    ldc = output_desc->dims[1];
  } else if (dim == 3) {
    size_a = input_desc->dims[1];
    lda = input_desc->dims[2];
    size_c = output_desc->dims[1];
    ldc = output_desc->dims[2];
  }

  PARAM_CHECK("mluOpCholesky", lda >= size_a);
  PARAM_CHECK("mluOpCholesky", ldc >= size_c);

  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  int jb;
  const float s_one = 1.0;
  const float s_neg_one = -1.0;

  if (dtype == MLUOP_DTYPE_FLOAT) {
    if (upper == true) {
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, d_input, d_output,
                             handle, dtype, workspace));
    } else {
      CNRT_CHECK(
          cnrtMemcpy(d_output, d_input,
                     type_size * size_a * lda * ((unsigned long)batch_size),
                     CNRT_MEM_TRANS_DIR_DEV2DEV));
    }
  } else {
    CHECK_RETURN("mluOpCholesky",
                 transpose(batch_size, size_a * size_a, 2, d_input, d_output,
                           handle, MLUOP_DTYPE_FLOAT, workspace));
  }

  cnrtQueueSync(queue);
  int stride = size_a * lda;

  if (dtype == MLUOP_DTYPE_FLOAT) {
    int row = is_row_major ? lda : size_a;
    int nb = NB;
    set_half_zero(batch_size, stride, d_output, lda, lda, handle);
    cnrtQueueSync(queue);
    for (int j = 0; j < row; j += nb) {
      jb = std::min(nb, row - j);
      CHECK_RETURN("mluOpCholesky",
                   ssyrk(batch_size, stride, false, is_row_major, jb, j,
                         OFFSET_ROW(d_output, j, 0), lda,
                         OFFSET_ROW(d_output, j, j), lda, handle, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   mlu_spotrf_rectile(batch_size, stride, is_row_major, false,
                                      jb, recnb, OFFSET_ROW(d_output, j, j),
                                      lda, j, handle, workspace));
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            sgemm(batch_size, !is_row_major, is_row_major, row - j - jb, jb, j,
                  -1.0f, 1.0f, OFFSET_ROW(d_output, j + jb, 0), lda, stride,
                  OFFSET_ROW(d_output, j, 0), lda, stride,
                  OFFSET_ROW(d_output, j + jb, j), lda, stride, handle,
                  workspace));
        cnrtQueueSync(queue);
      }
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            strsm(batch_size, stride, false, is_row_major, jb, row - j - jb,
                  OFFSET_ROW(d_output, j, j), lda,
                  OFFSET_ROW(d_output, j + jb, j), lda, handle, workspace));
        cnrtQueueSync(queue);
      }
    }

    if (upper) {
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, d_output, workspace,
                             handle, dtype, workspace));
      cnrtQueueSync(queue);
      CNRT_CHECK(
          cnrtMemcpy(d_output, workspace,
                     type_size * size_a * lda * ((unsigned long)batch_size),
                     CNRT_MEM_TRANS_DIR_DEV2DEV));
    }
  } else {
    recnb = CREC_NB;
    int nb = CNB;
    int row = lda;
    float* r_start = d_output;
    float* i_start = d_output + size_a * lda;
    stride *= 2;

    set_half_zero(batch_size, stride, r_start, lda, lda, handle);
    set_half_zero(batch_size, stride, i_start, lda, lda, handle);
    cnrtQueueSync(queue);

    for (int j = 0; j < row; j += nb) {
      jb = std::min(nb, row - j);
      CHECK_RETURN("mluOpCholesky",
                   cherk(batch_size, stride, jb, j, r_start + j * lda,
                         i_start + j * lda, lda, r_start + j * lda + j,
                         i_start + j * lda + j, lda, handle, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   mlu_cpotrf_rectile(
                       batch_size, stride, jb, recnb, r_start + j * lda + j,
                       i_start + j * lda + j, lda, handle, workspace));
      cnrtQueueSync(queue);
      if (j + jb < row) {
        CHECK_RETURN("mluOpCholesky",
                     cgemm(batch_size, false, true, row - j - jb, jb, j, -1.0f,
                           1.0f, OFFSET_ROW(r_start, j + jb, 0),
                           OFFSET_ROW(i_start, j + jb, 0), lda, stride,
                           OFFSET_ROW(r_start, j, 0), OFFSET_ROW(i_start, j, 0),
                           lda, stride, OFFSET_ROW(r_start, j + jb, j),
                           OFFSET_ROW(i_start, j + jb, j), lda, stride, handle,
                           workspace));

        cnrtQueueSync(queue);
      }
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            ctrsm(batch_size, stride, jb, row - j - jb,
                  OFFSET_ROW(r_start, j, j), OFFSET_ROW(i_start, j, j), lda,
                  OFFSET_ROW(r_start, j + jb, j),
                  OFFSET_ROW(i_start, j + jb, j), lda, handle, workspace));
        cnrtQueueSync(queue);
      }
    }

    CHECK_RETURN("mluOpCholesky",
                 transpose(batch_size, 2, size_a * size_a, d_output, workspace,
                           handle, MLUOP_DTYPE_FLOAT, workspace));
    cnrtQueueSync(queue);

    if (upper) {
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, workspace, d_output,
                             handle, dtype, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky", conj_complex(batch_size, size_a, size_a,
                                                 d_output, d_output, handle));
      cnrtQueueSync(queue);
    } else {
      if (batch_size > 16) {
        CNRT_CHECK(cnrtMemcpy(d_output, workspace,
                              type_size * size_a * lda * 16,
                              CNRT_MEM_TRANS_DIR_DEV2DEV));
        CNRT_CHECK(cnrtMemcpy(
            d_output + type_size / 4 * size_a * lda * 16,
            workspace + type_size / 4 * size_a * lda * 16,
            type_size * size_a * lda * ((unsigned long)batch_size - 16),
            CNRT_MEM_TRANS_DIR_DEV2DEV));
      } else {
        CNRT_CHECK(
            cnrtMemcpy(d_output, workspace,
                       type_size * size_a * lda * ((unsigned long)batch_size),
                       CNRT_MEM_TRANS_DIR_DEV2DEV));
      }
    }
  }

  cnrtQueueSync(queue);

  return MLUOP_STATUS_SUCCESS;
}

// computes the Cholesky decomposition.
// This function is designed to handle both single and batch processing of
// matrices in either 2D or 3D formats. The function ensures that the input
// matrices are either float or complex float types and performs the
// decomposition either on the upper or lower triangular part of the matrix,
// based on the 'upper' boolean flag.
mluOpStatus_t MLUOP_WIN_API mluOpStatus_t MLUOP_WIN_API
mluOpCholesky(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
              float* d_input, const mluOpTensorDescriptor_t output_desc,
              float* d_output, bool upper, float* workspace) {
  PARAM_CHECK("mluOpCholesky", handle != NULL);
  PARAM_CHECK("mluOpCholesky", input_desc != NULL);
  PARAM_CHECK("mluOpCholesky", d_input != NULL);
  PARAM_CHECK("mluOpCholesky", output_desc != NULL);
  PARAM_CHECK("mluOpCholesky", d_output != NULL);

  PARAM_CHECK("mluOpCholesky", input_desc->dim == 2 || input_desc->dim == 3);
  PARAM_CHECK("mluOpCholesky", output_desc->dim == input_desc->dim);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);
  PARAM_CHECK("mluOpCholesky", output_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", output_desc->dims[1] > 0);

  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  if (input_desc->dim == 3) {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[2] > 0);
  }

  mluOpDataType_t dtype = input_desc->dtype;
  PARAM_CHECK("mluOpCholesky", dtype == output_desc->dtype);
  PARAM_CHECK("mluOpCholesky",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

  int dim = input_desc->dim;
  int size_a = 0, lda = 0, size_c = 0, ldc = 0;

  int batch_size = 1;
  if (dim == 2) {
    size_a = input_desc->dims[0];
    lda = input_desc->dims[1];
    size_c = output_desc->dims[0];
    ldc = output_desc->dims[1];
  } else if (dim == 3) {
    batch_size = input_desc->dims[0];
    size_a = input_desc->dims[1];
    lda = input_desc->dims[2];
    size_c = output_desc->dims[1];
    ldc = output_desc->dims[2];
  }

  unsigned long type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));
  if (type_size == 8 && batch_size > 16 && size_a > 2000) {
    int stride = 2 * size_a * lda;
    calculate_body(handle, 16, input_desc, d_input, output_desc, d_output,
                   upper, workspace);
    cnrtQueueSync(queue);
    calculate_body(handle, ((unsigned long)batch_size) - 16, input_desc,
                   d_input + 16 * stride, output_desc, d_output + 16 * stride,
                   upper, workspace);
  } else {
    calculate_body(handle, batch_size, input_desc, d_input, output_desc,
                   d_output, upper, workspace);
  }

  return MLUOP_STATUS_SUCCESS;
}