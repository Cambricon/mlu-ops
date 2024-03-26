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
#include <string>
#include "kernels/fft/fft.h"
#include "kernels/fft/rfft/rfft.h"
#include "kernels/fft/irfft/irfft.h"
#include "kernels/fft/c2c_fft/c2c_fft.h"

// May be use a common function is a better choice?
static inline bool supportFloatConv(mluOpHandle_t handle) {
  switch (handle->arch) {
    case MLUOP_MLU370:
      return true;
    default:
      return true;
  }
}

// Calculate whether the optimization strategy can be
// entered(CNFFT_FUNC_STOCKHAM and CNFFT_FUNC_COOLEY_TUKEY). If it can enter,
// select the optimal strategy and calculate corresponding parameters.
mluOpStatus_t selectFFTStrategy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                const std::string make_plan_api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  fft_plan->fft_strategy = CNFFT_FUNC_MATMUL;
  // The basic conditions for entering the optimization.
  if (fft_plan->n[0] > 4096) {
    bool find_stockham = 0;
    // CNFFT_FUNC_STOCKHAM optimizaion currently has more retrictions as
    // follows:
    if (handle->arch >= 300 &&
        (fft_plan->execution_dtype == MLUOP_DTYPE_HALF ||
         fft_plan->execution_dtype == MLUOP_DTYPE_FLOAT)) {
      find_stockham = true;
    }
    // strategy_status: 0 means select MLUOP_FUNC_STOCKHAM, 1 means selelct
    // COOLEY_TUKEY,
    //                  -1 means still select CNFFT_FUNC_MATMUL.
    int strategy_status =
        findFFTOptLimit(handle, fft_plan->n[0], fft_plan->m, fft_plan->L,
                        fft_plan->s, fft_plan->L_sub, find_stockham);
    if (strategy_status == 1) {
      fft_plan->fft_strategy = CNFFT_FUNC_COOLEY_TUKEY;
    } else if (strategy_status == 0) {
      fft_plan->fft_strategy = CNFFT_FUNC_STOCKHAM;
    }
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreateFFTPlan(mluOpFFTPlan_t *fft_plan) {
  mluOpFFTStruct *ts = new (std::nothrow) mluOpFFTStruct();
  if (ts == nullptr) {
    LOG(ERROR) << "[mluOpCreateFFTPlan]: alloc failed";
    return MLUOP_STATUS_ALLOC_FAILED;
  }
  *fft_plan = ts;
  return MLUOP_STATUS_SUCCESS;
}

/**
 * This function
 * 1. receives parameters from user;
 * 2. checks the validity of the parameters;
 * 3. picks up the supported circumstances;
 * 4. make decisions which strategy should use.
 */
mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanMany(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n, size_t *reservespace_size,
    size_t *workspace_size) {
  // bad param check
  const std::string make_plan_api = "[mluOpMakeFFTPlanMany]";
  // plan NULL check
  PARAM_CHECK_NE(make_plan_api, handle, NULL);
  if (fft_plan == NULL) {
    LOG(ERROR) << make_plan_api + ": plan is not allocated.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  PARAM_CHECK_NE(make_plan_api, input_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, output_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, n, NULL);
  PARAM_CHECK_NE(make_plan_api, reservespace_size, NULL);
  PARAM_CHECK_NE(make_plan_api, workspace_size, NULL);

  // plan rank can only be 1, 2, 3
  if (rank < 1 || rank > FFT_DIM_MAX) {
    LOG(ERROR) << make_plan_api + ": invalid rank, should be 1, 2 or 3. Now is "
               << rank << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (auto i = 0; i < rank; i++) {
    if (n[i] <= 0) {
      LOG(ERROR)
          << make_plan_api +
                 ": fourier transform length should be greater than 0. Now n["
          << i << "] is " << n[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  fft_plan->rank = rank;
  for (auto i = 0; i < rank; i++) {
    fft_plan->n[i] = n[i];
  }

  // dimension check
  fft_plan->idim = input_desc->dim;
  fft_plan->odim = output_desc->dim;
  fft_plan->inum = mluOpGetTensorElementNum(input_desc);
  fft_plan->onum = mluOpGetTensorElementNum(output_desc);
  PARAM_CHECK_GT(make_plan_api, input_desc->dim, 0);
  PARAM_CHECK_EQ(make_plan_api, fft_plan->idim, fft_plan->odim,
                 ": input and output dimension mismatch.");

  if (!(fft_plan->idim == rank || fft_plan->idim == rank + 1)) {
    LOG(ERROR) << make_plan_api << ": invalid input dimension, should be "
               << rank << " or " << rank + 1 << ". Now is " << fft_plan->idim
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // batch check
  if (fft_plan->idim == rank) {
    fft_plan->batch = 1;
  } else {  // idim == rank + 1
    fft_plan->batch = input_desc->dims[0];
    PARAM_CHECK_EQ(make_plan_api, fft_plan->batch, output_desc->dims[0],
                   ": batch size mismatch.");
  }

  // The FFT Struct is designed after cufftXtMakePlanMany.
  // An element of coordinates [z, y, x] in signal number b in the batch will
  // be associated with the following addresses in the memory
  // 1-D:
  //   input[b * idist + x * istride]
  //   output[b * odist + x * ostride]
  // 2-D:
  //   input[b * idist + (x * inembed[1] + y) * istride]
  //   output[b * odist + (x * onembed[1] + y) * istride]
  // 3-D:
  //   input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
  //   output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
  // Thus, cufft and fftw advanced data layout is a subset of mluOp advanced
  // data layout with tensor dim strides. 2-D and 3-D should pay attention.
  // stride check, if an in-place fft is adopted check, `istride` should be
  // equal to `ostride`.
  fft_plan->istride = input_desc->strides[fft_plan->idim - 1];
  fft_plan->ostride = output_desc->strides[fft_plan->odim - 1];

  PARAM_CHECK_GE(make_plan_api, fft_plan->istride, 0,
                 ": input stride should be greater than or equal to 0.");
  PARAM_CHECK_GE(make_plan_api, fft_plan->ostride, 0,
                 ": output stride should be greater than or equal to 0.");

  for (auto i = 0; i < fft_plan->rank; i++) {
    fft_plan->inembed[i] = input_desc->dims[fft_plan->idim - rank + i];
    fft_plan->onembed[i] = output_desc->dims[fft_plan->odim - rank + i];
  }
  if (fft_plan->idim == rank + 1) {
    fft_plan->idist = input_desc->strides[0];
    fft_plan->odist = output_desc->strides[0];
  } else {  // batch == 1
    fft_plan->idist = mluOpGetTensorElementNum(input_desc) / fft_plan->batch;
    fft_plan->odist = mluOpGetTensorElementNum(output_desc) / fft_plan->batch;
  }
  fft_plan->is_input_contiguous = !mluop::ifNeedTensorStrideProcess(input_desc);
  fft_plan->is_output_contiguous =
      !mluop::ifNeedTensorStrideProcess(output_desc);

  // dtype check
  mluOpDataType_t input_dtype = input_desc->dtype;
  mluOpDataType_t output_dtype = output_desc->dtype;
  const mluOpDataType_t f_c_dtype = MLUOP_DTYPE_COMPLEX_FLOAT;
  const mluOpDataType_t f_r_dtype = MLUOP_DTYPE_FLOAT;
  const mluOpDataType_t hf_c_dtype = MLUOP_DTYPE_COMPLEX_HALF;
  const mluOpDataType_t hf_r_dtype = MLUOP_DTYPE_HALF;
  if (input_dtype == hf_r_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2HALF;
  } else if (input_dtype == f_r_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2FLOAT;
  } else {
    LOG(ERROR) << make_plan_api
               << ": invalid data type combination. Now input data type is "
               << mluOpGetNameOfDataType(input_dtype)
               << ", and output data type is "
               << mluOpGetNameOfDataType(output_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  fft_plan->input_dtype = input_desc->dtype;
  fft_plan->output_dtype = output_desc->dtype;
  fft_plan->execution_dtype = input_desc->onchip_dtype;

  VLOG(5) << "input data type: "
          << mluOpGetNameOfDataType(fft_plan->input_dtype);
  VLOG(5) << "output data type: "
          << mluOpGetNameOfDataType(fft_plan->output_dtype);
  VLOG(5) << "execution data type: "
          << mluOpGetNameOfDataType(fft_plan->execution_dtype);

  // fft length check
  for (auto i = 0; i < fft_plan->rank - 1; i++) {  // except the last dim
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->inembed[i],
        ": the signal lengths of fft and input mismatch in dimention ", i, ".");
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->onembed[i],
        ": the signal lengths of fft and output mismatch in dimension ", i,
        ".");
  }
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1] / 2 + 1,
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in R2C.");
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2C.");
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2R.");
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid fft type.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  mluOpDataType_t execution_dtype = fft_plan->execution_dtype;
  switch (fft_plan->fft_type) {
    // half
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF: {
      if (supportFloatConv(handle)) {
        if (!(execution_dtype == hf_r_dtype ||
              execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      } else {
        if (!(execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    // float
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (supportFloatConv(handle)) {
        if (execution_dtype != f_r_dtype) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid execution dtype.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // unsupported param
  if (fft_plan->rank != 1) {
    LOG(ERROR)
        << make_plan_api
        << ": 2-dimensional and 3-dimensional FFT are not supported currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (fft_plan->fft_type == CNFFT_HALF2COMPLEX_HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2COMPLEX_HALF) {
    if ((n[0] & (n[0] - 1)) != 0) {
      LOG(ERROR) << make_plan_api
                 << ": the signal lengths of half-precision FFT are"
                 << " restriced to power of two only, but now is " << n[0]
                 << ".";
      return MLUOP_STATUS_NOT_SUPPORTED;
    }
  }

  // create input and output descriptor for gen_case
  // because mluOpExecFFT don't have input and output descriptor
  mluOpTensorDescriptor_t fft_input_desc, fft_output_desc;
  CHECK_RETURN(make_plan_api, mluOpCreateTensorDescriptor(&fft_input_desc));
  CHECK_RETURN(make_plan_api, mluOpCreateTensorDescriptor(&fft_output_desc));
  CHECK_RETURN(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_input_desc, input_desc->layout, input_desc->dtype,
                     input_desc->dim, input_desc->dims,
                     input_desc->strides));
  CHECK_RETURN(make_plan_api, mluOpSetTensorDescriptorOnchipDataType(
                                    fft_input_desc, input_desc->onchip_dtype));
  CHECK_RETURN(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_output_desc, output_desc->layout, output_desc->dtype,
                     output_desc->dim, output_desc->dims,
                     output_desc->strides));
  fft_plan->input_desc = fft_input_desc;
  fft_plan->output_desc = fft_output_desc;

  /*
   * decision part
   */
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make RFFT1d Policy";
        status = makeRFFT1dPolicy(handle, fft_plan);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make IRFFT1d Policy";
        status = makeIRFFT1dPolicy(handle, fft_plan);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make FFT1d Policy";
        status = makeFFT1dPolicy(handle, fft_plan);
      }
    }; break;
  }
  if (status != MLUOP_STATUS_SUCCESS) {
    return status;
  }

  *reservespace_size = fft_plan->reservespace_size;
  *workspace_size = fft_plan->workspace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroyFFTPlan(mluOpFFTPlan_t fft_plan) {
  const std::string destroy_api = "[mluOpDestroyFFTPlan]";
  PARAM_CHECK_NE("[mluOpDestroyFFTPlan]", fft_plan, NULL);
  if (fft_plan->input_desc != NULL) {
    CHECK_RETURN(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->input_desc));
  }
  if (fft_plan->output_desc != NULL) {
    CHECK_RETURN(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->output_desc));
  }
  delete fft_plan;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetFFTReserveArea(mluOpHandle_t handle,
                                                   mluOpFFTPlan_t fft_plan,
                                                   void *reservespace) {
  const std::string api = "[mluOpSetReserveArea]";
  PARAM_CHECK_NE(api, handle, NULL);
  PARAM_CHECK_NE(api, fft_plan, NULL);
  PARAM_CHECK_NE(api, reservespace, NULL);
  fft_plan->reservespace_addr = reservespace;
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        status = setRFFT1dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        status = setFFT1dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (fft_plan->rank == 1) {
        status = setIRFFT1dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpExecFFT(
    mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan, const void *input,
    const float scale_factor, void *workspace, void *output, int direction) {
  const std::string exec_api = "[mluOpExecFFT]";
  PARAM_CHECK_NE(exec_api, handle, NULL);
  PARAM_CHECK_NE(exec_api, fft_plan, NULL);
  VLOG(5) << "input contiguous ? " << fft_plan->is_input_contiguous;
  VLOG(5) << "output contiguous ? " << fft_plan->is_output_contiguous;

  if (fft_plan->batch == 0) {
    VLOG(5) << "[mluOpExecFFT] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }
  // generate mluOpFFTExec prototxt start!
  {
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->input_desc),
                     LARGE_TENSOR_NUM, "");
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->output_desc),
                     LARGE_TENSOR_NUM, "");
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("fft", "FFT");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, fft_plan->input_desc, 1, 0);
    GEN_CASE_DATA(false, "output", output, fft_plan->output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "fft", "rank", fft_plan->rank);
    GEN_CASE_OP_PARAM_ARRAY(1, "fft", "n", fft_plan->n, fft_plan->rank);
    GEN_CASE_OP_PARAM_SINGLE(1, "fft", "direction", direction);
    GEN_CASE_OP_PARAM_SINGLE(2, "fft", "scale_factor", scale_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  if (fft_plan->workspace_size > 0) {
    PARAM_CHECK_NE(exec_api, workspace, NULL);
  }
  if (fft_plan->inum > 0) {
    PARAM_CHECK_NE(exec_api, input, NULL);
  }
  PARAM_CHECK_NE(exec_api, output, NULL);
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  bool is_in_place = (input == output);
  VLOG(5) << exec_api << ": in place ? " << is_in_place;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < (fft_plan->odist * 2)) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "real-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execRFFT1d(handle, fft_plan, input, scale_factor, workspace,
                            output);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execFFT1d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (((fft_plan->idist * 2) < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-real fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execIRFFT1d(handle, fft_plan, input, scale_factor, workspace,
                             output);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }

  GEN_CASE_END();
  return status;
}
