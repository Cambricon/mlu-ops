#ifndef KERNELS_LOGSPACE_LOGSPACE_H
#define KERNELS_LOGSPACE_LOGSPACE_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API
KernelLogspace(const cnrtDim3_t k_dim, const cnrtFunctionType_t k_type, const cnrtQueue_t queue, const mluOpDataType_t d_type,
                        const float start, const float end, const int steps, const float base, void *res);

#endif