#ifndef KERNEL_LGAMMA_LGAMMA_H
#define KERNEL_LGAMMA_LGAMMA_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API
KernelLgamma(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, mluOpDataType_t d_type,
            const void *x, void *y, const int num);

#endif