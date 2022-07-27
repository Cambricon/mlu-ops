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
#ifndef CORE_CONTEXT_H_
#define CORE_CONTEXT_H_

<<<<<<< 8c359e38da44822464dcd513c997e7ed5ce933f0
#include <string>

#include "cn_api.h"
#include "core/logging.h"
#include "mlu_op.h"
=======
#include "cn_api.h"
#include "core/logging.h"
#include "core/mlu_op_core.h"
#include "mlu_op_core.h"
>>>>>>> [Fix](bangc-ops):Add several functional functions

#define CONTEXT_DEVICENAME_BUFFER_SIZE 64
#define CONTEXT_DEVICENAME_LEAST_SIZE 6

/*
Tested version dependency:

| MLUOP version | CNTOOLKIT version | CNRT version | DRIVER version  |
---------------------------------------------------------------------
| MLUOP V1.2    | CNTOOLKIT V1.6    | CNRT V4.9    | DRIVER V4.8     |
| MLUOP V1.1    | CNTOOLKIT V1.5    | CNRT V4.8    | DRIVER V4.7     |
| MLUOP V1.0    | CNTOOLKIT V1.4    | CNRT V4.7    | DRIVER V4.6     |
*/
#define MLUOP_DEP_CNRT_MIN_MAJOR 5
#define MLUOP_DEP_CNRT_MIN_MINOR 0
#define MLUOP_DEP_CNRT_MIN_PATCHLEVEL 0

// Compatible with higher version CNRT by default.
#define MLUOP_DEP_CNRT_MAX_MAJOR 999
#define MLUOP_DEP_CNRT_MAX_MINOR 999
#define MLUOP_DEP_CNRT_MAX_PATCHLEVEL 999

typedef enum {
  MLUOP_UNKNOWN_DEVICE = 0,
  // MLUOP_MLU100 = 100,
  MLUOP_MLU220 = 220,
  MLUOP_MLU270 = 270,
  MLUOP_MLU290 = 290,
  MLUOP_MLU370 = 372,
} mluOpDevType_t;

struct deviceName {
  char name[CONTEXT_DEVICENAME_BUFFER_SIZE];
  mluOpDevType_t type;
};

struct mluOpContext {
  CNdev device;
  cnrtQueue_t queue;
  mluOpDevType_t arch;  // return arch type. e.g. MLUOP_MLU270
  int32_t cluster_num;
  int32_t core_num_per_cluster;
  int32_t nram_size;
  int32_t wram_size;
  int32_t sram_size;
  int32_t capability_cluster_num;
  int32_t capability_job_limit;
  mluOpQuantizeRoundMode_t round_mode;
  mluOpAtomicsMode_t atomics_mode;
};

typedef enum {
  WARNING = 1,
  ERROR   = 2,
} DepCheckLevel;  // related to include/cnlog.h

mluOpStatus_t mluOpCheckDependency(bool need_check_min = true,
                                   bool need_check_max = false,
                                   DepCheckLevel level = WARNING);

#endif  // CORE_CONTEXT_H_
