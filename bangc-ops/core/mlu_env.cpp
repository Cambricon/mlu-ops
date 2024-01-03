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

#include "core/mlu_env.h"

namespace mluop {

// Get CAMBRICON_TF32_OVERRIDE from env.
// CAMBRICON_TF32_OVERRIDE=0: do not use tf32.
// CAMBRICON_TF32_OVERRIDE=1: use tf32 for fp32 compute.
__attribute__((__unused__)) int cambricon_tf32_override_ =
    mluop::getUintEnvVar("CAMBRICON_TF32_OVERRIDE", 1);

// Get MLUOP_CHECK_DEP_VERSION from env.
// MLUOP_CHECK_DEP_VERSION=0: do not check dependency version in mluOpCreate().
// MLUOP_CHECK_DEP_VERSION=1:  check dependency version in mluOpCreate().
__attribute__((__unused__)) int mluop_check_dep_version_ =
    mluop::getUintEnvVar("MLUOP_CHECK_DEP_VERSION", 1);

namespace mlu_env {
int getCarmbriconTF32Override() { return cambricon_tf32_override_; }
int getCheckDepVersion() { return mluop_check_dep_version_; }
}  // namespace mlu_env
}  // namespace mluop
