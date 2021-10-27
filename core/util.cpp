/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
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
#include <stdexcept>
#include "core/mlu_op_core.h"
#include "core/logging.h"

const char *mluOpGetErrorString(mluOpStatus_t status) {
  CHECK_GE(status, 0);

  switch (status) {
    default: { return (char *)"MLUOP_STATUS_UNKNOWN"; }
    case MLUOP_STATUS_SUCCESS: {
      return (char *)"MLUOP_STATUS_SUCCESS";
    }
    case MLUOP_STATUS_NOT_INITIALIZED: {
      return (char *)"MLUOP_STATUS_NOT_INITIALIZED";
    }
    case MLUOP_STATUS_ALLOC_FAILED: {
      return (char *)"MLUOP_STATUS_ALLOC_FAILED";
    }
    case MLUOP_STATUS_BAD_PARAM: {
      return (char *)"MLUOP_STATUS_BAD_PARAM";
    }
    case MLUOP_STATUS_INTERNAL_ERROR: {
      return (char *)"MLUOP_STATUS_INTERNAL_ERROR";
    }
    case MLUOP_STATUS_ARCH_MISMATCH: {
      return (char *)"MLUOP_STATUS_MISMATCH";
    }
    case MLUOP_STATUS_EXECUTION_FAILED: {
      return (char *)"MLUOP_STATUS_EXECUTION_FAILED";
    }
    case MLUOP_STATUS_NOT_SUPPORTED: {
      return (char *)"MLUOP_STATUS_NOT_SUPPORTED";
    }
    case MLUOP_STATUS_NUMERICAL_OVERFLOW: {
      return (char *)"MLUOP_STATUS_NUMERICAL_OVERFLOW";
    }
  }
}
