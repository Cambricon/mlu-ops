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
#ifndef CORE_TYPE_H_
#define CORE_TYPE_H_

#include "core/logging.h"
#include "core/mlu_op_core.h"
#include <string>

template <typename T>
static mluOpStatus_t
    getLowAndHighValueFrom64Bits(T value, uint32_t *high, int32_t *low) {
  if (sizeof(T) != sizeof(int64_t)) {
    VLOG(5) << "getLowAndHighValueFrom64Bits() only supports 64 bits data type";
    return MLUOP_STATUS_INTERNAL_ERROR;
  }
  uint64_t temp = *(uint64_t *)&value;
  // get the high 32bit value
  *high = temp >> 32;
  // get the low 32bit value
  *low = temp;
  return MLUOP_STATUS_SUCCESS;
}

size_t getSizeOfDataType(mluOpDataType_t dtype);

std::string getNameOfDataType(mluOpDataType_t dtype);

std::string getNameOfTensorLayout(mluOpTensorLayout_t layout);
#endif // CORE_TYPE_H_
