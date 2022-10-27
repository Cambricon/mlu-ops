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
#ifndef KERNELS_KERNEL_H_
#define KERNELS_KERNEL_H_

/******************************************************************************
 * Macros for host and device side
 ******************************************************************************/
#define NFU_ALIGN_SIZE 128          // Byte
#define REM_FOR_STACK (128 * 1024)  // 128KB reserved for cncc
#define CEIL_ALIGN(x, align) (((x) + (align)-1) / (align) * (align))
#define FLOOR_ALIGN(x, align) ((x) / (align) * (align))

#if defined(__BANG__)
#include <mlu.h>
#endif  // defined(__BANG__)

/******************************************************************************
 * Macros for device side
 ******************************************************************************/
#define CORE_DIM 4
#if defined(__BANG__)
#define MAX_NRAM_SIZE \
  (__MLU_NRAM_SIZE__ * 1024 - 128 * 1024)  // 128KB reserved for cncc
#define MAX_SRAM_SIZE \
  (__MLU_SRAM_SIZE__ * 1024 - 128 * 1024)  // 128KB reserved for cncc
#define MAX_WRAM_SIZE (__MLU_WRAM_SIZE__ * 1024)

__mlu_func__ void pvLock() {
#if __BANG_ARCH__ == 270
  if (coreId != 0x80) {
    __bang_lock(0, 0);
  }
#endif
}

__mlu_func__ void pvUnlock() {
#if __BANG_ARCH__ == 270
  if (coreId != 0x80) {
    __bang_unlock(0, 0);
  }
#endif
}
#endif  // defined(__BANG__)

#endif  // KERNELS_KERNEL_H_
