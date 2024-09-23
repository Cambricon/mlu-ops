/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_DEVICE_CHECK_H_
#define KERNELS_DEVICE_CHECK_H_

#ifdef NDEBUG
#ifdef __BANG__
#include "bang.h"
__mlu_device__ __mlu_builtin__ __attribute__((noinline)) void __assert_fail(
    const char *__message, const char *__file, unsigned int __line,
    const char *__function);
#endif
#endif

// The variable names on the device side are usually different from the input
// parameters in the API function, so the logging information needs to be
// written as needed. The detailed information of the "__assert_fail"
// instruction can be found in
// "llvm-project/-/blob/master/docs_bang/design_docs/assert/BANGAssert.md"
#define MLU_KERNEL_ASSERT(cond, message)                                  \
  if (!(cond)) {                                                          \
    __assert_fail(message, __FILE__, static_cast<unsigned int>(__LINE__), \
                  __func__);                                              \
  }

#endif  // KERNELS_DEVICE_CHECK_H_
