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
#ifndef KERNELS_DEBUG_H_
#define KERNELS_DEBUG_H_

#include <stdio.h>
#include <sys/time.h>

#ifdef NDEBUG

#define MLULOG(fmt, ...)
#define PERF_TIME_BEGIN()
#define PERF_TIME_END()

#else  // NDEBUG

/******************************************************************************
 * Debug Level
 ******************************************************************************/
#define DEBUG_LEVEL_INFO 0
#define DEBUG_LEVEL_WARNING 1
#define DEBUG_LEVEL_ERROR 2
#define DEBUG_LEVEL_FATAL 3

/******************************************************************************
 * Log for BANG
 ******************************************************************************/
#define MLUBLUE "\033[1;34m"
#define MLUGREEN "\033[32m"
#define COLOUREND "\033[0m"
#define MLULOG(fmt, ...)                                \
  do {                                                  \
    __bang_printf("[taskId : %d]", taskId);             \
    __bang_printf(MLUGREEN " [%s:%d]" COLOUREND MLUBLUE \
                           " [MLULOG]:" COLOUREND fmt,  \
                  __FILE__, __LINE__, ##__VA_ARGS__);   \
  } while (0)

/******************************************************************************
 * Hardware Timer for BANG
 ******************************************************************************/
#define PERF_TIME_BEGIN() \
  struct timeval t_start; \
  struct timeval t_end;   \
  gettimeofday(&t_start, NULL);

#define PERF_TIME_END()                                                    \
  do {                                                                     \
    gettimeofday(&t_end, NULL);                                            \
    printf("[taskId : %d]", taskId);                                       \
    printf(MLUGREEN " [%s:%d]" COLOUREND MLUBLUE " [MLULOG]:" COLOUREND    \
                    "taskId = %d: Kernel Hardware Time: %u us\n",          \
           __FILE__, __LINE__, taskId,                                     \
           (1000000U * (uint32_t)t_end.tv_sec + (uint32_t)t_end.tv_usec) - \
               (1000000U * (uint32_t)t_start.tv_sec +                      \
                (uint32_t)t_start.tv_usec));                               \
  } while (0)

#endif  // NDEBUG

#endif  // KERNELS_DEBUG_H_
