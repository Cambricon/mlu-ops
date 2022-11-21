/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 *input_cpu_ptr
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

#include <iostream>
#include <random>

#include "math.h"

#include "cnrt.h"
#include "mlu_op.h"

const double EPSILON = 1e-9;
const double EPSILON_FLOAT = 1e-6;
const double EPSILON_HALF = 1e-3;

void mluOpCheck(mluOpStatus_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::string error = "\"" + std::string(mluOpGetErrorString(result)) +
                        " in " + std::string(func) + "\"";
    throw std::runtime_error(error);
  }
}

#define MLUOP_CHECK(val) mluOpCheck((val), #val, __FILE__, __LINE__)

struct HostTimer {
  struct timespec t0 = {0, 0};
  struct timespec t1 = {0, 0};
  double tv_nsec = 0.0;
  double tv_sec = 0.0;
  double tv_usec = 0.0;
  void start() { clock_gettime(CLOCK_MONOTONIC, &t0); }
  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    tv_nsec = (double)t1.tv_nsec - (double)t0.tv_nsec;
    tv_sec = (double)t1.tv_sec - (double)t0.tv_sec;
    tv_usec = tv_nsec / 1000 + tv_sec * 1000 * 1000;
  }
};

float *mallocDataRandf(int size, int low, int hight) {
  float *data = (float *)malloc(size * sizeof(float));
  std::uniform_real_distribution<float> dist(low, hight);
  std::default_random_engine random(time(NULL));
  for (int i = 0; i < size; i++) {
    data[i] = dist(random);
  }
  return data;
}

bool hasNanOrInf(float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (std::isinf(data[i]) || std::isnan(data[i])) {
      return true;
    }
  }
  return false;
}

double computeDiff3(float *baseline_result, float *mlu_result, size_t count,
                    mluOpDataType_t dtype) {
  if (hasNanOrInf(baseline_result, count) || hasNanOrInf(mlu_result, count)) {
    printf("Found NaN or Inf when compute diff, return __DBL_MAX__ instead.");
    return __DBL_MAX__;
  }
  double max_value = 0.0;
  for (int i = 0; i < count; ++i) {
    float numerator = fabs(mlu_result[i] - baseline_result[i]);
    double ratio = 0;
    if (((MLUOP_DTYPE_HALF == dtype) &&
         (fabs(baseline_result[i]) < EPSILON_HALF)) ||
        ((MLUOP_DTYPE_FLOAT == dtype) &&
         (fabs(baseline_result[i]) < EPSILON_FLOAT))) {
      ratio = numerator;
    } else {
      ratio = numerator / (fabs(baseline_result[i]) + EPSILON);
    }
    max_value = (ratio > max_value) ? ratio : max_value;
  }
  return max_value;
}

void cpuCompute(const float *input, const int input_length, float *output) {
  for (int i = 0; i < input_length; ++i) {
    output[i] = (input[i] >= 0) ? input[i] : -input[i];
  }
}

void initDevice(int &dev, cnrtQueue_t &queue, mluOpHandle_t &handle) {
  CNRT_CHECK(cnrtGetDevice(&dev));
  CNRT_CHECK(cnrtSetDevice(dev));

  CNRT_CHECK(cnrtQueueCreate(&queue));

  mluOpCreate(&handle);
  mluOpSetQueue(handle, queue);
}

struct ShapeParam {
  int shape[8] = {0};
  int dims = 0;
  int length = 0;
};

bool parserParam(int argc, char *argv[], ShapeParam &param) {
  argc -= 1;
  argv++;
  if (argc < 2 || atoi(argv[0]) != argc - 1) {
    printf("Please enter correct parameters.\n");
    printf("e.g.\n./abs_sample [dims_value] [shape0] [shape1] ...\n");
    printf("e.g.\n./abs_sample 4 10 10 10 10 \n");
    return false;
  }

  param.length = 1;
  param.dims = atoi(argv[0]);
  for (int i = 1; i <= param.dims; ++i) {
    param.shape[i - 1] = atoi(argv[i]);
    param.length *= param.shape[i - 1];
  }
  return true;
}

void printTestCaseShape(const ShapeParam &param) {
  printf("---------------------------------\n");
  printf("input dims is : %d \n", param.dims);
  printf("input shape is : [");
  for (int i = 0; i < param.dims - 1; ++i) {
    printf(" %d,", param.shape[i]);
  }
  printf(" %d]\n", param.shape[param.dims - 1]);
}

int main(int argc, char *argv[]) {
  ShapeParam abs_param;
  HostTimer interface_timer;
  HostTimer cpu_compute_timer;
  float mlu_time = 0.0f;
  cnrtNotifier_t start = nullptr, end = nullptr;

  // prase input param
  if (!parserParam(argc, argv, abs_param)) {
    return 0;
  }
  printTestCaseShape(abs_param);

  // init device
  int dev;
  mluOpHandle_t handle = nullptr;
  cnrtQueue_t queue = nullptr;
  initDevice(dev, queue, handle);

  // create input/output tensors
  float *input_cpu_ptr = mallocDataRandf(abs_param.length, -1000, 1000);
  int dimNb = abs_param.dims;
  int *dimSize = abs_param.shape;
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
  mluOpDataType_t type = MLUOP_DTYPE_FLOAT;

  mluOpTensorDescriptor_t input_tensor_desc;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_tensor_desc));
  MLUOP_CHECK(mluOpSetTensorDescriptor(input_tensor_desc, MLUOP_LAYOUT_ARRAY,
                                       type, dimNb, dimSize));

  mluOpTensorDescriptor_t output_tensor_desc;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_tensor_desc));
  MLUOP_CHECK(mluOpSetTensorDescriptor(output_tensor_desc, MLUOP_LAYOUT_ARRAY,
                                       type, dimNb, dimSize));

  // cpu compute
  cpu_compute_timer.start();
  float *cpu_output = (float *)malloc(abs_param.length * sizeof(float));
  cpuCompute(input_cpu_ptr, abs_param.length, cpu_output);
  cpu_compute_timer.stop();

  // create input device ptr
  void *input_tensor_ptr;
  CNRT_CHECK(
      cnrtMalloc((void **)&input_tensor_ptr, abs_param.length * sizeof(float)));
  // copy host data to device
  CNRT_CHECK(cnrtMemcpy(input_tensor_ptr, input_cpu_ptr,
                        abs_param.length * sizeof(float), cnrtMemcpyHostToDev));

  // create output device ptr
  void *output_tensor_ptr;
  CNRT_CHECK(cnrtMalloc((void **)&output_tensor_ptr,
                        abs_param.length * sizeof(float)));

  // get device time
  CNRT_CHECK(cnrtNotifierCreate(&start));
  CNRT_CHECK(cnrtNotifierCreate(&end));
  CNRT_CHECK(cnrtPlaceNotifier(start, queue));

  // call mluOpAbs interface
  interface_timer.start();
  MLUOP_CHECK(mluOpAbs(handle, input_tensor_desc, input_tensor_ptr,
                       output_tensor_desc, output_tensor_ptr));
  interface_timer.stop();

  CNRT_CHECK(cnrtPlaceNotifier(end, queue));
  CNRT_CHECK(cnrtQueueSync(queue));
  CNRT_CHECK(cnrtNotifierDuration(start, end, &mlu_time));

  // call end, copy device data to host
  float *mlu_out_cpu_ptr = (float *)malloc(abs_param.length * sizeof(float));
  CNRT_CHECK(cnrtMemcpy(mlu_out_cpu_ptr, output_tensor_ptr,
                        abs_param.length * sizeof(float), cnrtMemcpyDevToHost));

  // compute diff3
  double diff3 = computeDiff3(cpu_output, mlu_out_cpu_ptr, abs_param.length,
                              MLUOP_DTYPE_FLOAT);

  printf("[interface time        ]: %lf us \n", interface_timer.tv_usec);
  printf("[MLU hardware time     ]: %f us \n", mlu_time);
  printf("[CPU compute time      ]: %lf us \n", cpu_compute_timer.tv_usec);
  printf("[diff3                 ]: %lf \n", diff3);

  // destory resource
  CNRT_CHECK(cnrtFree(input_tensor_ptr));
  CNRT_CHECK(cnrtFree(output_tensor_ptr));

  MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_tensor_desc));
  MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_tensor_desc));

  CNRT_CHECK(cnrtNotifierDestroy(start));
  CNRT_CHECK(cnrtNotifierDestroy(end));

  CNRT_CHECK(cnrtQueueDestroy(queue));
  MLUOP_CHECK(mluOpDestroy(handle));
  return 0;
}
