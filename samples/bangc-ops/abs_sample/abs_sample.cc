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

#include <iostream>

#include "mlu_op.h"

void initDevice(int &dev, cnrtQueue_t &queue, mluOpHandle_t &handle) {
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));

    CNRT_CHECK(cnrtQueueCreate(&queue));

    mluOpCreate(&handle);
    mluOpSetQueue(handle, queue);
}

int main(int argc, char *argv[]) {
    int dev;
    mluOpHandle_t handle = nullptr;
    cnrtQueue_t queue = nullptr;

    initDevice(dev, queue, handle);
    printf("init device\n");

    // construct input data 
    constexpr int INPUT_SIZE = 10;
    float a[INPUT_SIZE] = {-5.5, -3.5, -10, 3.6, -999.23, 234.324, -2456.136, 0.0, 214535.4, 89.7};

    // create mluOp tensor descriptor
    int dimNb = 1;
    int dimSize[1] = {INPUT_SIZE};
    mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
    mluOpDataType_t type = MLUOP_DTYPE_FLOAT;

    mluOpTensorDescriptor_t input_tensor_desc;
    mluOpCreateTensorDescriptor (&input_tensor_desc);
    mluOpSetTensorDescriptor(input_tensor_desc, MLUOP_LAYOUT_ARRAY,type, dimNb, dimSize);

    mluOpTensorDescriptor_t output_tensor_desc;
    mluOpCreateTensorDescriptor (&output_tensor_desc);
    mluOpSetTensorDescriptor(output_tensor_desc, MLUOP_LAYOUT_ARRAY,type, dimNb, dimSize);

    // create input device ptr
    void *input_tensor_ptr;
    cnrtMalloc((void **)&input_tensor_ptr, INPUT_SIZE * sizeof(float));
    // copy host data to device
    cnrtMemcpy(input_tensor_ptr, a, INPUT_SIZE * sizeof(float), cnrtMemcpyHostToDev);

    // create output device ptr
    void *output_tensor_ptr;
    cnrtMalloc((void **)&output_tensor_ptr, INPUT_SIZE * sizeof(float));

    // call mluops abs
    mluOpAbs(handle, input_tensor_desc, input_tensor_ptr, output_tensor_desc, output_tensor_ptr);

    // sync queue
    cnrtQueueSync(queue);

    // call end, copy device data to host
    float *b = new float [INPUT_SIZE];
    cnrtMemcpy(b, output_tensor_ptr, INPUT_SIZE * sizeof(float),  cnrtMemcpyDevToHost);
    for(int i = 0 ; i < INPUT_SIZE; ++i){
        printf("input[%d]:%f, output[%d]:%f\n", i, a[i], i, b[i]);
    }

    // destory resource
    cnrtFree(input_tensor_ptr);
    cnrtFree(output_tensor_ptr);

    mluOpDestroyTensorDescriptor(input_tensor_desc);
    mluOpDestroyTensorDescriptor(output_tensor_desc);

    cnrtQueueDestroy(queue);
    mluOpDestroy(handle);
    return 0;
}