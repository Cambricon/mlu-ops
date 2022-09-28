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

    constexpr int shape0 = 3;
    constexpr int shape1 = 9;
    constexpr int INPUT_SIZE = shape0 * shape1;
    float a[INPUT_SIZE] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.65, 0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.85, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 0.35};


    // create mluOp tensor descriptor
    int dim_num = 2;
    int dim_size[2] = {shape0, shape1};
    mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
    mluOpDataType_t type = MLUOP_DTYPE_FLOAT;

    mluOpTensorDescriptor_t input_tensor_desc;
    mluOpCreateTensorDescriptor (&input_tensor_desc);
    mluOpSetTensorDescriptor(input_tensor_desc, layout, type, dim_num, dim_size);


    int output_dim_nb = 1;
    int output_dim_size[1] = {shape0};
    mluOpTensorLayout_t output_layout = MLUOP_LAYOUT_ARRAY;
    mluOpDataType_t output_type = MLUOP_DTYPE_INT32;

    mluOpTensorDescriptor_t output_tensor_desc;
    mluOpCreateTensorDescriptor (&output_tensor_desc);
    mluOpSetTensorDescriptor(output_tensor_desc, output_layout, output_type, output_dim_nb, output_dim_size);

    // get workspace size
    size_t workspace_size = 0;
    mluOpGetPolyNmsWorkspaceSize(handle, input_tensor_desc, &workspace_size);

    // create workspace ptr
    void *workspace_ptr = nullptr;
    cnrtMalloc((void **)&workspace_ptr, workspace_size);

    // create input device ptr
    void *input_tensor_ptr;
    cnrtMalloc((void **)&input_tensor_ptr, INPUT_SIZE * sizeof(float));
    // copy host data to device
    cnrtMemcpy(input_tensor_ptr, a, INPUT_SIZE * sizeof(float), cnrtMemcpyHostToDev);

    // create output device ptr
    void *output_tensor_ptr;
    cnrtMalloc((void **)&output_tensor_ptr, shape0 * sizeof(float));

    void *output_size_ptr;
    cnrtMalloc((void **)&output_size_ptr, sizeof(float));

    const float iou_threshold = 0.5;

    // call mluOpPolyNms
    mluOpPolyNms(handle, input_tensor_desc, input_tensor_ptr, iou_threshold, workspace_ptr,  workspace_size, output_tensor_desc, output_tensor_ptr, output_size_ptr);

    // sync queue
    cnrtQueueSync(queue);

    // call end, copy device data to host
    int32_t *b = new int32_t [shape0];
    cnrtMemcpy(b, output_tensor_ptr, shape0 * sizeof(int32_t),  cnrtMemcpyDevToHost);

    int32_t *output_size =  new int32_t();
    cnrtMemcpy(output_size, output_size_ptr, sizeof(int32_t),  cnrtMemcpyDevToHost);

    printf("poly_nms intput:\n");
    for(int i = 0; i < shape0; ++i) {
        printf("box[%d]:[", i);
        for(int j = 0; j < shape1; ++j) {
            printf(" %f,", a[i*shape1 + j]);   
        }
        printf("]\n");
    }
    printf("poly_nms output size = %d\n", *output_size);
    for(int i = 0 ; i < *output_size; ++i){
        printf("poly_nms output[%d]:%d\n", i, b[i]);
    }

    // destory resource
    cnrtFree(input_tensor_ptr);
    cnrtFree(output_tensor_ptr);
    cnrtFree(workspace_ptr);
    cnrtFree(output_size_ptr);

    mluOpDestroyTensorDescriptor(input_tensor_desc);
    mluOpDestroyTensorDescriptor(output_tensor_desc);

    cnrtQueueDestroy(queue);
    mluOpDestroy(handle);
    return 0;
}