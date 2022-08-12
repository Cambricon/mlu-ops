/*************************************************************************
	> File Name: main.cpp
	> Author: wenzhengyin
	> Mail: jones980116@163.com 
	> Created Time: Tue Apr 19 14:35:06 2022
 ************************************************************************/
#include "cnrt.h"
#include "cndev.h"
//#include "cnrt_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "core/mlu_op_core.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op_kernel.h"
#include "kernels/unary_op/unary_op_host.h"
using namespace std;
typedef uint16_t half;


mluOpStatus_t MLUOP_WIN_API mluOpMlNms(mluOpHandle_t handle,
		const mluOpTensorDescriptor_t boxes_data_ptr_desc, void* boxes_data_ptr, void* scores_max_boxes_data_ptr,
		int input_boxes_num, float iou_threshold, uint8_t* output_boxes_index) {
	
	int setoff;
	bool zero_element = false;
	mluOpDataType_t data_type = MLUOP_DTYPE_HALF;
	mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};

	cnrtDim3_t k_dim = {4, 1, 1};
	cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;

	mluOpStatus_t param_check = unaryOpNmsParamCheck(
			"[mluOpMlNms]", boxes_data_ptr_desc, boxes_data_ptr, scores_max_boxes_data_ptr, support_type, 2, zero_element);
	if(param_check != MLUOP_STATUS_SUCCESS){
		return param_check;
	}

	void (*mluOpFuncKernel)(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, mluOpDataType_t data_type,
			void* boxes_data_ptr, void* scores_max_boxes_data_ptr, float nmsThres, int input_boxes_num, uint8_t* output_boxes_index);
	//mluOpFuncKernel = NULL;
	if (k_type == CNRT_FUNC_TYPE_BLOCK){
		if (boxes_data_ptr_desc->dtype == MLUOP_DTYPE_HALF){
			mluOpFuncKernel = mluBlockKernelMlNmsHalfFast;
		}else {
			mluOpFuncKernel = mluBlockKernelMlNmsFloatFast;
		}
	}else {
		if (boxes_data_ptr_desc->dtype == MLUOP_DTYPE_HALF){
			mluOpFuncKernel = mluUnionKernelMlNmsHalfFast;
		}else{
			mluOpFuncKernel = mluUnionKernelMlNmsFloatFast;
		}
	}

	KERNEL_CHECK(
			(mluOpFuncKernel(k_dim, k_type, handle->queue, boxes_data_ptr_desc->dtype, boxes_data_ptr, scores_max_boxes_data_ptr, iou_threshold, input_boxes_num, output_boxes_index)));
	GEN_CASE_END();

	return MLUOP_STATUS_SUCCESS;

}



