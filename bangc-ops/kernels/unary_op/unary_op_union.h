/*************************************************************************
	> File Name: union_task.h
	> Author: wenzhengyin
	> Mail: jones980116@163.com 
	> Created Time: Sun Jun 19 18:58:04 2022
 ************************************************************************/
#ifndef UNARY_OP_UNION_H_
#define UNARY_OP_UNION_H_

#define UNION_OP_KERNEL_DECLARE(Op, DType, Prefer)           \
  __mlu_global__ void MLUUnionKernel##Op##DType##Prefer(\
	cnrtFunctionType_t k_type, mluOpDataType_t data_type, void* boxes_data_ptr, void* boxes_scores_ptr, float nms_thres, int input_boxes_num, uint8_t* output_boxes_index);\

#define UNION_OP_KERNEL_IMPLE(Op, DType, Prefer)                 \
  __mlu_global__ void MLUUnionKernel##Op##DType##Prefer(     \
	cnrtFunctionType_t k_type, mluOpDataType_t data_type, void* boxes_data_ptr, void* boxes_scores_ptr, float nms_thres, int input_boxes_num, uint8_t* output_boxes_index){\
		int setoff;\
		getOffsetNum##Op##Prefer(input_boxes_num, data_type, &setoff, k_type);                     \
		unionImple<DType, compute##Op##Prefer>(\
			(DType*)boxes_data_ptr, (DType*)boxes_scores_ptr, (DType)nms_thres, setoff, input_boxes_num, output_boxes_index);}

template <typename T, void (*OpFunc)(T*, T*, T, int, int, uint8_t*)>
__mlu_device__ void unionImple(T* boxes_data_ptr, T* boxes_scores_ptr, T nms_thres, int setoff, int input_boxes_num, uint8_t* output_boxes_index){

	OpFunc(boxes_data_ptr, boxes_scores_ptr, nms_thres, setoff, input_boxes_num, output_boxes_index);
}

#endif
