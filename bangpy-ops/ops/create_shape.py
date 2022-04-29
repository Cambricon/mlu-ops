import random
def random_int_list(max_dim_length, each_dim_max_length):#生成随机元组
    random_list = []
    for i in range(max_dim_length):
        random_list.append(random.randint(2, each_dim_max_length))
    return  tuple(random_list)


#nram_single_buffer_size_by_byte 核上单个buffer得空间单位字节
#append_test_count 随机生成shape的个数
#max_dim_length 最大维度数
#each_dim_max_length 每个维度最大多少个元素
def CreatShapeList(nram_single_buffer_size_by_byte, append_test_count = 50, max_dim_length = 5, each_dim_max_length = 64):
	const_float32_128_align_element_count = 32  # float32 下 128字节对应元素个数
	const_float16_128_align_element_count = 64  # float16 下 128字节对应元素个数
	const_current_mlu_single_buffer_float32_max_element_size = int(
	    nram_single_buffer_size_by_byte / 4)  # float32下 单个nram_buffer的最大元素数
	const_current_mlu_single_buffer_float16_max_element_size = int(
	    nram_single_buffer_size_by_byte / 2)  # float16下 单个nram_buffer的最大元素数
	#内置固定检测shape
	test_shape_list = [
	    (1,),
	    (2,),
	    # 128字节对齐边界测试
	    (const_float32_128_align_element_count - 1,),  # 不足128对齐
	    (const_float32_128_align_element_count    ,),  # 满足128对齐
	    (const_float32_128_align_element_count + 1,),  # 128对齐后多1个
	    (const_float16_128_align_element_count - 1,),
	    (const_float16_128_align_element_count    ,),
	    (const_float16_128_align_element_count + 1,),
	    # nram_buffer边界测试
	    (const_current_mlu_single_buffer_float32_max_element_size - 1,),  # 比空间大小少一个元素
	    (const_current_mlu_single_buffer_float32_max_element_size    ,),  # 刚好用完空间
	    (const_current_mlu_single_buffer_float32_max_element_size + 1,),  # 比空间大小多一个元素
	    (const_current_mlu_single_buffer_float16_max_element_size - 1,),
	    (const_current_mlu_single_buffer_float16_max_element_size    ,),
	    (const_current_mlu_single_buffer_float16_max_element_size + 1,),
	]
	for i in range(append_test_count):
		test_shape_list.append(random_int_list(random.randint(2,max_dim_length),random.randint(2,each_dim_max_length)))
	return  test_shape_list
