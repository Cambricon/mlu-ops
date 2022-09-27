.. _fusionsample:

示例代码
=================

本章描述寒武纪BANGC OPS编程示例代码。


算子融合示例代码
-----------------

本节介绍了算子融合过程的示例代码。

FusedOps示例代码
>>>>>>>>>>>>>>>>>

::

  /* FusedOps */
  /*
  * A test which shows how to run FusedOps op
  *
  * define fusion_type: CNNL_CONV_SCALE_BN_ACTIVATION
  * contain single compute: convolution, scale, batchnorm, ACTIVATION_RELU
  */

  struct ParamPosition {
    static void updateParamPosition(int *position_arr[], int pos, bool place_holder) {
      int pos_tmp = pos;
      int begin_pos = -1;
      while (pos_tmp-- > 0) {
        if (*(position_arr[pos_tmp]) > -1) {
          begin_pos = *(position_arr[pos_tmp]);
          break;
        }
      }
      *(position_arr[pos]) = place_holder ? begin_pos + 1 : -1;
    }

    int input_position = -1;
    int filter_position = -1;
    int output_position = -1;
    int bias_position = -1;
    int bn_mean_position = -1;
    int bn_var_position = -1;
    int bn_filter_position = -1;
    int bn_bias_position = -1;
    int scale_alpha_position = -1;
    int scale_beta_position = -1;
    int prelu_alpha_position = -1;

    int *conv_bn_scale_active[11] = {&input_position, &filter_position, &bias_position,
      &scale_alpha_position, &scale_beta_position, &bn_mean_position, &bn_var_position,
      &bn_filter_position, &bn_bias_position, &prelu_alpha_position, &output_position};
  };

  struct DataBlock {
    DataBlock(void *h, void *d, void *g, void *p, size_t s, bool o, bool n) :
      host_ptr(h), device_ptr(d), device_origin_ptr(g), device_perf_ptr(p), size(s), is_output(o),
      is_null(n) {}
    bool is_null = false;
    bool is_output = false;
    void *host_ptr = nullptr;  // host pointer;
    void *device_ptr = nullptr;  // device pointer
    size_t size = 0;        // data_size (count * sizeof[dtype])
    void *device_origin_ptr = nullptr;  // device pointer of origin
    void *device_perf_ptr = nullptr;  // device pointer for perf test
  };

  // Init Fused Param
  cnnlFusedOps_t fusion_type;
  cnnlFusedOpsPlan_t fusion_plan = nullptr;
  cnnlFusedOpsConstParamPack_t cparam_pack = nullptr;
  cnnlFusedOpsVariantParamPack_t vparam_pack = nullptr;

  // Create FusedOps Plan, Create VarianParamPack, Create ConstParamPack
  CNNL_CHECK(cnnlCreateFusedOpsPlan(&fusion_plan, fusion_type));
  CNNL_CHECK(cnnlCreateFusedOpsVariantParamPack(&vparam_pack, fusion_type));
  CNNL_CHECK(cnnlCreateFusedOpsConstParamPack(&cparam_pack, fusion_type));

  bool conv_place_holder = true;
  bool bn_place_holder = true;
  bool scale_place_holder = true;
  bool prelu_place_holder = false;
  bool has_bias = true;
  bool has_bn_filter_bias = true;

  param_position_.input_position = 0;
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*filter_position = */ 1, conv_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*bias_position = */ 2, has_bias);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*scale_alpha_position = */ 3, scale_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*scale_beta_position = */ 4, scale_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*bn_mean_position = */ 5, bn_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*bn_var_position = */ 6, bn_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*bn_filter_position = */ 7, has_bn_filter_bias);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*bn_bias_position = */ 8, has_bn_filter_bias);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*prelu_alpha_position = */ 9, prelu_place_holder);
  ParamPosition::updateParamPosition(param_position_.conv_bn_scale_active, /*output_position = */ 10, true);

  // Create Tensors
  // Create Conv Input Tensor
  cnnlDataType_t conv_input_dtype = CNNL_DTYPE_INT16;
  cnnlDataType_t conv_input_oc_dtype = CNNL_DTYPE_INT16;
  cnnlTensorLayout_t conv_input_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> conv_input_shape = {1, 28, 28, 512};
  cnnlTensorDescriptor_t conv_input_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&conv_input_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(conv_input_desc, conv_input_layout, conv_input_dtype, conv_input_shape.size(), conv_input_shape.data()));

  // Create Filter Input Tensor
  cnnlDataType_t conv_filter_dtype = CNNL_DTYPE_INT16;
  cnnlDataType_t conv_filter_oc_dtype = CNNL_DTYPE_INT16;
  cnnlTensorLayout_t conv_filter_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> conv_filter_shape = {512, 3, 3, 512};
  cnnlTensorDescriptor_t conv_filter_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&conv_filter_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(conv_filter_desc, conv_filter_layout, conv_filter_dtype, conv_filter_shape.size(), conv_filter_shape.data()));

  // Create Conv Bias Tensor
  cnnlDataType_t conv_bias_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t conv_bias_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t conv_bias_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> conv_bias_shape = {1, 1, 1, 512};
  cnnlTensorDescriptor_t conv_bias_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&conv_bias_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(conv_bias_desc, conv_bias_layout, conv_bias_dtype, conv_bias_shape.size(), conv_bias_shape.data()));

  // Create Scale Alpha Tensor
  cnnlDataType_t scale_alpha_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t scale_alpha_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t scale_alpha_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> scale_alpha_shape = {1, 1, 1, 1};
  cnnlTensorDescriptor_t scale_alpha_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&scale_alpha_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(scale_alpha_desc, scale_alpha_layout, scale_alpha_dtype, scale_alpha_shape.size(), scale_alpha_shape.data()));

  // Create Scale Beta Tensor
  cnnlDataType_t scale_beta_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t scale_beta_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t scale_beta_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> scale_beta_shape = {1, 1, 1, 1};
  cnnlTensorDescriptor_t scale_beta_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&scale_beta_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(scale_beta_desc, scale_beta_layout, scale_beta_dtype, scale_beta_shape.size(), scale_beta_shape.data()));

  // Create BatchNorm Mean Tensor
  cnnlDataType_t bn_mean_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t bn_mean_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t bn_mean_layout = CNNL_LAYOUT_ARRAY;
  std::vector<int> bn_mean_shape = {512};
  cnnlTensorDescriptor_t bn_mean_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&bn_mean_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(bn_mean_desc, bn_mean_layout, bn_mean_dtype, bn_mean_shape.size(), bn_mean_shape.data()));

  // Create BatchNorm Var Tensor
  cnnlDataType_t bn_var_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t bn_var_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t bn_mean_layout = CNNL_LAYOUT_ARRAY;
  std::vector<int> bn_mean_shape = {512};
  cnnlTensorDescriptor_t bn_var_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&bn_var_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(bn_var_desc, bn_var_layout, bn_var_dtype, bn_var_shape.size(), bn_var_shape.data()));

  // Create BatchNorm Filter Tensor
  cnnlDataType_t bn_filter_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t bn_filter_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t bn_filter_layout = CNNL_LAYOUT_ARRAY;
  std::vector<int> bn_filter_shape = {512};
  cnnlTensorDescriptor_t bn_filter_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&bn_filter_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(bn_filter_desc, bn_filter_layout, bn_filter_dtype, bn_filter_shape.size(), bn_filter_shape.data()));

  // Create BatchNorm Bias Tensor
  cnnlDataType_t bn_bias_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t bn_bias_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t bn_bias_layout = CNNL_LAYOUT_ARRAY;
  std::vector<int> bn_bias_shape = {512};
  cnnlTensorDescriptor_t bn_bias_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&bn_bias_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(bn_bias_desc, bn_bias_layout, bn_bias_dtype, bn_bias_shape.size(), bn_bias_shape.data()));

  // Create Output Tensor
  cnnlDataType_t output_dtype = CNNL_DTYPE_FLOAT;
  cnnlDataType_t output_oc_dtype = CNNL_DTYPE_FLOAT;
  cnnlTensorLayout_t output_layout = CNNL_LAYOUT_NHWC;
  std::vector<int> output_shape = {1, 28, 28, 512};
  cnnlTensorDescriptor_t output_desc;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&output_desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(output_desc, output_layout, output_dtype, output_shape.size(), output_shape.data()));
  
  // Set Const Param
  if (param_position_.input_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_XDESC, (void*)conv_input_desc);
  }

  if (param_position_.filter_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_WDESC, (void*)conv_filter_desc);
  }

  if (param_position_.output_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_YDESC, (void*)output_desc);
  }

  if (param_position_.bias_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_BIAS_DESC, (void*)conv_bias_desc);
  }

  if (param_position_.bn_mean_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_BN_FILTER_BIAS_MEAN_VAR_DESC, (void*)bn_mean_desc);
  }

  if (param_position_.scale_alpha_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALE_ALPHA_DESC, (void*)scale_alpha_desc);
  }

  if (param_position_.scale_beta_position) {
    CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALE_BETA_DESC, (void*)scale_beta_desc);
  }

  // Init Device
  cnnlHandle_t handle = nullptr;
  CNNL_CHECK(cnnlCreate(&handle));

  // FusedOps Set Conv Desc
  int conv_dim = 4;
  int conv_group_count = 1;
  int conv_pad = {1, 1, 1, 1};
  int conv_stride = {1, 1};
  int conv_dilation = {1, 1};
  int workspace_size = 0;
  cnnlDataType_t compute_dtype = CNNL_DTYPE_FLOAT;

  CPURuntime cpu_runtime;
  MLURuntime mlu_runtime;
  cnnlConvolutionDescriptor_t conv_desc = nullptr;
  cnnlActivationDescriptor_t active_desc = nullptr;
  std::shared_ptr<cntest::CPUMemoryPool> cmp = std::make_shared<cntest::CPUMemoryPool>();
  std::shared_ptr<cntest::MLUMemoryPool> mmp = std::make_shared<cntest::MLUMemoryPool>();
  cpu_runtime.init(cmp);
  mlu_runtime.init(mmp);
  conv_desc = cpu_runtime.allocate(cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor);
  CNNL_CHECK(cnnlSetConvolutionDescriptor(conv_desc, conv_dim, conv_pad, conv_stride, conv_dilation, conv_group_count, compute_type));
  CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_CONV_DESC, (void *)conv_desc));

  cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
  cnnlConvolutionFwdPreference_t preference = CNNL_CONVOLUTION_FWD_FASTEST;
  CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(handle, conv_desc, conv_input_desc, conv_filter_desc, output_desc, preference, &algo));

  cnnlConvolutionCastMode_t cast_mode = CNNL_OFFLINE_SYMMETRIC_QUANTIZE;
  CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALAR_CONV_FWD_CAST_MODE, (void *)&cast_mode));

  cnnlActivationMode_t mode = CNNL_ACTIVATION_RELU;
  active_desc = cpu_runtime.allocate(cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor);
  CNNL_CHECK(cnnlSetActivationDescriptor(active_desc, mode, CNNL_NOT_PROPAGATE_NAN, 0.0));
  CNNL_CHECK(cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_ACTIVATION_DESC, (void *)active_desc));

  // Make Fused Ops Plan
  CNNL_CHECK(cnnlMakeFusedOpsPlan(handle, fusion_plan, cparam_pack, &workspace_size));

  if (workspace_size > 0) {
    char* workspace_ptr = (char *)mlu_runtime.allocate(workspace_size);
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_WORKSPACE, (void *)workspace_ptr));
  }
  CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_SCALAR_WORKSPACE_SIZE, (void *)(&workspace_size)));

  // Create Conv Input Ptr
  if (param_position_.input_position) {
    void* conv_input_ptr = mlu_runtime.allocate(1 * 28 * 28 * 512 * 2, "CONV_INPUT");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_X, (void*)conv_input_ptr);
  }

  if (param_position_.filter_position) {
    void* conv_filter_ptr = mlu_runtime.allocate(512 * 3 * 3 * 512 * 2, "CONV_FILTER");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_W, (void*)conv_filter_ptr);
  }

  if (param_position_.output_position) {
    void* output_ptr = mlu_runtime.allocate(1 * 28 * 28 * 512 * 4, "OUTPUT");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_Y, (void*)output_ptr);
  }

  if (param_position_.bias_position) {
    void* conv_bias_ptr = mlu_runtime.allocate(1 * 1 * 1 * 512 * 4, "CONV_BIAS");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_BIAS, (void*)conv_bias_ptr);
  }

  if (param_position_.bn_mean_position) {
    void* bn_mean_ptr = mlu_runtime.allocate(512 * 4, "BN_MEAN");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_BN_MEAN, (void*)bn_mean_ptr);
  }

  if (param_position_.bn_var_position) {
    void* bn_var_ptr = mlu_runtime.allocate(512 * 4, "BN_VAR");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_BN_VAR, (void*)bn_var_ptr);
  }

  if (param_position_.bn_filter_position) {
    void* bn_filter_ptr = mlu_runtime.allocate(512 * 4, "BN_FILTER");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_BN_FILTER, (void*)bn_filter_ptr);
  }

  if (param_position_.bn_bias_position) {
    void* bn_bias_ptr = mlu_runtime.allocate(512 * 4, "BN_BIAS");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_BN_BIAS, (void*)bn_bias_ptr);
  }

  if (param_position_.scale_alpha_position) {
    void* scale_alpha_ptr = mlu_runtime.allocate(1 * 4, "SCALE_ALPHA");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_SCALE_ALPHA, (void*)scale_alpha_ptr);
  }

  if (param_position_.scale_beta_position) {
    void* scale_beta_ptr = mlu_runtime.allocate(1 * 4, "SCALE_BETA");
    CNNL_CHECK(cnnlSetFusedOpsVariantParamPackAttribute(vparam_pack, CNNL_PTR_SCALE_BETA, (void*)scale_beta_ptr);
  }

  // Excute Fusion Plan
  CNNL_CHECK(cnnlFusedOpsExecute(handle, fusion_plan, vparam_pack));

  // Destroy Resources
  cnnlDestroyTensorDescriptor(conv_input_desc);
  cnnlDestroyTensorDescriptor(conv_filter_desc);
  cnnlDestroyTensorDescriptor(conv_bias_desc);
  cnnlDestroyTensorDescriptor(scale_alpha_desc);
  cnnlDestroyTensorDescriptor(scale_beta_desc);
  cnnlDestroyTensorDescriptor(bn_mean_desc);
  cnnlDestroyTensorDescriptor(bn_var_desc);
  cnnlDestroyTensorDescriptor(bn_filter_desc);
  cnnlDestroyTensorDescriptor(bn_bias_desc);
  cnnlDestroyTensorDescriptor(output_desc);
  cnnlDestroyConvolutionDescriptor(conv_desc);
  cnnlDestroyActivationDescriptor(active_desc);
  cnnlDestroyFusedOpsConstParamPack(cparam_pack);
  cnnlDestroyFusedOpsVariantParamPack(vparam_pack);
  cnnlDestroyFusedOpsPlan(fusion_plan);

