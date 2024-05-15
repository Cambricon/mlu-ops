//    cd mlu-ops( cd /mnt )
//    ./build.sh --filter="logspace"
//    cd build/test/
//    ./mluop_gtest --gtest_filter=*logspace*


#include "logspace.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"



void LogspacePolicyFunc(const mluOpHandle_t &handle, const int steps,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_in_cluster = handle->core_num_per_cluster;
  uint32_t core_max=cluster_num*core_in_cluster;
  uint32_t core_used=core_max>steps? steps:core_max;

  k_dim->x = core_used;
  //k_dim->x = 12;
  k_dim->y = 1;
  k_dim->z = 1;
}

static inline bool isSupportType(const mluOpDataType_t check_type,
                                 const mluOpDataType_t support_type[],
                                 const int len) {
  for (int i = 0; i < len; ++i) {
    if (check_type == support_type[i]) {
      return true;
    }
  }
  return false;
}


mluOpStatus_t LogspaceParamCheck(
    const mluOpHandle_t &handle, const float start, const float end, const int steps, const float base,
    const mluOpTensorDescriptor_t &res_desc, const void *res) {

  PARAM_CHECK("[mluOpLogspace]", handle != nullptr);
  PARAM_CHECK("[mluOpLogspace]", res_desc != nullptr);

  //float参数不能是nan或inf，且base大于0
  PARAM_CHECK("[mluOpLogspace]", (start != NAN)&&(start != INFINITY));
  PARAM_CHECK("[mluOpLogspace]", (end != NAN)&&(end != INFINITY));
  PARAM_CHECK("[mluOpLogspace]", (base != NAN)&&(base != INFINITY));
  PARAM_CHECK("[mluOpLogspace]", base > 0);

  PARAM_CHECK("[mluOpLogspace]", steps > 0);
  size_t element_num = mluOpGetTensorElementNum(res_desc);
  PARAM_CHECK("[mluOpLogspace]", steps <= element_num);

  //数据类型检查
  mluOpDataType_t support_type[4] = {MLUOP_DTYPE_FLOAT, MLUOP_DTYPE_BFLOAT16, MLUOP_DTYPE_HALF, MLUOP_DTYPE_INT32};
  if (!isSupportType(res_desc->dtype, support_type, 4)) {
    LOG(ERROR) << "[mluOpLogspace]" << ":res_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}


mluOpStatus_t MLUOP_WIN_API mluOpLogspace(mluOpHandle_t handle, const float start, const float end, const int steps, const float base, 
                                      const mluOpTensorDescriptor_t res_desc, void *res) {
  // param check
  //由于只有单向量输入unaryOpParamCheck和双向量输入的参数检查api，因此需要自己去写
  mluOpStatus_t param_check =
      LogspaceParamCheck(handle, start, end, steps, base, res_desc, res);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }


  // policy select
  //任务维度和任务类型选择策略
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  LogspacePolicyFunc(handle, steps, &k_dim, &k_type);
  //unaryOpPolicyFunc(handle, res_desc, &k_dim, &k_type);
  VLOG(5) << "[mluOpLogspace] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";



  //设备类型检查
  if (handle->arch != MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpLogspace] now only support <MLU370>\n";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }



  //调用函数
  VLOG(5) << "kernel KernelLogspace.";
  CHECK_RETURN("[mluOpLogspace] ",
               KernelLogspace(k_dim, k_type, handle->queue, res_desc->dtype,
                start, end, steps, base, res));
  // GEN_CASE_END();

  return MLUOP_STATUS_SUCCESS;
}    