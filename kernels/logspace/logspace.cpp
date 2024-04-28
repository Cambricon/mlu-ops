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
  k_dim->x = 12;
  k_dim->y = 1;
  k_dim->z = 1;
}


mluOpStatus_t MLUOP_WIN_API mluOpLogspace(mluOpHandle_t handle, const float start, const float end, const int steps, const float base, 
                                      const mluOpTensorDescriptor_t res_desc, void *res) {
  // param check
  //由于只有单向量输入unaryOpParamCheck和双向量输入的参数检查api，因此需要自己去写
  //todo



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