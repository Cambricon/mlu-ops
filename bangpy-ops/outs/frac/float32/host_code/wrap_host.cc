#include <vector>
#include "runtime_tensor.h"
#include "host.h"
#include "cnrt.h"
extern "C" void call_host(
  cnrtDev_t dev,
  cnrtQueue_t queue,
  std::vector<RuntimeTensorDescriptor_t> tensor_structs,
  std::vector<void*> tensor_addr,
  std::vector<BangArgUnion32*> scalar_addr
){
    RuntimeHandle_t handle = new (std::nothrow) RuntimeContext();
    handle->device = dev;
    handle->queue = queue;
    int dim[RUNTIME_DIM_MAX] = {0};
    RuntimeSetTensorDescriptor(tensor_structs[0], RUNTIME_DTYPE_FLOAT, 4, dim);
    RuntimeSetTensorDescriptor(tensor_structs[1], RUNTIME_DTYPE_FLOAT, 4, dim);
    tensor_structs[0]->dims[0] = int(scalar_addr[0]->v_int32);
    tensor_structs[1]->dims[0] = int(scalar_addr[0]->v_int32);
    tensor_structs[0]->dims[1] = int(scalar_addr[1]->v_int32);
    tensor_structs[1]->dims[1] = int(scalar_addr[1]->v_int32);
    tensor_structs[0]->dims[2] = int(scalar_addr[2]->v_int32);
    tensor_structs[1]->dims[2] = int(scalar_addr[2]->v_int32);
    tensor_structs[0]->dims[3] = int(scalar_addr[3]->v_int32);
    tensor_structs[1]->dims[3] = int(scalar_addr[3]->v_int32);
    MluOpFrac(handle, tensor_structs[0], tensor_addr[0], tensor_structs[1], tensor_addr[1]);
}
