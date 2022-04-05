#include "runtime_tensor.h"
RuntimeStatus_t MluOpFrac(
  RuntimeHandle_t handle,
  const RuntimeTensorDescriptor_t desc_INPUT,
  const void *INPUT,
  const RuntimeTensorDescriptor_t desc_OUTPUT,
  const void *OUTPUT);
