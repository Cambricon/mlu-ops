extern "C" __mlu_global__ void frac_float32_kernel0(float* INPUT, float* OUTPUT, int dim_0, int dim_1, int dim_2, int dim_3) {
  __nram__ float INPUT_N[1664];
  for (int i = 0; i < ((((dim_0 * dim_1) * dim_2) * dim_3) >> 8); ++i) {
    __memcpy(((float *)INPUT_N + (0)), ((float *)INPUT + ((i * 256))), 1024, GDRAM2NRAM);
    __bang_active_abs(((float *)INPUT_N + (256)), ((float *)INPUT_N + (0)), 256);
    __bang_float2int16_tz(((short *)INPUT_N + (1024)), ((float *)INPUT_N + (256)), 256, 0);
    __bang_int162float(((float *)INPUT_N + (640)), ((short *)INPUT_N + (1024)), 256, 0);
    __bang_active_sign(((float *)INPUT_N + (896)), ((float *)INPUT_N + (0)), 256);
    __bang_mul(((float *)INPUT_N + (1152)), ((float *)INPUT_N + (640)), ((float *)INPUT_N + (896)), 256);
    __bang_sub(((float *)INPUT_N + (1408)), ((float *)INPUT_N + (0)), ((float *)INPUT_N + (1152)), 256);
    __memcpy(((float *)OUTPUT + ((i * 256))), ((float *)INPUT_N + (1408)), 1024, NRAM2GDRAM);
  }
  if ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 8) * 256) != ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 8) * 256) + (((((dim_0 * dim_1) * dim_2) * dim_3) % 256) * 4))) {
    __memcpy(((float *)INPUT_N + (0)), ((float *)INPUT + ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 8) * 256))), (((((dim_0 * dim_1) * dim_2) * dim_3) % 256) * 16), GDRAM2NRAM);
    __bang_active_abs(((float *)INPUT_N + (256)), ((float *)INPUT_N + (0)), 256);
    __bang_float2int16_tz(((short *)INPUT_N + (1024)), ((float *)INPUT_N + (256)), 256, 2);
    __bang_int162float(((float *)INPUT_N + (640)), ((short *)INPUT_N + (1024)), 256, 2);
    __bang_active_sign(((float *)INPUT_N + (896)), ((float *)INPUT_N + (0)), 256);
    __bang_mul(((float *)INPUT_N + (1152)), ((float *)INPUT_N + (640)), ((float *)INPUT_N + (896)), 256);
    __bang_sub(((float *)INPUT_N + (1408)), ((float *)INPUT_N + (0)), ((float *)INPUT_N + (1152)), 256);
    __memcpy(((float *)OUTPUT + ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 8) * 256))), ((float *)INPUT_N + (1408)), (((((dim_0 * dim_1) * dim_2) * dim_3) % 256) * 16), NRAM2GDRAM);
  }
}

