extern "C" __mlu_global__ void frac_float32_kernel0(float* INPUT, float* OUTPUT, int dim_0, int dim_1, int dim_2, int dim_3) {
  __nram__ float INPUT_N[34816];
  if (262143 < (((dim_0 * dim_1) * dim_2) * dim_3)) {
    __memcpy_async(((float *)INPUT_N + (0)), ((float *)INPUT + ((((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6)))), 16384, GDRAM2NRAM);
    __mlvm_sync();
  }
  if (524287 < (((dim_0 * dim_1) * dim_2) * dim_3)) {
    __bang_active_abs(((float *)INPUT_N + (8192)), ((float *)INPUT_N + (0)), 4096);
    __bang_float2int16_tz(((short *)INPUT_N + (24576)), ((float *)INPUT_N + (8192)), 4096, 0);
    __bang_int162float(((float *)INPUT_N + (14336)), ((short *)INPUT_N + (24576)), 4096, 0);
    __bang_active_sign(((float *)INPUT_N + (18432)), ((float *)INPUT_N + (0)), 4096);
    __bang_mul(((float *)INPUT_N + (22528)), ((float *)INPUT_N + (14336)), ((float *)INPUT_N + (18432)), 4096);
    __bang_sub(((float *)INPUT_N + (26624)), ((float *)INPUT_N + (0)), ((float *)INPUT_N + (22528)), 4096);
    __memcpy_async(((float *)INPUT_N + (4096)), ((float *)INPUT + (((((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6)) + 4096))), 16384, GDRAM2NRAM);
    __mlvm_sync();
  }
  for (int i = 0; i < (((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) - 2); ++i) {
    __memcpy_async(((float *)OUTPUT + (((i * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))))), ((float *)INPUT_N + ((((i % 2) * 4096) + 26624))), 16384, NRAM2GDRAM);
    __bang_active_abs(((float *)INPUT_N + (8192)), ((float *)INPUT_N + ((((i + 1) % 2) * 4096))), 4096);
    __bang_float2int16_tz(((short *)INPUT_N + (24576)), ((float *)INPUT_N + (8192)), 4096, 0);
    __bang_int162float(((float *)INPUT_N + (14336)), ((short *)INPUT_N + (24576)), 4096, 0);
    __bang_active_sign(((float *)INPUT_N + (18432)), ((float *)INPUT_N + ((((i + 1) % 2) * 4096))), 4096);
    __bang_mul(((float *)INPUT_N + (22528)), ((float *)INPUT_N + (14336)), ((float *)INPUT_N + (18432)), 4096);
    __bang_sub(((float *)INPUT_N + (((((i + 1) % 2) * 4096) + 26624))), ((float *)INPUT_N + ((((i + 1) % 2) * 4096))), ((float *)INPUT_N + (22528)), 4096);
    __memcpy_async(((float *)INPUT_N + (((i % 2) * 4096))), ((float *)INPUT + ((((i * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))) + 8192))), 16384, GDRAM2NRAM);
    __mlvm_sync();
  }
  if (524287 < (((dim_0 * dim_1) * dim_2) * dim_3)) {
    __memcpy_async(((float *)OUTPUT + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))) - 8192))), ((float *)INPUT_N + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) % 2) * 4096) + 26624))), 16384, NRAM2GDRAM);
  }
  if (262143 < (((dim_0 * dim_1) * dim_2) * dim_3)) {
    __bang_active_abs(((float *)INPUT_N + (8192)), ((float *)INPUT_N + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) + 1) % 2) * 4096))), 4096);
    __bang_float2int16_tz(((short *)INPUT_N + (24576)), ((float *)INPUT_N + (8192)), 4096, 0);
    __bang_int162float(((float *)INPUT_N + (14336)), ((short *)INPUT_N + (24576)), 4096, 0);
    __bang_active_sign(((float *)INPUT_N + (18432)), ((float *)INPUT_N + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) + 1) % 2) * 4096))), 4096);
    __bang_mul(((float *)INPUT_N + (22528)), ((float *)INPUT_N + (14336)), ((float *)INPUT_N + (18432)), 4096);
    __bang_sub(((float *)INPUT_N + (((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) - 1) % 2) * 4096) + 26624))), ((float *)INPUT_N + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) + 1) % 2) * 4096))), ((float *)INPUT_N + (22528)), 4096);
  }
  __mlvm_sync();
  if (262143 < (((dim_0 * dim_1) * dim_2) * dim_3)) {
    __memcpy_async(((float *)OUTPUT + ((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))) - 4096))), ((float *)INPUT_N + (((((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) + 1) % 2) * 4096) + 26624))), 16384, NRAM2GDRAM);
  }
  __mlvm_sync();
  if ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 6) % 4096) != 0) {
    __memcpy(((float *)INPUT_N + (8192)), ((float *)INPUT + (((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))))), ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 6) % 4096) * 4), GDRAM2NRAM);
    __bang_active_abs(((float *)INPUT_N + (14336)), ((float *)INPUT_N + (8192)), 4096);
    __bang_float2int16_tz(((short *)INPUT_N + (24576)), ((float *)INPUT_N + (14336)), 4096, 0);
    __bang_int162float(((float *)INPUT_N + (18432)), ((short *)INPUT_N + (24576)), 4096, 0);
    __bang_active_sign(((float *)INPUT_N + (22528)), ((float *)INPUT_N + (8192)), 4096);
    __bang_mul(((float *)INPUT_N + (0)), ((float *)INPUT_N + (18432)), ((float *)INPUT_N + (22528)), 4096);
    __bang_sub(((float *)INPUT_N + (26624)), ((float *)INPUT_N + (8192)), ((float *)INPUT_N + (0)), 4096);
    __memcpy(((float *)OUTPUT + (((((((dim_0 * dim_1) * dim_2) * dim_3) >> 18) * 4096) + (((int)taskId) * ((((dim_0 * dim_1) * dim_2) * dim_3) >> 6))))), ((float *)INPUT_N + (26624)), ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 6) % 4096) * 4), NRAM2GDRAM);
  }
  if (((((dim_0 * dim_1) * dim_2) * dim_3) % 64) != 0) {
    if (((int)taskId) == 63) {
      __memcpy(((float *)INPUT_N + (8192)), ((float *)INPUT + ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 6) * (((int)taskId) + 1)))), (((((dim_0 * dim_1) * dim_2) * dim_3) % 64) * 4), GDRAM2NRAM);
      __bang_active_abs(((float *)INPUT_N + (14336)), ((float *)INPUT_N + (8192)), 4096);
      __bang_float2int16_tz(((short *)INPUT_N + (24576)), ((float *)INPUT_N + (14336)), 4096, 0);
      __bang_int162float(((float *)INPUT_N + (18432)), ((short *)INPUT_N + (24576)), 4096, 0);
      __bang_active_sign(((float *)INPUT_N + (22528)), ((float *)INPUT_N + (8192)), 4096);
      __bang_mul(((float *)INPUT_N + (0)), ((float *)INPUT_N + (18432)), ((float *)INPUT_N + (22528)), 4096);
      __bang_sub(((float *)INPUT_N + (26624)), ((float *)INPUT_N + (8192)), ((float *)INPUT_N + (0)), 4096);
      __memcpy(((float *)OUTPUT + ((((((dim_0 * dim_1) * dim_2) * dim_3) >> 6) * (((int)taskId) + 1)))), ((float *)INPUT_N + (26624)), (((((dim_0 * dim_1) * dim_2) * dim_3) % 64) * 4), NRAM2GDRAM);
    }
  }
}

