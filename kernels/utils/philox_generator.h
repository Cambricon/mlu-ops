/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_UTILS_PHILOX_GENERATOR_H_
#define KERNELS_UTILS_PHILOX_GENERATOR_H_
#include <type_traits>
#define ONCE_COMPUTE_NUM 512
#define COUNTER_GEN_NUM 128
#define ROUND_UPPER 10

__nram__ uint32_t nram_counter[ONCE_COMPUTE_NUM];

__mlu_func__ void gen_random_u32(
    vv_uint32 *key0, vv_uint32 *key1, uint32_t *nram_counter0,
    uint32_t *nram_counter1, uint32_t *nram_counter2, uint32_t *nram_counter3,
    uint32_t &offset_low, uint32_t &offset_high, int32_t thread_begin,
    int32_t &thread_acc, int32_t thread_cur_core, vv_uint32 index) {
  const uint32_t kPhiloxM4xA = 0xD2511F53;
  const uint32_t kPhiloxM4xB = 0xCD9E8D57;
  vv_uint32 cx, cy, cz, cw, h0, h1, l0, l1;

  // init cx cy cz cw by offset and thread
  uint32_t counterz = thread_acc + thread_begin;
  __vv_move(cx, offset_low);
  __vv_move(cy, offset_high);
  __vv_move(cz, index);
  __vv_move(cw, 0);
  __vv_add(cz, cz, counterz);

  // compute loop, each loop repeat compute twice
  for (int32_t round_idx = 0; round_idx < ROUND_UPPER; round_idx += 2) {
    // First. For better performence, dst is different from second.
    __vv_mulh(h0, cx, kPhiloxM4xA);
    __vv_mulh(h1, cz, kPhiloxM4xB);
    __vv_mul(l0, cx, kPhiloxM4xA);
    __vv_mul(l1, cz, kPhiloxM4xB);
#if __BANG_ARCH__ > 520
    __vv_trixor(cx, h1, cy, key0[round_idx]);
    __vv_trixor(cz, h0, cw, key1[round_idx]);
#else
    __vv_xor(cz, h0, cw);
    __vv_xor(cz, cz, key1[round_idx]);
    __vv_xor(cx, h1, cy);
    __vv_xor(cx, cx, key0[round_idx]);
#endif

    // Second. Dst is cx, cy, cz, cw
    __vv_mulh(h0, cx, kPhiloxM4xA);
    __vv_mulh(h1, cz, kPhiloxM4xB);
    __vv_mul(cw, cx, kPhiloxM4xA);
    __vv_mul(cy, cz, kPhiloxM4xB);
#if __BANG_ARCH__ > 520
    __vv_trixor(cx, h1, l1, key0[round_idx + 1]);
    __vv_trixor(cz, h0, l0, key1[round_idx + 1]);
#else
    __vv_xor(cz, h0, l0);
    __vv_xor(cz, cz, key1[round_idx + 1]);
    __vv_xor(cx, h1, l1);
    __vv_xor(cx, cx, key0[round_idx + 1]);
#endif
  }

  // store to nram
  __vv_store(nram_counter0, cx, 64);
  __vv_store(nram_counter1, cy, 64);
  __vv_store(nram_counter2, cz, 64);
  __vv_store(nram_counter3, cw, 64);

  // loop unrolling, compute next 64
  vv_uint32 cx2, cy2, cz2, cw2, h02, h12, l02, l12;
  __vv_move(cx2, offset_low);
  __vv_move(cy2, offset_high);
  __vv_move(cz2, index);
  __vv_move(cw2, 0);
  __vv_add(cz2, cz2, counterz + 64);

  for (int32_t round_idx = 0; round_idx < ROUND_UPPER; round_idx += 2) {
    __vv_mulh(h02, cx2, kPhiloxM4xA);
    __vv_mulh(h12, cz2, kPhiloxM4xB);
    __vv_mul(l02, cx2, kPhiloxM4xA);
    __vv_mul(l12, cz2, kPhiloxM4xB);
#if __BANG_ARCH__ > 520
    __vv_trixor(cx2, h12, cy2, key0[round_idx]);
    __vv_trixor(cz2, h02, cw2, key1[round_idx]);
#else
    __vv_xor(cz2, h02, cw2);
    __vv_xor(cz2, cz2, key1[round_idx]);
    __vv_xor(cx2, h12, cy2);
    __vv_xor(cx2, cx2, key0[round_idx]);
#endif

    __vv_mulh(h02, cx2, kPhiloxM4xA);
    __vv_mulh(h12, cz2, kPhiloxM4xB);
    __vv_mul(cw2, cx2, kPhiloxM4xA);
    __vv_mul(cy2, cz2, kPhiloxM4xB);
#if __BANG_ARCH__ > 520
    __vv_trixor(cx2, h12, l12, key0[round_idx + 1]);
    __vv_trixor(cz2, h02, l02, key1[round_idx + 1]);
#else
    __vv_xor(cz2, h02, l02);
    __vv_xor(cz2, cz2, key1[round_idx + 1]);
    __vv_xor(cx2, h12, l12);
    __vv_xor(cx2, cx2, key0[round_idx + 1]);
#endif
  }

  __vv_store(nram_counter0 + 64, cx2, 64);
  __vv_store(nram_counter1 + 64, cy2, 64);
  __vv_store(nram_counter2 + 64, cz2, 64);
  __vv_store(nram_counter3 + 64, cw2, 64);

  // deal offset and thread_acc
  uint32_t offset_tmp =
      offset_low + (thread_acc == (thread_cur_core - COUNTER_GEN_NUM) ? 1 : 0);
  if (offset_low > offset_tmp) {
    ++offset_high;
  }
  offset_low = offset_tmp;
  thread_acc += COUNTER_GEN_NUM;
  thread_acc %= thread_cur_core;
}

template <typename RANGE_TYPE>
__mlu_func__ void cvtUniform(uint32_t *output, int32_t num, RANGE_TYPE max,
                             RANGE_TYPE min, bool is_int) {
  RANGE_TYPE range = max - min;
  if (is_int && std::is_same<RANGE_TYPE, int32_t>::value) {
    __bang_rem((uint32_t *)output, (uint32_t *)output, (uint32_t)range, num);
    __bang_add_scalar((int32_t *)output, (int32_t *)output, min, num);
  } else if (is_int && std::is_same<RANGE_TYPE, int64_t>::value) {
    __cn_vector_mod_scalar_u64(num / 2, (uint64_t *)output, (uint64_t *)output,
                               (uint64_t)range);
    __bang_add_scalar((int64_t *)output, (int64_t *)output, min, num);
  } else {
    __bang_band_scalar((uint32_t *)output, (uint32_t *)output, 0x007fffff, num);
    int32_t temp = 0x7e800000;
    float mul_float = *(float *)&temp;
    __bang_mul_scalar((float *)output, (float *)output, mul_float, num);
    __bang_fusion(FUSION_FMA, (float *)output, (float *)output, range, min,
                  num);
  }
}

/******************************************************************************
 * MLUOP FUNC: genUniform.
 * Generate random data in half, bfloat16 or float data type.
 *
 * The philox algorithm has three parameters: seed, offset and subsequence.
 * Seed specifies the key in the algorithm.
 * Offset specifies the counter in the algorithm.
 * On MLU devices, we use index [0, THREAD_NUM) to bind to subsequence.
 * \p Thread_begin, \p thread_acc and \p thread_cur_core specify the id of
 * index, that is, the value of subsequence.
 *
 * param 'key0_begin' and 'key1_begin' are parameters from seed.
 * param 'output' is the destination pointer in NRAM.
 * param 'num' is the number of random data which must be 512 aligned.
 * param 'offset_low' and 'offset_high' are parameters from offset.
 * param 'max' is the maximum value of random numbers.
 * param 'min' is the minimum value of random numbers.
 * param 'thread_begin' is the start id of current core.
 * param 'thread_acc' is the record of thread_id.
 * param 'thread_cur_core' is the total thread number of current core.
 * param 'is_int' indicates whether the generated data is fixed-point type.
 * Note: The space size of \p output is \p num * sizeof(float).
 ******************************************************************************/
template <typename DST_TYPE, typename RANGE_TYPE>
__mlu_func__ void genUniform(uint32_t key0_begin, uint32_t key1_begin,
                             DST_TYPE *output, int num, uint32_t &offset_low,
                             uint32_t &offset_high, RANGE_TYPE max,
                             RANGE_TYPE min, int32_t thread_begin,
                             int32_t &thread_acc, int32_t thread_cur_core,
                             bool is_int) {
  if (num > 0) {
    // counter nram contribute
    uint32_t *nram_counter0 = nram_counter;
    uint32_t *nram_counter1 = nram_counter + 1 * COUNTER_GEN_NUM;
    uint32_t *nram_counter2 = nram_counter + 2 * COUNTER_GEN_NUM;
    uint32_t *nram_counter3 = nram_counter + 3 * COUNTER_GEN_NUM;

    // prepare key in compute
    const uint32_t kPhiloxW32A = 0x9E3779B9;
    const uint32_t kPhiloxW32B = 0xBB67AE85;
    vv_uint32 index;
    __vv_index(index, 0, 1);
    vv_uint32 vv_key0[ROUND_UPPER];
    vv_uint32 vv_key1[ROUND_UPPER];
    __vv_move(vv_key0[0], key0_begin);
    __vv_move(vv_key1[0], key1_begin);
    for (int32_t i = 1; i < ROUND_UPPER; i++) {
      __vv_add(vv_key0[i], vv_key0[i - 1], kPhiloxW32A);
      __vv_add(vv_key1[i], vv_key1[i - 1], kPhiloxW32B);
    }

    // repeat compute gen random
    int32_t repeat_num = num / ONCE_COMPUTE_NUM;
    for (int32_t i = 0; i < repeat_num; i++) {
      gen_random_u32(vv_key0, vv_key1, nram_counter0, nram_counter1,
                     nram_counter2, nram_counter3, offset_low, offset_high,
                     thread_begin, thread_acc, thread_cur_core, index);
      __bang_transpose((uint32_t *)output + i * ONCE_COMPUTE_NUM, nram_counter,
                       4, COUNTER_GEN_NUM);
    }
    // cvt random to uniform
    cvtUniform((uint32_t *)output, num, (float)max, (float)min, is_int);

    if (std::is_same<DST_TYPE, half>::value) {
      __bang_float2half_tz((half *)output, (float *)output, num);
    }
    if (std::is_same<DST_TYPE, bfloat16_t>::value) {
      __bang_float2bfloat16_tz((bfloat16_t *)output, (float *)output, num);
    }
  }
}

#endif  // KERNELS_UTILS_PHILOX_GENERATOR_H_
