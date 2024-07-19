/*******************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modif y, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS for A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE for ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#pragma once

#define N_ALIGN 128
#define CoreCapacity MAX_NRAM_SIZE / 81920 * 81920
    // memory length of input per core
#define ClusterCapacity MAX_NRAM_SIZE / 81920 * 81920 * 4
    // memory length of input per cluster
#define DimOneDealLength 147456  // size of one NRAM in dim-one

__nram__ char nram_buffer[MAX_NRAM_SIZE];
__mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];

// removing the dependence between columns
template <typename T>
__mlu_func__ void removeDataDependence(T *dst,
                                       T *src,
                                       int deal_num,
                                       int unroll_num) {
  if (deal_num < unroll_num) {
    __bang_add(dst, dst, src, deal_num);
    return;
  }
  int deal_per_loop = deal_num / unroll_num;
  int rem_num       = deal_num % unroll_num;
  for (int i = 0; i < unroll_num; i++) {
    int deal_this_loop = (i == unroll_num - 1) ?
      deal_per_loop + rem_num : deal_per_loop;
    __bang_add(dst + deal_per_loop * i, dst + deal_per_loop * i,
               src + deal_per_loop * i, deal_this_loop);
  }
}

// cumsum execution
template <typename T>
__mlu_func__ void dimOneCumsum(T *src0_nram, T *src1_nram, int deal_length) {
  int seg_num = (deal_length + N_ALIGN - 1) / N_ALIGN;
  T pre_sum = 0;
  __bang_transpose(src1_nram, src0_nram, N_ALIGN, seg_num);

  for (int j = 1; j < seg_num; j++) {
    removeDataDependence(src1_nram + j * N_ALIGN,
                         src1_nram + (j - 1) * N_ALIGN,
                         N_ALIGN, 4);
  }
  src0_nram[0] = pre_sum;
  int index_offset = (seg_num - 1) * N_ALIGN - 1;
  for (int k = 1; k < N_ALIGN; k++) {
    pre_sum      = pre_sum + src1_nram[index_offset + k];
    src0_nram[k] = pre_sum;
  }
  __bang_cycle_add(src1_nram, src1_nram, src0_nram, seg_num * N_ALIGN, N_ALIGN);
  __bang_transpose(src0_nram, src1_nram, seg_num, N_ALIGN);
}

// To compute the offset between clusters
// "offset_type 0" means calculate the offset with the array
// that pointed by "cluster_offsets"
// "offset_type 1" means calculate the offset with the scalar "cum_offset"
template <typename T>
__mlu_func__ void offsetCompute(T *nram_src0, T *cluster_offsets,
                                T cum_offset, const int32_t data_size,
                                bool offset_type) {
    if (offset_type == 0) {
        if (coreId == 3) {
            cluster_offsets[clusterId] = nram_src0[data_size - 1];
        }
        __sync_all();
        if (taskId == 0) {
            if (clusterDim == 8) {
          // [0],[1],[2]...[7] are clusterDim offsets for clusterDim clusters;
          // the [9] records and saves the cumulative offset for next round;
          // the [8] is to support the offset calculate;
                cluster_offsets[8] = cluster_offsets[7];
                cluster_offsets[7] = cluster_offsets[6];
                cluster_offsets[6] = cluster_offsets[5];
                cluster_offsets[5] = cluster_offsets[4];
                cluster_offsets[4] = cluster_offsets[3];
                cluster_offsets[3] = cluster_offsets[2];
                cluster_offsets[2] = cluster_offsets[1];
                cluster_offsets[1] = cluster_offsets[0];
                cluster_offsets[0] = cluster_offsets[9];
                cluster_offsets[1] = cluster_offsets[0] + cluster_offsets[1];
                cluster_offsets[2] = cluster_offsets[1] + cluster_offsets[2];
                cluster_offsets[3] = cluster_offsets[2] + cluster_offsets[3];
                cluster_offsets[4] = cluster_offsets[3] + cluster_offsets[4];
                cluster_offsets[5] = cluster_offsets[4] + cluster_offsets[5];
                cluster_offsets[6] = cluster_offsets[5] + cluster_offsets[6];
                cluster_offsets[7] = cluster_offsets[6] + cluster_offsets[7];
                cluster_offsets[9] = cluster_offsets[7] + cluster_offsets[8];
            } else if (clusterDim == 4) {
                cluster_offsets[4] = cluster_offsets[3];
                cluster_offsets[3] = cluster_offsets[2];
                cluster_offsets[2] = cluster_offsets[1];
                cluster_offsets[1] = cluster_offsets[0];
                cluster_offsets[0] = cluster_offsets[5];
                cluster_offsets[1] = cluster_offsets[0] + cluster_offsets[1];
                cluster_offsets[2] = cluster_offsets[1] + cluster_offsets[2];
                cluster_offsets[3] = cluster_offsets[2] + cluster_offsets[3];
                cluster_offsets[5] = cluster_offsets[3] + cluster_offsets[4];
            } else if (clusterDim == 2) {
                cluster_offsets[2] = cluster_offsets[1];
                cluster_offsets[1] = cluster_offsets[0];
                cluster_offsets[0] = cluster_offsets[3];
                cluster_offsets[1] = cluster_offsets[0] + cluster_offsets[1];
                cluster_offsets[3] = cluster_offsets[1] + cluster_offsets[2];
            } else {
                return;
            }
    }
    __sync_all();
    __bang_add_scalar(nram_src0, nram_src0,
                      cluster_offsets[clusterId], data_size);
  } else {
    __sync_all();
    __sync_all();
    __bang_add_scalar(nram_src0, nram_src0, cum_offset, data_size);
  }
}
// inclusive execution
template <typename T>
__mlu_func__ void inclusiveScan(T* nram_src,
                                const int32_t data_size,
                                bool offset_type,
                                T* cluster_offsets,
                                T cum_offset) {
  // parameter preparing
  __mlu_shared__ T core_offsets[4];
  T *nram_src0 = nram_src;
  T *nram_src1 = nram_src0 + DimOneDealLength / sizeof(T);

  __mluop_exp(nram_src0, nram_src0, nullptr, 0, data_size);
  dimOneCumsum(nram_src0, nram_src1, data_size);
  if (__is_ipu()) {
    core_offsets[coreId] = nram_src0[data_size-1];
  }
  __sync_cluster();
  // offset between coresx
  if (coreId == 0) {
    core_offsets[1] = core_offsets[0] + core_offsets[1];
    core_offsets[3] = core_offsets[1] + core_offsets[2];
    core_offsets[2] = core_offsets[1];
    core_offsets[1] = core_offsets[0];
    core_offsets[0] = 0;
  }
  __sync_cluster();
  if (__is_ipu()) {
    __bang_add_scalar(nram_src0, nram_src0,
                      core_offsets[coreId], data_size);
  }
  __sync_cluster();
  // offset between clusters
  offsetCompute(nram_src0, cluster_offsets, cum_offset,
                data_size, offset_type);
  // log computing
  __mluop_log(nram_src0, nram_src0, nullptr, 0, data_size);
}

// one dimension executing kernel==========================================
template <typename T>
__mlu_global__ void
dimOneKernel(const T *input,
             T *output,
             int32_t data_size) {
    // parameters preparing
    const int32_t n_core = DimOneDealLength / sizeof(T);
    const int32_t n_cluster = n_core * 4;
    const int32_t n_round = n_cluster * clusterDim;
    const int32_t rounds = (data_size + n_round - 1) / n_round;
    const int32_t last_round_length = data_size - (rounds - 1) * n_round;

    T *cluster_offsets = output + data_size - 10;
    cluster_offsets[9] = 0;

    T basenum = 0;
    int32_t round = 0;
    T *sram_src0 = (T *)sram_buffer;
    T *sram_src1 = (T *)(sram_buffer + ClusterCapacity);
    T *nram_src0 = (T *)nram_buffer;
    T *nram_src1 = nram_src0 + ((MAX_NRAM_SIZE/sizeof(T)) >> 1);
    int32_t totalId = clusterId;  // clusters' Id in entire process
    int32_t last_round_clusters
      = (last_round_length + n_cluster - 1) / n_cluster;
    int32_t last_cluster_length
      = last_round_length - (last_round_clusters - 1) * n_cluster;
    int32_t n_last_core = last_cluster_length >> 2;
    const int32_t length_offset = last_cluster_length % 4;
    const int32_t copy_offset = coreId * n_last_core
                    + __mluop_min(coreId, length_offset);
    if (coreId < length_offset) {
      n_last_core += 1;
    }

    // first memory copy GDRAM2NRAM
    if (rounds > 1) {
      __memcpy(nram_src0,
      input + totalId * n_cluster + coreId * n_core,
      n_core * sizeof(T), GDRAM2NRAM);
    }

    T *this_sram = sram_src0;
    T *next_sram = sram_src1;
    T *this_nram = nram_src0;
    T *next_nram = nram_src1;

    // pipeline execute
    while (round < rounds - 1) {
      totalId = round * clusterDim + clusterId;
      this_sram = (round % 2 == 0) ? sram_src0 : sram_src1;
      next_sram = (round % 2 == 0) ? sram_src1 : sram_src0;
      this_nram = (round % 2 == 0) ? nram_src0 : nram_src1;
      next_nram = (round % 2 == 0) ? nram_src1 : nram_src0;
      // data copy for next round

      if (round < rounds - 2) {
        __memcpy_async(next_nram,
          input + (totalId + clusterDim) * n_cluster + n_core * coreId,
          n_core * sizeof(T), GDRAM2NRAM);
      } else {
        if (clusterId < last_round_clusters - 1) {
          __memcpy_async(next_nram,
            input + (totalId + clusterDim) * n_cluster + coreId * n_core,
            n_core * sizeof(T), GDRAM2NRAM);
        } else if (clusterId == last_round_clusters - 1) {
          __memcpy_async(next_nram,
            input + (totalId + clusterDim) * n_cluster + copy_offset,
            n_last_core * sizeof(T), GDRAM2NRAM);
        }
      }

      // compute
      inclusiveScan(this_nram, n_core, 0, cluster_offsets, basenum);

      __memcpy(this_sram + n_core * coreId, this_nram,
               n_core * sizeof(T), NRAM2SRAM);
      __sync_cluster();
      if (__is_mpu()) {
        __memcpy_async(output + totalId * n_cluster, this_sram,
                 n_cluster * sizeof(T), SRAM2GDRAM);
      }
      round++;
    }

    this_sram = next_sram;
    this_nram = next_nram;
    totalId = round * clusterDim + clusterId;

    // the last round
    if (last_round_clusters == 1) {
      if (clusterId == 0) {
        if (rounds == 1) {
          __memcpy(this_nram, input + totalId * n_cluster + copy_offset,
                   n_last_core * sizeof(T), GDRAM2NRAM);
        }
        inclusiveScan(this_nram, n_last_core, 1,
                      output, cluster_offsets[9]);
        __memcpy(output + totalId * n_cluster + copy_offset, this_nram,
                 n_last_core * sizeof(T), NRAM2GDRAM);
      }
    } else {
      if (clusterId < last_round_clusters - 1) {
        if (rounds == 1)__memcpy(this_nram,
          input + totalId * n_cluster + coreId * n_core,
          n_core * sizeof(T), GDRAM2NRAM);
        inclusiveScan(this_nram, n_core, 0, cluster_offsets, basenum);
        __memcpy(output + totalId * n_cluster + coreId * n_core,
                 this_nram, n_core * sizeof(T), NRAM2GDRAM);
      } else if (clusterId == last_round_clusters - 1) {
        if (rounds == 1) {
          __memcpy(this_nram, input + totalId * n_cluster + copy_offset,
                   n_last_core * sizeof(T), GDRAM2NRAM);
        }
        inclusiveScan(this_nram, n_last_core, 0,
                      cluster_offsets, basenum);
        __memcpy(output + totalId * n_cluster + copy_offset, this_nram,
                 n_last_core * sizeof(T), NRAM2GDRAM);
      } else {
        __sync_all();
        __sync_all();
      }
    }
}
