//
// Created by root on 7/22/22.
//

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cnrt.h>
#include <sys/time.h>

#include "core/context.h"
#include "mlu_op.h"


bool readMask(uint32_t *mask, int i, int j, int mask_col_num) {
  int pos_j = j / 32;
  int offset = j % 32;
  constexpr uint32_t DEFAULT_MASK = 0x80000000; // 0b 1000 0000 0000 0000 ...
  return mask[i * mask_col_num + pos_j] & (DEFAULT_MASK >> offset);
}

struct BoxData {
  BoxData(int n, bool cw = true, bool all_convex = true,
          bool discreate_offset = true) {
    host.resize(n * 9);
    float offset = 1.0 / n;
    if (discreate_offset) {
      offset = 0.6;
    }
    for (auto i = 0; i < n; ++i) {
      float *line = host.data() + 9 * i;

      if (!all_convex && i % 2 == 0) {
        FillNonConex(cw, line);
        area.push_back(1.6);
      } else {
        FillSquare(cw, line);
        area.push_back(1);
      }

      line[8] = i;

      for (int j = 0; j < 8; ++j) {
        line[j] += offset * i;
      }
    }
    mluOpCreateTensorDescriptor(&desc);
    int shape[] = {n, 9};
    mluOpSetTensorDescriptor(desc, mluOpTensorLayout_t::MLUOP_LAYOUT_ARRAY,
                             mluOpDataType_t::MLUOP_DTYPE_FLOAT, 2, shape);

    cnrtMalloc(&dev_data, host.size() * sizeof(float));
    cnrtMemcpy(dev_data, host.data(), host.size() * sizeof(float),
               cnrtMemcpyHostToDev);
  }
  void FillSquare(bool cw, float *line) const {
    line[0] = 0;
    line[1] = 0;

    line[4] = 1;
    line[5] = 1;
    if (cw) {
      line[2] = 0;
      line[3] = 1;
      line[6] = 1;
      line[7] = 0;
    } else {
      line[2] = 1;
      line[3] = 0;
      line[6] = 0;
      line[7] = 1;
    }
  }

  void FillNonConex(bool cw, float *line) const {
    line[0] = 1;
    line[1] = 0;

    if (cw) {
      line[2] = 0.6;
      line[3] = 0.6;
      line[4] = 0;
      line[5] = 1;
      line[6] = 1;
      line[7] = 1;
    } else {
      line[2] = 1;
      line[3] = 1;
      line[4] = 0;
      line[5] = 1;
      line[6] = 0.6;
      line[7] = 0.6;
    }
    for (int j = 0; j < 8; ++j) {
      line[j] *= 2;
      line[j] -= 1;
    }
  }

  std::vector<float> host;
  std::vector<float> area;
  void *dev_data;
  mluOpTensorDescriptor_t desc;
};

int RunTest(mluOpHandle_t handle, int NBox, BoxData &box, size_t workspace_size,
            void *workspace, void *dev_output_index, void *dev_output_count,
            bool check_result) {
  mluOpTensorDescriptor_t out_desc;
  mluOpCreateTensorDescriptor(&out_desc);
  int shape[]={NBox};
  mluOpSetTensorDescriptor(out_desc, mluOpTensorLayout_t::MLUOP_LAYOUT_ARRAY,
                           mluOpDataType_t::MLUOP_DTYPE_INT32, 1, shape);
  cnrtMemset(dev_output_index, 0, NBox * sizeof(int));
  cnrtMemset(dev_output_count, 0, sizeof(int));
  cnrtMemset(workspace, 0, workspace_size);
  timeval tic;
  timeval toc;
  gettimeofday(&tic, NULL);
  mluOpPolyNms(handle, box.desc, box.dev_data, 0.0, workspace,1,out_desc, dev_output_index,
               dev_output_count);
  cnrtQueueSync(handle->queue);
  gettimeofday(&toc, NULL);
  printf("input size: %d host latency : %f us\n", NBox,
         (toc.tv_sec - tic.tv_sec) * 1e6 + toc.tv_usec - tic.tv_usec);
  fflush(stdout);
  if (!check_result) {
    printf("Check Ignored \n");
    return 0;
  }
  std::vector<float> area(NBox);
  float *dev_area = (float *)workspace;
  cnrtMemcpy(area.data(), dev_area, NBox * sizeof(float), cnrtMemcpyDevToHost);

  std::vector<int> sort_info(NBox);
  int *dev_sort_info = (int *)dev_area + NBox;
  cnrtMemcpy(sort_info.data(), dev_sort_info, NBox * sizeof(int),
             cnrtMemcpyDevToHost);

  int mask_col_num = (NBox + 31) / 32;
  std::vector<uint32_t> masks(NBox * mask_col_num);
  uint32_t *dev_mask = (uint32_t *)dev_sort_info + NBox;
  cnrtMemcpy(masks.data(), dev_mask, NBox * mask_col_num * sizeof(int),
             cnrtMemcpyDevToHost);

  std::vector<int32_t> n_box(1);
  cnrtMemcpy(n_box.data(), dev_output_count, sizeof(int), cnrtMemcpyDevToHost);
  std::vector<int32_t> final_boxes;
  int should_id = NBox - 1;

  for (int i = 0; i < NBox; ++i) {
    if (std::abs(area[i] - box.area[i]) > 1e-2) {
      printf("Err area %f \n", area[i]);
      goto EXIT;
    }
  }

  for (int i = 0; i < sort_info.size(); ++i) {
    if (sort_info[i] != (sort_info.size() - 1 - i)) {
      printf("Err sort\n");
      goto EXIT;
    }
  }

  for (int i = 0; i < NBox; ++i) {
    bool check_ok = true;
    for (int j = 0; j < NBox; ++j) {
      if ((i - j) == 1) {
        check_ok = check_ok && !readMask(masks.data(), i, j, mask_col_num);
      } else {
        check_ok = check_ok && readMask(masks.data(), i, j, mask_col_num);
      }
      if (!check_ok) {
        printf("Err mask\n");
        goto EXIT;
      }
    }
  }

  if (n_box[0] != (NBox + 1) / 2) {
    printf("Err output count\n");
    goto EXIT;
  }

  final_boxes.resize(n_box[0]);
  cnrtMemcpy(final_boxes.data(), dev_output_index, n_box[0] * sizeof(int),
             cnrtMemcpyDevToHost);
  std::sort(final_boxes.begin(),final_boxes.end(),[](const  int lhs,const int rhs){
    return lhs > rhs;
  });
  for (auto v : final_boxes) {
    if (should_id != v) {
      printf("Err output\n");
      goto EXIT;
    }
    should_id -= 2;
  }

  printf("All Good \n");
  return 0;
EXIT:
  return -1;
}

int Test(int NBox, mluOpHandle_t handle, bool test_worst_case) {
  const char *extramark = test_worst_case ? "Worst case" : "Average case";

  BoxData box(NBox, true, true, !test_worst_case);
  void *workspace = nullptr;
  size_t wkspace_size;
  mluOpGetPolyNmsWorkspaceSize(handle, box.desc, &wkspace_size);
  cnrtMalloc(&workspace, wkspace_size);
  void *dev_output_index = nullptr;
  cnrtMalloc(&dev_output_index, NBox * sizeof(int));
  void *dev_output_count = nullptr;
  cnrtMalloc(&dev_output_count, sizeof(int));

  printf("Testing %s CW, all convex,\n", extramark);
  if (RunTest(handle, NBox, box, wkspace_size, workspace, dev_output_index,
              dev_output_count, !test_worst_case)) {
    return -1;
  }
  printf("Testing %s CCW,all convex\n", extramark);
  BoxData box_ccw(NBox, false, true, !test_worst_case);
  if (RunTest(handle, NBox, box_ccw, wkspace_size, workspace, dev_output_index,
              dev_output_count, !test_worst_case)) {
    return -1;
  }
  BoxData box_cw_half(NBox, true, false, !test_worst_case);
  printf("Testing %s CW, half convex\n", extramark);
  if (RunTest(handle, NBox, box_cw_half, wkspace_size, workspace,
              dev_output_index, dev_output_count, !test_worst_case)) {
    return -1;
  }
  BoxData box_ccw_half(NBox, false, false, !test_worst_case);
  printf("Testing %s CCW, half convex\n", extramark);
  if (RunTest(handle, NBox, box_ccw_half, wkspace_size, workspace,
              dev_output_index, dev_output_count, !test_worst_case)) {
    return -1;
  }
  cnrtFree(workspace);
  cnrtFree(dev_output_count);
  cnrtFree(dev_output_index);
  return 0;
}
int main(int argc, char **argv) {
  cnrtSetDevice(0);
  mluOpHandle_t handle;
  mluOpCreate(&handle);

  int NBox = 2000;
  if (Test(NBox, handle, false)) {
    return -1;
  }
  if (Test(NBox, handle, true)) {
    return -1;
  }
  return 0;
}