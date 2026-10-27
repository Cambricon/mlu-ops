/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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
#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "mlu_op.h"

namespace mluopapitest {

class DynamicPointToVoxelForwardRegression : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cnrtSuccess, cnrtQueueCreate(&queue_));
    ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpCreate(&handle_));
    ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpSetQueue(handle_, queue_));
  }

  void TearDown() override {
    if (queue_) cnrtQueueSync(queue_);
    for (void *data : data_) {
      if (data) cnrtFree(data);
    }
    if (workspace_) cnrtFree(workspace_);
    for (auto desc : desc_) {
      if (desc) mluOpDestroyTensorDescriptor(desc);
    }
    if (handle_) mluOpDestroy(handle_);
    if (queue_) cnrtQueueDestroy(queue_);
  }

  void run(int features, int invalid_points, mluOpReduceMode_t mode) {
    const std::vector<std::vector<int>> shapes = {
        {3, features}, {3, 3}, {1, features}, {1, 3}, {3}, {1}, {1}};
    const std::array<size_t, 7> elements = {
        3 * static_cast<size_t>(features), 9,
        static_cast<size_t>(features), 6, 3, 2, 1};
    for (size_t i = 0; i < desc_.size(); ++i) {
      ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpCreateTensorDescriptor(&desc_[i]));
      ASSERT_EQ(MLUOP_STATUS_SUCCESS,
                mluOpSetTensorDescriptor(
                    desc_[i], MLUOP_LAYOUT_ARRAY,
                    i == 0 || i == 2 ? MLUOP_DTYPE_FLOAT : MLUOP_DTYPE_INT32,
                    shapes[i].size(), shapes[i].data()));
      ASSERT_EQ(cnrtSuccess, cnrtMalloc(&data_[i], elements[i] * sizeof(int)));
    }
    std::vector<float> input(3 * features);
    std::vector<int> coors(9, 0);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < features; ++j) input[i * features + j] = 2 * i + 1;
      if (i >= 3 - invalid_points) coors[i * 3 + 1] = -1;
    }
    // The output descriptor has one voxel. The extra allocated row detects
    // writes beyond that capacity even when running without memcheck.
    const int sentinel = -777;
    std::vector<int> output_coors(6, sentinel), counts(2, sentinel);
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(data_[0], input.data(),
                                    input.size() * sizeof(float),
                                    cnrtMemcpyHostToDev));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(data_[1], coors.data(),
                                    coors.size() * sizeof(int),
                                    cnrtMemcpyHostToDev));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(data_[3], output_coors.data(),
                                    output_coors.size() * sizeof(int),
                                    cnrtMemcpyHostToDev));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(data_[5], counts.data(),
                                    counts.size() * sizeof(int),
                                    cnrtMemcpyHostToDev));
    size_t workspace_size = 0;
    ASSERT_EQ(MLUOP_STATUS_SUCCESS,
              mluOpGetDynamicPointToVoxelForwardWorkspaceSize(
                  handle_, desc_[0], desc_[1], &workspace_size));
    ASSERT_EQ(cnrtSuccess, cnrtMalloc(&workspace_, workspace_size));
    ASSERT_EQ(MLUOP_STATUS_SUCCESS,
              mluOpDynamicPointToVoxelForward(
                  handle_, mode, desc_[0], data_[0], desc_[1], data_[1],
                  workspace_, workspace_size, desc_[2], data_[2], desc_[3],
                  data_[3], desc_[4], data_[4], desc_[5], data_[5], desc_[6],
                  data_[6]));
    ASSERT_EQ(cnrtSuccess, cnrtQueueSync(queue_));
    int voxel_num = -1;
    std::array<int, 3> point_map;
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(&voxel_num, data_[6], sizeof(int),
                                    cnrtMemcpyDevToHost));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(point_map.data(), data_[4],
                                    3 * sizeof(int), cnrtMemcpyDevToHost));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(output_coors.data(), data_[3],
                                    6 * sizeof(int), cnrtMemcpyDevToHost));
    ASSERT_EQ(cnrtSuccess, cnrtMemcpy(counts.data(), data_[5],
                                    2 * sizeof(int), cnrtMemcpyDevToHost));
    EXPECT_EQ(invalid_points == 3 ? 0 : 1, voxel_num);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(i >= 3 - invalid_points ? -1 : 0, point_map[i]);
      EXPECT_EQ(sentinel, output_coors[i + 3]);
    }
    EXPECT_EQ(sentinel, counts[1]);
    if (invalid_points == 3) {
      EXPECT_EQ(sentinel, counts[0]);
      for (int i = 0; i < 3; ++i) EXPECT_EQ(sentinel, output_coors[i]);
    } else {
      EXPECT_EQ(3 - invalid_points, counts[0]);
      for (int i = 0; i < 3; ++i) EXPECT_EQ(0, output_coors[i]);
      std::vector<float> output(features);
      ASSERT_EQ(cnrtSuccess, cnrtMemcpy(output.data(), data_[2],
                                      features * sizeof(float),
                                      cnrtMemcpyDevToHost));
      const float expected = mode == MLUOP_REDUCE_DMEAN
                                 ? 3 - invalid_points
                                 : 5 - 2 * invalid_points;
      for (int i = 0; i < features; ++i) {
        ASSERT_FLOAT_EQ(expected, output[i]) << "feature " << i;
      }
    }
  }

  mluOpHandle_t handle_ = nullptr;
  cnrtQueue_t queue_ = nullptr;
  std::array<mluOpTensorDescriptor_t, 7> desc_{};
  std::array<void *, 7> data_{};
  void *workspace_ = nullptr;
};

TEST_F(DynamicPointToVoxelForwardRegression, InvalidVoxelMean) {
  run(11, 1, MLUOP_REDUCE_DMEAN);
}

TEST_F(DynamicPointToVoxelForwardRegression, InvalidVoxelMax) {
  run(15, 1, MLUOP_REDUCE_DMAX);
}

TEST_F(DynamicPointToVoxelForwardRegression, AllInvalidVoxels) {
  run(11, 3, MLUOP_REDUCE_DMEAN);
}

TEST_F(DynamicPointToVoxelForwardRegression, LargeFeaturesValidVoxelsMean) {
  run(400001, 0, MLUOP_REDUCE_DMEAN);
}

TEST_F(DynamicPointToVoxelForwardRegression, LargeFeaturesValidVoxelsMax) {
  run(400001, 0, MLUOP_REDUCE_DMAX);
}

}  // namespace mluopapitest
