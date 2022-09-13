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

#include "generate_proposals_v2_impl.h"

#include <iostream>
#include <float.h>

using namespace std;

namespace GenerateProposalsV2 {

#define PNMS_MIN (-(float)FLT_MAX)
static const double kBBoxClipDefault = std::log(1000.0 / 16.0);
// static const float kBBoxClipDefault = 4.135166556742356f;

template <typename T>
void quickSort(T *arr, int low, int high) {
  if (high <= low) return;
  int i = low;
  int j = high;
  T key = arr[low];
  while (true) {
    while (arr[i] <= key) {
      i++;
      if (i == high) {
        break;
      }
    }
    while (arr[j] >= key) {
      j--;
      if (j == low) {
        break;
      }
    }

    if (i >= j) break;
    T temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }

  arr[low] = arr[j];
  arr[j] = key;
  quickSort(arr, low, j - 1);
  quickSort(arr, j + 1, high);
}
int g_count = 0;
template <typename T>
bool isRealBox(const T xmin, const T ymin, const T xmax, const T ymax,
               const T im_h, const T im_w, bool pixel_offset, const T min_size,
               T *area) {
  bool is_real_box = false;
  float real_min_size = min_size > 1.0 ? min_size : 1.0;
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  T w = xmax - xmin + offset;
  T h = ymax - ymin + offset;

  if (pixel_offset) {
    T cx = xmin + w / 2.;
    T cy = ymin + h / 2.;

    if (w >= real_min_size && h >= real_min_size && cx <= im_w && cy <= im_h) {
      is_real_box = true;
    }
  } else {
    if (w >= real_min_size && h >= real_min_size) {
      is_real_box = true;
    }
  }

  if (is_real_box) {
    *area = w * h;
  }
  return is_real_box;
}

bool testf = true;
template <typename T>
void creatAndFilterProposalsBox(T *anchors_slice, T *bbox_deltas_slice,
                                T *im_shape_slice, T *variances_slice,
                                T *out_scores, T *out_proposals, T *out_areas,
                                const int A, const int H, const int W, const float min_size,
                                const float max_score, const int max_score_id,
                                bool pixel_offset, int *proposals_num) {
  // if(testf){
  //   testf = false;
  //   int test_indx = 4;
  //   printf("anchors[%d] (%f, %f, %f, %f)\n",test_indx,
  //   anchors_slice[4*test_indx], anchors_slice[4*test_indx+1],
  //   anchors_slice[4*test_indx + 2], anchors_slice[4*test_indx+3] );
  //   printf("variances_slice[%d] (%f, %f, %f, %f)\n",test_indx,
  //   variances_slice[4*test_indx], variances_slice[4*test_indx+1],
  //   variances_slice[4*test_indx + 2], variances_slice[4*test_indx+3] );
  //   printf("bbox_deltals[%d] (%f, %f, %f, %f)\n",
  //   test_indx,bbox_deltas_slice[test_indx], bbox_deltas_slice[1 * HWA
  //   +test_indx], bbox_deltas_slice[2 * HWA +test_indx], bbox_deltas_slice[3 *
  //   HWA +test_indx] );
  // }
  const int HWA = A*H*W;
  int k = max_score_id;
  // T axmin = anchors_slice[k];  // [A, H, W, 4]
  // T aymin = anchors_slice[1 * A * W * H + k];
  // T axmax = anchors_slice[2 * A * W * H + k];
  // T aymax = anchors_slice[3 * A * W * H + k];

  T axmin = anchors_slice[k*4];  // [A, H, W, 4]
  T aymin = anchors_slice[k*4+1];
  T axmax = anchors_slice[k*4+2];
  T aymax = anchors_slice[k*4+3];

  T offset = pixel_offset ? static_cast<T>(1.0) : 0;

  T w = axmax - axmin + offset;
  T h = aymax - aymin + offset;
  T cx = axmin + 0.5 * w;
  T cy = aymin + 0.5 * h;

  // int ta = 4 *(k%A);
  // T dxmin = bbox_deltas_slice[ta + 0 * W * H + k/A];  // [ 1, 4A , H, W]
  // T dymin = bbox_deltas_slice[ta + 1 * W * H + k/A];
  // T dxmax = bbox_deltas_slice[ta + 2 * W * H + k/A];
  // T dymax = bbox_deltas_slice[ta + 3 * W * H + k/A];

  T dxmin = bbox_deltas_slice[4*k];  // [ 1, 4 , 1, HWA]
  T dymin = bbox_deltas_slice[4*k+1];
  T dxmax = bbox_deltas_slice[4*k+2];
  T dymax = bbox_deltas_slice[4*k+3];

  T bbox_clip_default = static_cast<T>(kBBoxClipDefault);

  T d_cx, d_cy, d_w, d_h;
  if (variances_slice) {
    // printf("------var-----\n");
    d_cx = cx +
           dxmin * w * variances_slice[4 * k];  // variances_slice: [A, H, W, 4]
    d_cy = cy + dymin * h * variances_slice[4 * k + 1];
    d_w = exp(std::min(dxmax * variances_slice[4*k+2], bbox_clip_default)) *
          w;
    d_h = exp(std::min(dymax * variances_slice[4*k+3], bbox_clip_default)) *
          h;

  } else {
    d_cx = cx + dxmin * w;
    d_cy = cy + dymin * h;
    d_w = exp(std::min(dxmax, bbox_clip_default)) * w;
    d_h = exp(std::min(dymax, bbox_clip_default)) * h;
    // printf("variances\n");
  }
  // printf("anchor(%f, %f, %f, %f), deltal(%f, %f, %f, %f), var(%f, %f, %f, %f)\n",axmin, aymin, axmax, aymax, dxmin, dymin, dxmax, dymax, variances_slice[k*4], variances_slice[1 +k*4], variances_slice[2 +k*4], variances_slice[3 +k*4]);
  // printf("d_cx(%f, %f, %f, %f)\n", d_cx, d_cy, d_w, d_h);
  T oxmin = d_cx - d_w * 0.5;
  T oymin = d_cy - d_h * 0.5;
  T oxmax = d_cx + d_w * 0.5 - offset;
  T oymax = d_cy + d_h * 0.5 - offset;

  T p_xmin = std::max(std::min(oxmin, im_shape_slice[1] - offset), (T)0.);
  T p_ymin = std::max(std::min(oymin, im_shape_slice[0] - offset), (T)0.);
  T p_xmax = std::max(std::min(oxmax, im_shape_slice[1] - offset), (T)0.);
  T p_ymax = std::max(std::min(oymax, im_shape_slice[0] - offset), (T)0.);
  // printf("CPU dcxy(%f, %f) d_wh(%f, %f) ox(%f, %f, %f, %f), box(%f, %f, %f, %f), score= %f\n", d_cx, d_cy, d_w, d_h, oxmin, oymin, oxmax, oymax, p_xmin, p_ymin, p_xmax, p_ymax, max_score);

  T area = 0;
  // h =im_shape_slice[0], w = im_shape_slice[1]
  bool isValidBox = isRealBox(p_xmin, p_ymin, p_xmax, p_ymax, im_shape_slice[0],
                              im_shape_slice[1], pixel_offset, min_size, &area);
  // printf("isValidBox:%d \n", isValidBox);
  // printf("proposals_num:%d \n", proposals_num[0]);
  if (isValidBox) {
    int proposals_count = *proposals_num;
    // printf("proposals_count:%d ,max_score=%f\n", proposals_count,max_score);

    out_proposals[proposals_count * 4] = p_xmin;
    out_proposals[proposals_count * 4 + 1] = p_ymin;
    out_proposals[proposals_count * 4 + 2] = p_xmax;
    out_proposals[proposals_count * 4 + 3] = p_ymax;
    out_scores[proposals_count] = max_score;
    out_areas[proposals_count] = area;
    *proposals_num = *proposals_num + 1;
  } else {
     printf("remove box score:%f \n", max_score);
  }
}

template <typename T>
void findMaxScore(T *h_scores_buf, int size, T *max_score, int *max_score_id) {
  if (size == 0) {
    return;
  }
  T max_score_local = h_scores_buf[0];
  int max_score_id_local = 0;
  for (int i = 1; i < size; ++i) {
    if (h_scores_buf[i] > max_score_local) {
      max_score_local = h_scores_buf[i];
      max_score_id_local = i;
    }
  }
  *max_score = max_score_local;
  *max_score_id = max_score_id_local;
  return;
}

template <typename T>
T calcIoU(T *a, T *b, bool pixel_offset) {
  float offset = pixel_offset ? static_cast<float>(1.0) : 0;
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + offset, 0.f),
        height = max(bottom - top + offset, 0.f);
  float inter_s = width * height;
  float s_a = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float s_b = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return inter_s / (s_a + s_b - inter_s);
}

bool equal(float a, float b) { return abs(a - b) < 0.001; }

template <typename T>
void ProposalForOneImage(T *scores_slice, T *bbox_deltas_slice,
                         T *im_shape_slice, T *anchors_slice,
                         T *variances_slice, int H, int W, int A, T *rpn_rois,
                         T *rpn_roi_probs, int *one_image_proposal_num,
                         const int rpn_rois_batch_num, const int pre_nms_top_n,
                         const int post_nms_top_n, const T nms_thresh,
                         const T min_size, const bool pixel_offset) {
  const int HWA = A * H * W;
  int proposals_num = 0;

  int pre_nms_num =
      (pre_nms_top_n <= 0 || pre_nms_top_n > HWA) ? HWA : pre_nms_top_n;
  int post_nms_num = post_nms_top_n;
  if (post_nms_num > pre_nms_num) {
    post_nms_num = pre_nms_num;
  }
  printf("kBBoxClipDefault = %lf \n", kBBoxClipDefault);

  printf("cpu: HWA= %d, pre_nms_top_n = %d , pre_nms_num = %d\n", HWA,
         pre_nms_top_n, pre_nms_num);
  T k_score = 0.0f;
  T *out_scores_buf = new T[pre_nms_num];  // 设置四个buf, 从最上层接口传过来
  T *out_box_buf = new T[pre_nms_num * 4];
  T *out_area_buf = new T[pre_nms_num];
  T *temp_scores = new T[HWA];
  memcpy(temp_scores, scores_slice, HWA * sizeof(T));
  // top k, creatbox, filter box
  for (int top_id = 0; top_id < pre_nms_num; ++top_id) {
    int max_score_id = top_id;
    T max_score = scores_slice[max_score_id];

    findMaxScore(temp_scores, HWA, &max_score, &max_score_id);
    temp_scores[max_score_id] = PNMS_MIN;

    creatAndFilterProposalsBox<T>(
        anchors_slice, bbox_deltas_slice, im_shape_slice, variances_slice,
        out_scores_buf, out_box_buf, out_area_buf, A, H, W, min_size, max_score,
        max_score_id, pixel_offset, &proposals_num);
  }

  printf("creatAndFilterProposalsBox end: proposals_num: %d \n", proposals_num);
  // for( int i = 0; i < proposals_num; ++i){
  //   printf("score[%d]:%f, box(%f, %f, %f, %f)\n", i,  out_scores_buf[i],out_box_buf[i*4 +0],out_box_buf[i*4 + 1],out_box_buf[i*4+2],out_box_buf[i*4+3]);
  // }

  // for( int i = 0; i < proposals_num; ++i){
  //   printf("scores[%d]: %f \n", i, out_scores_buf[i]);
  // }

  // for( int i = 0; i < proposals_num; ++i){
  //   printf("box_area[%d]: %f \n", i, out_area_buf[i]);
  // }
  // nms
  // one_image_proposal_num;

  float offset = pixel_offset ? 1 : 0;
  int real_proposal_num = 0;
  int nms_num = std::min(proposals_num, post_nms_top_n);
  printf("cpu nms_num: %d ,proposals_num:%d\n", nms_num, proposals_num);
  for (int nms_id = 0; nms_id < nms_num; ++nms_id) {
    // Find max score
    float max_score = 0.0f;
    int max_score_id = 0;
    findMaxScore(out_scores_buf, proposals_num, &max_score, &max_score_id);
    out_scores_buf[max_score_id] = PNMS_MIN;

    // max_score < k_score
    if (max_score <= PNMS_MIN) { /**/
      break;
    }
    // Compute max area
    float max_score_x0 = out_box_buf[max_score_id * 4 + 0];
    float max_score_y0 = out_box_buf[max_score_id * 4 + 1];
    float max_score_x1 = out_box_buf[max_score_id * 4 + 2];
    float max_score_y1 = out_box_buf[max_score_id * 4 + 3];
    // save max score and box to output
    rpn_rois[(rpn_rois_batch_num + real_proposal_num) * 4 + 0] = max_score_x0;
    rpn_rois[(rpn_rois_batch_num + real_proposal_num) * 4 + 1] = max_score_y0;
    rpn_rois[(rpn_rois_batch_num + real_proposal_num) * 4 + 2] = max_score_x1;
    rpn_rois[(rpn_rois_batch_num + real_proposal_num) * 4 + 3] = max_score_y1;

    rpn_roi_probs[rpn_rois_batch_num + real_proposal_num] = max_score;
    real_proposal_num++;

    float max_score_area = out_area_buf[max_score_id];
    printf(
        "nms max_score: %f , index = %d, corrd: (%f, %f, %f, %f) "
        "max_score_area=%f\n",
        max_score, max_score_id, max_score_x0, max_score_y0, max_score_x1,
        max_score_y1, max_score_area);

    for (int inner_id = 0; inner_id < proposals_num; ++inner_id) {
      if (inner_id == max_score_id) {
        continue;
      }

      float *a = out_box_buf + max_score_id * 4;
      float *b = out_box_buf + inner_id * 4;
      float iou = calcIoU(a, b, pixel_offset);
      if (iou > nms_thresh) {
        if(out_scores_buf[inner_id] == 27.0) {
          printf("iou[%d] = %f, max_score = %f, nms_thresh = %f, scores=%f\n",inner_id, iou, out_scores_buf[inner_id], nms_thresh, out_scores_buf[inner_id]);
          printf("max_coor(%f, %f, %f, %f)\n",max_score_x0, max_score_y0, max_score_x1, max_score_y1);
          printf("ioubox coor(%f, %f, %f, %f)\n", out_box_buf[inner_id * 4 +
          0],  out_box_buf[inner_id * 4 + 1],  out_box_buf[inner_id * 4 + 2],
          out_box_buf[inner_id * 4 + 3]);
        }
        out_scores_buf[inner_id] = PNMS_MIN;
      }
    }
  }
  printf("nms end\n");
  *one_image_proposal_num = real_proposal_num;
  printf("nms end ,outputnum=%d\n", real_proposal_num);

  delete[] out_scores_buf;
  delete[] out_box_buf;
  delete[] out_area_buf;
  delete[] temp_scores;

  out_scores_buf = nullptr;
  out_box_buf = nullptr;
  out_area_buf = nullptr;
  temp_scores = nullptr;
}

void generateProposalsV2CPUImpl(
    float *scores, float *bbox_deltas, float *im_shape, float *anchors,
    float *variances, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    bool pixel_offset, const int N, const int A, const int H, const int W,
    float *rpn_rois, float *rpn_roi_probs, float *rpn_rois_num,
    float *rpn_rois_batch_size) {
  printf(
      "N,A,H,W: (%d, %d, %d, %d), pre_nms_top_n = %d, post_nms_top_n = %d \n",
      N, A, H, W, pre_nms_top_n, post_nms_top_n);
  printf("nms_thresh = %f, min_size = %f, pixel_offset = %d\n", nms_thresh,
         min_size, pixel_offset);

  const int HWA = A * H * W;
  int rpn_rois_batch_num = 0;

  for (int i = 0; i < N; ++i) {
    float *scores_slice = scores + i * HWA;
    float *bbox_deltas_slice = bbox_deltas + i * HWA * 4;
    float *im_shape_slice = im_shape + 2 * i;
    float *anchors_slice = anchors;      // [H, W, A, 4]
    float *variances_slice = variances;  // [H, W, A, 4]
    float *rpn_rois_slice = rpn_rois + rpn_rois_batch_num * 4;
    float *rpn_roi_probs_slice = rpn_roi_probs + rpn_rois_batch_num * 1;
    float *rpn_rois_num_slice = rpn_rois_num + rpn_rois_batch_num * 1;

    int one_image_proposal_num = 0;
    ProposalForOneImage<float>(
        scores_slice, bbox_deltas_slice, im_shape_slice, anchors_slice,
        variances_slice, H, W, A, rpn_rois, rpn_roi_probs,
        &one_image_proposal_num, rpn_rois_batch_num, pre_nms_top_n,
        post_nms_top_n, nms_thresh, min_size, pixel_offset);

    rpn_rois_batch_num += one_image_proposal_num;
    rpn_rois_num[i] = one_image_proposal_num;
    // for(int loop = 0; loop < rpn_rois_batch_num; ++loop) {
    //     printf("rpn_roi_probs[%d]: %f \n", loop, rpn_roi_probs[loop]);
    // }
  }
  *rpn_rois_batch_size = rpn_rois_batch_num;

  // printf("rpn_rois_batch_num: %d\n", rpn_rois_batch_num);
  // for( int i = 0; i < rpn_rois_batch_num; ++i){
  //   printf("proposal[%d]: corrd: (%f, %f, %f, %f) \n", i, rpn_rois[i*4 + 0],
  //   rpn_rois[i*4 + 1], rpn_rois[i*4+2], rpn_rois[i*4+3]);
  // }

  // for( int i = 0; i < rpn_rois_batch_num; ++i){
  //   printf("rpn_roi_probs[%d]: %f \n", i, rpn_roi_probs[i]);
  // }

  // for( int i = 0; i < N; ++i){
  //   printf("rpn_rois_num[%d]: %f \n", i, rpn_rois_num[i]);
  // }
}

}  // namespace GenerateProposalsV2