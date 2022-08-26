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

template <typename T>
bool isRealBox(const T xmin, const T ymin, const T xmax, const T ymax, const T im_h, const T im_w, bool pixel_offset, const T min_size, T *area){
    bool is_real_box = false;
    T offset = pixel_offset ? static_cast<T>(1.0) : 0;
    T w = xmax - xmin + offset;
    T h = ymax - ymin + offset;
    if (pixel_offset) {
      T cx = xmin + w / 2.;
      T cy = ymin + h / 2.;

      if (w >= min_size && h >= min_size && cx <= im_w && cy <= im_h) {
        is_real_box = true;
      }
    } else {
      if (w >= min_size && h >= min_size) {
        is_real_box = true;
      }
    }

    if(is_real_box){
        *area = w * h;
    }
    return is_real_box;
}

template <typename T>
void creatAndFilterProposalsBox(T *anchors_slice,
                                T *bbox_deltas_slice,
                                T *im_shape_slice,
                                T *variances_slice,
                                T *scores_buffer,
                                T *proposals,
                                T *area_buffer,
                                const int AHW,
                                float min_size,
                                float max_score,
                                int max_score_id,
                                bool pixel_offset,
                                int * proposals_num){
    T axmin = anchors_slice[max_score_id];
    T aymin = anchors_slice[1 * AHW + max_score_id];
    T axmax = anchors_slice[2 * AHW + max_score_id];
    T aymax = anchors_slice[3 * AHW + max_score_id];

    T offset  = pixel_offset ? 1.0 : 0;

    T w = axmax - axmin + offset;
    T h = aymax - aymin + offset;
    T cx = axmin + 0.5 * w;
    T cy = aymin + 0.5 * h;

    T dxmin = bbox_deltas_slice[max_score_id];
    T dymin = bbox_deltas_slice[1 * AHW + max_score_id];
    T dxmax = bbox_deltas_slice[2 * AHW + max_score_id];
    T dymax = bbox_deltas_slice[3 * AHW + max_score_id];

    T bbox_clip_default = static_cast<T>(kBBoxClipDefault);

    T d_cx, d_cy, d_w, d_h;
    if (variances_slice) {
      d_cx = cx + dxmin * w * variances_slice[0];
      d_cy = cy + dymin * h * variances_slice[1];
      d_w = exp(std::min(dxmax * variances_slice[2], bbox_clip_default)) * w;
      d_h = exp(std::min(dymax * variances_slice[3], bbox_clip_default)) * h;
    } else {
      d_cx = cx + dxmin * w;
      d_cy = cy + dymin * h;
      d_w = exp(std::min(dxmax, bbox_clip_default)) * w;
      d_h = exp(std::min(dymax, bbox_clip_default)) * h;
    }
    T oxmin = d_cx - d_w * 0.5;
    T oymin = d_cy - d_h * 0.5;
    T oxmax = d_cx + d_w * 0.5 - offset;
    T oymax = d_cy + d_h * 0.5 - offset;

    T p_xmin = std::max(std::min(oxmin, im_shape_slice[1] - offset), (T)0.);
    T p_ymin = std::max(std::min(oymin, im_shape_slice[0] - offset), (T)0.);
    T p_xmax = std::max(std::min(oxmax, im_shape_slice[1] - offset), (T)0.);
    T p_ymax = std::max(std::min(oymax, im_shape_slice[0] - offset), (T)0.);

    T area = 0;
    bool isValidBox = isRealBox(p_xmin, p_ymin, p_xmax, p_ymax, im_shape_slice[0], im_shape_slice[1], pixel_offset, min_size, &area);
    // printf("isValidBox:%d \n", isValidBox);
    // printf("proposals_num:%d \n", proposals_num[0]);
    if(isValidBox){
        int proposals_count = *proposals_num;
        // printf("proposals_count:%d ,max_score=%f\n", proposals_count,max_score);

        proposals[proposals_count * 4] = p_xmin;
        proposals[proposals_count * 4 + 1] = p_ymin;
        proposals[proposals_count * 4 + 2] = p_xmax;
        proposals[proposals_count * 4 + 3] = p_ymax;
        scores_buffer[proposals_count] = max_score;
        area_buffer[proposals_count] = area;
        *proposals_num = *proposals_num + 1;
    }
}


template <typename T>
void findMaxScore(T *h_scores_buf,
                int size,
                T *max_score,
                int *max_score_id) {
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
void  ProposalForOneImage(T *scores_slice,
                        T *bbox_deltas_slice,
                        T *im_shape_slice,
                        T *anchors_slice,
                        T *variances_slice,
                        int H,
                        int W,
                        int A,
                        T *rpn_rois,
                        T *rpn_roi_probs,
                        int *one_image_proposal_num,
                        const int rpn_rois_batch_num,
                        const int pre_nms_top_n,
                        const int post_nms_top_n,
                        const T nms_thresh,
                        const T min_size,
                        const bool pixel_offset){
  const int AHW = A*H*W;
  int proposals_num = 0;
  int space_size = min(AHW, pre_nms_top_n);
  T *temp_scores_buf = new T [space_size]; // 设置四个buf, 从最上层接口传过来
  T *temp_box_buf = new T [space_size*4]; 
  T *temp_area_buf  = new T [space_size]; 

  // top k, creatbox, filter box
  int top_num = std::min(pre_nms_top_n, AHW);
  for (int top_id = 0; top_id < top_num; ++top_id) {
    T max_score = 0.0f;
    int max_score_id = 0;
    findMaxScore(scores_slice, AHW, &max_score, &max_score_id);
    scores_slice[max_score_id] = PNMS_MIN;

    // printf("findMaxScore: topId: %d, max_score_index: %d, max_score: %f \n", top_id, max_score_id, max_score);

    creatAndFilterProposalsBox<T>(anchors_slice,
                                  bbox_deltas_slice,
                                  im_shape_slice,
                                  variances_slice,
                                  temp_scores_buf,
                                  temp_box_buf,
                                  temp_area_buf,
                                  AHW,
                                  min_size,
                                  max_score,
                                  max_score_id,
                                  pixel_offset,
                                  &proposals_num);
  }
  //  printf("creatAndFilterProposalsBox: proposals_num: %d \n", proposals_num);
  // for( int i = 0; i < proposals_num; ++i){
  //   printf("b ox[%d]: corrd: (%f, %f, %f, %f) \n", i, temp_box_buf[i*4 + 0],temp_box_buf[i*4 + 1],temp_box_buf[i*4+2],temp_box_buf[i*4+3]);  
  // }

  // for( int i = 0; i < proposals_num; ++i){
  //   printf("scores[%d]: %f \n", i, temp_scores_buf[i]);  
  // }

  // for( int i = 0; i < proposals_num; ++i){
  //   printf("box_area[%d]: %f \n", i, temp_area_buf[i]);  
  // }
  // nms
  // one_image_proposal_num;

  float offset = pixel_offset ? 1 : 0;
  int real_proposal_num = 0;
  int nms_num = std::min(proposals_num, post_nms_top_n);
  printf("nms_num: %d \n", nms_num);
  for (int nms_id = 0; nms_id < nms_num; ++nms_id) {
    // Find max score
    float max_score = 0.0f;
    int max_score_id = 0;
    findMaxScore(temp_scores_buf, proposals_num, &max_score, &max_score_id);
    if (max_score <= PNMS_MIN) {
      break;
    }
    // printf("nms max_score: %f \n", max_score);

    rpn_rois[rpn_rois_batch_num + real_proposal_num * 4 + 0] = temp_box_buf[max_score_id * 4 + 0];
    rpn_rois[rpn_rois_batch_num + real_proposal_num * 4 + 1] = temp_box_buf[max_score_id * 4 + 1];
    rpn_rois[rpn_rois_batch_num + real_proposal_num * 4 + 2] = temp_box_buf[max_score_id * 4 + 2];
    rpn_rois[rpn_rois_batch_num + real_proposal_num * 4 + 3] = temp_box_buf[max_score_id * 4 + 3];

    rpn_roi_probs[rpn_rois_batch_num + real_proposal_num] = max_score;
    real_proposal_num++;

    temp_scores_buf[max_score_id] = PNMS_MIN;

    // Compute max area
    float max_score_x0 = temp_box_buf[max_score_id * 4 + 0];
    float max_score_y0 = temp_box_buf[max_score_id * 4 + 1];
    float max_score_x1 = temp_box_buf[max_score_id * 4 + 2];
    float max_score_y1 = temp_box_buf[max_score_id * 4 + 3];

    float max_score_area = temp_area_buf[max_score_id];
    // if(max_score_area == 0){
    //     printf("corrd: (%f, %f, %f, %f) \n", max_score_x0, max_score_y0, max_score_x1, max_score_y1);  
    // }

    for (int inner_id = 0; inner_id < proposals_num; ++inner_id) {
      if (inner_id == max_score_id) {
        continue;
      }

      // Compute inter area
      float inter_x0 = std::max(max_score_x0, temp_box_buf[inner_id * 4 + 0]);
      float inter_y0 = std::max(max_score_y0, temp_box_buf[inner_id * 4 + 1]);
      float inter_x1 = std::min(max_score_x1, temp_box_buf[inner_id * 4 + 2]);
      float inter_y1 = std::min(max_score_y1, temp_box_buf[inner_id * 4 + 3]);

      float inter_area = std::max(0.0f, inter_x1 - inter_x0 + offset) * std::max(0.0f, inter_y1 - inter_y0 + offset) ;
      // printf("inter_area:%f , max_score_area : %f, temp_area_buf[inner_id]: %f, inter_area: %f\n", inter_area, max_score_area, temp_area_buf[inner_id], inter_area);
      float iou = inter_area / (max_score_area + temp_area_buf[inner_id] - inter_area);
      // printf("iou:%f\n", iou);
      // if(iou < 0 ){
      //   printf("inter_area:%f , max_score_area : %f, temp_area_buf[inner_id]: %f, inter_area: %f\n", inter_area, max_score_area, temp_area_buf[inner_id], inter_area);
      //   printf("corrd max: (%f, %f, %f, %f) \n", max_score_x0, max_score_y0, max_score_x1, max_score_y1);  
      //   printf("corrd temp: (%f, %f, %f, %f) \n", temp_box_buf[inner_id * 4 + 0], temp_box_buf[inner_id * 4 + 1], temp_box_buf[inner_id * 4 + 2], temp_box_buf[inner_id * 4 + 3]);  
      //   printf("u:%f\n",  (max_score_area + temp_area_buf[inner_id] - inter_area));
      //   return;
      // }
      if (iou > nms_thresh) {
          temp_scores_buf[inner_id] = PNMS_MIN;
      }
    }
  }
  printf("nms end\n");
  *one_image_proposal_num = real_proposal_num;

  delete[] temp_scores_buf;
  delete[] temp_box_buf;
  delete[] temp_area_buf;
  temp_scores_buf = nullptr;
  temp_box_buf = nullptr;
  temp_area_buf = nullptr;
  printf("nms end 1\n");

}

void generateProposalsV2CPUImpl(float *scores,
                               float *bbox_deltas,
                               float * im_shape,
                               float * anchors,
                               float * variances,
                               const int pre_nms_top_n,
                               const int post_nms_top_n,
                               const float nms_thresh,
                               const float min_size,
                               const float eta,
                               bool pixel_offset,
                               const int N,
                               const int A,
                               const int H,
                               const int W,
                               float* rpn_rois,
                               float* rpn_roi_probs,
                               float* rpn_rois_num,
                               float* rpn_rois_batch_size){
  cout << "N:" << N << endl;
  cout << "A:" << A << endl;
  cout << "H:" << H << endl;
  cout << "W:" << W << endl;

  const int AHW = A*W*H;
  int rpn_rois_batch_num = 0;

  for(int i = 0 ; i < N; ++i){
      float *scores_slice = scores + i * AHW;
      float *bbox_deltas_slice = bbox_deltas + i * AHW * 4;
      float *im_shape_slice = im_shape + 2 * i;
      float *anchors_slice = anchors; // [A, H, W, 4]
      float *variances_slice = variances; // [A, H, W, 4]
      float *rpn_rois_slice = rpn_rois + rpn_rois_batch_num*4;
      float *rpn_roi_probs_slice = rpn_roi_probs + rpn_rois_batch_num*1;
      float *rpn_rois_num_slice = rpn_rois_num + rpn_rois_batch_num*1;

      int one_image_proposal_num = 0;
      ProposalForOneImage<float>(scores_slice,
                                bbox_deltas_slice,
                                im_shape_slice,
                                anchors_slice,
                                variances_slice,
                                H,
                                W,
                                A,
                                rpn_rois,
                                rpn_roi_probs,
                                &one_image_proposal_num,
                                rpn_rois_batch_num,
                                pre_nms_top_n,
                                post_nms_top_n,
                                nms_thresh,
                                min_size,
                                pixel_offset);

      rpn_rois_batch_num += one_image_proposal_num;
      rpn_rois_num[i] = one_image_proposal_num;
  }
  *rpn_rois_batch_size = rpn_rois_batch_num;

  // printf("rpn_rois_batch_num: %d\n", rpn_rois_batch_num);
  // for( int i = 0; i < rpn_rois_batch_num; ++i){
  //   printf("proposal[%d]: corrd: (%f, %f, %f, %f) \n", i, rpn_rois[i*4 + 0], rpn_rois[i*4 + 1], rpn_rois[i*4+2], rpn_rois[i*4+3]);  
  // }

  // for( int i = 0; i < rpn_rois_batch_num; ++i){
  //   printf("rpn_roi_probs[%d]: %f \n", i, rpn_roi_probs[i]);  
  // }

  // for( int i = 0; i < N; ++i){
  //   printf("rpn_rois_num[%d]: %f \n", i, rpn_rois_num[i]);  
  // }
}

} // namespace GenerateProposalsV2