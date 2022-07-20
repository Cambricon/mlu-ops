/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef KERNELS_PNMS_PNMS_DETECTION_H_
#define KERNELS_PNMS_PNMS_DETECTION_H_
#include "float.h"
#define NMS_SIZE 64
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y

#define PNMS_MIN (-(float)FLT_MAX)

template <typename IN_DT>
__mlu_func__ void quickSort(IN_DT *arr, int low, int high) {
  if (high <= low) return;
  int i = low;
  int j = high;
  int key = arr[low];
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
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }

  arr[low] = arr[j];
  arr[j] = key;
  quickSort(arr, low, j - 1);
  quickSort(arr, j + 1, high);
}

template <typename IN_DT>
__mlu_func__ void absBoxesArea(IN_DT *area, const int length) {
  __bang_active_abs((IN_DT *)area, (IN_DT *)area, length);
}

// calculate max_score_box_area
// output: max_score| index | coordinate | sign box area | box area
// box_area=1/2 * ((x1*y2 - y1*x2) + (x2*y3-y2*x3) + (x3*y4 - y3*x4) + (x4*y1 -
// y4*x1))
template <typename IN_DT>
__mlu_func__ void calculateMaxScoreBoxArea(IN_DT *max_box) {
  auto max_box_coordinate = max_box + 2;
  max_box_coordinate[8] = max_box_coordinate[0];
  max_box_coordinate[9] = max_box_coordinate[1];
  auto max_area = 0.0;
  for (int j = 0; j < 8; j = j + 2) {
    max_area += (max_box_coordinate[j] * max_box_coordinate[j + 3] -
                 max_box_coordinate[j + 1] * max_box_coordinate[j + 2]);
  }
  max_area = max_area / 2;
  max_box_coordinate[8] = max_area;
  max_area = max_area > 0 ? max_area : -max_area;
  max_box_coordinate[9] = max_area;

  // if (max_area < 0) reverse coordinate  ABCD-->DCBA
  max_area = max_box[10];  //  max_box[10] sign max score box area
  if (max_area < 0) {
    auto tmp_x = max_box[2];
    auto tmp_y = max_box[3];
    max_box[2] = max_box[8];
    max_box[3] = max_box[9];
    max_box[8] = tmp_x;
    max_box[9] = tmp_y;

    tmp_x = max_box[4];
    tmp_y = max_box[5];
    max_box[4] = max_box[6];
    max_box[5] = max_box[7];
    max_box[6] = tmp_x;
    max_box[7] = tmp_y;
  }
}

// >0 = 1, <0 = -1, ==0 = 0
// const float eps = 1E-8;  return (d > eps) - (d < -eps);
template <typename IN_DT>
__mlu_func__ void sig(IN_DT *output, IN_DT *input, IN_DT *tmp1, IN_DT *tmp2,
                      const int max_seg_num, const int actual_box_num) {
  int length = max_seg_num;
  // __bang_write_value(output, length, float(0.0));
  // __bang_write_value(tmp1, length, float(0.0));
  // __bang_write_value(tmp2, length, float(0.0));

  // __bang_gt(tmp1, input, output, length);
  // __bang_lt(tmp2, input, output, length);
  // __bang_sub(output, tmp1, tmp2, length);

  __bang_write_value(output, length, float(1E-8));
  __bang_write_value(tmp1, length, float(-1E-8));
  __bang_write_value(tmp2, length, float(0.0));

  __bang_gt(tmp2, input, output, length);
  __bang_lt(output, input, tmp1, length);
  __bang_sub(output, tmp2, output, length);
}

template <typename IN_DT>
__mlu_func__ void crossP3(IN_DT *result, IN_DT *p1_x, IN_DT *p1_y, IN_DT *p2_x,
                          IN_DT *p2_y, IN_DT *p3_x, IN_DT *p3_y,
                          const int length, IN_DT *temp1_ram,
                          IN_DT *temp2_ram) {
  // crossP3<T>(o, a, b) = (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y -
  // o.y);
  // a.x - o.x
  __bang_sub((IN_DT *)result, (IN_DT *)p2_x, (IN_DT *)p1_x, length);
  // b.y - o.y
  __bang_sub((IN_DT *)temp2_ram, (IN_DT *)p3_y, (IN_DT *)p1_y, length);
  // A =  (a.x - o.x) * (b.y - o.y)
  __bang_mul((IN_DT *)result, (IN_DT *)result, (IN_DT *)temp2_ram, length);
  // (b.x - o.x)
  __bang_sub((IN_DT *)temp1_ram, (IN_DT *)p3_x, (IN_DT *)p1_x, length);
  // (a.y - o.y)
  __bang_sub((IN_DT *)temp2_ram, (IN_DT *)p2_y, (IN_DT *)p1_y, length);
  // B = (b.x - o.x) * (a.y - o.y)
  __bang_mul((IN_DT *)temp1_ram, (IN_DT *)temp1_ram, (IN_DT *)temp2_ram,
             length);
  // A-B
  __bang_sub((IN_DT *)result, (IN_DT *)result, (IN_DT *)temp1_ram, length);
}

template <typename IN_DT>
__mlu_func__ void updatePAndPCount(IN_DT *p_x[], IN_DT *p_y[], IN_DT *p_count,
                                   IN_DT *pp_x, IN_DT *pp_y, IN_DT *px_ram,
                                   IN_DT *py_ram, IN_DT *buffer,
                                   const int pp_count, const int max_seg_num,
                                   const int actual_box_num) {
  // n = 0;
  // for (int i = 0; i < m; i++)
  //     if (!i || !(pp[i] == pp[i - 1]))
  //         p[n++] = pp[i];
  // while (n > 1 && p[n - 1] == p[0])
  //     n--;
  __bang_write_value(p_count, max_seg_num, float(0));
  __bang_write_value(px_ram, 30 * max_seg_num, float(0));
  __bang_write_value(py_ram, 30 * max_seg_num, float(0));
  bool pp_vaild = false;
  for (int i = 0; i < actual_box_num; ++i) {
    bool first_loop = true;
    for (int j = 0; j < pp_count; ++j) {
      float valid_pp0_x = 0.0;
      float valid_pp0_y = 0.0;
      for (int t = j; t < pp_count; t++) {
        if (pp_x[i + t * max_seg_num] == PNMS_MIN ||
            pp_y[i + t * max_seg_num] == PNMS_MIN) {
          continue;
        } else {
          valid_pp0_x = pp_x[i + t * max_seg_num];
          valid_pp0_y = pp_y[i + t * max_seg_num];
          j = t;
          pp_vaild = true;
          break;
        }
      }

      if (pp_vaild == false) {
        break;
      }

      if (first_loop) {
        first_loop = false;
        px_ram[uint32_t(p_count[i]) * max_seg_num + i] = valid_pp0_x;
        py_ram[uint32_t(p_count[i]) * max_seg_num + i] = valid_pp0_y;
        p_count[i] = p_count[i] + 1;
        j--;
        continue;
      }

      if (j == pp_count - 1) {
        break;
      }
      j++;
      pp_vaild = false;
      auto valid_pp1_x = 0.0;
      auto valid_pp1_y = 0.0;

      for (int t = j; t < pp_count; t++) {
        if (pp_x[i + t * max_seg_num] == PNMS_MIN ||
            pp_y[i + t * max_seg_num] == PNMS_MIN) {
          continue;
        } else {
          valid_pp1_x = pp_x[i + t * max_seg_num];
          valid_pp1_y = pp_y[i + t * max_seg_num];
          pp_vaild = true;
          j = t;
          break;
        }
      }

      if (pp_vaild == false) {
        break;
      }

      if (valid_pp0_x != valid_pp1_x || valid_pp0_y != valid_pp1_y) {
        px_ram[uint32_t(p_count[i]) * max_seg_num + i] = valid_pp1_x;
        py_ram[uint32_t(p_count[i]) * max_seg_num + i] = valid_pp1_y;
        p_count[i] = p_count[i] + 1;
      }
      j--;
    }
  }

  // while (n > 1 && p[n - 1] == p[0]) n--;
  for (int i = 0; i < actual_box_num; ++i) {
    int n = uint32_t(p_count[i]);
    while (n > 1 && px_ram[(n - 1) * max_seg_num + i] == px_ram[i] &&
           py_ram[(n - 1) * max_seg_num + i] == py_ram[i]) {
      p_count[i] = p_count[i] - 1;
      n--;
    }
  }
  for (int i = 0; i < actual_box_num; ++i) {
    int n = uint32_t(p_count[i]);
    px_ram[n * max_seg_num + i] = px_ram[i];
    py_ram[n * max_seg_num + i] = py_ram[i];
  }

  uint32_t p_count_max = 0;
  __bang_max(buffer, p_count, max_seg_num);
  p_count_max = uint32_t(((float *)buffer)[0]);
  for (int j = 0; j < p_count_max + 1; ++j) {
    p_x[j] = px_ram + j * max_seg_num;
    p_y[j] = py_ram + j * max_seg_num;
  }
}

template <typename IN_DT>
__mlu_func__ void points_swap(IN_DT *boxes_pts_x0, IN_DT *boxes_pts_y0,
                              IN_DT *boxes_pts_x1, IN_DT *boxes_pts_y1,
                              const int idx) {
  auto tmp = boxes_pts_x0[idx];
  boxes_pts_x0[idx] = boxes_pts_x1[idx];
  boxes_pts_x1[idx] = tmp;

  tmp = boxes_pts_y0[idx];
  boxes_pts_y0[idx] = boxes_pts_y1[idx];
  boxes_pts_y1[idx] = tmp;
}

template <typename IN_DT>
__mlu_func__ void points_reverse(IN_DT *boxes_pts_x, IN_DT *boxes_pts_y,
                                 const int idx, const int max_seg_num) {
  auto tmp_x = boxes_pts_x[idx];
  auto tmp_y = boxes_pts_y[idx];
  boxes_pts_x[idx] = boxes_pts_x[idx + 3 * max_seg_num];
  boxes_pts_y[idx] = boxes_pts_y[idx + 3 * max_seg_num];
  boxes_pts_x[idx + 3 * max_seg_num] = tmp_x;
  boxes_pts_y[idx + 3 * max_seg_num] = tmp_y;

  tmp_x = boxes_pts_x[idx + 1 * max_seg_num];
  tmp_y = boxes_pts_y[idx + 1 * max_seg_num];
  boxes_pts_x[idx + 1 * max_seg_num] = boxes_pts_x[idx + 2 * max_seg_num];
  boxes_pts_y[idx + 1 * max_seg_num] = boxes_pts_y[idx + 2 * max_seg_num];
  boxes_pts_x[idx + 2 * max_seg_num] = tmp_x;
  boxes_pts_y[idx + 2 * max_seg_num] = tmp_y;
}

template <typename IN_DT>
__mlu_func__ void calPolygonSignArea(IN_DT *ret, IN_DT *p_x[], IN_DT *p_y[],
                                     IN_DT *count, const int actual_box_num) {
  __bang_write_value(ret, actual_box_num, float(0));
  for (int j = 0; j < actual_box_num; j++) {
    uint32_t n = uint32_t(count[j]);
    p_x[n][j] = p_x[0][j];
    p_y[n][j] = p_y[0][j];

    for (int i = 0; i < n; i++) {
      ret[j] += p_x[i][j] * p_y[i + 1][j] - p_y[i][j] * p_x[i + 1][j];
    }
    ret[j] = 0.5 * ret[j];
  }
}

template <typename IN_DT>
__mlu_func__ void computeDiv(IN_DT *result, IN_DT *melo, IN_DT *denom,
                             IN_DT *denom_tmp, const int actual_box_num) {
#if __BANG_ARCH__ == 372
  __bang_recip((float *)denom_tmp, (float *)denom, actual_box_num);
  __bang_mul((float *)result, (float *)melo, (float *)denom_tmp,
             actual_box_num);
#elif __BANG_ARCH__ >= 322
  __bang_div((float *)result, (float *)melo, (float *)denom, actual_box_num);
#else
  __bang_active_reciphp((float *)denom_tmp, (float *)denom, actual_box_num);
  __bang_mul((float *)result, (float *)melo, (float *)denom_tmp,
             actual_box_num);
#endif
}

template <typename IN_DT>
__mlu_func__ void lineCross(IN_DT *a_x, IN_DT *a_y, IN_DT *b_x, IN_DT *b_y,
                            IN_DT *p1_x, IN_DT *p1_y, IN_DT *p2_x, IN_DT *p2_y,
                            IN_DT *pp_x, IN_DT *pp_y, IN_DT *valid_pts,
                            uint32_t pp_count, IN_DT *cross_s1, IN_DT *cross_s2,
                            IN_DT *sig_cross_s1, IN_DT *sig_cross_s2,
                            IN_DT *nram_tmp, const int actual_box_num,
                            const int max_seg_num) {
  // 7
  // __bang_printf("line cross start, a(%f, %f), b(%f, %f), c(%f, %f)m, d(%f,
  // %f)\n", a_x[0],
  // a_y[0], b_x[0], b_y[0],p1_x[0], p1_y[0],p2_x[0], p2_y[0]);
  IN_DT *p_tmp1;
  IN_DT *p_tmp2;
  IN_DT *p_melo;
  IN_DT *p_denom;
  IN_DT *tmp_zero;
  IN_DT *p_denom_tmp;
  IN_DT *mask_sig_eq0;

  p_tmp1 = nram_tmp;
  p_tmp2 = p_tmp1 + 1 * max_seg_num;
  p_melo = p_tmp1 + 2 * max_seg_num;
  p_denom = p_tmp1 + 3 * max_seg_num;
  tmp_zero = p_tmp1 + 4 * max_seg_num;
  p_denom_tmp = p_tmp1 + 5 * max_seg_num;
  mask_sig_eq0 = p_tmp1 + 6 * max_seg_num;

  // if (sig(s1) == 0 && sig(s2) == 0) return
  __bang_write_value(tmp_zero, max_seg_num, float(0));
  __bang_ne(p_tmp1, sig_cross_s1, tmp_zero, max_seg_num);
  __bang_ne(p_tmp2, sig_cross_s2, tmp_zero, max_seg_num);
  __bang_or(mask_sig_eq0, p_tmp1, p_tmp2, max_seg_num);

  //  if (sig(s2 - s1) == 0) return
  __bang_sub(p_tmp1, cross_s2, cross_s1, max_seg_num);
  sig(p_tmp2, p_tmp1, p_melo, p_denom, max_seg_num, actual_box_num);
  __bang_ne(valid_pts, p_tmp2, tmp_zero, max_seg_num);
  //  if (sig(s2 - s1) == 0) return  ||  (sig(s1) == 0 && sig(s2) == 0) return
  __bang_mul(valid_pts, valid_pts, mask_sig_eq0, max_seg_num);  // and->or

  // line cross a,b,p0,p1
  // p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  __bang_mul((float *)p_tmp1, (float *)p1_x, (float *)cross_s2, actual_box_num);
  __bang_mul(p_tmp2, p2_x, cross_s1, actual_box_num);
  __bang_sub(p_melo, p_tmp1, p_tmp2, actual_box_num);

  // s2 - s1
  __bang_sub(p_denom, cross_s2, cross_s1, actual_box_num);
  // set 1 with (s2 -s1 == 0)
  // //s2-s1=0的box无效，计算时候先让s2-s1=0，然后再用mask把这部分去掉
  __bang_eq(p_denom_tmp, p_denom, tmp_zero, actual_box_num);
  __bang_add(p_denom, p_denom, p_denom_tmp, actual_box_num);

  // compute div
  computeDiv((float *)(pp_x + pp_count * max_seg_num), (float *)p_melo,
             (float *)p_denom, (float *)p_denom_tmp, actual_box_num);
  // p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  __bang_mul(p_tmp1, p1_y, cross_s2, actual_box_num);
  __bang_mul(p_tmp2, p2_y, cross_s1, actual_box_num);
  __bang_sub(p_melo, p_tmp1, p_tmp2, actual_box_num);
  // compute div
  computeDiv((float *)(pp_y + pp_count * max_seg_num), (float *)p_melo,
             (float *)p_denom, (float *)p_denom_tmp, actual_box_num);
}

template <typename IN_DT>
__mlu_func__ void polygon_cut(IN_DT *p_x[], IN_DT *p_y[], IN_DT *p_count,
                              IN_DT *a_x, IN_DT *a_y, IN_DT *b_x, IN_DT *b_y,
                              IN_DT *pp_x, IN_DT *pp_y, IN_DT *buffer,
                              const int actual_box_num, const int max_seg_num) {
  IN_DT *cross_s1;      // 1
  IN_DT *cross_s2;      // 1
  IN_DT *sig_cross_s1;  // 1
  IN_DT *sig_cross_s2;  // 1
  IN_DT *tmp_zero;      // 1
  IN_DT *mask_sig_ne;   // 1
  IN_DT *valid_pts;     // 1
  IN_DT *px_ram;        // 10
  IN_DT *py_ram;        // 10
  IN_DT *s1_tmp1;       // 1
  IN_DT *s1_tmp2;       // 1
  IN_DT *tmp1;          // 1
  IN_DT *invalid_pts;   // 1
  IN_DT *nram_tmp;

  cross_s1 = buffer;
  cross_s2 = cross_s1 + 1 * max_seg_num;
  sig_cross_s1 = cross_s1 + 2 * max_seg_num;
  sig_cross_s2 = cross_s1 + 3 * max_seg_num;
  tmp_zero = cross_s1 + 4 * max_seg_num;
  mask_sig_ne = cross_s1 + 5 * max_seg_num;
  valid_pts = cross_s1 + 6 * max_seg_num;
  px_ram = cross_s1 + 7 * max_seg_num;   // 10
  py_ram = cross_s1 + 17 * max_seg_num;  // 10
  s1_tmp1 = cross_s1 + 27 * max_seg_num;
  s1_tmp2 = cross_s1 + 28 * max_seg_num;
  tmp1 = cross_s1 + 29 * max_seg_num;
  invalid_pts = cross_s1 + 30 * max_seg_num;
  nram_tmp = cross_s1 + 31 * max_seg_num;

  uint32_t p_count_max = 0;
  __bang_max(tmp1, p_count, max_seg_num);
  p_count_max = uint32_t(((float *)tmp1)[0]);

  int pp_count = 0;

  // 按照p_count最大值取循环，保证每个位置对应的点都计算在内。
  // 需要考虑的是pp_count都不相同, 需要把无效值置为PNMS_MIN
  for (int n = 0; n < p_count_max; n++) {
    // cross(a,b,p[i])
    crossP3(cross_s1, a_x, a_y, b_x, b_y, p_x[n], p_y[n], actual_box_num,
            s1_tmp1, s1_tmp2);
    // cross(a,b,p[i+1])
    crossP3(cross_s2, a_x, a_y, b_x, b_y, p_x[n + 1], p_y[n + 1],
            actual_box_num, s1_tmp1, s1_tmp2);

    sig(sig_cross_s1, cross_s1, s1_tmp1, s1_tmp2, max_seg_num, actual_box_num);
    sig(sig_cross_s2, cross_s2, s1_tmp1, s1_tmp2, max_seg_num, actual_box_num);

    //  if (sig(cross(a, b, p[i])) > 0) pp[m++] = p[i]
    __bang_write_value(tmp_zero, max_seg_num, float(0));
    __bang_gt(valid_pts, sig_cross_s1, tmp_zero, max_seg_num);
    // pp = sig_gt_ret *  p[n];
    __bang_mul(pp_x + pp_count * max_seg_num, valid_pts, p_x[n],
               actual_box_num);
    __bang_mul(pp_y + pp_count * max_seg_num, valid_pts, p_y[n],
               actual_box_num);

    // s1 <= 0, set PNMS_MIN with unvailds pp_x, pp_y
    __bang_eq(invalid_pts, valid_pts, tmp_zero, max_seg_num);
    __bang_mul_const(invalid_pts, invalid_pts, float(PNMS_MIN), max_seg_num);
    __bang_add(pp_x + pp_count * max_seg_num, pp_x + pp_count * max_seg_num,
               invalid_pts, actual_box_num);
    __bang_add(pp_y + pp_count * max_seg_num, pp_y + pp_count * max_seg_num,
               invalid_pts, actual_box_num);
    pp_count++;

    // if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
    __bang_ne(mask_sig_ne, sig_cross_s1, sig_cross_s2, actual_box_num);

    lineCross(a_x, a_y, b_x, b_y, p_x[n], p_y[n], p_x[n + 1], p_y[n + 1], pp_x,
              pp_y, valid_pts, pp_count, cross_s1, cross_s2, sig_cross_s1,
              sig_cross_s2, nram_tmp, actual_box_num, max_seg_num);

    // valid_pts = valid_pts || if (sig(cross(a, b, p[i])) != sig(cross(a, b,
    // p[i + 1])))
    __bang_mul(valid_pts, valid_pts, mask_sig_ne, max_seg_num);
    __bang_mul(pp_x + pp_count * max_seg_num, pp_x + pp_count * max_seg_num,
               valid_pts, actual_box_num);
    __bang_mul(pp_y + pp_count * max_seg_num, pp_y + pp_count * max_seg_num,
               valid_pts, actual_box_num);

    // set PNMS_MIN with unvailds pp_x, pp_y
    __bang_eq(invalid_pts, valid_pts, tmp_zero, max_seg_num);
    __bang_mul_const(invalid_pts, invalid_pts, float(PNMS_MIN), max_seg_num);
    __bang_add(pp_x + pp_count * max_seg_num, pp_x + pp_count * max_seg_num,
               invalid_pts, actual_box_num);
    __bang_add(pp_y + pp_count * max_seg_num, pp_y + pp_count * max_seg_num,
               invalid_pts, actual_box_num);
    pp_count++;
  }  // for(int n = 0;n<p_count;n++)

  // px,py,pcount
  updatePAndPCount(p_x, p_y, p_count, pp_x, pp_y, px_ram, py_ram, nram_tmp,
                   pp_count, max_seg_num, actual_box_num);
  // __bang_printf("--polygon cut out, p_count[0]=%d\n",  uint32_t(p_count[0]));
}

template <typename IN_DT>
__mlu_func__ void intersectArea(
    IN_DT *area, const IN_DT *max_box_pts_x0_tmp,
    const IN_DT *max_box_pts_y0_tmp, const IN_DT *max_box_pts_x1_tmp,
    const IN_DT *max_box_pts_y1_tmp, const IN_DT *box_pts_x0_tmp,
    const IN_DT *box_pts_y0_tmp, const IN_DT *box_pts_x1_tmp,
    const IN_DT *box_pts_y1_tmp, IN_DT *buffer, const int max_seg_num,
    const int actual_box_num) {
  // 14 + 60 + buffer
  IN_DT *o_x;             // 1
  IN_DT *o_y;             // 1
  IN_DT *box_pts_x0;      // 1 数值需要交换
  IN_DT *box_pts_y0;      // 1
  IN_DT *box_pts_x1;      // 1
  IN_DT *box_pts_y1;      // 1
  IN_DT *max_box_pts_x0;  // 1
  IN_DT *max_box_pts_y0;  // 1
  IN_DT *max_box_pts_x1;
  IN_DT *max_box_pts_y1;
  IN_DT *s1;
  IN_DT *s2;
  IN_DT *p_count;
  IN_DT *mask_s1_s2_eqf1;
  IN_DT *pp_x;  // 30
  IN_DT *pp_y;  // 30
  IN_DT *p_x[10];
  IN_DT *p_y[10];
  IN_DT *nram_tmp;

  IN_DT *s_c1;
  IN_DT *s_c2;
  IN_DT *s_tmp1;
  IN_DT *s_tmp2;
  IN_DT *temp3_ram;
  IN_DT *mask_vaild_pts;
  IN_DT *mask_s1_eq0;   // = s_tmp1;
  IN_DT *mask_s2_eq0;   // = s_tmp2;
  IN_DT *mask_s1_eqF1;  // = s_tmp1;
  IN_DT *mask_s2_eqF1;  // = s_tmp2;

  o_x = (IN_DT *)buffer;
  o_y = o_x + max_seg_num;
  box_pts_x0 = o_y + max_seg_num;                 // 1
  box_pts_y0 = box_pts_x0 + max_seg_num;          // 1
  box_pts_x1 = box_pts_y0 + max_seg_num;          // 1
  box_pts_y1 = box_pts_x1 + max_seg_num;          // 1
  max_box_pts_x0 = box_pts_y1 + max_seg_num;      // 1
  max_box_pts_y0 = max_box_pts_x0 + max_seg_num;  // 1
  max_box_pts_x1 = max_box_pts_y0 + max_seg_num;  // 1
  max_box_pts_y1 = max_box_pts_x1 + max_seg_num;  // 1

  s1 = max_box_pts_y1 + max_seg_num;
  s2 = s1 + max_seg_num;
  p_count = s2 + max_seg_num;
  mask_s1_s2_eqf1 = p_count + max_seg_num;
  mask_vaild_pts = mask_s1_s2_eqf1 + max_seg_num;
  pp_x = mask_vaild_pts + max_seg_num;
  pp_y = pp_x + 30 * max_seg_num;
  nram_tmp = pp_y + 30 * max_seg_num;

  s_c1 = nram_tmp;
  s_c2 = s_c1 + max_seg_num;
  s_tmp1 = s_c2 + max_seg_num;
  s_tmp2 = s_tmp1 + max_seg_num;
  temp3_ram = s_tmp2 + max_seg_num;
  mask_s1_eq0 = temp3_ram + max_seg_num;
  mask_s2_eq0 = mask_s1_eq0 + max_seg_num;
  mask_s1_eqF1 = mask_s2_eq0 + max_seg_num;
  mask_s2_eqF1 = mask_s1_eqF1 + max_seg_num;

  __bang_write_value(o_x, 74 * max_seg_num, float(0.0));

  __memcpy(box_pts_x0, box_pts_x0_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(box_pts_y0, box_pts_y0_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(box_pts_x1, box_pts_x1_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(box_pts_y1, box_pts_y1_tmp, max_seg_num * 4, NRAM2NRAM);

  __memcpy(max_box_pts_x0, max_box_pts_x0_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(max_box_pts_y0, max_box_pts_y0_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(max_box_pts_x1, max_box_pts_x1_tmp, max_seg_num * 4, NRAM2NRAM);
  __memcpy(max_box_pts_y1, max_box_pts_y1_tmp, max_seg_num * 4, NRAM2NRAM);

  // a b max_box_pts, cd  box_pts
  // corss_oab
  crossP3(s_c1, o_x, o_y, max_box_pts_x0, max_box_pts_y0, max_box_pts_x1,
          max_box_pts_y1, actual_box_num, s_tmp1, s_tmp2);
  // corss_ocd
  crossP3(s_c2, o_x, o_y, box_pts_x0, box_pts_y0, box_pts_x1, box_pts_y1,
          actual_box_num, s_tmp1, s_tmp2);
  sig(s1, s_c1, s_tmp1, s_tmp2, max_seg_num, actual_box_num);
  sig(s2, s_c2, s_tmp1, s_tmp2, max_seg_num, actual_box_num);

  // if (s1 == 0 || s2 == 0) return valid pts mask
  __bang_write_value((void *)temp3_ram, max_seg_num, float(0.0));
  __bang_ne((float *)mask_s1_eq0, (float *)s1, (float *)temp3_ram, max_seg_num);
  __bang_ne((float *)mask_s2_eq0, (float *)s2, (float *)temp3_ram, max_seg_num);
  __bang_and((float *)mask_vaild_pts, (float *)mask_s1_eq0,
             (float *)mask_s2_eq0, max_seg_num);

  // swap boxes_points with s1=-1
  __bang_write_value((void *)temp3_ram, max_seg_num, float(-1.0));
  __bang_eq((float *)mask_s1_eqF1, (float *)s1, (float *)temp3_ram,
            max_seg_num);
  // point swap 标量操作
  int loop_num = actual_box_num;
  while (loop_num--) {
    __bang_max(temp3_ram, mask_s1_eqF1, max_seg_num);
    if (uint32_t(temp3_ram[0]) <= uint32_t(0)) break;
    uint32_t idx = ((uint32_t *)temp3_ram)[1];
    mask_s1_eqF1[idx] = -1;
    points_swap(max_box_pts_x0, max_box_pts_y0, max_box_pts_x1, max_box_pts_y1,
                idx);
  }

  // swap boxes_points with s2 = -1
  __bang_write_value(temp3_ram, max_seg_num, float(-1.0));
  __bang_eq(mask_s2_eqF1, s2, temp3_ram, actual_box_num);
  loop_num = actual_box_num;
  while (loop_num--) {
    __bang_max(temp3_ram, mask_s2_eqF1, actual_box_num);
    if (uint32_t(temp3_ram[0]) <= uint32_t(0)) break;
    uint32_t idx = ((uint32_t *)temp3_ram)[1];
    mask_s2_eqF1[idx] = -1;
    points_swap(box_pts_x0, box_pts_y0, box_pts_x1, box_pts_y1, idx);
  }

  // polygon cut
  // p(o,a,b) 3 o c
  p_x[0] = o_x;
  p_x[1] = max_box_pts_x0;  // box_pts_x0;
  p_x[2] = max_box_pts_x1;  // box_pts_x1;
  p_x[3] = o_x;

  p_y[0] = o_y;
  p_y[1] = max_box_pts_y0;  // box_pts_y0;
  p_y[2] = max_box_pts_y1;  // box_pts_y1;
  p_y[3] = o_y;

  IN_DT *a_x;
  IN_DT *a_y;
  IN_DT *b_x;
  IN_DT *b_y;

  a_x = o_x;
  a_y = o_y;
  b_x = box_pts_x0;
  b_y = box_pts_y0;
  __bang_write_value(p_count, max_seg_num, float(3));
  __bang_write_value(pp_x, 30 * max_seg_num, float(0));
  __bang_write_value(pp_y, 30 * max_seg_num, float(0));

  //  for (int i = 0; i < actual_box_num; i++) {
  //     __bang_printf(
  //         "intersectArea p[%d]:
  //         max_box_pts_0:(%f,%f),max_box_pts_1:(%f,%f),box_pts_0:(%f,%f), "
  //         "box_pts_1:(%f,%f), \n",
  //         i, max_box_pts_x0[i], max_box_pts_y0[i], max_box_pts_x1[i],
  //         max_box_pts_y1[i],
  //         box_pts_x0[i], box_pts_y0[i], box_pts_x1[i], box_pts_y1[i]);
  //   }

  // __bang_printf("- polygon_cut -1-\n");
  polygon_cut(p_x, p_y, p_count, a_x, a_y, b_x, b_y, pp_x, pp_y, nram_tmp,
              actual_box_num, max_seg_num);

  a_x = box_pts_x0;
  a_y = box_pts_y0;
  b_x = box_pts_x1;
  b_y = box_pts_y1;
  // __bang_printf("- polygon_cut -2-\n");
  polygon_cut(p_x, p_y, p_count, a_x, a_y, b_x, b_y, pp_x, pp_y, nram_tmp,
              actual_box_num, max_seg_num);

  a_x = box_pts_x1;
  a_y = box_pts_y1;
  b_x = o_x;
  b_y = o_y;
  // __bang_printf("- polygon_cut -3-\n");
  polygon_cut(p_x, p_y, p_count, a_x, a_y, b_x, b_y, pp_x, pp_y, nram_tmp,
              actual_box_num, max_seg_num);
  // __bang_printf("- polygon_cut -all end-\n");
  // for(int i = 0;i< p_count[0];i++){
  //   __bang_printf("- polygon_cut -all end- p[%f, %f]\n", p_x[i][0],
  //   p_y[i][0]);
  // }

  calPolygonSignArea<float>((float *)area, p_x, p_y, p_count, actual_box_num);
  absBoxesArea((float *)area, actual_box_num);
  __bang_mul((float *)area, (float *)mask_vaild_pts, (float *)area,
             actual_box_num);

  // f (s1 * s2 == -1) res = -res;
  __bang_write_value((void *)temp3_ram, max_seg_num, float(-1.0));
  __bang_mul(mask_s1_s2_eqf1, s1, s2, actual_box_num);
  __bang_eq(s1, mask_s1_s2_eqf1, (float *)temp3_ram, actual_box_num);
  __bang_ne(s2, mask_s1_s2_eqf1, (float *)temp3_ram, actual_box_num);

  __bang_mul(mask_s1_s2_eqf1, s1, (float *)temp3_ram, actual_box_num);
  __bang_add(mask_s1_s2_eqf1, mask_s1_s2_eqf1, s2, actual_box_num);

  __bang_mul(area, area, mask_s1_s2_eqf1, actual_box_num);

  // for(int w = 0; w < actual_box_num; w++){
  //   __bang_printf("-- intersectArea end-area=%f, mask_s1_s2_eqf1= %f\n",
  //   area[w],
  //   mask_s1_s2_eqf1[w]);
  // }
}
template <typename IN_DT, typename OUT_DT>
__mlu_func__ void calculateBoxesArea(const IN_DT *x1, const IN_DT *y1,
                                     const IN_DT *x2, const IN_DT *y2,
                                     const IN_DT *x3, const IN_DT *y3,
                                     const IN_DT *x4, const IN_DT *y4,
                                     OUT_DT *area, OUT_DT *tmp,
                                     const int input_stride) {
  // calculate polygon area
  // polygon vertexs:(x1,y1),(x2,y2),(x3,y3),(x4,y4)
  // polygon_area= abs(1/2 * ((x1*y2 - y1*x2) + (x2*y3-y2*x3) + (x3*y4 - y3*x4)
  // + (x4*y1 - y4*x1)))

  // x1*y2
  __bang_mul((IN_DT *)area, (IN_DT *)x1, (IN_DT *)y2,
             input_stride * sizeof(IN_DT));
  __bang_mul((IN_DT *)tmp, (IN_DT *)y1, (IN_DT *)x2,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2
  __bang_sub((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3
  __bang_mul((IN_DT *)tmp, (IN_DT *)x2, (IN_DT *)y3,
             input_stride * sizeof(IN_DT));
  __bang_add((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3-y2*x3
  __bang_mul((IN_DT *)tmp, (IN_DT *)y2, (IN_DT *)x3,
             input_stride * sizeof(IN_DT));
  __bang_sub((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3-y2*x3 + x3*y4
  __bang_mul((IN_DT *)tmp, (IN_DT *)x3, (IN_DT *)y4,
             input_stride * sizeof(IN_DT));
  __bang_add((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3-y2*x3) + x3*y4 - y3*x4
  __bang_mul((IN_DT *)tmp, (IN_DT *)y3, (IN_DT *)x4,
             input_stride * sizeof(IN_DT));
  __bang_sub((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3-y2*x3) + x3*y4 - y3*x4 + x4*y1
  __bang_mul((IN_DT *)tmp, (IN_DT *)x4, (IN_DT *)y1,
             input_stride * sizeof(IN_DT));
  __bang_add((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // x1*y2 - y1*x2 + x2*y3-y2*x3) + x3*y4 - y3*x4 + x4*y1 - y4*x1
  __bang_mul((IN_DT *)tmp, (IN_DT *)y4, (IN_DT *)x1,
             input_stride * sizeof(IN_DT));
  __bang_sub((IN_DT *)area, (IN_DT *)area, (IN_DT *)tmp,
             input_stride * sizeof(IN_DT));
  // (x1*y2 - y1*x2 + x2*y3-y2*x3) + x3*y4 - y3*x4 + x4*y1 - y4*x1)*0.5
  __bang_mul_scalar((IN_DT *)area, (IN_DT *)area, (IN_DT)0.5,
                    input_stride * sizeof(IN_DT));
  // __bang_printf("calculateBoxesArea end\n");
}

template <typename IN_DT>
__mlu_func__ void calculateOverlapArea(IN_DT *box_pts_x, IN_DT *box_pts_y,
                                       const IN_DT *scores, IN_DT *max_box,
                                       IN_DT *max_box_pts_x,
                                       IN_DT *max_box_pts_y, IN_DT *boxes_area,
                                       IN_DT *intersection_area,
                                       IN_DT *nram_tmp, const int max_seg_num,
                                       const int actual_box_num) {
  // __bang_printf("calculateOverlapArea begin\n");

  // 12 + buffer
  // IN_DT *box_pts_x;      // 4
  // IN_DT *box_pts_y;      // 4
  IN_DT *temp1_ram;  // 1
  IN_DT *temp2_ram;  // 1
  IN_DT *temp3_ram;  // 1
  IN_DT *area_tmp;
  IN_DT *buffer;  //

  // box_pts_x     = (IN_DT *)nram_tmp;
  // box_pts_y     = box_pts_x + 4 * max_seg_num;
  temp1_ram = nram_tmp;
  temp2_ram = temp1_ram + max_seg_num;
  temp3_ram = temp2_ram + max_seg_num;
  area_tmp = temp3_ram + max_seg_num;
  buffer = area_tmp + max_seg_num;

  // reverse boxes_points with area < 0
  __bang_write_value(temp1_ram, max_seg_num, float(0.0));
  __bang_lt((IN_DT *)temp2_ram, boxes_area, (IN_DT *)temp1_ram, max_seg_num);

  int loop_num = actual_box_num;
  while (loop_num--) {
    __bang_max(temp3_ram, (float *)temp2_ram, max_seg_num);
    if (uint32_t(temp3_ram[0]) <= uint32_t(0)) break;
    uint32_t idx = ((uint32_t *)temp3_ram)[1];
    temp2_ram[idx] = -1;
    points_reverse(box_pts_x, box_pts_y, idx, max_seg_num);
  }
  // __bang_printf("---before reverse:\n");
  // for(int j =0;j<5;j++){
  //   __bang_printf("%f, ", boxes_area[j]);
  // }
  // __bang_printf("\n");
  // after reverse points coord, update box_area
  calculateBoxesArea(box_pts_x, box_pts_y, box_pts_x + 1 * max_seg_num,
                     box_pts_y + 1 * max_seg_num, box_pts_x + 2 * max_seg_num,
                     box_pts_y + 2 * max_seg_num, box_pts_x + 3 * max_seg_num,
                     box_pts_y + 3 * max_seg_num, boxes_area, temp1_ram,
                     actual_box_num);
  // __bang_printf("---after reverse:\n");
  // for(int j =0;j<5;j++){
  //   __bang_printf("%f, ", boxes_area[j]);
  // }
  // __bang_printf("\n");

  __bang_write_value(intersection_area, max_seg_num, float(0));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int m = i == 3 ? 0 : (i + 1);
      int n = j == 3 ? 0 : (j + 1);
      // __bang_printf("intersectArea: i = %d, j = %d\n", i, j);

      // res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
      intersectArea<float>(
          area_tmp, max_box_pts_x + i * max_seg_num,
          max_box_pts_y + i * max_seg_num, max_box_pts_x + m * max_seg_num,
          max_box_pts_y + m * max_seg_num, box_pts_x + j * max_seg_num,
          box_pts_y + j * max_seg_num, box_pts_x + n * max_seg_num,
          box_pts_y + n * max_seg_num, buffer, max_seg_num, actual_box_num);
      __bang_add(intersection_area, intersection_area, area_tmp,
                 actual_box_num);
      // __bang_printf("area_tmp = %f, overlaps=%f \n", area_tmp[0],
      // overlaps[0]);
    }
  }
}

// template <typename IN_DT, typename OUT_DT>
// __mlu_func__ void dealNanAndInfWithInputBoxes(IN_DT *input_box_ptr,
//                                               IN_DT *input_score_ptr,
//                                               IN_DT *buffer,
//                                               OUT_DT *output_data,
//                                               OUT_DT *nram_save,
//                                               const int input_stride,
//                                               const int buffer_size,
//                                               const int
//                                               nram_save_limit_count,
//                                               int &output_save_count,
//                                               int &nram_save_count,
//                                               int &output_box_num){
//   if(coreId != 0){
//     return;
//   }
//   const int INPUT_TYPE_SIZE = sizeof(IN_DT);
//   const int NFU_ALIGN_SIZE  = 128;
//   int align_num             = NFU_ALIGN_SIZE / INPUT_TYPE_SIZE;

//   IN_DT *boxes;
//   IN_DT *scores;
//   IN_DT *ram_lowest;
//   IN_DT *ram_highest;
//   IN_DT *ge_ret;
//   IN_DT *le_ret;
//   IN_DT *valid_boxes;

//   int32_t max_buffer_num   = buffer_size / INPUT_TYPE_SIZE ;
//   int32_t max_seg_num      = FLOOR_ALIGN((max_buffer_num / 14), align_num *
//   9);
//   int32_t max_seg_pad      = max_seg_num * INPUT_TYPE_SIZE;
//   int32_t repeat           = input_stride * 9 / max_seg_num;
//   int32_t remain           = input_stride * 9 % max_seg_num;
//   int32_t remain_num_align = CEIL_ALIGN(remain, align_num);
//   int32_t remain_pad       = remain_num_align * INPUT_TYPE_SIZE;

//   boxes  = buffer;
//   scores = boxes + 8 * max_seg_num;
//   ram_lowest = scores + max_seg_num;
//   ram_highest= ram_lowest + max_seg_num;
//   ge_ret = ram_highest + max_seg_num;
//   le_ret = ge_ret + max_seg_num;
//   valid_boxes = le_ret + max_seg_num;

//   const uint32_t float32_highest = 0x7f7fffff;
//   const uint32_t float32_lowest = 0xff6fffff; // 0xff7fffff
//   __bang_write_value(ram_lowest, max_seg_num, float32_lowest);
//   __bang_write_value(ram_highest, max_seg_num, float32_highest);
//   for(int i = 0; i <=repeat; i++){
//     if(i == repeat && remain == 0){
//       break;
//     }
//     int actual_num = i == repeat ? remain: max_seg_num;
//     int autual_pad = i == repeat ? remain_pad:max_seg_pad;
//     __memcpy(boxes, input_score_ptr + i * max_seg_num, autual_pad,
//     GDRAM2NRAM);
//       __bang_write_value(valid_boxes, align_num, float(1));

//     for( int w = 0; w < 8;w++){
//       __memcpy(boxes, input_box_ptr + w * input_stride, autual_pad,
//       GDRAM2NRAM);
//       __bang_ge(ge_ret, boxes, ram_lowest, autual_pad/4);
//       __bang_le(le_ret, boxes, ram_highest, autual_pad/4);

//       __bang_and(le_ret, le_ret, ge_ret, autual_pad/4);
//       __bang_and(valid_boxes, valid_boxes, le_ret, autual_pad/4);
//     }

//     for(int j = 0; j < actual_num; j++){
//       __bang_printf("valid_boxes[%d]:%f\n", j, valid_boxes[j]);
//     }
//     __bang_mul(scores, scores, valid_boxes, actual_num);

//     __bang_write_value(le_ret, autual_pad/4, 0);
//     __bang_ne(valid_boxes, valid_boxes, le_ret, autual_pad);
//     __bang_write_value(le_ret, autual_pad/4, PNMS_MIN);
//     __bang_mul(valid_boxes, valid_boxes, le_ret, actual_num);

//     __bangadd(scores,scores,valid_boxes,actual_num);
//     __memcpy(input_score_ptr + i * max_seg_num, scores, autual_pad,
//     NRAM2GDRAM);

//     for (size_t i = 0; i < actual_num; i++)
//     {
//       if(valid_boxes[i] == 1){
//         nram_save[nram_save_count] = (uint32_t)(valid_boxes[i]);
//         nram_save_count++;
//         output_box_num++;
//         if (nram_save_count == nram_save_limit_count) {
//           __bang_printf("nram_save_count == nram_save_limit_count=%d\n",
//           nram_save_count);
//           __memcpy(output_data + output_save_count * nram_save_limit_count,
//           nram_save,
//                     nram_save_count * sizeof(uint32_t), NRAM2GDRAM);
//           output_save_count++;
//           nram_save_count = 0;
//         }
//       }
//     } // for (size_t i = 0; i < actual_num; i++)
//   } // for repeat
// }

template <typename IN_DT>
__mlu_func__ void getMaxScoreIndex(IN_DT *input_box_ptr, IN_DT *input_score_ptr,
                                   IN_DT *scores, IN_DT *max_box,
                                   IN_DT *sram_buffer, IN_DT *buffer,
                                   int32_t *nan_inf_flag, const int buffer_size,
                                   const int scores_len, const int input_offset,
                                   const int input_stride,
                                   const int core_limit) {
  //  | nram_save| max_box|  max_box_tmp     |  scores |
  //  | 1| NFU_ALIGN_SIZE |  NFU_ALIGN_SIZE  |  N      |
  const int INPUT_TYPE_SIZE = sizeof(IN_DT);
  int align_num = NFU_ALIGN_SIZE / INPUT_TYPE_SIZE;

  int32_t nan_inf_op_flag = nan_inf_flag[0];

  IN_DT *max_box_tmp;
  IN_DT *tmp1;
  IN_DT *tmp2;

  int32_t max_buffer_num = buffer_size / INPUT_TYPE_SIZE - 2 * align_num;
  int32_t max_seg_num = FLOOR_ALIGN(max_buffer_num, align_num);
  int32_t max_seg_pad = max_seg_num * INPUT_TYPE_SIZE;
  int32_t repeat = scores_len / max_seg_num;
  int32_t remain = scores_len % max_seg_num;
  int32_t remain_num_align = CEIL_ALIGN(remain, align_num);
  int32_t remain_pad = remain_num_align * INPUT_TYPE_SIZE;

  if (nan_inf_op_flag == 0) {
    // |max_box| max_box_tmp|scores|
    max_box = buffer;
    max_box_tmp = max_box + align_num;
    scores = max_box_tmp + align_num;
  } else {
    // |max_box| max_box_tmp|tmp1|tmp2|scores|
    max_buffer_num = buffer_size / INPUT_TYPE_SIZE - 2 * align_num;
    max_seg_num = FLOOR_ALIGN((max_buffer_num / 3), align_num);
    max_seg_pad = max_seg_num * INPUT_TYPE_SIZE;
    repeat = scores_len / max_seg_num;
    remain = scores_len % max_seg_num;
    remain_num_align = CEIL_ALIGN(remain, align_num);
    remain_pad = remain_num_align * INPUT_TYPE_SIZE;

    max_box = buffer;
    max_box_tmp = max_box + align_num;
    tmp1 = max_box_tmp + max_seg_num;
    tmp2 = tmp1 + max_seg_num;
    scores = tmp2 + max_seg_num;
  }

  int32_t max_index = 0;
  max_box[0] = (float)PNMS_MIN;

  for (int i = 0; i <= repeat; i++) {
    if (i == repeat && remain == 0) {
      break;
    }
    int actual_scores_pad = 0;
    int actual_scores_num = 0;

    actual_scores_num = (i == repeat) ? remain : max_seg_num;
    actual_scores_pad = (i == repeat) ? remain_pad : max_seg_pad;

    __bang_write_value((float *)scores, actual_scores_pad / 4, (float)PNMS_MIN);
    __memcpy(scores, input_score_ptr + input_offset + i * max_seg_num,
             actual_scores_num * INPUT_TYPE_SIZE, GDRAM2NRAM);

    // scores nan,inf,-inf
    if (nan_inf_op_flag == 1) {
      const uint32_t float32_highest = 0x7f7fffff;
      const uint32_t float32_lowest = 0xff6fffff;  // 0xff7fffff
      __bang_printf("--nan_inf_op_flag--\n");
      __bang_write_value(tmp1, actual_scores_pad, float32_lowest);
      __bang_write_value(tmp2, actual_scores_pad, float32_highest);

      // -inf/nan to float_lowest
      __bang_maxequal(scores, tmp1, scores, actual_scores_num);
      // inf to float_highest
      __bang_minequal(scores, tmp2, scores, actual_scores_num);
      __memcpy(input_score_ptr + input_offset + i * max_seg_num, scores,
               actual_scores_num * INPUT_TYPE_SIZE, NRAM2GDRAM);
    }

    __bang_max(max_box_tmp, scores, actual_scores_pad / 4);
    if (max_box_tmp[0] > max_box[0]) {
      max_box[0] = max_box_tmp[0];
      max_index = ((uint32_t *)max_box_tmp)[1] + input_offset + i * max_seg_num;
      max_box[1] = max_index;
    }
  }  // for repeat

  // __bang_printf("---coreid: %d, max_score_index1: %d, max_score1: %f\n",
  // coreId,
  // (uint32_t)(max_box[1]), max_box[0]);

  if (core_limit == 1) {
    // get max_score_box coordinate
    // max_box = | max_score| max_score_index | max_box_coordinate |
    // max_score_box_area|
    //           | 1        |       1         |          8         |     1 |
    // input_box_ptr:x1---, y1---, x2---, y2---, x3---, y3---, x4---,
    // y4---,scores---
    __memcpy(max_box + 2, input_box_ptr + max_index, 1 * INPUT_TYPE_SIZE,
             GDRAM2NRAM, 1 * sizeof(uint32_t), input_stride * sizeof(uint32_t),
             8);

    // calculate max_score_box_area
    // output: max_score| index | coordinate | sign box area | box area
    calculateMaxScoreBoxArea(max_box);
    input_score_ptr[uint32_t(max_box[1])] = PNMS_MIN;
    // __bang_printf("core id = %d, max_index = %d,  score=%f\n", coreId,
    // max_index, max_box[0]);
  } else {
    // sram_buffer: max_score1 | index1 | max_score2 | index2 ...
    __memcpy(sram_buffer + 2 * taskId, max_box, 2 * INPUT_TYPE_SIZE, NRAM2SRAM);
    __sync_cluster();

    __bang_write_value(scores, NFU_ALIGN_SIZE / 4, PNMS_MIN);
    for (int j = 0; j < core_limit; j++) {
      scores[j] = sram_buffer[j * 2];
    }

    __bang_max(max_box_tmp, scores, NFU_ALIGN_SIZE / 4);
    max_box[0] = max_box_tmp[0];
    max_index = ((uint32_t *)max_box_tmp)[1];
    max_box[1] = (uint32_t)(sram_buffer[max_index * 2 + 1]);

    max_index = uint32_t(max_box[1]);
    input_score_ptr[max_index] = PNMS_MIN;

    __memcpy(max_box + 2, input_box_ptr + max_index, 1 * INPUT_TYPE_SIZE,
             GDRAM2NRAM, 1 * sizeof(uint32_t), input_stride * sizeof(uint32_t),
             8);

    // __bang_printf("core id = %d, max_index = %d, global max_index:%d,
    // max_box[x2]=%f, score=%f\n",
    //               coreId, max_index, uint32_t(max_box[1]), max_box[4],
    //               max_box[0]);
    calculateMaxScoreBoxArea(max_box);
  }  // if (core_limit == 4)
}

template <typename IN_DT, typename OUT_DT>
__mlu_func__ void pnms_detection(OUT_DT *result_num, OUT_DT *output_data,
                                 IN_DT *input_data_box, IN_DT *buffer,
                                 const int buffer_size, IN_DT *sram_buffer,
                                 void *workspace, const int core_limit,
                                 const int input_data_num,
                                 const int input_stride,
                                 const float thresh_iou) {
  __bang_printf("launch pnms_detection\n");
  __bang_printf("core_limit: %d, input_stride = %d,input_data_num = %d\n",
                core_limit, input_stride, input_data_num);
  uint32_t output_box_num = 0;
  // NRAM N=max_seg_pad
  // |nram_save| max_box(max_score,max_box,max_index,+-max_area, max_area)|
  // tranx_box | scores |
  // max_box_tmp
  // |box_area|| nram_tmp(box,box_area_tmp)|
  // |  COMPUTE_COUNT_ALIGN                         |    N*8    |  N     |
  // COMPUTE_COUNT_ALIGN|
  // N|
  // N
  // |          213*N        |

  const int INPUT_TYPE_SIZE = sizeof(IN_DT);
  const int COMPUTE_COUNT_ALIGN = 64;
  // global value
  int32_t *loop_end_flag = (int32_t *)(workspace);
  loop_end_flag[0] = 0;

  int input_boxes_num = input_stride;
  int input_offset_num = 0;
  if (core_limit == 1) {
    input_boxes_num = input_stride;
    input_offset_num = 0;
  } else {
    int avg_core = input_boxes_num / core_limit;
    int rem = input_boxes_num % core_limit;
    input_boxes_num = avg_core + (taskId < rem ? 1 : 0);
    input_offset_num = avg_core * taskId + (taskId <= rem ? taskId : rem);
  }
  // 保证每次repeat的数是9的倍数
  int limit = (buffer_size - NFU_ALIGN_SIZE) / 137;  // 133+12-8
  int max_seg_pad = FLOOR_ALIGN(limit, COMPUTE_COUNT_ALIGN * 9);
  int max_seg_num = max_seg_pad / INPUT_TYPE_SIZE;

  int repeat = input_boxes_num / max_seg_num;
  int remain_num = input_boxes_num % max_seg_num;

  __bang_printf("coreId: %d \n", coreId);
  __bang_printf("input_offset: %d \n", input_offset_num);
  __bang_printf("buffer_size: %d \n", buffer_size);
  __bang_printf("limit: %d \n", limit);
  __bang_printf("max_seg_pad: %d \n", max_seg_pad);
  __bang_printf("max_seg_num: %d \n", max_seg_num);

  __bang_printf("repeat: %d \n", repeat);
  __bang_printf("remain_num: %d \n", remain_num);

  IN_DT *input_box_ptr;
  IN_DT *input_score_ptr;

  input_box_ptr = input_data_box;
  input_score_ptr = input_box_ptr + 8 * input_stride;

  // init nram ptr
  IN_DT *nan_inf_flag;
  IN_DT *box_pts_x;
  IN_DT *box_pts_y;
  IN_DT *scores;
  IN_DT *max_box;
  OUT_DT *nram_save;
  IN_DT *nram_tmp;
  IN_DT *box_area_tmp;
  IN_DT *box_area;
  IN_DT *max_box_pts_x;
  IN_DT *max_box_pts_y;
  IN_DT *intersection_area;

  nram_save = (OUT_DT *)((char *)buffer);
  nan_inf_flag = (IN_DT *)((char *)nram_save + max_seg_pad);
  max_box = nan_inf_flag + NFU_ALIGN_SIZE / INPUT_TYPE_SIZE;
  max_box_pts_x = max_box + NFU_ALIGN_SIZE / INPUT_TYPE_SIZE;
  max_box_pts_y = max_box_pts_x + 4 * max_seg_num;
  box_pts_x = max_box_pts_y + 4 * max_seg_num;
  box_pts_y = box_pts_x + 4 * max_seg_num;
  scores = box_pts_y + 4 * max_seg_num;
  box_area = scores + max_seg_num;
  intersection_area = box_area + max_seg_num;
  nram_tmp = intersection_area + max_seg_pad;
  box_area_tmp = nram_tmp;

  int nram_save_count = 0;
  const int nram_save_limit_count = max_seg_num;
  int max_output_size = input_stride;
  int output_save_count = 0;
  __bang_printf("max_output_size = %d\n", max_output_size);
  ((int32_t *)nan_inf_flag)[0] = 0;  // =1处理nan inf

  // max_output_size   = 2; //debug
  int test_loop_count = 0;
  for (int loop = 0; loop < max_output_size; loop++) {
    if (core_limit != 1) {
      __sync_cluster();  // sync before current loop
    }
    test_loop_count++;

    // look for max_score
    // 1 get_max_box_index();
    // output: max_box (max_score, max_index, max_box_coordinate, sign box_area,
    // max_score_box_area)
    uint32_t scoreIndexBufSize = buffer_size - max_seg_pad;
    getMaxScoreIndex(input_box_ptr, input_score_ptr, scores, max_box,
                     sram_buffer, max_box, (int32_t *)nan_inf_flag,
                     scoreIndexBufSize, input_boxes_num, input_offset_num,
                     input_stride, core_limit);
    scores = box_pts_y + 4 * max_seg_num;
    nan_inf_flag[0] = 0;
    // if(nan_inf_flag[0] == 1){
    //   scoreIndexBufSize = buffer_size - max_seg_pad - NFU_ALIGN_SIZE;

    //   dealNanAndInfWithInputBoxes<IN_DT, OUT_DT>(input_box_ptr,
    //   input_score_ptr, buffer, (OUT_DT
    //   *)output_data, (OUT_DT *)nram_save, input_stride, scoreIndexBufSize,
    //   nram_save_limit_count,
    //   output_save_count, nram_save_count, output_box_num);
    //   nan_inf_flag[0] = 0;
    // }

    //  __bang_printf("coreId = %d, max_score_index = %d \n", coreId,
    //  uint32_t(max_box[1]));

    // store max_score_index to nram_save, and store nram_save to
    // output_data(gdram).
    if (coreId == 0) {
      if (float(max_box[0]) > (float)PNMS_MIN) {
        nram_save[nram_save_count] = (uint32_t)(max_box[1]);
        nram_save_count++;
        output_box_num++;

        if (nram_save_count == nram_save_limit_count) {
          __bang_printf("nram_save_count == nram_save_limit_count=%d\n",
                        nram_save_count);
          __memcpy(output_data + output_save_count * nram_save_limit_count,
                   nram_save, nram_save_count * sizeof(uint32_t), NRAM2GDRAM);
          output_save_count++;
          nram_save_count = 0;
        }
      }  // if (float(max_box[0]) >= (float)PNMS_MIN)
    }    // if (coreId == 0)

    // if the max score <= 0, end
    if (core_limit == 1) {
      if (float(max_box[0]) <= PNMS_MIN || (loop == max_output_size - 1)) {
        __memcpy(output_data + output_save_count * nram_save_limit_count,
                 nram_save, nram_save_count * sizeof(uint32_t), NRAM2GDRAM);
        __bang_printf("pnms end:nram_save_count = %d\n", nram_save_count);
        break;
      }
    } else {
      // __bang_printf("coreId = %d, mx_box[0]=%f \n", coreId,
      // float(max_box[0]));
      if (float(max_box[0]) <= PNMS_MIN || (loop == max_output_size - 1)) {
        if (coreId == 0) {
          __memcpy(output_data + output_save_count * nram_save_limit_count,
                   nram_save, nram_save_count * sizeof(uint32_t), NRAM2GDRAM);
          __bang_printf(
              "2 end condition :coreId == 0, mx_box[0]=%f <=PNMS_MIN\n",
              float(max_box[0]));
          loop_end_flag[0] = 1;
        }
      }
      __sync_cluster();
      if (loop_end_flag[0] == 1) {
        break;
      }
    }

    // max_score_box 扩维 max_box_pts_x,y, 提前copy好，减少在repeat中copy次数
    // max_box: max_score, max_score_index, max_score_box coordinate(x1, y1, x2,
    // y2, x3, y3, x4,
    // y4), sign max_score_box_area, unsign max_score_box_area
    // max_box_pts_x: x1---, x2---, x3---, x4---
    // max_box_pts_y: y1---, y2---, y3---, y4---
    __bang_write_value(max_box_pts_x, max_seg_num * 8, float(0));
    int max_box_count = repeat == 0 ? remain_num : max_seg_num;
    for (int j = 0; j < 4; j++) {
      __bang_write_value(max_box_pts_x + j * max_seg_num, max_box_count,
                         float(max_box[2 + j * 2]));
      __bang_write_value(max_box_pts_y + j * max_seg_num, max_box_count,
                         float(max_box[3 + j * 2]));
    }

    for (int i = 0; i <= repeat; i++) {
      if (i == repeat && remain_num == 0) {
        break;
      }

      int actual_box_num = 0;
      int actual_compute_box_num = 0;

      actual_box_num = (i == repeat) ? remain_num : max_seg_num;
      actual_compute_box_num = max_seg_num;

      // input_box_ptr: x1---, y1---, x2---, y2---, x3---, y3---, x4---, y4---,
      // scores---
      // box_pts_x: x1---, x2---, x3---, x4---
      // box_pts_y: y1---, y2---, y3---, y4---
      __bang_write_value(box_pts_x, 11 * max_seg_num, (float)0.0);
      __memcpy((IN_DT *)box_pts_x,
               input_box_ptr + input_offset_num + i * max_seg_num,
               actual_box_num * INPUT_TYPE_SIZE, GDRAM2NRAM,
               max_seg_num * INPUT_TYPE_SIZE,
               2 * input_stride * INPUT_TYPE_SIZE, 4);

      __memcpy(
          (IN_DT *)box_pts_y,
          input_box_ptr + input_offset_num + i * max_seg_num + input_stride,
          actual_box_num * INPUT_TYPE_SIZE, GDRAM2NRAM,
          max_seg_num * INPUT_TYPE_SIZE, 2 * input_stride * INPUT_TYPE_SIZE, 4);

      // scores
      __memcpy(scores,
               input_score_ptr + input_offset_num + i * actual_compute_box_num,
               max_seg_num * INPUT_TYPE_SIZE, GDRAM2NRAM);

      // scoreIndexBufSize = buffer_size - max_seg_pad - NFU_ALIGN_SIZE;
      // dealWithNanAndInfWithBoxes(input_box_ptr, box_pts_x, input_stride,
      // scoreIndexBufSize,
      // core_limit);

      calculateBoxesArea(
          box_pts_x, box_pts_y, box_pts_x + 1 * max_seg_num,
          box_pts_y + 1 * max_seg_num, box_pts_x + 2 * max_seg_num,
          box_pts_y + 2 * max_seg_num, box_pts_x + 3 * max_seg_num,
          box_pts_y + 3 * max_seg_num, box_area, box_area_tmp, actual_box_num);

      calculateOverlapArea<float>(box_pts_x, box_pts_y, scores, max_box,
                                  max_box_pts_x, max_box_pts_y, box_area,
                                  intersection_area, nram_tmp,
                                  actual_compute_box_num, actual_box_num);
      absBoxesArea(box_area, actual_box_num);

      // if((uint32_t)(max_box[1])==2){
      // __bang_printf("--11--loop:%d-(index, scores):(%d, %f), boxarea=%f\n",
      // loop,
      // (uint32_t)(max_box[1]), max_box[0], max_box[11]);
      // for(int j = 0; j<3; j++){
      //     __bang_printf("-(%d, intersection_area=%f, box_area=%f \n", j,
      //     ((float
      //     *)intersection_area)[j], box_area[j]);
      // }
      // }

      //  // scores为pnms_min的areai无效
      // __bang_write_value((float *)nram_tmp, actual_compute_box_num,
      // (float)PNMS_MIN);
      // __bang_ne((float *)(nram_tmp + actual_compute_box_num), (float
      // *)scores, nram_tmp,
      //           actual_compute_box_num);
      // __bang_mul(intersection_area, intersection_area, (float *)(nram_tmp +
      // actual_compute_box_num),
      //            actual_box_num);

      // 4 compare iou with thresh_iou(); iou>thresh_iou, 将其对应的score置0；
      // area_U = box_area + max_area - area_I
      __bang_add_const((float *)box_area, (float *)box_area, (float)max_box[11],
                       actual_box_num);
      __bang_sub((float *)box_area, (float *)box_area,
                 (float *)intersection_area, actual_box_num);
      // if (union_area == 0)  iou = (inter_area + 1) / (union_area + 1);
      __bang_write_value(nram_tmp, actual_box_num, (float)0.0);
      __bang_eq(nram_tmp, box_area, nram_tmp, actual_box_num);
      __bang_add(box_area, box_area, nram_tmp, actual_box_num);
      __bang_add(intersection_area, intersection_area, nram_tmp,
                 actual_box_num);

      // compute div  iou = intersection_area
      computeDiv((float *)intersection_area, (float *)intersection_area,
                 (float *)box_area, nram_tmp, actual_box_num);

      //  __bang_printf("---loo:%d--index, scores:(%d, %f),iou=:\n", loop,
      //  (uint32_t)(max_box[1]),
      //  max_box[0]);
      // for(int j = 0;j<actual_box_num; j++){
      //       __bang_printf("-%f-",intersection_area[j]);
      // }
      // __bang_printf("\n");
      // masked = intersection_area = iou <= thresh_iou
      __bang_write_value(nram_tmp, actual_box_num, (float)thresh_iou);
      __bang_le((float *)intersection_area, (float *)intersection_area,
                (float *)nram_tmp, actual_box_num);

      // __bang_printf("---loo:%d--index, scores:(%d, %f)\n", loop,
      // (uint32_t)(max_box[1]),
      // max_box[0]);
      // for(int j = 0;j<actual_box_num; j++){
      //       __bang_printf("-%d-", j);
      // }
      // __bang_printf("\n");

      // scores = scores * intersection_area; >iou_thresh的scores置为0
      __bang_mul((float *)scores, (float *)scores, (float *)intersection_area,
                 actual_box_num);

      // compare scores with float 0 -> intersection_area
      __bang_write_value(nram_tmp, actual_box_num,
                         (float)0.0);  // actual_box_num * INPUT_TYPE_SIZE
      __bang_eq((float *)intersection_area, (float *)scores, (float *)nram_tmp,
                actual_box_num);

      // intersection_area = intersection_area*FLT_MIN  (masked *FLT_MIN )
      __bang_mul_const((float *)intersection_area, (float *)intersection_area,
                       (float)PNMS_MIN, actual_box_num);

      // scores  = scores + intersection_area
      __bang_add((float *)scores, (float *)scores, (float *)intersection_area,
                 actual_box_num);
      __memcpy((float *)input_score_ptr + input_offset_num + i * max_seg_num,
               (float *)scores, actual_box_num * INPUT_TYPE_SIZE, NRAM2GDRAM);
    }  // for repeat
  }    // for loop : max_output_size

  __bang_printf("--end-- coreId = %d ,test_loop_count = %d\n", coreId,
                test_loop_count);

  if (coreId == 0) {
    __bang_printf("result_num1 = %d\n", output_box_num);
    ((uint32_t *)result_num)[0] = output_box_num;
    __bang_printf("result_num 2 = %d, result_box_num=%d \n",
                  ((uint32_t *)result_num)[0], output_box_num);
    __memcpy(buffer, output_data, output_box_num * 4, GDRAM2NRAM);
    quickSort((uint32_t *)buffer, 0, output_box_num - 1);
    __memcpy(output_data, buffer, output_box_num * 4, NRAM2GDRAM);
    __bang_printf("--end--\n");
  }
}
#endif  // KERNELS_NMS_NMS_DETECTION_H_
