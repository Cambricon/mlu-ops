#include "pnms_impl.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;

namespace PNMS {

#define maxn 51
const float eps = 1E-8;
int sig(float d) {
  return (d > eps) - (d < -eps);
}

struct Point {
  float x, y;
  Point() {}
  Point(float x, float y) : x(x), y(y) {}
  bool operator==(const Point &p) const { return sig(x - p.x) == 0 && sig(y - p.y) == 0; }
};

float cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

float area(Point *ps, int n) {
  ps[n] = ps[0];
  float res = 0;
  for (int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res / 2.0;
}

int lineCross(Point a, Point b, Point c, Point d, Point &p) {
  float s1, s2;
  s1 = cross(a, b, c);
  s2 = cross(a, b, d);
  if (sig(s1) == 0 && sig(s2) == 0)
    return 2;
  if (sig(s2 - s1) == 0)
    return 0;
  p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
  p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
  return 1;
}


void polygon_cut(Point *p, int &n, Point a, Point b, Point *pp) {
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    if (sig(cross(a, b, p[i])) > 0)
      pp[m++] = p[i];
    if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
      lineCross(a, b, p[i], p[i + 1], pp[m++]);
  }

  n = 0;
  for (int i = 0; i < m; i++)
    if (!i || !(pp[i] == pp[i - 1]))
      p[n++] = pp[i];
  while (n > 1 && p[n - 1] == p[0])
    n--;
}

float intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));

  int s2 = sig(cross(o, c, d));

  if (s1 == 0 || s2 == 0)
    return 0.0;
  if (s1 == -1)
    swap(a, b);
  if (s2 == -1)
    swap(c, d);
  Point p[10] = {o, a, b};
  int n = 3;
  Point pp[maxn];

  polygon_cut(p, n, o, c, pp);
  polygon_cut(p, n, c, d, pp);
  polygon_cut(p, n, d, o, pp);

  float res = fabs(area(p, n));
  if (s1 * s2 == -1)
    res = -res;
  return res;
}

float intersectArea(Point *ps1, int n1, Point *ps2, int n2) {
  float area1 = area(ps1, n1);
  if (area(ps1, n1) < 0) {
    reverse(ps1, ps1 + n1);
  }

  if (area(ps2, n2) < 0){
    reverse(ps2, ps2 + n2);
  }

  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  float res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
    }
  }
  return res;
}

float iou_poly(vector<float> p, vector<float> q) {
  Point ps1[maxn], ps2[maxn];
  int n1 = 4;
  int n2 = 4;
  for (int i = 0; i < 4; i++) {
    ps1[i].x = p[i * 2];
    ps1[i].y = p[i * 2 + 1];

    ps2[i].x = q[i * 2];
    ps2[i].y = q[i * 2 + 1];
  }

  float inter_area = intersectArea(ps1, n1, ps2, n2);
  float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
 
  float iou = 0;
  if (union_area == 0) {
    iou = (inter_area + 1) / (union_area + 1);
  } else {
    iou = inter_area / union_area;
  }
  return iou;
}

bool cmp_value(const pair<vector<float>, int> &a, pair<vector<float>, int> &b) {
  return (*(a.first.end() - 1) > *(b.first.end() - 1));
}

vector<int> pnms_impl(vector<vector<float>> &p, const float thresh) {
  vector<pair<vector<float>, int>> vp;
  for (int i = 0; i < p.size(); i++) {
    vp.push_back(make_pair(p[i], i));
  }

  sort(vp.begin(), vp.end(), cmp_value);

  vector<int> keep;
  while (vp.size()) {
    keep.push_back(vp.begin()->second);
    auto box = vp.begin()->first;
    auto box_index = vp.begin()->second;
    vp.erase(vp.begin());
    for (int i = 0; i < vp.size(); i++) {
      float iou = iou_poly(box, vp[i].first);
      if (iou > thresh) {
        vp.erase(vp.begin() + i);
        i--;
      }
    }
  }

  sort(keep.begin(), keep.end(), [&](int a, int b) { return a < b; });
  return keep;
}
}  // namespace PNMS