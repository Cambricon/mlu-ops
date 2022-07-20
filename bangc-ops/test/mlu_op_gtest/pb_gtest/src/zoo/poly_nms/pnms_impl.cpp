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

float cross(Point o, Point a, Point b) {  //叉积
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
  // printf("line coress: p(%f, %f)\n", p.x, p.y);
  return 1;
}

//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
void polygon_cut(Point *p, int &n, Point a, Point b, Point *pp) {
  //    static Point pp[maxn];
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
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
float intersectArea(Point a, Point b, Point c, Point d) {
  Point o(0, 0);
  int s1 = sig(cross(o, a, b));
  // printf("s1=%d\n", s1);

  int s2 = sig(cross(o, c, d));
  // printf("s2=%d\n", s2);

  if (s1 == 0 || s2 == 0)
    return 0.0;  //退化，面积为0
  if (s1 == -1)
    swap(a, b);
  if (s2 == -1)
    swap(c, d);
  Point p[10] = {o, a, b};
  int n = 3;
  Point pp[maxn];
  // // printf("polygon_cut before:\n");
  // for(int i = 0;i<10;i++)
  // {
  //     printf("p[%d]:(%f, %f) ,", i, p[i].x, p[i].y);
  // }
  //  printf("\n");
  polygon_cut(p, n, o, c, pp);
  polygon_cut(p, n, c, d, pp);
  polygon_cut(p, n, d, o, pp);
  // printf("polygon_cut after:\n");
  // for(int i = 0;i<10;i++)
  // {
  //     printf("p[%d]:(%f, %f) ,", i, p[i].x, p[i].y);
  // }
  // printf("\n");
  float res = fabs(area(p, n));
  if (s1 * s2 == -1)
    res = -res;
  // printf("res=:%lf\n",res);
  return res;
}
bool falg = false;
//求两多边形的交面积
float intersectArea(Point *ps1, int n1, Point *ps2, int n2) {
  float area1 = area(ps1, n1);
  // printf("area1 = %f\n", area1);
  // printf("area2 = %f\n", area(ps2, n2));
  if (area(ps1, n1) < 0) {
    //falg = true;
    auto f1 = area(ps1, n1);
    // printf("reverse ps1 before:area1:%f\n", area(ps1, n1));
    // for(int i = 0;i<n1;i++)
    // {
    //     printf("(%f, %f), ", ps1[i].x, ps1[i].y);
    // }
    reverse(ps1, ps1 + n1);
    // printf("\n reverse area1 after:\n");
    //  for(int i = 0;i<n1;i++)
    // {
    //     printf("(%f, %f), ", ps1[i].x, ps1[i].y);
    // }
    auto f2 = area(ps1, n1);
    // if(fabs(f1) != fabs(f2)){
    //   printf("reverse ps1 before:area1:%f\n", f1);
    //   printf("reverse ps1 after:area1:%f\n", f2);
    // }
  }

  if (area(ps2, n2) < 0)
    reverse(ps2, ps2 + n2);

  ps1[n1] = ps1[0];
  ps2[n2] = ps2[0];
  float res = 0;
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
      // printf("intersectArea 11 res:%f\n", res);
    }
  }
  return res;  // assumeresispositive!
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
 
  // printf("fabs(area1 (ps1, n1)): %f \n", fabs(area(ps1, n1)));
  // printf("fabs(area2 (ps2, n2)): %f \n", fabs(area(ps2, n2)));
  // printf("areaI: %f \n", inter_area);
  // printf("areaU: %f \n", union_area);
  // printf("-------\n");
  float iou = 0;
  if (union_area == 0) {
    iou = (inter_area + 1) / (union_area + 1);
  } else {
    iou = inter_area / union_area;
  }
  // printf("-------\n");
  // cout << "inter_area:" << inter_area << endl;
  // cout << "area1:" << area(ps1, n1)<< endl;
  // cout << "area2:" << area(ps2, n2) << endl;
  // cout << "union_area:" << union_area << endl;
  // cout << "iou:" << iou << endl;

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
  //  printf("after soft: \n");
  // for(int i = 0;i<vp.size(); i++)
  // {
  //    printf("%d ", vp[i].second);
  // }
  // printf("\n");
  vector<int> keep;
  vector<int> test;

  while (vp.size()) {
    // printf("vp.size()=%ld, index=%d\n", vp.size(), vp.begin()->second);
    keep.push_back(vp.begin()->second);
    auto box = vp.begin()->first;
    auto box_index = vp.begin()->second;
    vp.erase(vp.begin());
    // printf("-----(p, scores):(%d:%f) ----------\n",box_index, box[8]);
    for (int i = 0; i < vp.size(); i++) {
      float iou = iou_poly(box, vp[i].first);
      if(box_index == 134 && vp[i].second == 0){
        printf(" -134 && 0-iou=%f\n", iou);
      }
      // printf("(%d, %f), iou=%f\n", vp[i].second, vp[i].first[8], iou);
      if (iou > thresh) {
        test.push_back(vp[i].second);
        vp.erase(vp.begin() + i);
        i--;
      }
    }
    sort(test.begin(), test.end(), [&](int a,int b){return a<b;});
    // for(int i = 0;i<test.size();i++){
    //     printf("%d, ", test[i]);
    // }
    // printf("\n");
    test.clear();
  }
  //  printf("before keep sort: \n");
  //  for(int i = 0;i< keep.size(); i++)
  // {
  //    printf("%d ", keep[i]);
  // }
  //  printf("\n after keep sort: \n");
  sort(keep.begin(), keep.end(), [&](int a, int b) { return a < b; });
  // printf("\n");

  // for(int i = 0;i< keep.size(); i++)
  // {
  //    printf("%d ", keep[i]);
  // }
  // printf("\n");

  // cout << "keep:" << keep.size() << endl;
  return keep;
}

// #include<time.h>

// #define random(x) (rand()%x)
// int main()
// {
//     double thresh = 0.1;
//     // double p[9] = {0, 0, 1, 1, -1, -1, 0, -1,1};
//     // double q[9] = {1, 0, 2, 0, 1, 1, 0, 1,2};

//     //  e r t 不相交，输出 【0 1 2】
//     double e[9] = {0, 0, 1, 0, 1, 1, 0, 1, 3};
//     double r[9] = {1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5, 2};
//     double t[9] = {0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 1};

//     // // e r相交 et,rt 不相交，输出 【0 2】
//     // double e[9] = {0, 0, 1, 0, 1, 1, 0, 1, 3};
//     // double r[9] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 2};
//     // double t[9] = {0, 0, -0.5, 0, -0.5, -0.5, 0, -0.5, 1};

//     // // e r t相交 e输出 【0】
//     // double e[9] = {0, 0, 1, 0, 1, 1, 0, 1, 3};
//     // double r[9] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 2};
//     // double t[9] = {0, 0, 0.5, 0, 0.5, 0.5, 0, 0.5, 1};

//     // vector<vector<double>>  m;
//     // vector<double> P(e, e + 9);
//     // vector<double> Q(r, r + 9);
//     // vector<double> R(t, t + 9);
//     // vector<vector<double>>  m;
//     // m.push_back(P);
//     // m.push_back(Q);
//     // m.push_back(R);

//     vector<vector<double>>  m;
//     vector<double> tmp;
//     srand((int)time(0));
//     for(int x=0;x<1000;x++)
//     {
//         if(x ==0 || x%9 != 0)
//         {
//             tmp.push_back(random(100));
//         }
//         else{
//             m.push_back(tmp);
//             tmp.clear();
//             tmp.push_back(random(100));
//         }
//     }
//     printf("m.size :%d\n", m.size());
//     for(int i = 0;i<m.size();i++)
//     {
//         for(int j = 0;j<m[i].size();j++)
//         {
//             printf(" : %f",m[i][j]);
//         }
//         printf("\n new :");
//     }

// // return 0;

//     vector<int> ret_index = pnms(m,thresh);
//    // printf("iou_poly: %f\n", ret);
//     for(int i = 0; i<ret_index.size();i++)
//     {
//         printf("ret_index:%i\n", ret_index[i]);
//     }
//     return 0;
// }

}  // namespace PNMS