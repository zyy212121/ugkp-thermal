#include "gpu/PolyhedralGeometry.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
using namespace ugkp::gpu::geometry;
int main(){
 double previous=-1;
 for(double h:{1e-7,1e-5,1e-3,1.})for(double recession:{-.4,.2}){
  const std::vector<Point> p{{0,0,0},{0,h,0},{0,3*h,0},{0,7*h,0}};
  std::vector<int> rows{0,1,3,5,6},adj{1,0,2,1,3,2};std::vector<unsigned char> fixed{1,0,0,1};
  std::vector<Point> b(4);b[0].y=recession*h;
  std::vector<MotionPenalty> penalty(4);addNormalHeight(penalty[1],{0,1,0},recession*h,h);
  HarmonicPointMotion owner(p,rows,adj,fixed,1e-14*h,20000,penalty);const auto out=owner.solve(b);
  assert(out.errorPoint<0&&out.residual<=1e-14*h);
  // Independent scalar 2x2 stationarity solve for the two free points.
  const double w01=1/h,w12=1/(2*h),w23=1/(4*h),k=1000/h;
  const double a=w01+w12+k,bb=-w12,c=w12+w23;
  const double rhs=(w01+k)*recession*h;
  const double u1=rhs*c/(a*c-bb*bb),u2=-bb*rhs/(a*c-bb*bb);
  assert(std::abs(out.points[1].y-(h+u1))<2e-13*h);
  assert(std::abs(out.points[2].y-(3*h+u2))<2e-13*h);
  const double ratio=(out.points[1].y-out.points[0].y)/h;
  assert(std::abs(ratio-1)<1e-4);
  assert(out.points[0].y==b[0].y&&out.points[3].y==p[3].y);
  std::cout<<std::setprecision(17)<<h<<' '<<recession<<' '<<ratio<<' '<<out.residual/h<<'\n';
 }
 // Two independent normals at a corner; no determinant-based cancellation.
 MotionPenalty q;addNormalHeight(q,{1,0,0},.3,1e-4);addNormalHeight(q,{0,1,0},-.2,2e-4);
 const auto v=motionUpdate({2,4,6},2,q);assert(std::abs(v.x-(2+3e6)/(2+1e7))<1e-14);
 assert(std::abs(v.y-(4-1e6)/(2+5e6))<1e-14&&std::abs(v.z-3)<1e-14);
 std::cout<<"NORMAL_HEIGHT_PASS\n";
}
