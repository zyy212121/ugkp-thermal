#include "gpu/PolyhedralGeometry.H"
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>
using namespace ugkp::gpu::geometry;
int main(){
 PolyhedralGeometry gpu;std::mt19937 gen(1789);std::uniform_real_distribution<double> unit(-1.,1.);
 int queries=0;
 for(double scale:{1e-200,1e-6,1.,1e100,1e200,1e307}){
  Topology t;t.faceOffsets.push_back(0);std::vector<Point> points,centres;std::vector<int> walls;
  for(int f=0;f<257;++f){const double x=unit(gen)*scale,y=unit(gen)*scale,z=unit(gen)*scale;
   for(Point q:std::vector<Point>{{x,y,z},{x+.01*scale,y,z},{x,y+.02*scale,z}}){t.facePoints.push_back(points.size());points.push_back(q);}
   t.faceOffsets.push_back(points.size());walls.push_back(f);
  }
  // Degenerate triangles are still valid distance queries.
  points[3]=points[4]=points[5];
  for(int c=0;c<33;++c)centres.push_back({unit(gen)*scale,unit(gen)*scale,unit(gen)*scale});
  centres.push_back(points[0]);
  for(int order=0;order<2;++order){if(order)std::reverse(walls.begin(),walls.end());
   auto result=gpu.wallDistance(t,points,centres,walls);
   for(auto q:result){assert(q.invalidFace<0);std::cout<<std::hexfloat<<q.distance<<'\n';++queries;}}
  auto empty=gpu.wallDistance(t,points,centres,{});for(auto q:empty)assert(q.distance==0&&q.invalidFace<0);
  points[11].x=std::numeric_limits<double>::quiet_NaN();points[60].z=std::numeric_limits<double>::infinity();
  auto invalid=gpu.wallDistance(t,points,centres,walls);for(auto q:invalid){assert(q.invalidFace==20);std::cout<<"invalid "<<q.invalidFace<<'\n';}
 }
 std::cout<<"WALL_DISTANCE_PASS queries="<<queries<<" empty_and_invalid=6\n";
}
