#include "gpu/PolyhedralGeometry.H"
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
using namespace ugkp::gpu::geometry;
// The baseline and candidate executables print the actual device result bits.
// Repeated calls exercise scratch lifetime without a second distance formula.
void emitResults(const char* name,const std::vector<WallDistanceResult>& result){
 for(const auto& q:result){std::uint64_t bits=0;static_assert(sizeof(bits)==sizeof(q.distance),"FP64 distance required");
  std::memcpy(&bits,&q.distance,sizeof(bits));
  std::cout<<name<<" distance_bits="<<std::hex<<bits<<std::dec<<" invalid="<<q.invalidFace<<'\n';}
}
void exactResults(const std::vector<WallDistanceResult>& a,const std::vector<WallDistanceResult>& b){
 assert(a.size()==b.size());for(std::size_t i=0;i<a.size();++i){
  assert(a[i].invalidFace==b[i].invalidFace);
  assert(std::memcmp(&a[i].distance,&b[i].distance,sizeof(double))==0);}
}
void repeatedCoordinateQueries(PolyhedralGeometry& gpu){
 Topology t;t.faceOffsets={0,3,6};t.facePoints={0,1,2,3,4,5};
 const std::vector<Point> original={{-2,-2,0},{2,-2,0},{0,2,0},{-2,-2,8},{2,-2,8},{0,2,8}};
 const std::vector<Point> centres={{0,0,1}};auto points=original;
 const auto query=[&](const char* name,const std::vector<int>& walls){
  auto result=gpu.wallDistance(t,points,centres,walls);emitResults(name,result);return result;};
 const auto initial=query("reuse-initial",{0,1});assert(initial[0].invalidFace<0&&initial[0].distance==1);
 // The formerly distant wall becomes closest. Its old box would incorrectly
 // prune it against the unchanged wall, so this specifically tests invalidation.
 for(int i=3;i<6;++i)points[i].z=1.25;
 const auto moved=query("reuse-moved",{0,1});assert(moved[0].invalidFace<0&&moved[0].distance==.25);
 exactResults(moved,query("reuse-reordered",{1,0}));
 const auto empty=query("reuse-empty",{});assert(empty[0].invalidFace<0&&empty[0].distance==0);
 exactResults(moved,query("reuse-after-empty",{0,1}));
 const auto subset=query("reuse-subset",{0});exactResults(initial,subset);
 const auto valid=points;points[0].x=std::numeric_limits<double>::quiet_NaN();
 points[3].z=std::numeric_limits<double>::infinity();
 const auto invalidForward=query("reuse-invalid-forward",{0,1});assert(invalidForward[0].invalidFace==0);
 const auto invalidReverse=query("reuse-invalid-reverse",{1,0});assert(invalidReverse[0].invalidFace==1);
 points[3]=valid[3];const auto partlyRepaired=query("reuse-one-repaired",{1,0});assert(partlyRepaired[0].invalidFace==0);
 points[0]=valid[0];exactResults(moved,query("reuse-all-repaired",{1,0}));
 points=original;exactResults(initial,query("reuse-original-restored",{0,1}));
}
void repeatedRebuildQueries(PolyhedralGeometry& gpu){
 Topology t;t.faceOffsets={0,4,8,12,16,20,24};
 t.facePoints={0,3,2,1,4,5,6,7,0,4,7,3,1,2,6,5,0,1,5,4,3,7,6,2};
 t.cellOffsets={0,6};t.cellFaces={0,1,2,3,4,5};t.cellSigns={1,1,1,1,1,1};
 t.vertexOffsets={0,8};t.vertices={0,1,2,3,4,5,6,7};
 const std::vector<Point> original={{0,0,0},{8,0,0},{8,8,0},{0,8,0},{0,0,8},{8,0,8},{8,8,8},{0,8,8}};
 auto points=original;
 const auto query=[&](const char* name,const std::vector<int>& walls){
  auto result=gpu.rebuild(t,points,walls);assert(result.cells.size()==1&&!result.cells[0].invalid);
  emitResults(name,result.distance);return result.distance;};
 const auto initial=query("rebuild-initial",{2,1});assert(initial[0].invalidFace<0&&std::abs(initial[0].distance-4)<1e-13);
 // Keep x=0 as an unchanged alternative while the upper z wall moves closer.
 for(int i=4;i<8;++i)points[i].z=2;
 const auto moved=query("rebuild-moved",{2,1});assert(moved[0].invalidFace<0&&std::abs(moved[0].distance-1)<1e-13);
 exactResults(moved,query("rebuild-reordered",{1,2}));
 const auto empty=query("rebuild-empty",{});assert(empty[0].invalidFace<0&&empty[0].distance==0);
 // A distinct endpoint after the empty call forces fresh work, not an LRU hit.
 for(int i=4;i<8;++i)points[i].z=3;
 const auto next=query("rebuild-after-empty",{1,2});assert(next[0].invalidFace<0&&std::abs(next[0].distance-1.5)<1e-13);
 points=original;exactResults(initial,query("rebuild-original-restored",{2,1}));
}
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
 repeatedCoordinateQueries(gpu);repeatedRebuildQueries(gpu);
 std::cout<<"WALL_DISTANCE_PASS queries="<<queries<<" empty_and_invalid=6 repeated_coordinate_calls=11 repeated_rebuild_calls=6\n";
}
