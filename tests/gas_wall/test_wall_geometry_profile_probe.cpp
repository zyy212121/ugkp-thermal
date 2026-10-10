// Standalone host accuracy probe. Compile with -I<repo> -I<repo>/common.

#include "common/gasWall/WallGeometryBuilder.H"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
using namespace ugkwp::gaswall;
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double b){return {a[0]*b,a[1]*b,a[2]*b};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<r*(1+std::abs(b));}
struct MeshBuilder {
 WallGeometryMesh<double> m;
 std::map<std::vector<int>,int> seen;
 MeshBuilder(){m.faceOffsets.push_back(0);}
 int point(V p){for(int i=0;i<int(m.points.size());++i)if(m.points[i]==p)return i;m.points.push_back(p);return int(m.points.size())-1;}
 void cell(const std::vector<std::vector<V>>& faces,V centre,double volume){
  int c=int(m.cellCentres.size());m.cellCentres.push_back(centre);m.volumes.push_back(volume);
  for(auto f:faces){
   if(dot(cross(sub(f[1],f[0]),sub(f[2],f[0])),sub(f[0],centre))<0)std::reverse(f.begin(),f.end());
   std::vector<int> ids;for(auto p:f)ids.push_back(point(p));auto key=ids;std::sort(key.begin(),key.end());
   auto old=seen.find(key);if(old!=seen.end()){assert(m.neighbour[old->second]<0);m.neighbour[old->second]=c;continue;}
   int idx=int(m.owner.size());seen[key]=idx;m.owner.push_back(c);m.neighbour.push_back(-1);m.physicalWall.push_back(0);
   m.facePoints.insert(m.facePoints.end(),ids.begin(),ids.end());m.faceOffsets.push_back(int(m.facePoints.size()));
  }
 }
};
WallGeometryMesh<double> cubes(int count=3,double skew=0){
 MeshBuilder b;
 for(int k=0;k<count;++k){
  V a={skew*k,0,double(k)},bb={1+skew*k,0,double(k)},c={1+skew*k,1,double(k)},d={skew*k,1,double(k)};
  V e={skew*(k+1),0,double(k+1)},f={1+skew*(k+1),0,double(k+1)},g={1+skew*(k+1),1,double(k+1)},h={skew*(k+1),1,double(k+1)};
  b.cell({{a,bb,c,d},{e,f,g,h},{a,bb,f,e},{bb,c,g,f},{c,d,h,g},{d,a,e,h}}, {0.5+skew*(k+.5),.5,k+.5},1);
 }
 b.m.physicalWall[0]=1;return b.m;
}
WallGeometryMesh<double> tetra(){
 MeshBuilder b;V a={0,0,0},bb={2,0,0},c={0,2,0},d={.2,.3,1};
 b.cell({{a,bb,c},{a,bb,d},{bb,c,d},{c,a,d}}, {.55,.575,.25},2./3.);
 V e=add(bb,{0,0,3}),f=add(c,{0,0,3}),g=add(d,{0,0,3});
 b.cell({{bb,c,d},{e,f,g},{bb,c,f,e},{c,d,g,f},{d,bb,e,g}}, {2.2/3,2.3/3,11./6},4.5);
 b.m.physicalWall[0]=1;return b.m;
}
void checkMoments(const WallGeometryDescriptor<double>& d,V expected,double volume){
 double v=0,y=0,y2=0;V p={0,0,0};
 assert(d.distance.size()==d.volumeWeight.size());assert(d.distance.size()==d.quadraturePoints.size());
 for(size_t i=0;i<d.distance.size();++i){
  assert(d.volumeWeight[i]>0);assert(d.distance[i]>0);assert(d.distance[i]<d.matchingDistance);
  v+=d.volumeWeight[i];y+=d.volumeWeight[i]*d.distance[i];y2+=d.volumeWeight[i]*d.distance[i]*d.distance[i];
  p=add(p,mul(d.quadraturePoints[i],d.volumeWeight[i]));
 }
 assert(near(v,volume));assert(near(d.volume,volume));assert(near(y,d.firstMoment));
 for(int k=0;k<3;++k){assert(near(p[k],volume*expected[k]));assert(near(d.firstMomentVector[k],p[k]));}
 assert(d.matchingDistance>d.maximumOwnerDistance);assert(d.matchingCells.size()==1);
 assert(d.matchingWeights[0]>0&&near(d.matchingWeights[0],1));
 auto q=d.quadrature();assert(q.count==int(d.distance.size()));assert(q.distance==d.distance.data());
 assert(near(q.volume,volume));assert(near(q.firstMoment,y));
}

#include <iomanip>
int main(){
 const char* names[]={"cube","skew_prism","tetra"};int meshIndex=0;
 for(auto mesh:{cubes(),cubes(3,.2),tetra()}){
  for(int nodes:{24,48,128}){
   WallGeometryOptions<double>opt;opt.profileNodes=nodes;opt.profileStretch=2;
   WallGeometryDescriptor<double>d;WallStatus status;
   if(!buildWallGeometry(mesh,0,d,status,opt)){std::cerr<<"geometry failure "<<int(status.code)<<" face "<<status.node<<"\n";return 2;}
   checkMoments(d,mesh.cellCentres[0],mesh.volumes[0]);
   std::vector<double>y(nodes),source(nodes);
   for(int i=0;i<nodes;++i){y[i]=d.matchingDistance*std::pow(double(i)/(nodes-1),2);source[i]=std::sin(2.7*i)+.1*i;}
   auto evaluate=[&](double p){int i=0;while(i+1<nodes-1&&y[i+1]<p)++i;return source[i]+(source[i+1]-source[i])*(p-y[i])/(y[i+1]-y[i]);};
   double actual=0;for(size_t q=0;q<d.distance.size();++q)actual+=d.volumeWeight[q]*evaluate(d.distance[q]);
   const bool tet=meshIndex==2;double expected=0;
   for(int i=0;i<nodes-1&&y[i]<1;++i){
    const double lo=y[i],hi=std::min(1.,y[i+1]),b=(source[i+1]-source[i])/(y[i+1]-y[i]),a=source[i]-b*y[i];
    auto primitive=[&](double p){return tet?2*(a*p+(b-2*a)*p*p/2+(a-2*b)*p*p*p/3+b*p*p*p*p/4):a*p+b*p*p/2;};
    expected+=primitive(hi)-primitive(lo);
   }
   const double error=std::abs(actual-expected),threshold=2e-11*(1+std::abs(expected));
   const int pointBound=2*(nodes+(tet?4:8)+1)+8;
   std::cout<<std::setprecision(17)<<"{\"geometry\":\""<<names[meshIndex]<<"\",\"profile_nodes\":"<<nodes
    <<",\"quadrature_points\":"<<d.distance.size()<<",\"point_bound\":"<<pointBound
    <<",\"source_integral\":"<<actual<<",\"independent_exact\":"<<expected
    <<",\"absolute_error\":"<<error<<",\"threshold\":"<<threshold<<"}\n";
   if(error>threshold||d.distance.size()>size_t(pointBound)||d.quadraturePointsAreInterior)return 3;
  }
  ++meshIndex;
 }
}
