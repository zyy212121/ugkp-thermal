
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
int failures=0;
void run(const char*name,WallGeometryMesh<double> m){
 wallGeometryDetail::Context<double> ctx;WallStatus status;WallGeometryOptions<double> options;
 bool prep=wallGeometryDetail::prepare(m,ctx,status,options);V seed{};double scale=0;bool seedok=prep&&wallGeometryDetail::cellSeed(m,ctx,0,seed,scale);
 double violation=0,inside=1e300;if(prep)for(int f:ctx.cellFaces[0]){auto n=wallGeometryDetail::mul(ctx.faces[f].normal,m.owner[f]==0?1.:-1.);double dist=wallGeometryDetail::dot(wallGeometryDetail::sub(seed,ctx.faces[f].centre),n);violation=std::max(violation,dist);inside=std::min(inside,-dist);}
 WallGeometryDescriptor<double>d;ugkp::diagnostics::DeviceContext context{};options.diagnostics=&context;options.constructQuadrature=false;bool ok=buildWallGeometry(m,0,d,status,options);double sum=0,minweight=1e300;V moment{};for(size_t i=0;i<d.volumeWeight.size();++i){sum+=d.volumeWeight[i];minweight=std::min(minweight,d.volumeWeight[i]);moment=add(moment,mul(d.quadraturePoints[i],d.volumeWeight[i]));}
 double err=0;for(int k=0;k<3;++k)err=std::max(err,std::abs(moment[k]-m.volumes[0]*m.cellCentres[0][k]));
 if(!prep||!ok||!d.volumeWeight.empty()||!d.distance.empty()||context.fatalClaim||context.warnings[1].count||std::abs(d.geometryVolume-m.volumes[0])>1e-10*m.volumes[0])++failures;
 std::cout<<name<<","<<prep<<","<<seedok<<","<<seed[0]<<","<<seed[1]<<","<<seed[2]<<","<<inside<<","<<violation<<","<<ok<<","<<int(status.code)<<","<<d.origin[0]<<","<<d.origin[1]<<","<<d.origin[2]<<","<<d.matchingDistance<<","<<sum<<","<<err<<","<<minweight<<"\n";
}

int main(){std::cout<<std::setprecision(17);std::cout<<"case,prepare,seed_ok,seed_x,seed_y,seed_z,min_interior_distance,halfspace_violation,build_ok,status,origin_x,origin_y,origin_z,matching_distance,quadrature_volume,first_moment_error,min_weight\n";
for(double L:{10.,100.,200.,300.,1000.}){MeshBuilder b;std::vector<V>p={{0,0,0},{L,0,0},{L,1,0},{1,1,0},{1,L,0},{0,L,0}};double area=2*L-1,c=(L*L+L-1)/(2*(2*L-1));
for(int z=0;z<3;++z){std::vector<V>lo,hi;for(auto q:p){q[2]=z*L;lo.push_back(q);q[2]=(z+1)*L;hi.push_back(q);}std::vector<std::vector<V>>f={lo,hi};for(size_t j=0;j<p.size();++j)f.push_back({lo[j],lo[(j+1)%p.size()],hi[(j+1)%p.size()],hi[j]});b.cell(f,{.5,.5,(z+.5)*L},area*L);b.m.cellCentres.back()={c,c,(z+.5)*L};}b.m.physicalWall[0]=1;run(("L_"+std::to_string(L)).c_str(),b.m);}for(double thin:{1e-3,1e-8,1e-12,1e-14}){auto m=cubes();for(auto&p:m.points)p[0]*=thin;for(auto&p:m.cellCentres)p[0]*=thin;for(auto&v:m.volumes)v*=thin;run(("thin_"+std::to_string(thin)).c_str(),m);}run("cube",cubes());run("skew",cubes(3,.2));run("tetra",tetra());
 for(int shape=0;shape<2;++shape){MeshBuilder b;std::vector<V> p=shape?std::vector<V>{{0,0,0},{4,0,0},{4,1,0},{1,1,0},{1,4,0},{0,4,0}}:std::vector<V>{{0,0,0},{2,0,0},{2,1,0},{1.5,1,0},{1.5,2,0},{0,2,0}};
 for(int z=0;z<3;++z){std::vector<V>lo,hi;for(auto q:p){q[2]=z;lo.push_back(q);q[2]=z+1;hi.push_back(q);}std::vector<std::vector<V>>f={lo,hi};for(size_t j=0;j<p.size();++j)f.push_back({lo[j],lo[(j+1)%p.size()],hi[(j+1)%p.size()],hi[j]});b.cell(f,{.5,.5,z+.5},shape?7:3.5);b.m.cellCentres.back()=shape?V{9.5/7,9.5/7,z+.5}:V{3.125/3.5,3.25/3.5,z+.5};}b.m.physicalWall[0]=1;run(shape?"concave_centroid_outside":"concave_centroid_inside",b.m);}
return failures?1:0;}
