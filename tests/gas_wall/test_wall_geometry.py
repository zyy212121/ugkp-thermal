"""Real polyhedral geometry contracts, compiled against the host-only builder."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = r'''
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
'''

def compile_run(tmp_path, body):
    src = tmp_path / 'geometry.cpp'
    src.write_text(BASE+'\nint main(){\n'+body+'\n}\n')
    exe = tmp_path / 'geometry'
    result = subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-pedantic','-I',str(ROOT),'-I',str(ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_real_cube_skew_prism_and_tetra_volume_quadrature(tmp_path):
    compile_run(tmp_path, r'''
 for(auto m:{cubes(),cubes(3,.2),tetra()}){
  WallGeometryDescriptor<double>d;WallStatus s;
  if(!buildWallGeometry(m,0,d,s)){std::cerr<<"geometry "<<int(s.code)<<" face "<<s.node<<"\n";return 2;}
  checkMoments(d,m.cellCentres[0],m.volumes[0]);assert(d.matchingCells[0]==1);
  assert(near(d.ownerDistance,m.cellCentres[0][2]));assert(near(d.normal[2],1));
  double y2=0;for(size_t i=0;i<d.distance.size();++i)y2+=d.volumeWeight[i]*d.distance[i]*d.distance[i];
  assert(near(y2,m.volumes[0]*(m.volumes[0]<1?.1:1./3.)));
 }
 ''')


def test_rotation_covariance_and_positive_refinement_convergence(tmp_path):
    compile_run(tmp_path, r'''
 auto m=tetra();WallGeometryDescriptor<double>d,a;WallStatus s;
 assert(buildWallGeometry(m,0,d,s));
 auto rotate=[](V p){return V{(p[0]-p[1])/std::sqrt(2.),(p[0]+p[1]-2*p[2])/std::sqrt(6.),(p[0]+p[1]+p[2])/std::sqrt(3.)};};
 for(auto& p:m.points)p=rotate(p);for(auto& p:m.cellCentres)p=rotate(p);
 assert(buildWallGeometry(m,0,a,s));checkMoments(a,m.cellCentres[0],m.volumes[0]);
 assert(near(a.matchingDistance,d.matchingDistance));assert(near(a.ownerDistance,d.ownerDistance));
 auto n=rotate(d.normal);for(int k=0;k<3;++k)assert(near(a.normal[k],n[k]));
 double previous=1;
 for(int level=0;level<3;++level){
  WallGeometryOptions<double>opt;opt.quadratureRefinement=level;
  assert(buildWallGeometry(m,0,a,s,opt));double fourth=0;
  for(size_t i=0;i<a.distance.size();++i)fourth+=a.volumeWeight[i]*std::pow(a.distance[i],4);
  double error=std::abs(fourth-m.volumes[0]/35.);assert(error<previous);previous=error;
 }
 ''')


def test_connected_matching_ray_and_constraint_exclusion(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();WallGeometryDescriptor<double>d;WallStatus s;
 m.omegaConstrained={1,1,0};assert(buildWallGeometry(m,0,d,s));assert(d.matchingCells[0]==2);assert(d.matchingDistance>2);
 double keep=d.matchingDistance;m.omegaConstrained={1,1,1};assert(!buildWallGeometry(m,0,d,s));assert(d.matchingDistance==keep);
 m.omegaConstrained={1,0,0};m.gasCell={1,0,1};assert(!buildWallGeometry(m,0,d,s));
 m=cubes(1);assert(!buildWallGeometry(m,0,d,s));
 m=cubes();m.physicalWall[1]=1;assert(!buildWallGeometry(m,0,d,s));
 m=cubes();WallGeometryOptions<double>o;o.maxRayCells=1;assert(!buildWallGeometry(m,0,d,s,o));
 ''')


def test_corners_bad_moments_and_wall_support_rejected_atomically(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(m,0,d,s));double keep=d.volume;
 m.physicalWall[2]=1;assert(!buildWallGeometry(m,0,d,s));assert(d.volume==keep);
 m=cubes();m.volumes[0]=.9;assert(!buildWallGeometry(m,0,d,s));
 m=cubes();m.cellCentres[0][2]=.4;assert(!buildWallGeometry(m,0,d,s));
 m=cubes();m.points[m.facePoints[0]][2]=.1;assert(!buildWallGeometry(m,0,d,s));
 m=cubes();m.facePoints[m.faceOffsets[2]]=999;assert(!buildWallGeometry(m,0,d,s));
 ''')


def test_geometry_version_rebuild_tracks_deformed_stage(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();WallGeometryCache<double> c;WallStatus s;
 assert(c.rebuild(m,{0},s));assert(c.descriptors.size()==1);checkMoments(c.descriptors[0],m.cellCentres[0],1);
 auto old=c.descriptors[0].matchingDistance;
 for(auto& p:m.points)p[2]*=1.3;for(auto& p:m.cellCentres)p[2]*=1.3;for(auto& v:m.volumes)v*=1.3;++m.geometryVersion;
 assert(c.rebuild(m,{0},s));checkMoments(c.descriptors[0],m.cellCentres[0],1.3);assert(near(c.descriptors[0].matchingDistance,1.3*old));
 assert(c.descriptors[0].geometryVersion==m.geometryVersion);
 auto* first=c.descriptors[0].distance.data();assert(c.rebuild(m,{0},s));assert(c.descriptors[0].distance.data()==first);
 m.volumes[0]=-1;++m.geometryVersion;assert(!c.rebuild(m,{0},s));assert(near(c.descriptors[0].volume,1.3));
 ''')


def test_concave_star_polyhedron_with_centroid_outside_kernel(tmp_path):
    compile_run(tmp_path,r'''
 // Long-arm L section: the exact volume centroid is outside the cell, but its
 // kernel is nonempty. The wall's area centroid is outside its polygon too.
 MeshBuilder b;
 const std::vector<V> base={{0,0,0},{4,0,0},{4,1,0},{1,1,0},{1,4,0},{0,4,0}};
 for(int layer=0;layer<3;++layer){
  std::vector<V> low,high;for(auto p:base){p[2]=layer;low.push_back(p);p[2]=layer+1;high.push_back(p);}
  std::vector<std::vector<V>> faces={low,high};
  for(size_t j=0;j<base.size();++j)faces.push_back({low[j],low[(j+1)%base.size()],high[(j+1)%base.size()],high[j]});
  // Give the helper a kernel seed solely to orient the test mesh, then replace
 // with the analytically exact centroid required by the production contract.
  b.cell(faces,{.5,.5,layer+.5},7);b.m.cellCentres.back()={19./14,19./14,layer+.5};
 }
 b.m.physicalWall[0]=1;WallGeometryDescriptor<double>d;WallStatus s;
 if(!buildWallGeometry(b.m,0,d,s)){std::cerr<<int(s.code)<<" "<<s.node<<"\n";return 2;}
 checkMoments(d,b.m.cellCentres[0],7);
 for(auto p:d.quadraturePoints){assert(p[0]>=0&&p[1]>=0&&p[0]<=4&&p[1]<=4);assert(p[0]<=1+1e-10||p[1]<=1+1e-10);}
 assert(d.origin[0]<=1||d.origin[1]<=1);
 ''')


def test_cache_rejects_duplicate_wall_ownership(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();WallGeometryCache<double> cache;WallStatus s;
 assert(!cache.rebuild(m,{0,0},s));assert(cache.descriptors.empty());
 ''')


def test_matching_stencil_is_explicitly_piecewise_constant(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes(3,.2);WallGeometryDescriptor<double>d;WallStatus s;
 assert(buildWallGeometry(m,0,d,s));assert(d.samplingOrder==0);
 const auto donor=m.cellCentres[d.matchingCells[0]];
 // A positive containing-cell sample is a zeroth-order reconstruction. It
 // must not be mistaken for a claimed affine-exact point interpolation.
 assert(std::abs(donor[0]-d.matchingPoint[0])>.1);
 assert(near(d.matchingPoint[0],d.origin[0]+d.matchingDistance*d.normal[0]));
 ''')


def test_matching_constant_preservation_and_first_order_mesh_error(tmp_path):
    compile_run(tmp_path,r'''
 double previous=0;
 for(double h:{1.,.5,.25,.125}){
  auto m=cubes(3,.2);for(auto&p:m.points)p=mul(p,h);for(auto&p:m.cellCentres)p=mul(p,h);for(auto&v:m.volumes)v*=h*h*h;
  WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(m,0,d,s));
  const auto donor=m.cellCentres[d.matchingCells[0]];
  const auto delta=sub(donor,d.matchingPoint);
  assert(near(d.matchingDonorDistance,std::sqrt(dot(delta,delta))));
  double constant=0,sampled=0;
  auto linear=[](V p){return 2+3*p[0]+4*p[1]+5*p[2];};
  for(size_t i=0;i<d.matchingWeights.size();++i){constant+=7*d.matchingWeights[i];sampled+=linear(m.cellCentres[d.matchingCells[i]])*d.matchingWeights[i];}
  assert(near(constant,7));double error=std::abs(sampled-linear(d.matchingPoint));assert(error>0);
  if(previous>0)assert(near(error,.5*previous));
  previous=error;
 }
 ''')


def test_profile_cut_quadrature_integrates_source_and_real_moments(tmp_path):
    compile_run(tmp_path,r'''
 for(auto mesh:{cubes(),cubes(3,.2),tetra()}){
  WallGeometryOptions<double> opt;opt.profileNodes=48;opt.profileStretch=2;
  WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(mesh,0,d,s,opt));
  checkMoments(d,mesh.cellCentres[0],mesh.volumes[0]);
  std::vector<double> y(opt.profileNodes),source(opt.profileNodes);
  for(int i=0;i<opt.profileNodes;++i){y[i]=d.matchingDistance*std::pow(double(i)/(opt.profileNodes-1),opt.profileStretch);source[i]=std::sin(2.7*i)+.1*i;}
  auto evaluate=[&](double p){int i=0;while(i+1<opt.profileNodes-1&&y[i+1]<p)++i;return source[i]+(source[i+1]-source[i])*(p-y[i])/(y[i+1]-y[i]);};
  double actual=0;for(size_t q=0;q<d.distance.size();++q)actual+=d.volumeWeight[q]*evaluate(d.distance[q]);
  // Independent antiderivative: true horizontal section area is one for the
  // (possibly skewed) cube, 2*(1-y)^2 for the wall tetrahedron.
  const bool tet=mesh.volumes[0]<1;double expected=0;
  for(int i=0;i<opt.profileNodes-1&&y[i]<1;++i){
   double lo=y[i],hi=std::min(1.,y[i+1]),b=(source[i+1]-source[i])/(y[i+1]-y[i]),a=source[i]-b*y[i];
   auto primitive=[&](double p){return tet?2*(a*p+(b-2*a)*p*p/2+(a-2*b)*p*p*p/3+b*p*p*p*p/4):a*p+b*p*p/2;};
   expected+=primitive(hi)-primitive(lo);
  }
  assert(near(actual,expected,2e-11));
  // Linear complexity in tetra count and profile nodes, independent of the
  // legacy recursive quadratureRefinement exponential.
  const int tets=tet?4:12;
  assert(d.distance.size()<=size_t(tets*(2*(opt.profileNodes+3)+8)));
  auto e=d.distance;opt.quadratureRefinement=4;assert(buildWallGeometry(mesh,0,d,s,opt));assert(e==d.distance);
 }
 ''')


def test_profile_first_segment_positive_gauss_converges(tmp_path):
    compile_run(tmp_path,r'''
 for(auto mesh:{cubes(),cubes(3,.2),tetra()})for(double ellFactor:{0.,1e-3,.4}){
  double previous=1;
  for(int order:{4,8,16}){
   WallGeometryOptions<double>opt;opt.profileNodes=24;opt.profileStretch=2;opt.firstSegmentGaussOrder=order;
   WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(mesh,0,d,s,opt));checkMoments(d,mesh.cellCentres[0],mesh.volumes[0]);
   double y1=d.matchingDistance/std::pow(23.,2),ell=ellFactor*y1,p=3.229468812791236;
   auto source=[&](double y){
    if(y>=y1)return 0.;
    double k=ell==0?std::pow(y/y1,p):std::pow((y+ell)/(y1+ell),p)*(-std::expm1(-(2*p-1)*std::log1p(y/ell)))/(-std::expm1(-(2*p-1)*std::log1p(y1/ell)));
    return k*y1*y1/((y+ell)*(y+ell))*std::exp(.6*(y/y1-1));
   };
   double actual=0;for(size_t i=0;i<d.distance.size();++i)actual+=d.volumeWeight[i]*source(d.distance[i]);
   // Independent fine midpoint integration after y=y1*t^4 regularization.
   double reference=0;const int count=262144;
   for(int i=0;i<count;++i){double t=(i+.5)/count,y=y1*std::pow(t,4),area=mesh.volumes[0]<1?2*(1-y)*(1-y):1;
    reference+=source(y)*area*4*y1*t*t*t/count;}
   double error=std::abs(actual-reference)/reference;
   assert(error<previous);previous=error;
   if(order==16)assert(error<2e-8);
  }
 }
 ''')


def test_profile_configuration_cache_key_and_atomic_rejection(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();WallGeometryCache<double>c;WallStatus s;WallGeometryOptions<double>o;
 assert(c.rebuild(m,{0},s,o));auto legacy=c.descriptors[0].distance;
 o.profileNodes=24;assert(c.rebuild(m,{0},s,o));assert(c.descriptors[0].distance!=legacy);
 auto first=c.descriptors[0].distance.data();assert(c.rebuild(m,{0},s,o));assert(c.descriptors[0].distance.data()==first);
 auto y=c.descriptors[0].distance;o.profileStretch=3;assert(c.rebuild(m,{0},s,o));assert(c.descriptors[0].distance!=y);
 y=c.descriptors[0].distance;o.firstSegmentGaussOrder=16;assert(c.rebuild(m,{0},s,o));assert(c.descriptors[0].distance.size()>y.size());
 y=c.descriptors[0].distance;o.profileNodes=3;assert(!c.rebuild(m,{0},s,o));assert(c.descriptors[0].distance==y);
 o.profileNodes=24;o.firstSegmentGaussOrder=3;assert(!c.rebuild(m,{0},s,o));
 o.firstSegmentGaussOrder=8;o.profileStretch=.5;assert(!c.rebuild(m,{0},s,o));
 ''')


def test_profile_section_aggregation_bounds_resident_storage(tmp_path):
    compile_run(tmp_path,r'''
 for(auto mesh:{cubes(),cubes(3,.2),tetra()}){
  WallGeometryOptions<double>opt;opt.profileNodes=128;opt.profileStretch=2;
  WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(mesh,0,d,s,opt));
  checkMoments(d,mesh.cellCentres[0],mesh.volumes[0]);
  assert(!d.quadraturePointsAreInterior);
  const int ownerVertices=mesh.volumes[0]<1?4:8;
  assert(d.distance.size()<=size_t(2*(opt.profileNodes+ownerVertices+1)+8));
  // The transmitted view needs just these compact scalar samples. Full
  // tetrahedral geometry and cut searches are not repeated during a gas stage.
  assert(d.quadrature().count==int(d.distance.size()));
 }
 ''')


def test_profile_rotations_and_nearly_coincident_cuts(tmp_path):
    compile_run(tmp_path,r'''
 for(auto mesh:{cubes(),cubes(3,.2),tetra()}){
  auto rotate=[](V p){return V{(p[0]-p[1])/std::sqrt(2.),(p[0]+p[1]-2*p[2])/std::sqrt(6.),(p[0]+p[1]+p[2])/std::sqrt(3.)};};
  for(auto&p:mesh.points)p=rotate(p);for(auto&p:mesh.cellCentres)p=rotate(p);
  WallGeometryOptions<double>o;o.profileNodes=4;o.profileStretch=1+2*std::numeric_limits<double>::epsilon();
  WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(mesh,0,d,s,o));checkMoments(d,mesh.cellCentres[0],mesh.volumes[0]);
 }
 ''')


def test_concave_profile_quadrature_preserves_actual_section_moments(tmp_path):
    compile_run(tmp_path,r'''
 MeshBuilder b;const std::vector<V>base={{0,0,0},{4,0,0},{4,1,0},{1,1,0},{1,4,0},{0,4,0}};
 for(int layer=0;layer<3;++layer){
  std::vector<V>low,high;for(auto p:base){p[2]=layer;low.push_back(p);p[2]=layer+1;high.push_back(p);}
  std::vector<std::vector<V>>faces={low,high};for(size_t j=0;j<base.size();++j)faces.push_back({low[j],low[(j+1)%base.size()],high[(j+1)%base.size()],high[j]});
  b.cell(faces,{.5,.5,layer+.5},7);b.m.cellCentres.back()={19./14,19./14,layer+.5};
 }
 b.m.physicalWall[0]=1;WallGeometryOptions<double>o;o.profileNodes=48;
 WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(b.m,0,d,s,o));checkMoments(d,b.m.cellCentres[0],7);
 assert(!d.quadraturePointsAreInterior);
 // The diagnostic section centroid lies outside this L section. No interior
 // claim is made; positive normal measure still comes solely from real tetra.
 for(auto p:d.quadraturePoints)assert(p[0]>1&&p[1]>1);
 double integral=0;for(size_t i=0;i<d.distance.size();++i)integral+=d.volumeWeight[i]*(1+3*d.distance[i]);
 assert(near(integral,17.5));
 ''')


def test_standalone_profile_probe_reports_numeric_acceptance(tmp_path):
    import json
    exe = tmp_path / 'profile_probe'
    subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),'-I',str(ROOT/'common'),str(ROOT/'tests/gas_wall/test_wall_geometry_profile_probe.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True)
    result = subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 9
    assert {row['geometry'] for row in rows} == {'cube', 'skew_prism', 'tetra'}
    for row in rows:
        assert row['absolute_error'] <= row['threshold']
        assert row['threshold'] == 2e-11 * (1 + abs(row['independent_exact']))
        assert row['quadrature_points'] <= row['point_bound']


def test_mss7_tip_sections_are_translation_and_scale_stable(tmp_path):
    compile_run(tmp_path,r'''
 // Recorded certified tetra and normal cut from actual MSS7 wall face 9063.
 const std::array<V,4> original={{{.037112221270554802,.064554715745008068,-2.9735989201637231e-19},
  {.03801563464947668,.064938134403000003,-.0028352601790000002},
  {.03801563464947668,.064938134403000003,.0028352601789999998},
  {.038015634649476687,.064169772833611999,.0028017127883829536}}};
 const std::array<double,4> h0={{.0003834186579919352,0,0,.00076836156938800415}};
 for(double scale:{std::ldexp(1.,-12),1.,std::ldexp(1.,12)})for(V shift:{V{0,0,0},V{2,-3,5}}){
  std::array<V,4>p=original;std::array<double,4>h=h0;
  for(auto&v:p)v=add(mul(v,scale),shift);for(auto&v:h)v*=scale;
  WallGeometryDescriptor<double>d;d.normal={0,-1,0};d.tangent1={1,0,0};d.tangent2={0,0,1};
  for(double y:{.00076836156938800122*scale,std::nextafter(h[3],0.),.75*h[3]}){
   double area=0;V centre{};
   assert(wallGeometryDetail::tetraSection(p,h,y,d,area,centre));assert(area>0);
   // Independent triangle oracle: all three crossings meet at vertex 3.
   std::array<std::array<long double,3>,3> q;
   for(int i=0;i<3;++i)for(int k=0;k<3;++k)
    q[i][k]=((long double)p[i][k]-p[3][k])*(((long double)h[3]-y)/((long double)h[3]-h[i]));
   long double ax=q[1][0]-q[0][0],az=q[1][2]-q[0][2],bx=q[2][0]-q[0][0],bz=q[2][2]-q[0][2];
   const long double reference=std::abs(ax*bz-az*bx)/2;
   assert(std::abs((long double)area/reference-1)<1e-10L);
   for(int k=0;k<3;++k){long double expected=p[3][k]+(q[0][k]+q[1][k]+q[2][k])/3;
    assert(std::abs((long double)centre[k]-expected)<8*std::numeric_limits<double>::epsilon()*(1+std::abs(expected)));}
  }
  double area=0;V centre{};
  assert(!wallGeometryDetail::tetraSection(p,h,h[3],d,area,centre));
  for(auto&v:p)v[0]=p[3][0]; // Truly zero-area section must remain rejected.
  assert(!wallGeometryDetail::tetraSection(p,h,.75*h[3],d,area,centre));
 }
 ''')


def test_mss7_face_9063_owner_profile_preserves_positive_measure(tmp_path):
    compile_run(tmp_path,r'''
 // Actual imported MSS7 owner and its unconstrained matching donor. Preserve
 // input last-bit vertex differences that generated the near-tip cut.
 MeshBuilder b;
 const std::vector<V>p={
  {0.036208807891632916,0.064169772833612013,-0.0028017127883829536},
  {0.036208807891632916,0.064169772833612013,0.0028017127883829536},
  {0.038015634649476687,0.064169772833611999,0.0028017127883829536},
  {0.038015634649476687,0.064169772833611999,-0.0028017127883829536},
  {0.036208807891632916,0.064938134403000003,-0.0028352601790000002},
  {0.036208807891632916,0.064938134403000003,0.0028352601789999998},
  {0.03801563464947668,0.064938134403000003,-0.0028352601790000002},
  {0.03801563464947668,0.064938134403000003,0.0028352601789999998},
  {0.037112221270554802,0.064554715745008068,-2.9735989201637231e-19},
  {0.036208807891632916,0.063345481424703851,-0.0027657234482355224},
  {0.036208807891632916,0.063345481424703851,0.0027657234482355224},
  {0.03801563464947668,0.063345481424703837,0.0027657234482355228},
  {0.03801563464947668,0.063345481424703837,-0.0027657234482355228},
  {0.037112221270554802,0.063758515201112864,-2.3585359738380786e-20}
 };
 b.cell({{p[0],p[1],p[2],p[3]},{p[0],p[4],p[5],p[1]},{p[3],p[6],p[7],p[2]},{p[4],p[5],p[7],p[6]},{p[0],p[4],p[6],p[3]},{p[1],p[2],p[7],p[5]},},p[8],7.8257883940270517e-09);
 b.cell({{p[9],p[10],p[11],p[12]},{p[9],p[0],p[1],p[10]},{p[12],p[3],p[2],p[11]},{p[0],p[1],p[2],p[3]},{p[9],p[0],p[3],p[12]},{p[10],p[11],p[2],p[1]},},p[13],8.2918710349492206e-09);

 b.m.physicalWall[3]=1;
 for(int nodes:{4,24,48,128}){
  WallGeometryOptions<double>o;o.profileNodes=nodes;
  WallGeometryDescriptor<double>d;WallStatus status;
  assert(buildWallGeometry(b.m,3,d,status,o));
  checkMoments(d,b.m.cellCentres[0],b.m.volumes[0]);
  assert(d.matchingCells[0]==1);
  double volume=0,normalMoment=0;V offsetMoment{};
  for(size_t i=0;i<d.distance.size();++i){
   volume+=d.volumeWeight[i];normalMoment+=d.distance[i]*d.volumeWeight[i];
   offsetMoment=add(offsetMoment,mul(sub(d.quadraturePoints[i],b.m.cellCentres[0]),d.volumeWeight[i]));
  }
  assert(std::abs(volume/b.m.volumes[0]-1)<1e-12);
  assert(std::abs(normalMoment/(b.m.volumes[0]*d.ownerDistance)-1)<1e-12);
  assert(std::sqrt(dot(offsetMoment,offsetMoment))<1e-12*b.m.volumes[0]*.006);
 }
 ''')
