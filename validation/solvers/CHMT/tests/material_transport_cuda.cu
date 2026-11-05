// Preserved CPU transport is linked as a separate translation unit.
#include "materials/MaterialTransport.H"
#include "materials/MaterialTransportMath.H"
#if defined(__CUDACC__)
#include "gpu/MaterialTransportKernels.cuh"
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace chmt {
bool baselineEvaluateMaterialTransport(const HostMesh&,const std::vector<SolidQ>&,const PhysicsConfig&,
    const std::vector<Real>&,Real,Real,MaterialTransportResult&,std::string&,bool*);
}
using namespace chmt;
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
constexpr double pi=3.14159265358979323846;
int idx(int i,int j,int k,int n){return (k*n+j)*n+i;}
bool interior(int c,int n,int margin){int i=c%n,j=c/n%n,k=c/n/n;return i>=margin&&j>=margin&&k>=margin&&i<n-margin&&j<n-margin&&k<n-margin;}
Vec3 transform(Vec3 x,double shear){return {x.x+shear*x.y,x.y,x.z};}
HostMesh cube(int n,double shear=0,bool periodic=false,bool stretched=false){
 HostMesh m;const double h=1./n;
 auto edge=[&](int i){return i*h+(stretched?.035*std::sin(2*pi*i*h):0);};
 auto centre=[&](int i){return .5*(edge(i)+edge(i+1));};
 auto width=[&](int i){return edge(i+1)-edge(i);};std::vector<std::vector<std::pair<int,int>>> entries(n*n*n);
 for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){m.cellCentres.push_back(transform({centre(i),centre(j),centre(k)},shear));m.volumes.push_back(width(i)*width(j)*width(k));}
 auto face=[&](int o,int r,Vec3 fc,Vec3 S,BoundaryKind kind){int f=m.owner.size();m.owner.push_back(o);m.neighbour.push_back(r);m.faceCentres.push_back(fc);m.areaVectors.push_back(S);m.boundaryKind.push_back(kind);m.periodicPartner.push_back(-1);entries[o].push_back({f,1});if(r>=0)entries[r].push_back({f,-1});return f;};
 for(int axis=0;axis<3;++axis)for(int b=0;b<n;++b)for(int a=0;a<n;++a){int low=-1,high=-1;
  for(int t=0;t<=n;++t){int xyz[3];xyz[axis]=t;xyz[(axis+1)%3]=a;xyz[(axis+2)%3]=b;
   Vec3 fc={axis==0?edge(t):centre(xyz[0]),axis==1?edge(t):centre(xyz[1]),axis==2?edge(t):centre(xyz[2])};
   const double area=width(a)*width(b);
   Vec3 S=axis==0?Vec3{area,-shear*area,0}:(axis==1?Vec3{0,area,0}:Vec3{0,0,area});
   int l[3]={xyz[0],xyz[1],xyz[2]},r[3]={xyz[0],xyz[1],xyz[2]};l[axis]=t-1;
   int f;if(t==0){r[axis]=0;f=face(idx(r[0],r[1],r[2],n),-1,transform(fc,shear),S*(-1),periodic?BoundaryKind::Periodic:BoundaryKind::NoSlip);low=f;}
   else if(t==n){f=face(idx(l[0],l[1],l[2],n),-1,transform(fc,shear),S,periodic?BoundaryKind::Periodic:BoundaryKind::NoSlip);high=f;}
   else f=face(idx(l[0],l[1],l[2],n),idx(r[0],r[1],r[2],n),transform(fc,shear),S,BoundaryKind::Internal);
  }
  if(periodic){m.periodicPartner[low]=high;m.periodicPartner[high]=low;}
 }
 m.cellFaceOffsets.push_back(0);for(const auto& list:entries){for(auto e:list){m.cellFaces.push_back(e.first);m.cellFaceSigns.push_back(e.second);}m.cellFaceOffsets.push_back(m.cellFaces.size());}
 for(std::size_t c=0;c<m.volumes.size();++c){Vec3 sum{};require(m.cellFaceOffsets[c+1]-m.cellFaceOffsets[c]==6,"mesh has six faces per cell");for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e)sum+=m.areaVectors[m.cellFaces[e]]*m.cellFaceSigns[e];require(mag(sum)<1e-14,"mesh closure");}
 return m;
}

PhysicsConfig physics(int scenario){
 PhysicsConfig p;p.spatialOrder=scenario%2?2:1;p.permeability=2e-11;p.poreViscosity=1.8e-5;
 for(int c=0;c<Nc;++c){auto& t=p.condensed[c];t.rho=1200+200*c;t.cp0=900+100*c;t.cp1=.02*c;t.conductivity=1+10*c;t.Tmin=100;t.Tmax=2000;}
 for(int s=0;s<Ns;++s){auto& t=p.species[s];t.R=280+60*s;t.cp0=1050+100*s;t.cp1=.01*s;t.Tmin=100;t.Tmax=2000;p.poreEffectiveDiffusivity[s]=(s+1)*1e-5;}
 if(scenario==3){p.permeability=0;p.spatialOrder=1;}
 if(scenario==5){p.permeability=0;for(int s=0;s<Ns;++s)p.poreEffectiveDiffusivity[s]=0;}
 if(scenario==6)p.gravity={2,-3,4};
 if(scenario==7)p.permeability=0;
 return p;
}
std::vector<SolidQ> state(const HostMesh& m,const PhysicsConfig& p,int scenario){
 std::vector<SolidQ> q(m.volumes.size());
 for(std::size_t c=0;c<q.size();++c){
  const double V=m.volumes[c],T=scenario==5?650.:600.+3*c;
  q[c].condensed[0]=.5*p.condensed[0].rho*V;q[c].condensed[1]=.2*p.condensed[1].rho*V;
  q[c].porosity=.3;
  for(int k=0;k<Nc;++k)q[c].energy+=q[c].condensed[k]*condensedE(p.condensed[k],T);
  for(int k=0;k<Nr;++k)q[c].progress[k]=.001*(k+1)*V;
  for(int s=0;s<Ns;++s){
   double density=.1*(s+1)*(1+.01*c);
   if(scenario==3&&(c==0||c==1))density=0;
   if(scenario==4)density=s==int(c%Ns)?.3:0;
   if(scenario==5)density=.1*(s+1);
   q[c].pore[s]=V*density;q[c].energy+=q[c].pore[s]*speciesE(p.species[s],T);
  }
 }
 return q;
}
void compare(const MaterialTransportResult& a,const MaterialTransportResult& b,bool exact,double& maximum){
 auto equal=[&](double x,double y){
  require(std::isfinite(x)&&std::isfinite(y),"finite output");
  const double scaled=std::abs(x-y)/std::max(1.,std::max(std::abs(x),std::abs(y)));
  maximum=std::max(maximum,scaled);
  if(exact)require(x==y,"CPU extraction changed original arithmetic");
  else require(std::abs(x-y)<=2e-10+3e-12*std::max(std::abs(x),std::abs(y)),"CUDA/CPU transport mismatch");
 };
 auto solid=[&](const SolidQ& x,const SolidQ& y){
  for(int k=0;k<Nc;++k)equal(x.condensed[k],y.condensed[k]);
  for(int k=0;k<Ns;++k)equal(x.pore[k],y.pore[k]);
  for(int k=0;k<Nr;++k)equal(x.progress[k],y.progress[k]);
  equal(x.energy,y.energy);equal(x.porosity,y.porosity);
 };
 equal(a.dtLimit,b.dtLimit);require(a.rates.size()==b.rates.size()&&a.faceFlux.size()==b.faceFlux.size(),"result shape");
 for(std::size_t c=0;c<a.rates.size();++c){solid(a.rates[c],b.rates[c]);equal(a.poreVelocity[c].x,b.poreVelocity[c].x);equal(a.poreVelocity[c].y,b.poreVelocity[c].y);equal(a.poreVelocity[c].z,b.poreVelocity[c].z);}
 for(std::size_t f=0;f<a.faceFlux.size();++f)solid(a.faceFlux[f],b.faceFlux[f]);
}
void conservation(const HostMesh&m,const std::vector<SolidQ>&q,const PhysicsConfig&p,const MaterialTransportResult&r){
 long double sum[Nc+Ns+Nr+1]{},scale[Nc+Ns+Nr+1]{};
 auto add=[&](const SolidQ&value,double factor){for(int k=0;k<Nc+Ns+Nr+1;++k){const double x=materialTransport::component(value,k)*factor;sum[k]+=x;scale[k]+=std::abs(x);}};
 for(const auto&v:r.rates)add(v,1);
 for(std::size_t f=0;f<m.owner.size();++f)if(m.neighbour[f]<0&&m.boundaryKind[f]!=BoundaryKind::Periodic)add(r.faceFlux[f],1);
 for(std::size_t c=0;c<q.size();++c){double mass=0;for(int s=0;s<Ns;++s)mass+=q[c].pore[s];sum[Nc+Ns+Nr]-=mass*dot(p.gravity,r.poreVelocity[c]);}
 for(int k=0;k<Nc+Ns+Nr+1;++k)require(std::abs(sum[k])<=2e-10L+2e-13L*scale[k],"physical inventory balance");
 for(std::size_t f=0;f<m.owner.size();++f)if(m.boundaryKind[f]==BoundaryKind::Periodic)
  for(int k=0;k<Nc+Ns+Nr+1;++k)require(materialTransport::component(r.faceFlux[f],k)==-materialTransport::component(r.faceFlux[m.periodicPartner[f]],k),"periodic unique flux antisymmetry");
}
#if defined(__CUDACC__)
void cudaCheck(cudaError_t code){require(code==cudaSuccess,cudaGetErrorString(code));}
struct DeviceStorage{
 std::vector<void*> allocations;
 ~DeviceStorage(){for(void*p:allocations)cudaFree(p);}
 template<class T>T* allocate(std::size_t count){T* p=nullptr;if(count){cudaCheck(cudaMalloc(&p,count*sizeof(T)));allocations.push_back(p);}return p;}
 template<class T>T* upload(const std::vector<T>& host){T*p=allocate<T>(host.size());if(p)cudaCheck(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
 template<class T>T* one(const T& host){T*p=allocate<T>(1);cudaCheck(cudaMemcpy(p,&host,sizeof(T),cudaMemcpyHostToDevice));return p;}
};
MaterialTransportDeviceResult deviceEvaluate(const HostMesh&m,const std::vector<SolidQ>&q,const PhysicsConfig&p,const std::vector<Real>&swept,MaterialTransportResult&out){
 DeviceStorage d;const int nc=q.size(),nf=m.owner.size();
 // Independently build a canonical physical-face incidence by explicit lists.
 std::vector<std::vector<std::pair<int,int>>> list(nc);
 for(int f=0;f<nf;++f){int right=m.neighbour[f];const auto kind=m.boundaryKind[f];
  if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;
  if(kind==BoundaryKind::Periodic){if(f>m.periodicPartner[f])continue;right=m.owner[m.periodicPartner[f]];}
  if(right>=0){list[m.owner[f]].push_back({f,0});list[right].push_back({f,1});}
 }
 std::vector<int> offsets(1,0),faces,sides;
 for(const auto&cell:list){for(auto e:cell){faces.push_back(e.first);sides.push_back(e.second);}offsets.push_back(faces.size());}
 MaterialMeshView mesh{nc,nf,d.upload(m.volumes),d.upload(m.cellCentres),d.upload(m.owner),d.upload(m.neighbour),d.upload(m.areaVectors),d.upload(m.faceCentres),d.upload(m.boundaryKind),d.upload(m.periodicPartner),d.upload(m.cellFaceOffsets),d.upload(m.cellFaces),d.upload(m.cellFaceSigns),d.upload(offsets),d.upload(faces),d.upload(sides)};
 MaterialTransportWorkspace w{d.allocate<MaterialPrimitive>(nc),d.allocate<Vec3>(nc),d.allocate<Vec3>(nc),d.allocate<Real>(nc),d.allocate<Real>(nc),d.allocate<Real>(nf),d.allocate<Real>(nf),d.allocate<Real>(nc),d.allocate<SolidQ>(nf),d.allocate<SolidQ>(nc),d.allocate<int>(nc),d.allocate<int>(nf)};
 auto* result=d.allocate<MaterialTransportDeviceResult>(1);
 const auto* deviceQ=d.upload(q);const auto* deviceP=d.one(p);const auto* deviceSweep=d.upload(swept);
 MaterialTransportDeviceResult host;
 for(int repeat=0;repeat<2;++repeat){
  cudaCheck(launchMaterialTransport(mesh,deviceQ,deviceP,deviceSweep,.002,.4,w,result,nullptr));
  cudaCheck(cudaMemcpy(&host,result,sizeof(host),cudaMemcpyDeviceToHost));
 }
 if(host.errorCode)return host;
 out.rates.resize(nc);out.poreVelocity.resize(nc);out.faceFlux.resize(nf);out.dtLimit=host.dtLimit;
 cudaCheck(cudaMemcpy(out.rates.data(),w.rates,nc*sizeof(SolidQ),cudaMemcpyDeviceToHost));
 cudaCheck(cudaMemcpy(out.faceFlux.data(),w.faceFlux,nf*sizeof(SolidQ),cudaMemcpyDeviceToHost));
 cudaCheck(cudaMemcpy(out.poreVelocity.data(),w.poreVelocity,nc*sizeof(Vec3),cudaMemcpyDeviceToHost));return host;
}
#endif
void failureParity(const HostMesh&m,const std::vector<SolidQ>&q,const PhysicsConfig&p,const std::vector<Real>&swept,
    MaterialTransportError expected,int entity){
 MaterialTransportResult before,after;before.dtLimit=after.dtLimit=123;
 std::string oldError,newError;bool oldNumerical=false,newNumerical=false;
 const bool oldOk=baselineEvaluateMaterialTransport(m,q,p,swept,.002,.4,before,oldError,&oldNumerical);
 const bool newOk=evaluateMaterialTransport(m,q,p,swept,.002,.4,after,newError,&newNumerical);
 require(!oldOk&&!newOk&&oldError==newError&&oldNumerical==newNumerical,"preserved failure semantics");
 require(before.dtLimit==123&&after.dtLimit==123&&before.rates.empty()&&after.rates.empty(),"transactional failed output");
#if defined(__CUDACC__)
 auto status=deviceEvaluate(m,q,p,swept,after);
 require(status.errorCode==int(expected)&&status.entity==entity&&status.numerical,"deterministic first stage/entity failure");
#endif
}
int main(){
#if defined(__CUDACC__)
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
#endif
 double maximum=0;
 for(int scenario=0;scenario<8;++scenario){
  const bool periodic=scenario==2||scenario==4||scenario==7;
  auto m=cube(scenario==7?1:3,scenario?0.45:0,periodic,scenario!=7);
  auto p=physics(scenario);auto q=state(m,p,scenario);std::vector<Real>swept;
  // Reverse stencil incidence to ensure diffusion stability does not inherit
  // this order; original diffusion losses always used global face order.
  for(std::size_t c=0;c<q.size();++c){const int a=m.cellFaceOffsets[c],b=m.cellFaceOffsets[c+1];std::reverse(m.cellFaces.begin()+a,m.cellFaces.begin()+b);std::reverse(m.cellFaceSigns.begin()+a,m.cellFaceSigns.begin()+b);}
  if(scenario==5){for(std::size_t f=0;f<m.owner.size();++f){const auto x=m.faceCentres[f];swept.push_back(.002*dot({.2+.04*x.x,-.3+.07*x.y,.1+.03*x.z},m.areaVectors[f]));}}
  if(scenario==0){for(std::size_t f=0;f<m.owner.size();++f)if(m.neighbour[f]<0)m.boundaryKind[f]=f%2?BoundaryKind::Empty:BoundaryKind::Interface;}
  MaterialTransportResult baseline,current;std::string error;bool numerical=false;
  const bool baselineOk=baselineEvaluateMaterialTransport(m,q,p,swept,.002,.4,baseline,error,&numerical);
  require(baselineOk,"baseline scenario "+std::to_string(scenario)+": "+error);
  const bool currentOk=evaluateMaterialTransport(m,q,p,swept,.002,.4,current,error,&numerical);
  require(currentOk,"shared CPU scenario "+std::to_string(scenario)+": "+error);
  compare(baseline,current,true,maximum);conservation(m,q,p,current);
#if defined(__CUDACC__)
  MaterialTransportResult gpu;auto status=deviceEvaluate(m,q,p,swept,gpu);
  require(!status.errorCode,"GPU status "+std::to_string(status.errorCode)+" entity "+std::to_string(status.entity));
  compare(baseline,gpu,false,maximum);conservation(m,q,p,gpu);
  // Two simultaneous EOS failures: first cell is deterministic at every run.
  if(scenario==1){auto invalid=q;invalid[2].energy=invalid[5].energy=std::numeric_limits<Real>::quiet_NaN();
   for(int repeat=0;repeat<3;++repeat){auto bad=deviceEvaluate(m,invalid,p,swept,gpu);require(bad.errorCode==int(MaterialTransportError::CellEos)&&bad.entity==2&&bad.numerical,"stable first EOS error");}}
#endif
  if(scenario==3&&Ns>=3){
   // The preserved algorithm rejects this second-order Darcy/empty-pore
   // combination with dt=0. Keep that outcome, without changing the limiter.
   auto emptyDarcy=p;emptyDarcy.permeability=2e-11;emptyDarcy.spatialOrder=2;
   failureParity(m,q,emptyDarcy,swept,MaterialTransportError::InvalidTimestep,-1);
  }
  if(scenario==5){const double divergence=.14;
   for(std::size_t c=0;c<q.size();++c)for(int k=0;k<Nc+Ns+Nr+1;++k){
    const double before=materialTransport::component(q[c],k)/m.volumes[c];
    const double after=(materialTransport::component(q[c],k)+.002*materialTransport::component(current.rates[c],k))/(m.volumes[c]*(1+.002*divergence));
    require(std::abs(after-before)<=5e-12*std::max(1.,std::abs(before)),"ALE e/inventory transport GCL");
   }
  }
 }
 {
  auto mesh=cube(3,.45,false,true);auto p=physics(1);auto q=state(mesh,p,1);
  auto invalid=q;invalid[2].energy=invalid[5].energy=std::numeric_limits<Real>::quiet_NaN();
  failureParity(mesh,invalid,p,{},MaterialTransportError::CellEos,2);
  auto diffusivity=p;diffusivity.poreEffectiveDiffusivity[1]=-1;
  failureParity(mesh,invalid,diffusivity,{},MaterialTransportError::InvalidDiffusivity,1);
  auto viscosity=p;viscosity.poreViscosity=0;
  failureParity(mesh,q,viscosity,{},MaterialTransportError::DarcyViscosity,0);
  std::vector<Real> swept(mesh.owner.size());swept[1]=swept[5]=std::numeric_limits<Real>::quiet_NaN();
  failureParity(mesh,q,p,swept,MaterialTransportError::AleSweep,1);
 }
 std::printf("material transport Ns=%d scenarios=8 CPU baseline exact; max_scaled_cuda_error=%.17g\n",Ns,maximum);
}
