#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include "OpenFoamWallFunctions.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "gasTransport/GasStateView.H"
#define __device__
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
using R = GpuReal;
using Vec = ugkptransport::Vector3;
constexpr R OfVSmall=R(1e-30), OfSmall=R(1e-15);
R clampMin(R a,R b){return std::max(a,b);}
R finiteOr(R a,R b){return std::isfinite(a)?a:b;}
struct DeviceState {
 int turbulenceModel=3,sstWallTreatment=1;
 int riemannBoundaryUFix[1]={1},riemannBoundaryTFix[1]={1};
 R Ux[1]={20},Uy[1]={-8},Uz[1]={3},Tgas[1]={310};
 R riemannBoundaryUx[1]={2},riemannBoundaryUy[1]={-1},riemannBoundaryUz[1]={1};
 R riemannBoundaryT[1]={300},deltaCoeffs[1]={1700},sstWallDistance[1]={R(.002)};
 R nut[1]={R(.1)},p[1]={101325},Rgas=287,rhoMin=R(1e-12),TgasMin=1;
 R gasMu=R(1.8e-5),gasCp=1005,gasPrClamped=R(.71),turbulentPrandtl=R(.85);
 R sstWallCmu=R(.09),sstWallKappa=R(.41),sstWallE=R(9.8),k[1]={R(.2)};
 R sstJayatillekeP=ugkpwall::jayatillekeSmoothP(gasPrClamped/turbulentPrandtl);
 R sstThermalYPlus=ugkpwall::jayatillekeThermalYPlus(gasPrClamped/turbulentPrandtl);
};
struct Prim {R rho=R(1.3),ux=2,uy=-1,uz=1;};
Prim riemannFacePrimitiveForGradient(const DeviceState&,int,int){return {};}
bool useRiemannBoundaryVelocity(const DeviceState&s,int f,const Prim&){return s.riemannBoundaryUFix[f]!=0;}
R molecularGasConductivity(const DeviceState&s){return s.gasMu*s.gasCp/s.gasPrClamped;}
// PRODUCTION_GAS_FACE_CALLER
int failures=0;
void check(const char*name,long double got,long double expected,long double tol=0){
 if(!tol)tol=sizeof(R)==4?2e-5L:5e-12L;
 if(!std::isfinite(got)||std::abs(got-expected)>tol*std::max(1.L,std::abs(expected))){
  std::cerr<<name<<" got="<<got<<" expected="<<expected<<"\n";++failures;
 }
}
void require(const char*name,bool ok){if(!ok){std::cerr<<name<<"\n";++failures;}}
struct BoundaryResult {Vec grad[3],normal;R thermal;};
BoundaryResult boundaryFragment(const DeviceState&s,int boundaryKind,Vec n,const R original[3][3]){
 int own=0,f=0;R nx=n.x,ny=n.y,nz=n.z;
 Prim left{},right{};right.ux=s.riemannBoundaryUx[0];right.uy=s.riemannBoundaryUy[0];right.uz=s.riemannBoundaryUz[0];
 R gradUxX=original[0][0],gradUxY=original[0][1],gradUxZ=original[0][2];
 R gradUyX=original[1][0],gradUyY=original[1][1],gradUyZ=original[1][2];
 R gradUzX=original[2][0],gradUzY=original[2][1],gradUzZ=original[2][2];
 Vec compactSnGradU{};R normalTemperatureGradient=0;
 // PRODUCTION_BOUNDARY_FRAGMENT
 return {{{gradUxX,gradUxY,gradUxZ},{gradUyX,gradUyY,gradUyZ},{gradUzX,gradUzY,gradUzZ}},compactSnGradU,normalTemperatureGradient};
}
void tractionTests(){
 const R original[3][3]={{2,7,-3},{5,-11,13},{-17,19,23}};
 for(Vec n:{Vec{1,0,0},Vec{R(.6),R(.8),0},Vec{R(2)/3,-R(1)/3,R(2)/3}}){
  for(int kind:{0,2})for(int fixed:{0,1}){
   DeviceState s;s.riemannBoundaryUFix[0]=fixed;s.deltaCoeffs[0]=R(2.3);
   auto actual=boundaryFragment(s,kind,n,original);
   R normal[3]={n.x,n.y,n.z},target[3]={0,0,0};
   if(kind==2||fixed){target[0]=(s.riemannBoundaryUx[0]-s.Ux[0])*s.deltaCoeffs[0];target[1]=(s.riemannBoundaryUy[0]-s.Uy[0])*s.deltaCoeffs[0];target[2]=(s.riemannBoundaryUz[0]-s.Uz[0])*s.deltaCoeffs[0];}
   R want[3][3];
   for(int i=0;i<3;++i){
    R oldNormal=0;for(int j=0;j<3;++j)oldNormal+=original[i][j]*normal[j];
    for(int j=0;j<3;++j)want[i][j]=original[i][j]+normal[j]*(target[i]-oldNormal);
    check("boundary tensor matches target normal",ugkptransport::dot(actual.grad[i],n),target[i]);
    Vec tangent{-n.y,n.x,0};
    check("boundary tensor preserves tangent",ugkptransport::dot(actual.grad[i],tangent),-n.y*original[i][0]+n.x*original[i][1]);
   }
   Vec traction=ugkptransport::openFoamNewtonianTraction(R(.7),n,actual.normal,actual.grad[0],actual.grad[1],actual.grad[2]);
   R output[3]={traction.x,traction.y,traction.z},trace=want[0][0]+want[1][1]+want[2][2];
   for(int i=0;i<3;++i){R t=0;for(int j=0;j<3;++j)t+=(want[i][j]+want[j][i]-(i==j?R(2)/3*trace:0))*normal[j];check("consistent Newtonian traction",output[i],R(.7)*t);}
  }
 }
}
void spaldingTests(){
 for(R delta:{R(200),R(500),R(1700)}){
  DeviceState s;s.deltaCoeffs[0]=delta;
  R dux=s.Ux[0]-s.riemannBoundaryUx[0],duy=s.Uy[0]-s.riemannBoundaryUy[0],duz=s.Uz[0]-s.riemannBoundaryUz[0];
  R up=std::sqrt(dux*dux+duy*duy+duz*duz),rho=R(1.3),nu=s.gasMu/rho;
  auto reference=ugkpwall::spaldingWallState(up,s.sstWallDistance[0],nu,s.sstWallKappa,s.sstWallE);
  R expectedNut=std::max(R(0),reference.uTau*reference.uTau/(up*delta)-nu);
  auto separate=ugkpwall::spaldingWallStateFromNormalGradient(up,s.sstWallDistance[0],nu,up*delta,s.sstWallKappa,s.sstWallE);
  check("law uTau independent of patch gradient",separate.uTau,reference.uTau);
  check("law yPlus uses law distance",separate.yPlus,reference.yPlus);
  check("separate helper uses patch gradient for nut",separate.nut,expectedNut);
  R mu,k,q;int active=0;gasFaceSubgridTransportProperties(s,0,0,-1,2,rho,mu,k,q,active);
  check("gas caller converts uTau using patch snGrad",mu/rho,expectedNut, sizeof(R)==4?2e-7L:1e-10L);
  require("wall transport thermal flux finite",std::isfinite(q)&&active==1);
 }
 // The legacy overload remains the orthogonal y=1/delta specialization.
 auto zero=ugkpwall::spaldingWallState(0,R(.002),R(1.8e-5));
 check("zero speed nut",zero.nut,0);check("zero speed uTau",zero.uTau,0);
}
long double matchingResidual(long double y,long double ratio,long double P){return ratio*y-std::log(9.8L*y)/.41L-P;}
long double upperThermalRoot(long double ratio,long double P){
 long double lo=1/(.41L*ratio),hi=std::max(2*lo,11.L);
 while(matchingResidual(hi,ratio,P)<0)hi*=2;
 for(int i=0;i<200;++i){long double mid=(lo+hi)/2;if(matchingResidual(mid,ratio,P)>0)hi=mid;else lo=mid;}
 return (lo+hi)/2;
}
void thermalTests(){
 // Invalid roots must fail closed in both thermal consumers, even when the
 // local log-law tPlus is finite or zero turbulent speed would short-circuit.
 const R nan=std::numeric_limits<R>::quiet_NaN(),inf=std::numeric_limits<R>::infinity();
 for(R ratio:{R(0),R(-1),nan,inf})check("invalid Pr ratio rejected",ugkpwall::jayatillekeThermalYPlus(ratio),0);
 check("no physical intersection rejected",ugkpwall::jayatillekeThermalYPlus(1,10,R(.1)),0);
 for(R badRoot:{R(0),R(-1),nan,inf}){
  for(R kin:{R(0),R(.2)}){
   auto invalid=ugkpwall::sstJayatillekeThermalTransport(1,1005,R(1.8e-5),1,R(.85),R(.09),R(.41),R(9.8),0,badRoot,kin,R(.01),0,0,0);
   require("thermal transport rejects invalid root",invalid.valid==0);
  }
  auto invalid=ugkpwall::jayatillekeWallHeatFluxPrecomputed(1,1005,1,R(.85),R(.41),R(9.8),0,badRoot,1,100,310,300);
  require("precomputed heat rejects invalid root",invalid.valid==0);
 }
 auto invalidPlus=ugkpwall::jayatillekeWallHeatFluxPrecomputed(1,1005,1,R(.85),R(.41),R(9.8),-100,1,1,2,310,300);
 require("negative temperaturePlus is not hidden by denominator floor",invalidPlus.valid==0);
 // Gas-face integration keeps the outward sign from the boundary snGrad.
 // In the viscous branch with zero velocity difference C=0, so wall
 // heating/cooling cannot be disguised by a kinetic offset. The log branch
 // retains its kinetic offset and is checked independently below.
 for(R Pr:{R(.01),R(.1),R(.71)})for(R wallT:{R(285),R(315),R(300)-R(.0001),R(300),R(300)+R(.0001)}){
  DeviceState s;s.Tgas[0]=300;s.riemannBoundaryT[0]=wallT;s.sstWallDistance[0]=R(1e-5);
  s.Ux[0]=s.Uy[0]=s.Uz[0]=s.riemannBoundaryUx[0]=s.riemannBoundaryUy[0]=s.riemannBoundaryUz[0]=0;
  s.gasPrClamped=Pr;s.sstJayatillekeP=ugkpwall::jayatillekeSmoothP(Pr/s.turbulentPrandtl);
  s.sstThermalYPlus=ugkpwall::jayatillekeThermalYPlus(Pr/s.turbulentPrandtl);
  R mu,k,q;int active=0;gasFaceSubgridTransportProperties(s,0,0,-1,2,R(1.3),mu,k,q,active);
  require("gas-face heat path active and finite",active==1&&std::isfinite(q));
  if(wallT<s.Tgas[0])require("cold wall receives positive outward heat",q>0);
  else if(wallT>s.Tgas[0])require("hot wall sends negative outward heat",q<0);
  else check("equal temperatures zero thermal flux",q,0);
  check("gas-face signed Fourier closure",q,-(molecularGasConductivity(s)+k)*(wallT-s.Tgas[0])*s.deltaCoeffs[0]);
 }
 for(R Pr:{R(.001),R(.01),R(.1),R(.3),R(.71),R(.85),R(1),R(7),R(100)}){
  R Prt=R(.85),ratio=Pr/Prt,P=ugkpwall::jayatillekeSmoothP(ratio),yt=ugkpwall::jayatillekeThermalYPlus(ratio);
  long double root=upperThermalRoot(ratio,P);
  check("physical upper thermal root",yt,root,sizeof(R)==4?4e-6L:1e-12L);
  require("root above stationary point",yt>=R(1)/(.41L*ratio));
  check("matching residual",matchingResidual(yt,ratio,P),0,sizeof(R)==4?2e-4L:2e-11L);
  long double lower=1e-30L,upper=1/(.41L*ratio);
  for(int i=0;i<200;++i){long double mid=(lower+upper)/2;if(matchingResidual(mid,ratio,P)>0)lower=mid;else upper=mid;}
  require("upper branch is distinct from small positive root",yt>upper);
  // At the chosen outward intersection, log-law transport exceeds molecular
  // only above the transition; this identifies which of the two roots applies.
  long double above=root*1.01L;
  require("outer log branch permits positive turbulent conductivity",ratio*above>std::log(9.8L*above)/.41L+P);
  for(R multiple:{R(.5),R(1.01),R(3)}){
   R rho=R(1.2),cp=1005,mu=R(1.8e-5),cmu=R(.09),kin=R(.2),speed=R(12),wallSpeed=R(3);
   R uStar=std::sqrt(std::sqrt(cmu))*std::sqrt(kin),y=R(root)*multiple*mu/(uStar*rho),yp=uStar*y*rho/mu;
   bool visc=yp<root;
   R tp=visc?Pr*yp:Prt*(std::log(R(9.8)*yp)/R(.41)+P);
   R uc=visc?0:uStar/R(.41)*std::log(R(9.8)*R(root))-wallSpeed;
   R C=R(.5)*rho*uStar*(visc?Pr*speed*speed:Prt*speed*speed+(Pr-Prt)*uc*uc);
   R conductivity=std::max(mu*cp/Pr,cp*rho*uStar*y/tp);
   for(R gradient:{R(-10000),R(10000),R(-1e-10),R(0),R(1e-10)}){
    auto state=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,Pr,Prt,cmu,R(.41),R(9.8),P,yt,kin,y,speed,wallSpeed,gradient);
    require("hot/cold/zero gradient closure valid",state.valid==1);
    check("original conductivity formula retained",state.conductivity,conductivity,sizeof(R)==4?3e-5L:2e-11L);
    check("signed heat and kinetic C/tPlus retained",state.heatFlux,-conductivity*gradient+C/tp,sizeof(R)==4?4e-5L:2e-11L);
   }
  }
 }
}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::string which=argv[1];
 if(which=="traction")tractionTests();else if(which=="spalding")spaldingTests();else if(which=="thermal")thermalTests();else return 2;
 std::cout<<which<<" failures="<<failures<<"\n";return failures?1:0;
}
