#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasNumerics/OpenFoamWallFunctions.cuh"
#include <memory>
#include <iostream>
#include <iomanip>
int main(){
 std::cout<<std::setprecision(17)<<"owner_yplus_target,nodes,ok,code,iterations,residual,target_tau,wallfunction_tau,boundarylayer_tau,boundarylayer_ownerU,log_ownerU,boundarylayer_ownerk,log_ownerk,boundarylayer_ownerOmega,log_ownerOmega,WFproduction,stress_production\n";
 for(double yp:{30.,80.,200.})for(int nodes:{24,48,96,128}){
  CapturedSstWall model;auto&in=model.input;auto&c=model.config;
  constexpr double ut=1,nu=1e-5,kappa=.41,E=9.8,Cmu=.09;
  const double y=yp*nu/ut,L=3*y,kw=ut*ut/std::sqrt(Cmu),U=ut/kappa*std::log(E*yp),W=ut/(std::sqrt(Cmu)*kappa*y);
  in.temperature=300;in.pressure=ugkwp::universalGasConstant<double>()/model.species[0].molarMass*300;
  in.ownerDistance=y;in.matchingDistance=L;in.model.viscosity=nu;
  in.matching.temperature=300;in.matching.velocity[0]=ut/kappa*std::log(E*3*yp);in.matching.velocity[1]=in.matching.velocity[2]=0;
  in.matching.k=kw;in.matching.omega=W/3;in.matching.massFraction[0]=1;in.matching.massFraction[1]=0;
  c.nodes=nodes;
  auto w=std::make_unique<ugkwp::gaswall::WallWorkspace<double,2,128>>();ugkwp::gaswall::WallOutput<double,2>out;ugkwp::gaswall::WallStatus st;
  bool ok=evaluateWallModel(in,c,*w,out,st);
  const auto wf=ugkpwall::spaldingWallState(U,y,nu);const auto omega=ugkpwall::omegaWallFunctionState(kw,U/y,y,nu);
  double pu=0,pk=0;
  if(ok){int l=0;while(l+1<nodes-1&&w->y[l+1]<y)++l;double a=(y-w->y[l])/(w->y[l+1]-w->y[l]);ugkwp::gaswall::WallProfileNode<double,2>p0,p1;reactingWallProfileNode(in,c,*w,l,p0);reactingWallProfileNode(in,c,*w,l+1,p1);pu=(1-a)*p0.velocity[0]+a*p1.velocity[0];pk=(1-a)*p0.k+a*p1.k;}
  std::cout<<yp<<','<<nodes<<','<<ok<<','<<int(st.code)<<','<<out.iterations<<','<<out.residual<<','<<ut*ut<<','<<wf.uTau*wf.uTau<<','<<out.traction[0]<<','<<pu<<','<<U<<','<<pk<<','<<kw<<','<<out.ownerOmega<<','<<W<<','<<omega.production<<','<<ut*ut*ut/(kappa*y)<<'\n';
 }
}
