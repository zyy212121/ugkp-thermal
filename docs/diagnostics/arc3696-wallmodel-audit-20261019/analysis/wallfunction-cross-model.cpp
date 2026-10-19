#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasNumerics/OpenFoamWallFunctions.cuh"
#define ugkpwall legacywall
#include "old-OpenFoamWallFunctions.cuh"
#undef ugkpwall
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <memory>
int main(){
 std::string line;std::getline(std::cin,line);std::cout<<std::setprecision(17);
 std::cout<<"dataset,face,ok,code,iterations,residual,BL_heat,BL_shear,BL_ownerOmega,WF_heat,WF_shear,WF_omega,WF_production,BL_production,legacy_WF_heat,legacy_WF_shear,WF_root_residual,BL_profile_ownerU,BL_profile_ownerT,BL_profile_ownerk,WF_from_BLprofile_heat,WF_from_BLprofile_shear,WF_from_BLprofile_omega\n";
 while(std::getline(std::cin,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream stream(line);std::string label;int face;CapturedSstWall model;auto&in=model.input;
  double ownT,ownU[3],ownk,ownomega,ownrho,ownp;
  stream>>label>>face>>in.pressure>>in.temperature>>in.matchingDistance>>in.ownerDistance>>in.matching.temperature>>in.matching.k>>in.matching.omega;
  for(auto&x:in.matching.velocity)stream>>x;for(auto&x:in.matching.massFraction)stream>>x;stream>>ownT;for(auto&x:ownU)stream>>x;stream>>ownk>>ownomega>>ownrho>>ownp;
  auto work=std::make_unique<ugkwp::gaswall::WallWorkspace<double,2,24>>();ugkwp::gaswall::WallOutput<double,2>out;ugkwp::gaswall::WallStatus status;
  const bool ok=evaluateWallModel(in,model.config,*work,out,status);
  const double rho=ownp/(ugkwp::universalGasConstant<double>()/model.species[0].molarMass*in.temperature),nu=in.model.viscosity/rho,y=in.ownerDistance,U=std::sqrt(ownU[0]*ownU[0]+ownU[1]*ownU[1]+ownU[2]*ownU[2]);
  const auto wf=ugkpwall::spaldingWallState(U,y,nu);
  const auto oldwf=legacywall::spaldingWallState(U,y,nu);
  const auto omega=ugkpwall::omegaWallFunctionState(ownk,U/y,y,nu);
  const double P=ugkpwall::jayatillekeSmoothP(.72/.9),ypth=ugkpwall::jayatillekeThermalYPlus(.72/.9);
  const auto heat=ugkpwall::sstJayatillekeThermalTransport(rho,1004.675,in.model.viscosity,.72,.9,.09,.41,9.8,P,ypth,ownk,y,U,0,(in.temperature-ownT)/y);
  const auto oldheat=legacywall::sstJayatillekeThermalTransport(rho,1004.675,in.model.viscosity,.72,.9,.09,.41,9.8,P,ypth,ownk,y,U,0,(in.temperature-ownT)/y);
  const double root=ugkpwall::spaldingReynoldsRatio(U/wf.uTau,U*y/nu,std::log(U*y/nu),.41,9.8)-1;
  double pu=0,pt=0,pk=0,pomega=0,pq=0,pshear=0;
  if(ok){
   int l=0;while(l+1<23&&work->y[l+1]<y)++l;double a=(y-work->y[l])/(work->y[l+1]-work->y[l]);
   ugkwp::gaswall::WallProfileNode<double,2>p0,p1;reactingWallProfileNode(in,model.config,*work,l,p0);reactingWallProfileNode(in,model.config,*work,l+1,p1);
   pu=(1-a)*p0.velocity[0]+a*p1.velocity[0];pt=(1-a)*p0.temperature+a*p1.temperature;pk=(1-a)*p0.k+a*p1.k;
   const auto me=ugkpwall::spaldingWallState(pu,y,nu);
   const auto om=ugkpwall::omegaWallFunctionState(pk,pu/y,y,nu);pomega=om.omega;pshear=rho*me.uTau*me.uTau;
   pq=-ugkpwall::sstJayatillekeThermalTransport(rho,1004.675,in.model.viscosity,.72,.9,.09,.41,9.8,P,ypth,pk,y,pu,0,(in.temperature-pt)/y).heatFlux;
  }
  std::cout<<label<<','<<face<<','<<ok<<','<<int(status.code)<<','<<out.iterations<<','<<out.residual<<','<<out.conductiveHeatFlux<<','<<out.traction[0]<<','<<out.ownerOmega<<','<<-heat.heatFlux<<','<<rho*wf.uTau*wf.uTau<<','<<omega.omega<<','<<omega.production<<','<<out.traction[0]/ownrho*std::pow(.09,.25)*std::sqrt(ownk)/(.41*y)<<','<<-oldheat.heatFlux<<','<<rho*oldwf.uTau*oldwf.uTau<<','<<root<<','<<pu<<','<<pt<<','<<pk<<','<<pq<<','<<pshear<<','<<pomega<<'\n';
 }
}
