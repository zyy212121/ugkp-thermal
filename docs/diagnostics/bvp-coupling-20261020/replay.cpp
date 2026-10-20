
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
int main(int argc,char**argv){
 if(argc<3||argc>4)return 1;std::ifstream input(argv[1]);std::ofstream out(argv[2]),profiles(std::string(argv[2])+".profiles.csv");std::string line;std::getline(input,line);
 profiles<<std::setprecision(17)<<"face,owner,node,y,Ux,Uy,Uz,T,k,omega\n";
 out<<std::setprecision(17)<<"face,owner,ok,status,fatal,tauX,qIntoGas,ownerOmega,residual,iterations\n";
 while(std::getline(input,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int face,owner;double L,y,p,T,u,v,w,k,omega;
  if(!(row>>face>>owner>>L>>y>>p>>T>>u>>v>>w>>k>>omega))return 2;
  CapturedSstWall m;m.config.nodes=argc>3?std::atoi(argv[3]):24;auto&i=m.input;i.integrateOwnerKSource=false;
  i.matchingDistance=L;i.ownerDistance=y;i.pressure=p;i.matching.temperature=T;
  i.matching.velocity[0]=u;i.matching.velocity[1]=v;i.matching.velocity[2]=w;i.matching.k=k;i.matching.omega=omega;
  ugkp::diagnostics::DeviceContext diagnostics{};i.diagnostics=&diagnostics;i.model.thermo.diagnostics=&diagnostics;i.diagnosticEntity=face;
  ugkwp::gaswall::WallWorkspace<double,2,128>work{};ugkwp::gaswall::WallOutput<double,2>value;ugkwp::gaswall::WallStatus status;
  bool ok=ugkwp::gaswall::evaluateReactingWallLayer(i,m.config,work,value,status);
  out<<face<<','<<owner<<','<<ok<<','<<int(status.code)<<','<<diagnostics.fatalClaim<<','<<value.traction[0]<<','<<value.conductiveHeatFlux<<','<<value.ownerOmega<<','<<value.residual<<','<<value.iterations<<'\n';
  ugkwp::gaswall::detail::LayerContext<double,2>ctx;
  if(ok&&ugkwp::gaswall::detail::makeContext(i,m.config,ctx))for(int node=0;node<m.config.nodes;++node){
   ugkwp::gaswall::WallProfileNode<double,2>p;
   if(!ugkwp::gaswall::reactingWallProfileNode(i,m.config,work,node,ctx,p))return 3;
   profiles<<face<<','<<owner<<','<<node<<','<<p.y<<','<<p.velocity[0]<<','<<p.velocity[1]<<','<<p.velocity[2]<<','<<p.temperature<<','<<p.k<<','<<p.omega<<'\n';
  }
 }
}
