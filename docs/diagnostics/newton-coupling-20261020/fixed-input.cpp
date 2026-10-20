
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace ugkwp::gaswall;
void solve(const std::string& label,CapturedSstWall& m,int nodes){
 m.config.nodes=nodes;m.input.integrateOwnerKSource=false;m.input.volumeSstClosure=false;
 ugkp::diagnostics::DeviceContext diag{};m.input.diagnostics=&diag;m.input.model.thermo.diagnostics=&diag;
 WallWorkspace<double,2,128>w;WallOutput<double,2>o;WallStatus s;initprobe::stats={};
 auto t=std::chrono::steady_clock::now();bool ok=evaluateWallModel(m.input,m.config,w,o,s);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
 auto&a=initprobe::stats;
 printf("%s,%d,%d,%d,%u,%d,%.17g,%.17g,%.17g,%.17g,%d,%ld,%ld,%ld,%ld,%ld,%ld,%.17g,%.17g,%.17g",label.c_str(),nodes,INIT_POLICY,int(ok),diag.fatalClaim,o.iterations,o.residual,a.picard,a.candidate,a.initial,a.selected,a.point,a.interval,a.row,a.residual,a.newton,a.seed,us,o.traction[0],o.conductiveHeatFlux);
 printf(",%.17g,%.17g,%.17g,%.17g,%.17g\n",a.seedTime,a.newtonTime-a.linearTime,a.residualTime,a.linearTime,a.publishTime);
}
int main(int argc,char**argv){
 puts("case,nodes,policy,ok,fatal,iterations,final_residual,picard_initial,candidate_initial,chosen_initial,selected,point_calls,interval_calls,row_calls,residual_calls,newton_calls,picard_sweeps,microseconds,tauX,heat,seed_us,jacobian_us,residual_us,linear_us,publish_us");
 for(int nodes:{24,48,128}){CapturedSingularWall tiny;solve("actual-small-k",tiny,nodes);CapturedSstWall restart;solve("original-restart",restart,nodes);
 for(int arg=1;arg<argc;++arg){std::ifstream f(argv[arg]);std::string line;std::getline(f,line);int index=0;while(std::getline(f,line)){
 std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int face,owner;double L,y,p,T,u,v,w,k,omega;
 if(!(row>>face>>owner>>L>>y>>p>>T>>u>>v>>w>>k>>omega))return 2;
 CapturedSstWall m;auto&i=m.input;i.matchingDistance=L;i.ownerDistance=y;i.pressure=p;i.matching.temperature=T;i.matching.velocity[0]=u;i.matching.velocity[1]=v;i.matching.velocity[2]=w;i.matching.k=k;i.matching.omega=omega;
 solve("file"+std::to_string(arg)+"-face"+std::to_string(face),m,nodes);++index;
 }} }
}
