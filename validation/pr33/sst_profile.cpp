#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 using namespace ugkwp::gaswall;CapturedSstWall f;
 f.config.nodes=argc>1?std::atoi(argv[1]):24;f.config.maxIterations=200;
 WallWorkspace<double,2,128>w;WallOutput<double,2>o;WallStatus s;
 if(!evaluateWallModel(f.input,f.config,w,o,s)){std::cerr<<"failure "<<int(s.code)<<" residual "<<s.residual<<'\n';return 1;}
 std::cout<<std::setprecision(17)<<"{\"heat\":"<<o.conductiveHeatFlux<<",\"residual\":"<<o.residual<<",\"nodes\":[";
 for(int i=0;i<f.config.nodes;++i){if(i)std::cout<<',';std::cout<<'['<<w.y[i];for(int j=0;j<6;++j)std::cout<<','<<w.state[i][j];std::cout<<']';}
 std::cout<<"]}\n";
}
