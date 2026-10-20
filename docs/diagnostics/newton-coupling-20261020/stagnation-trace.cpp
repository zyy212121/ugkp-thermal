
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){using namespace ugkwp::gaswall;
 int count=argc>1?std::atoi(argv[1]):48;
 CapturedSingularWall m;m.config.nodes=count;m.config.maxIterations=60;
 m.input.integrateOwnerKSource=false;m.input.volumeSstClosure=false;
 ugkp::diagnostics::DeviceContext diag{};m.input.diagnostics=&diag;m.input.model.thermo.diagnostics=&diag;
 WallWorkspace<double,2,128>w;WallOutput<double,2>o;WallStatus s;
 bool ok=evaluateWallModel(m.input,m.config,w,o,s);
 std::printf("result,%d,%d,%u,%d,%.17g\n",count,int(ok),diag.fatalClaim,o.iterations,o.residual);
 for(int i=0;i<count;++i){std::printf("node,%d,%d,%.17g",count,i,w.y[i]);
  for(int j=0;j<6;++j)std::printf(",%.17g",w.state[i][j]);
  for(int j=0;j<6;++j)std::printf(",%.17g",w.residual[i][j]);
  for(int j=0;j<6;++j)std::printf(",%.17g",w.delta[i][j]);std::printf("\n");}
}
