#include "wall_composition_cases.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdlib>
int main(int argc,char**argv){CompositionWallCase m;m.in.integrateOwnerKSource=false;WallModelConfig<double> c;c.nodes=16;c.maxIterations=80;c.relativeTolerance=1e-8;c.enableSst=false;int index=argc>1?std::atoi(argv[1]):11;
configureCompositionCase(m,index);
ugkp::diagnostics::DeviceContext d{};m.in.diagnostics=&d;m.in.model.thermo.diagnostics=&d;WallWorkspace<double,2,128>w;WallOutput<double,2>out;WallStatus status;
bool ok=evaluateWallModel(m.in,c,w,out,status);detail::LayerContext<double,2> ctx;assert(detail::makeContext(m.in,c,ctx));double residual=0;
bool measured=detail::allResidual(m.in,c,ctx,w.state,w.y,w.trial,residual);double largest=0;for(int i=0;i<c.nodes;++i)largest=std::max(largest,w.state[i][3]);
std::cout<<std::setprecision(17)<<"case="<<index<<" ok="<<ok<<" fatal="<<d.fatalClaim<<" site="<<d.firstFatal.sourceSiteId<<" residual="<<residual<<" maxY="<<largest<<" iterations="<<out.iterations<<'\n';
assert(ok&&!d.fatalClaim&&measured&&residual<=c.absoluteTolerance+c.relativeTolerance);for(int i=0;i<c.nodes;++i)assert(w.state[i][3]>=0&&w.state[i][3]<=1);
}