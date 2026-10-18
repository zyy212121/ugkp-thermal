#include "validation/tests/gas_wall/fixtures/wall_composition_cases.H"
#include <algorithm>
#include <cmath>
#include <cstdio>
static CompositionWallCase model;
static WallModelConfig<double> config;
static detail::LayerContext<double,2> context;
static double y[16];
extern "C" int initialize(const double* v,double* x,double* f){
 model.in.integrateOwnerKSource=false;model.in.model.mode=GasMode::MixtureFrozen;
 model.in.matchingDistance=v[0];model.in.ownerDistance=v[1];model.in.temperature=v[2];model.in.matching.temperature=v[3];model.in.matching.velocity[0]=v[4];model.in.matching.k=v[5];model.in.matching.omega=v[6];model.in.massFlux[0]=.7*v[7];model.in.massFlux[1]=.3*v[7];
 config={};config.nodes=16;config.maxIterations=80;config.relativeTolerance=1e-8;config.enableSst=true;
 WallWorkspace<double,2,16>w{};WallOutput<double,2>o{};WallStatus status{};
 if(!evaluateWallModel(model.in,config,w,o,status))return 0;
 context={};if(!detail::makeContext(model.in,config,context))return 0;double norm=0;if(!detail::allResidual(model.in,config,context,w.state,w.y,w.residual,norm))return 0;
 for(int i=0;i<16;++i){y[i]=w.y[i];for(int j=0;j<6;++j){x[i*6+j]=j==4?w.state[i][j]*w.state[i][j]:w.state[i][j];f[i*6+j]=w.residual[i][j];}}return 1;
}
extern "C" int evaluate(const double* x,double* f){
 double state[16][6],residual[16][6];for(int i=0;i<16;++i)for(int j=0;j<6;++j){if(!std::isfinite(x[i*6+j])||(j==4&&x[i*6+j]<0))return 0;state[i][j]=j==4?std::sqrt(x[i*6+j]):x[i*6+j];}
 double norm=0;if(!detail::allResidual(model.in,config,context,state,y,residual,norm))return 0;
 for(int i=0;i<16;++i)for(int j=0;j<6;++j)f[i*6+j]=residual[i][j];return 1;
}
