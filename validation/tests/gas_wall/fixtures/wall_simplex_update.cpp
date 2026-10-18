#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cstdio>
using namespace ugkwp;using namespace ugkwp::gaswall;
template<int N> void exercise(){
 SpeciesThermoData<double> sp[N]{};double coefficients[N*3]{};WallInput<double,N> in{};
 for(int s=0;s<N;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;coefficients[3*s]=1000;in.model.diffusivity[s]=1e-5;}
 in.model.thermo.species=sp;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=3*N;
 in.pressure=101325;in.temperature=in.matching.temperature=500;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;in.matching.massFraction[0]=1;in.model.viscosity=2e-5;in.model.conductivity=.04;
 ugkp::diagnostics::DeviceContext d{};in.diagnostics=&d;in.model.thermo.diagnostics=&d;WallModelConfig<double> c;c.nodes=4;c.enableSst=false;
 detail::LayerContext<double,N> ctx;assert(detail::makeContext(in,c,ctx));WallWorkspace<double,N,4> w{};
 for(int i=0;i<4;++i){w.y[i]=.01*i/3;w.state[i][N+3]=1;for(int j=0;j<N-1;++j){w.state[i][3+j]=.2/(N-1);w.delta[i][3+j]=-1./(N-1);}}
 double norm=0,alpha=0;assert(detail::allResidual(in,c,ctx,w.state,w.y,w.residual,norm));assert(detail::applyLayerNewtonUpdate(in,c,ctx,w,norm,alpha));
 for(int i=0;i<4;++i){double sum=0;for(int j=0;j<N-1;++j){assert(w.state[i][3+j]>=0&&w.state[i][3+j]<=1);sum+=w.state[i][3+j];}assert(sum<=1);}
 assert(!d.fatalClaim);std::printf("N=%d alpha=%.17g norm=%.17g storedY=%.17g\n",N,alpha,norm,w.state[1][3]);
}
int main(){exercise<3>();exercise<10>();}
