#include <cstdio>
#include "../pr32/equilibrium_wall_fixture.H"
int main(){for(int angle=0;angle<3;++angle)for(int n=0;n<=200;++n){Fixture f;WallModelConfig<double>c;c.model=BoundaryLayerModel::EquilibriumOde;c.nodes=8;c.stretch=2.5;c.maxIterations=120;auto&in=f.in;
in.temperature=0.38893064424945384*0+281.184411372;in.pressure=31392.142296576414;
in.matchingDistance=0.00022500000000000045;in.matching.temperature=248.31792134965801;in.matching.velocity[0]=280.84062658762321;in.matching.velocity[1]=0;in.matching.velocity[2]=-4.6846490212946809e-06;double source[32]={0.040856007662394105,0.0029634379068673223,1.596330561693379e-09,0.0002857330683197016,-0.3703549080137935,0.019224319031710693,2.0896423099235885e-08,-0.0006371732698842082,-13.705300523838961,0.12474492388031509,3.240102291178356e-07,-3.9112382916427286,-83.53555996924823,0.36168101266330016,1.6858433235988541e-06,-193.59447905587322,-37.093709943265274,0.12032533084119716,7.236815587094378e-07,-185.7703651682218,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};for(auto&v:source)v*=n/100.;in.particleSource=source;const double theta=angle*.61,co=std::cos(theta),si=std::sin(theta);
{double x=in.matching.velocity[0],z=in.matching.velocity[2];in.matching.velocity[0]=co*x-si*z;in.matching.velocity[2]=si*x+co*z;}
for(int i=0;i<8;++i){double x=source[4*i],z=source[4*i+2];source[4*i]=co*x-si*z;source[4*i+2]=si*x+co*z;}
Workspace w;WallOutput<double,2>out;WallStatus status;bool ok=evaluateWallModel(in,c,w,out,status);if(!ok){std::cerr<<"FAILED scale="<<n/100.<<" norm="<<status.residual<<" iterations="<<status.iteration<<"\n";return 1;}
assert(out.residual<=c.absoluteTolerance+c.relativeTolerance);
assert(std::abs(out.particleEnergyResidual)<1e-4);
for(int d=0;d<2;++d)assert(std::abs(out.particleMomentumResidual[d])<1e-5);
}}