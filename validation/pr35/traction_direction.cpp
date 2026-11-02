#include <cstdio>
#include "../pr32/equilibrium_wall_fixture.H"
int main(){for(int angle=0;angle<3;++angle)for(int n=0;n<=201;++n){Fixture f;WallModelConfig<double>c;c.model=BoundaryLayerModel::EquilibriumOde;c.nodes=8;c.stretch=2.5;c.maxIterations=120;auto&in=f.in;
in.temperature=0.38893064424945384*0+281.184411372;in.pressure=31392.142296576414;
in.matchingDistance=0.00022500000000000045;in.matching.temperature=248.31792134965801;in.matching.velocity[0]=280.84062658762321;in.matching.velocity[1]=0;in.matching.velocity[2]=-4.6846490212946809e-06;double source[32]={0.040856007662394105,0.0029634379068673223,1.596330561693379e-09,0.0002857330683197016,-0.3703549080137935,0.019224319031710693,2.0896423099235885e-08,-0.0006371732698842082,-13.705300523838961,0.12474492388031509,3.240102291178356e-07,-3.9112382916427286,-83.53555996924823,0.36168101266330016,1.6858433235988541e-06,-193.59447905587322,-37.093709943265274,0.12032533084119716,7.236815587094378e-07,-185.7703651682218,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};for(auto&v:source)v*=n/100.;in.particleSource=source;// Also replay the exact first post-restart failure, before its direction
// was fixed. The remaining rows sweep the independent checkpoint source.
if(n==201){
in.pressure=31392.083487585311;in.temperature=281.184411372;in.matching.temperature=248.31927493626364;
in.matching.velocity[0]=280.83534523263552;
in.matching.velocity[1]=-0.58771686622593722;
in.matching.velocity[2]=-4.7172126529244213e-06;
const double captured[32]={0.04058945403476175,0.0029620194556789895,1.6712272862625136e-09,0.0002849638404776855,-0.372047330348719,0.019276635066088826,2.1578360372644726e-08,-0.0006579823903847094,-13.715830995946778,0.12494946500767613,3.30966949832298e-07,-3.9208440151891466,-83.0821109293178,0.3601224522320291,1.706431680374404e-06,-191.7792808187748,-37.896099506574764,0.12307279755080158,7.518265437345457e-07,-189.22116455698702,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};for(int j=0;j<32;++j)source[j]=captured[j];}
const double theta=angle*.61,co=std::cos(theta),si=std::sin(theta);
{double x=in.matching.velocity[0],z=in.matching.velocity[2];in.matching.velocity[0]=co*x-si*z;in.matching.velocity[2]=si*x+co*z;}
for(int i=0;i<8;++i){double x=source[4*i],z=source[4*i+2];source[4*i]=co*x-si*z;source[4*i+2]=si*x+co*z;}
Workspace w;WallOutput<double,2>out;WallStatus status;bool ok=evaluateWallModel(in,c,w,out,status);if(!ok){std::cerr<<"FAILED scale="<<n/100.<<" norm="<<status.residual<<" iterations="<<status.iteration<<"\n";return 1;}
assert(out.residual<=c.absoluteTolerance+c.relativeTolerance);
// Warm starts must retain the traction direction as well as its magnitude.
// With the same physical input a converged profile needs no Newton update.
if(n==100||n==201){
 double history[8*6];for(int i=0;i<8;++i)for(int j=0;j<6;++j)history[i*6+j]=w.state[i][j];
 detail::LayerContext<double,2>ctx;assert(detail::makeContext(in,c,ctx));double traction[2]{};
 for(int d=0;d<2;++d)traction[d]=detail::dot(out.traction,ctx.tangent[d]);
 const double stress=std::hypot(traction[0],traction[1]);auto warm=c;warm.maxIterations=1;
 WallOutput<double,2>again;WallStatus warmStatus;
 assert(evaluateEquilibriumOdeWallLayerImpl(in,warm,w,again,warmStatus,detail::SerialWallTeam{},history,stress,traction));
 assert(warmStatus.iteration==0);
 for(int d=0;d<3;++d)assert(std::abs(again.traction[d]-out.traction[d])<1e-6);
}

assert(std::abs(out.particleEnergyResidual)<1e-4);
for(int d=0;d<2;++d)assert(std::abs(out.particleMomentumResidual[d])<1e-5);
}}