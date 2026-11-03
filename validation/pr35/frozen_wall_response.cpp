#include "../pr32/equilibrium_wall_fixture.H"
using namespace ugkwp::gaswall::detail;
int main(){
 Fixture f;WallModelConfig<double> c;configure(17,f,c);c.nodes=8;c.stretch=2.5;c.maxIterations=200;
 NodeParticleSourceResponse<double> response[8];const double mass=2e-5,alpha=.2,betaDt=-std::log1p(-alpha),theta=.4,scale=500;
 const double pu[3]={3,-1,.5};
 for(int i=0;i<c.nodes;++i){response[i].sourceScale=scale;response[i].moments.addRaw(mass,pu,theta,1.2e-4);response[i].moments.setUniformBetaDt(betaDt);}
 f.in.particleResponse=response;Workspace w;WallOutput<double,2> out;WallStatus status;
 assert(evaluateWallModel(f.in,c,w,out,status));assert(status.code==WallCode::Success);assert(out.residual<1e-8);assert(std::abs(out.heatBalanceResidual)<1e-5);
 LayerContext<double,2> ctx;assert(makeContext(f.in,c,ctx));LayerPoint<double,2> p;
 assert(particleResponsePoint(f.in,c,ctx,w.state,w.y,3,p));
 const double h=1e-5;auto plus=p,minus=p;plus.u[0]+=h;minus.u[0]-=h;
 const double dp=(particleMomentumSourceAt(f.in,ctx,3,plus,ctx.tangent[0])-particleMomentumSourceAt(f.in,ctx,3,minus,ctx.tangent[0]))/(2*h);
 assert(std::abs(dp-(-response[3].sourceScale*response[3].moments.weightedMass))<1e-8);
 double ug[3];wallParticleGasVelocity(f.in,ctx,p,ug);double de=0;
 for(int d=0;d<3;++d)de+=nodeMomentGasEnergyDerivative(response[3].moments,d,ug[d])*ctx.tangent[0][d];
 de=de*response[3].sourceScale-ctx.wallU[0]*dp;
 const double fd=(particleEnergySourceAt(f.in,ctx,3,plus)-particleEnergySourceAt(f.in,ctx,3,minus))/(2*h);
 assert(std::abs(fd-de)<1e-8);
 const auto coupledTraction=out.traction;const double coupledResidual=out.residual;
 f.in.particleResponse=nullptr;Workspace baseW;WallOutput<double,2> baseOut;WallStatus baseStatus;
 assert(evaluateWallModel(f.in,c,baseW,baseOut,baseStatus));
 NodeParticleSourceResponse<double> zeroResponse[8]{};f.in.particleResponse=zeroResponse;
 Workspace zeroW;WallOutput<double,2> zeroOut;WallStatus zeroStatus;
 assert(evaluateWallModel(f.in,c,zeroW,zeroOut,zeroStatus));
 assert(std::abs(zeroOut.residual-baseOut.residual)<1e-14);
 for(int d=0;d<3;++d)assert(std::abs(zeroOut.traction[d]-baseOut.traction[d])<1e-12);
 assert(coupledResidual<1e-8&&std::abs(coupledTraction[0]-baseOut.traction[0])>1e-3);
 std::cout<<"response-active equilibrium wall solve, source derivatives and zero-response degeneration: PASS\n";
}
