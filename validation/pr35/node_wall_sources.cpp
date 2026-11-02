
#include "../pr32/equilibrium_wall_fixture.H"
int main(){
 std::cout<<std::setprecision(17);
 for(bool moving:{false,true})for(bool adiabatic:{false,true})for(bool turbulent:{false,true}){
  Fixture f;WallModelConfig<double> c;configure(17,f,c);c.nodes=8;c.stretch=2.5;c.maxIterations=200;
  f.in.adiabatic=adiabatic;f.in.matchingDistance=.01;f.in.ownerDistance=.005;
  if(turbulent)c.equilibriumKappa=.41;
  if(moving){f.in.velocity[0]=23;f.in.velocity[2]=-7;f.in.matching.velocity[0]+=23;f.in.matching.velocity[2]=-7;}
  double source[32]{};
  for(int i=0;i<c.nodes;++i){source[4*i]=.0005*(i+1);source[4*i+1]=.0002*(i-3);source[4*i+2]=-.0003*(i+1);source[4*i+3]=3*(i+1);}
  f.in.particleSource=source;Workspace w;WallOutput<double,2> out;WallStatus status;
  assert(evaluateWallModel(f.in,c,w,out,status));
  detail::LayerContext<double,2> ctx;assert(detail::makeContext(f.in,c,ctx));
  ctx.equilibriumWallStress=std::sqrt(detail::dot(out.traction,out.traction));
  for(int d=0;d<2;++d)ctx.equilibriumWallTraction[d]=detail::dot(out.traction,ctx.tangent[d]);
  detail::LayerFlux<double,2> first,previous,last;
  assert(detail::intervalFlux(f.in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first));previous=first;
  for(int i=1;i+1<c.nodes;++i){
   detail::LayerFlux<double,2> flux;
   assert(detail::intervalFlux(f.in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],flux));
   for(int d=0;d<2;++d)assert(std::abs(flux.momentum[d]-previous.momentum[d]-detail::particleMomentumSource(f.in,i,ctx.tangent[d]))<1e-7);
   assert(std::abs(flux.energy-previous.energy-detail::particleEnergySource(f.in,ctx,i))<1e-5);
   previous=flux;
  }last=previous;
  for(int d=0;d<2;++d){const double wall=-detail::dot(out.traction,ctx.tangent[d]);
   const double edge=last.momentum[d]+detail::particleMomentumSource(f.in,c.nodes-1,ctx.tangent[d]);
   double sum=0;for(int i=0;i<c.nodes;++i)sum+=detail::particleMomentumSource(f.in,i,ctx.tangent[d]);
   assert(std::abs(edge-wall-sum)<1e-7);assert(std::abs(out.particleMomentumResidual[d])<1e-7);
  }
  const double wallEnergy=out.conductiveHeatFlux+detail::wallTransportEnergy(f.in,ctx,[&](){detail::LayerPoint<double,2>p;assert(detail::pointState(f.in,c,ctx,w.state[0],p));return p;}());
  const double edgeEnergy=last.energy+detail::particleEnergySource(f.in,ctx,c.nodes-1);
  assert(std::abs(edgeEnergy-wallEnergy-out.particleEnergy)<1e-5);
  assert(std::abs(out.particleEnergyResidual)<1e-5);assert(std::abs(out.heatBalanceResidual)<1e-5);
  if(adiabatic)assert(std::abs(out.conductiveHeatFlux)<1e-5);
  assert(out.heatParticle!=0);
  // Prefix geometry includes both endpoint half CVs and all vector directions.
  for(int d=0;d<2;++d){double total=0;for(int i=0;i<c.nodes;++i)total+=detail::particleMomentumSource(f.in,i,ctx.tangent[d]);
   assert(std::abs(detail::particleMomentumPrefix(f.in,c,f.in.matchingDistance,ctx.tangent[d])-total)<1e-15);}
  std::cout<<"moving="<<moving<<" adiabatic="<<adiabatic<<" turbulent="<<turbulent<<" residual="<<out.residual
           <<" heatBalance="<<out.heatBalanceResidual<<" particleEnergyBalance="<<out.particleEnergyResidual<<'\n';
 }
}
