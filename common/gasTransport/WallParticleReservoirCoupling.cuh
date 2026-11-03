#pragma once
#include "../gasWall/ReactingWallLayer.H"
namespace mixtureParticles {
template<class State,class=void> struct HasWallReservoir:std::false_type{};
template<class State> struct HasWallReservoir<State,std::void_t<
 decltype(std::declval<State>().gasBoundaryLayerModel.particleSource),
 decltype(std::declval<State>().px)>>:std::true_type{};
template<class State> __device__ int wallReservoirSlot(const State&s,int cell){
 if constexpr(HasWallReservoir<State>::value){
  const auto&m=s.gasBoundaryLayerModel;
  if(s.gasBoundaryLayer.enabled&&m.particleSource&&int(m.config.model)==2)
   return s.gasBoundaryLayer.ownerSlot[cell];
 }return -1;
}
template<class State> struct WallReservoirSelection {
 const double* y;int nodes,node,face;double height;bool outside;
 __device__ bool operator()(const State&s,int i)const{
  const auto&in=s.gasBoundaryLayerModel.input[s.gasBoundaryLayer.faceSlot[face]];
  const double h=(s.px[i]-s.faceCx[face])*in.normal[0]
               +(s.py[i]-s.faceCy[face])*in.normal[1]
               +(s.pz[i]-s.faceCz[face])*in.normal[2];
  if(node==nodes)return outside&&h>y[nodes-1];
  if(outside&&h>y[nodes-1])return false;
  // Convex-cell/planar-wall reconstruction; clipping here only locates a parcel
  // infinitesimally outside the face due to tracking roundoff, never its state.
  const double position=fmin(fmax(h,0.),height);
  return ugkwp::particleNormalReservoir(y,nodes,position)==node;
 }
};
}
// Read-only per-node particle moments and heat moments for the wall ODE source.
// Macro gas closure and one ordinary parcel relaxation remain in the common stages.
template<class DragModel,class State>
__device__
#if defined(__CUDA_ARCH__)
__noinline__
#endif
void gatherWallParticleMomentResponse(State&s,int c,int slot,double dt,
 bool drag,bool csr,bool split){
 constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
 constexpr int capacity=129;
 auto&m=s.gasBoundaryLayerModel;const auto&in=m.input[slot];const int nodes=m.config.nodes;
 const int face=m.faces[slot];const double area=s.magSf[face],height=s.V[c]/area;
 const double volume=s.epsGPrev[c]*s.V[c];
 if(nodes<2||nodes>=capacity||!(height>0&&volume>0&&dt>0)){
  using namespace ugkp::diagnostics;recordFatal(s.diagnostics,
   makeRecord(Code::InvalidPhysicalState,30701,0,0,EntityKind::Cell,c));return;
 }
 const double scale=1/(s.epsGPrev[c]*area*dt),epsG=s.epsGPrev[c];
 for(int node=0;node<nodes;++node){auto&r=m.particleResponse[std::size_t(slot)*nodes+node];
  r=ugkwp::gaswall::detail::NodeParticleSourceResponse<double>{};r.sourceScale=scale;}
 const int end=mixtureParticles::count(s,c,csr,split);
 const bool heat=s.solveParticleTemperature!=0&&s.particleGasHeatTransferModelId!=0;
 if(end==0||(!drag&&!heat)||dt==0)return;
 double y[capacity];ugkwp::ParticleReservoirShape<Ns,double> shape[capacity];
 ugkwp::ParticleThermalReservoir<Ns,double> gas[capacity];
 ugkwp::gaswall::detail::LayerContext<double,Ns> ctx;
 bool available=ugkwp::gaswall::detail::makeContext(in,m.config,ctx);
 const double* profile=m.candidateHistory+std::size_t(slot)*nodes*(Ns+4);
 for(int i=0;i<nodes;++i)y[i]=in.matchingDistance*pow(double(i)/double(nodes-1),m.config.stretch);
 for(int i=0;i<nodes;++i){
  const double lo=i?.5*(y[i-1]+y[i]):0,hi=fmin(height,i+1<nodes?.5*(y[i]+y[i+1]):y[i]);
  auto&q=shape[i];q.volumeFraction=fmax(0.,hi-lo)/height;if(q.volumeFraction==0)continue;
  const double centre=.5*(lo+hi);int left=0;while(left+1<nodes-1&&y[left+1]<centre)++left;
  const double alpha=(centre-y[left])/(y[left+1]-y[left]);double state[Ns+4];
  for(int j=0;j<Ns+4;++j)state[j]=(1-alpha)*profile[left*(Ns+4)+j]+alpha*profile[(left+1)*(Ns+4)+j];
  ugkwp::gaswall::detail::LayerPoint<double,Ns> point;
  available=available&&ugkwp::gaswall::detail::pointState(in,m.config,ctx,state,point);
  q.density=point.rho;q.temperature=point.T;
  for(int j=0;j<Ns;++j)q.massFraction[j]=point.Y[j];
  for(int d=0;d<3;++d)q.velocity[d]=in.velocity[d]+point.u[0]*ctx.tangent[0][d]
      +point.u[1]*ctx.tangent[1][d]+(point.normalVelocity-ctx.wallNormal)*in.normal[d];
 }
 auto&outer=shape[nodes];outer.volumeFraction=fmax(0.,height-y[nodes-1])/height;
 outer.density=s.rho[c];outer.temperature=s.Tgas[c];
 for(int d=0;d<3;++d)outer.velocity[d]=d==0?s.Ux[c]:d==1?s.Uy[c]:s.Uz[c];
 for(int j=0;j<Ns;++j)outer.massFraction[j]=s.gasSpecies.rho[j*s.nCells+c]/s.rho[c];
 ugkwp::ParticleGasReservoir<double> macro;macro.mass=volume*s.rho[c];macro.energy=volume*s.rhoE[c];
 macro.momentum[0]=volume*s.rhoUx[c];macro.momentum[1]=volume*s.rhoUy[c];macro.momentum[2]=volume*s.rhoUz[c];
 double species[Ns];for(int j=0;j<Ns;++j)species[j]=volume*s.gasSpecies.rho[j*s.nCells+c];
 available=available&&ugkwp::reconstructParticleReservoirs(macro,species,volume,s.Tgas[c],shape,nodes+1,
   s.gasSpecies.thermo,s.gasSpecies.thermoControls,gas);
 if(!available){using namespace ugkp::diagnostics;recordFatal(s.diagnostics,
  makeRecord(Code::InvalidPhysicalState,30701,0,0,EntityKind::Cell,c));return;}
 for(int pos=0;pos<end;++pos){const int i=mixtureParticles::parcel(s,c,pos,csr,split);
  if(!mixtureParticles::member(s,i,c))continue;
  const double h=(s.px[i]-s.faceCx[face])*in.normal[0]+(s.py[i]-s.faceCy[face])*in.normal[1]+(s.pz[i]-s.faceCz[face])*in.normal[2];
  if(outer.volumeFraction>0&&h>y[nodes-1])continue;
  const double position=fmin(fmax(h,0.),height);const int node=ugkwp::particleNormalReservoir(y,nodes,position);
  if(!(gas[node].mass>0))continue;
  mixtureParticles::WallReservoirSelection<State> select{y,nodes,node,face,height,outer.volumeFraction>0};
  mixtureParticles::NodeReservoir<State,Ns,decltype(select)> reservoir{gas[node],select,nullptr};
  const double q=drag?reservoir.template beta<DragModel>(s,c,i,epsG)*dt:0.;
  const double velocity[3]={s.pux[i],s.puy[i],s.puz[i]};
#if GPU_OPERATOR_THERMAL
  const double theta=particleMomentThetaDevice(s,i);
#else
  const double theta=clampMin(finiteOr(s.pTheta[i],0.0),0.0);
#endif
  auto& response=m.particleResponse[std::size_t(slot)*nodes+node].moments;
  if(drag)response.addRaw(s.pm[i],velocity,theta,q);
  if(heat){
   const double tp=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
#if GPU_OPERATOR_THERMAL
   const double h=materialEnthalpyMoment(s.pm[i],tp);
#else
   const double heatFactor=particleHeatFactorDevice(s);
   const double h=s.pm[i]*heatFactor*tp;
#endif
   response.heatMass+=s.pm[i];response.heatEnthalpy+=h;
   response.heatDiameter+=s.pm[i]*clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),1.0e-12);
   response.heatMomentum[0]+=s.pm[i]*s.pux[i];
   response.heatMomentum[1]+=s.pm[i]*s.puy[i];
   response.heatMomentum[2]+=s.pm[i]*s.puz[i];
  }
 }
 if(s.solveParticleTemperature!=0&&s.particleGasHeatTransferModelId!=0){
  for(int node=0;node<nodes;++node){
   auto& response=m.particleResponse[std::size_t(slot)*nodes+node].moments;
   const double mass=response.heatMass;
   if(!(mass>0.)||!(gas[node].mass>0.))continue;
#if GPU_OPERATOR_THERMAL
   const double tpAvg=Foam::gpuThermal::aluminaTemperatureFromSpecificEnthalpyK(response.heatEnthalpy/(mass+OfSmall));
   const double cp=particleSpecificHeatDevice(tpAvg);
   const double rateCapacity=s.rhoSolid*cp;
#else
   const double cp=particleHeatFactorDevice(s);
   const double tpAvg=clampRange(response.heatEnthalpy/(mass*clampMin(cp,OfSmall)),s.TpMin,s.TpMax);
   const double rateCapacity=s.particleThermalRho*s.particleCp;
#endif
   if(!(cp>0.)||!(rateCapacity>0.))continue;
   const double diameter=clampMin(response.heatDiameter/mass,1.0e-12);
   const double usx=response.heatMomentum[0]/mass,usy=response.heatMomentum[1]/mass,usz=response.heatMomentum[2]/mass;
   const double gvx=gas[node].momentum[0]/gas[node].mass,gvy=gas[node].momentum[1]/gas[node].mass,gvz=gas[node].momentum[2]/gas[node].mass;
   const double slip=sqrt(sqr3(gvx-usx,gvy-usy,gvz-usz));
   const double gasTemperature=gas[node].temperature;
   const double gasDensity=gas[node].mass/gas[node].volume;
   const double mu=clampMin(ugkwp::gasCellMolecularViscosityAt(s,c,gasTemperature),1.0e-30);
   const double re=gasDensity*diameter*slip/mu;
   const double pr=ugkwp::gasCellMolecularPrandtl(s,c);
   const double conductivity=molecularGasConductivity(s,c,gasTemperature);
   const double nu=ugkwp::ranzMarshallNuFromPr(clampMin(re,0.0),clampMin(pr,1.0e-12));
   const double rate=6.0*nu*conductivity/(rateCapacity*diameter*diameter+1.0e-300);
   const double gasCv=s.Rgas/clampMin(s.gammaGas-1.0,OfSmall);
   const double gasCapacity=gas[node].mass*gasCv;
   const double solidCapacity=mass*cp;
   const double totalCapacity=gasCapacity+solidCapacity;
   const double effectiveCapacity=gasCapacity*solidCapacity/(totalCapacity+OfSmall);
   const double coupledRate=clampMin(rate,0.0)*(1.0+solidCapacity/(gasCapacity+OfSmall));
   response.particleTemperature=clampRange(tpAvg,s.TpMin,s.TpMax);
   response.heatGasCapacity=gasCapacity;response.heatSolidCapacity=solidCapacity;
   response.heatRate=clampMin(rate,0.0);response.heatDt=dt;
  }
 }
 for(int node=0;node<nodes;++node){
  auto& response=m.particleResponse[std::size_t(slot)*nodes+node].moments;
  response.gasMass=gas[node].mass;
  response.gasCv=s.Rgas/clampMin(s.gammaGas-1.0,OfSmall);
  response.gasTemperatureMin=s.TgasMin;
 }
}
