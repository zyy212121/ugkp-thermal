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
// One serial worker owns a wall cell and its reservoirs, so intermediate gas
// updates are immediately visible to the following heat transaction. Bulk
// cells retain the existing scalar/warp scheduling and transaction unchanged.
template<class DragModel,bool ConstantHeat,class State>
__device__
#if defined(__CUDA_ARCH__)
__noinline__
#endif
void mixtureParticleWallReservoirs(State&s,int c,int slot,double dt,
 bool drag,bool csr,bool split){
 if(mixtureParticles::count(s,c,csr,split)==0)return;
 constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
 constexpr int capacity=129; // same public wall capacity (128), plus outer cell remainder
 auto&m=s.gasBoundaryLayerModel;const auto&in=m.input[slot];const int nodes=m.config.nodes;
 const int face=m.faces[slot];const double area=s.magSf[face],height=s.V[c]/area;
 const double volume=s.epsGPrev[c]*s.V[c];
 if(dt==0)return;
 if(nodes<2||nodes>=capacity||!(height>0&&volume>0&&dt>0)){
  using namespace ugkp::diagnostics;recordFatal(s.diagnostics,
   makeRecord(Code::InvalidPhysicalState,30701,0,0,EntityKind::Cell,c));return;
 }
 double y[capacity];ugkwp::ParticleReservoirShape<Ns,double> shape[capacity];
 ugkwp::ParticleThermalReservoir<Ns,double> gas[capacity];
 ugkwp::gaswall::detail::LayerContext<double,Ns> ctx;
 bool available=ugkwp::gaswall::detail::makeContext(in,m.config,ctx);
 const double* profile=m.candidateHistory+std::size_t(slot)*nodes*(Ns+4);
 for(int i=0;i<nodes;++i)y[i]=in.matchingDistance*pow(double(i)/double(nodes-1),m.config.stretch);
 for(int i=0;i<nodes;++i){
  const double lo=i?.5*(y[i-1]+y[i]):0,hi=fmin(height,i+1<nodes?.5*(y[i]+y[i+1]):y[i]);
  auto&q=shape[i];q.volumeFraction=fmax(0.,hi-lo)/height;
  if(q.volumeFraction==0)continue;
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
 // A matching plane inside the owner volume leaves an ordinary macro bath
 // outside the wall layer. No parcel is exchanged with both baths.
 auto&outer=shape[nodes];outer.volumeFraction=fmax(0.,height-y[nodes-1])/height;
 outer.density=s.rho[c];outer.temperature=s.Tgas[c];
 const double u[3]={s.Ux[c],s.Uy[c],s.Uz[c]};
 for(int d=0;d<3;++d)outer.velocity[d]=u[d];
 for(int j=0;j<Ns;++j)outer.massFraction[j]=s.gasSpecies.rho[j*s.nCells+c]/s.rho[c];
 ugkwp::ParticleGasReservoir<double> macro;macro.mass=volume*s.rho[c];macro.energy=volume*s.rhoE[c];
 macro.momentum[0]=volume*s.rhoUx[c];macro.momentum[1]=volume*s.rhoUy[c];macro.momentum[2]=volume*s.rhoUz[c];
 double species[Ns];for(int j=0;j<Ns;++j)species[j]=volume*s.gasSpecies.rho[j*s.nCells+c];
 available=available&&ugkwp::reconstructParticleReservoirs(macro,species,volume,s.Tgas[c],shape,nodes+1,
   s.gasSpecies.thermo,s.gasSpecies.thermoControls,gas);
 if(!available){using namespace ugkp::diagnostics;recordFatal(s.diagnostics,
  makeRecord(Code::InvalidPhysicalState,30701,0,0,EntityKind::Cell,c));return;}
 double impulse[3]{},energy=0;
 for(int i=0;i<=nodes;++i){
  if(!(gas[i].mass>0))continue;
  mixtureParticles::WallReservoirSelection<State> select{y,nodes,i,face,height,outer.volumeFraction>0};
  mixtureParticles::NodeReservoir<State,Ns,decltype(select)> reservoir{gas[i],select};
  mixtureParticleCouplingReservoir<DragModel,ConstantHeat>(s,c,dt,drag,csr,split,
    mixtureParticles::SerialCouplingTeam{},reservoir);
  // Reverse only the actual represented parcel increments. Reconstruction is
  // conservative, but its floating-point sum is not another physical source.
  for(int d=0;d<3;++d)impulse[d]+=reservoir.reverseMomentum[d];
  energy+=reservoir.reverseEnergy;
  if(i<nodes){
   const std::size_t offset=std::size_t(slot)*nodes+i;double*src=m.particleSource+4*offset;
   const double scale=1/(s.epsGPrev[c]*area*dt);
   for(int d=0;d<3;++d)src[d]=reservoir.reverseMomentum[d]*scale;
   src[3]=reservoir.reverseEnergy*scale;m.particleHeat[offset]=reservoir.thermalExchange*scale;
   double*row=m.particleReservoir+offset*(Ns+7);row[0]=gas[i].volume;row[1]=gas[i].mass;
   for(int d=0;d<3;++d)row[2+d]=gas[i].momentum[d];row[5]=gas[i].energy;row[6]=gas[i].temperature;
   for(int j=0;j<Ns;++j)row[7+j]=gas[i].speciesMass[j];
  }
 }
 s.rhoUx[c]+=impulse[0]/volume;s.rhoUy[c]+=impulse[1]/volume;s.rhoUz[c]+=impulse[2]/volume;s.rhoE[c]+=energy/volume;
 recoverGasPrimitiveCell(s,c);
}
