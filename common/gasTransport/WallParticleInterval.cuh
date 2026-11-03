#pragma once
// Source buffers are scratch for this physical interval, reconstructed from
// current gas and particles after every restart. No lagged source is replayed.
template<class State> __global__ void resetWallParticleInterval(State*sp){
 auto&s=*sp;auto&m=s.gasBoundaryLayerModel;const int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot==0)m.particleCoupling=false;
 if(slot>=s.gasBoundaryLayer.count||!m.particleSource)return;
 for(int i=0;i<m.config.nodes;++i){
  const int offset=slot*m.config.nodes+i;
  for(int d=0;d<4;++d)m.particleSource[4*offset+d]=0;
  m.particleHeat[offset]=0;
  if(m.particleResponse)m.particleResponse[offset]=ugkwp::gaswall::detail::NodeParticleSourceResponse<double>{};
  for(int j=0;j<ugkwp::compiledGasSpecies+7;++j)m.particleReservoir[offset*(ugkwp::compiledGasSpecies+7)+j]=0;
 }
}
template<class State> __global__ void publishWallParticleInterval(State*sp,double dt,double time){
 if(blockIdx.x==0&&threadIdx.x==0){auto&m=sp->gasBoundaryLayerModel;
  m.particleCoupling=true;m.particleIntervalDt=dt;m.particleIntervalTime=time;
 }
}
