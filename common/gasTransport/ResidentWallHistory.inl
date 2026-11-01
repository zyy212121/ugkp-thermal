#pragma once
template<class State> __global__ void commitAcceptedWallHistory(State*state){
 if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
 auto&m=state->gasBoundaryLayerModel;const int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=state->gasBoundaryLayer.count||!m.acceptedHistory||!m.candidateHistory||!m.candidateFrame||!m.acceptedFrame||!m.candidateFrame[slot].valid)return;
 constexpr int B=ugkwp::compiledGasSpecies+4;const std::size_t offset=std::size_t(slot)*m.config.nodes*B;
 for(int i=0;i<m.config.nodes*B;++i)m.acceptedHistory[offset+i]=m.candidateHistory[offset+i];
 m.acceptedFrame[slot]=m.candidateFrame[slot];
}

// Shared completion hook for gas-only transactions and coupled particle steps.
// The device kernel declines publication after a fatal diagnostic.
template<class State> inline cudaError_t commitResidentWallHistory(State* s) {
 if(s->gasBoundaryLayer.enabled&&s->gasBoundaryLayerModel.acceptedHistory){
  commitAcceptedWallHistory<<<(s->gasBoundaryLayer.count+63)/64,64,0,s->gasCaptureStream>>>(s->deviceState);
  return cudaGetLastError();
 }
 return cudaSuccess;
}
