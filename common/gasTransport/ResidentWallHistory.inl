__global__ void commitAcceptedWallHistory(DeviceState*state){
 if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
 auto&m=state->gasBoundaryLayerModel;const int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=state->gasBoundaryLayer.count||!m.acceptedHistory||!m.candidateHistory||!m.candidateFrame||!m.acceptedFrame||!m.candidateFrame[slot].valid)return;
 constexpr int B=ugkwp::compiledGasSpecies+4;const std::size_t offset=std::size_t(slot)*m.config.nodes*B;
 for(int i=0;i<m.config.nodes*B;++i)m.acceptedHistory[offset+i]=m.candidateHistory[offset+i];
 m.acceptedFrame[slot]=m.candidateFrame[slot];
}
