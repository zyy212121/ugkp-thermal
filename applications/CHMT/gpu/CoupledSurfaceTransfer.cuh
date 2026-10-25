#ifndef CHMT_COUPLED_SURFACE_TRANSFER_CUH
#define CHMT_COUPLED_SURFACE_TRANSFER_CUH

namespace chmt {
struct CoupledSurfaceSample {
    GasPrimitive primitive;
    GasGradient gradient;
    GasQ flux;
};
__device__ inline GasQ readCoupledFaceFlux(const SharedGasDeviceView& v,int face,int activeSpecies){
    GasQ q;q.mass=v.gasPhiRho[face];q.momentum={v.gasPhiRhoUx[face],v.gasPhiRhoUy[face],v.gasPhiRhoUz[face]};q.energy=v.gasPhiRhoE[face];
    for(int s=0;s<Ns;++s)q.species[s]=v.gasSpecies.mode==ugkwp::GasMode::SingleLegacy
        ?(s==activeSpecies?q.mass:0):v.gasSpecies.flux[s*v.nFaces+face];
    return q;
}
__global__ void gatherCoupledSurface(const SharedGasDeviceView* state,const int* faces,
    int count,int activeSpecies,CoupledSurfaceSample* samples){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    const auto& v=*state;const int face=faces[slot],cell=v.faceOwner[face];
    auto& sample=samples[slot];auto& w=sample.primitive;auto& g=sample.gradient;
    w.rho=v.rho[cell];w.temperature=v.Tgas[cell];w.pressure=v.p[cell];w.velocity={v.Ux[cell],v.Uy[cell],v.Uz[cell]};
    const bool mixture=v.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy;
    w.soundSpeed=mixture?v.gasSpecies.soundSpeed[cell]:sqrt(v.gammaGas*w.pressure/w.rho);
    g.rho={v.gradRhoX[cell],v.gradRhoY[cell],v.gradRhoZ[cell]};
    g.temperature={v.gradTX[cell],v.gradTY[cell],v.gradTZ[cell]};
    g.pressure={v.gradPx[cell],v.gradPy[cell],v.gradPz[cell]};
    g.velocity[0]={v.gradUxX[cell],v.gradUxY[cell],v.gradUxZ[cell]};
    g.velocity[1]={v.gradUyX[cell],v.gradUyY[cell],v.gradUyZ[cell]};
    g.velocity[2]={v.gradUzX[cell],v.gradUzY[cell],v.gradUzZ[cell]};
    for(int s=0;s<Ns;++s){
        w.Y[s]=mixture?v.gasSpecies.rho[s*v.nCells+cell]/w.rho:(s==activeSpecies?1:0);
        g.Y[s]=mixture?Vec3{v.gasSpecies.gradX[s*v.nCells+cell],v.gasSpecies.gradY[s*v.nCells+cell],v.gasSpecies.gradZ[s*v.nCells+cell]}:Vec3{};
    }
    sample.flux=readCoupledFaceFlux(v,face,activeSpecies);
}
__global__ void gatherCoupledFlux(const SharedGasDeviceView* state,const int* faces,
    int count,int activeSpecies,GasQ* flux){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    flux[slot]=readCoupledFaceFlux(*state,faces[slot],activeSpecies);
}
__global__ void scatterCoupledFlux(SharedGasDeviceView* state,const int* faces,int count,const GasQ* flux){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    auto& v=*state;const int face=faces[slot];const auto& q=flux[slot];
    v.gasPhiRho[face]=q.mass;v.gasPhiRhoUx[face]=q.momentum.x;v.gasPhiRhoUy[face]=q.momentum.y;v.gasPhiRhoUz[face]=q.momentum.z;v.gasPhiRhoE[face]=q.energy;
    if(v.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)
        for(int s=0;s<Ns;++s)v.gasSpecies.flux[s*v.nFaces+face]=q.species[s];
}
}
#endif
