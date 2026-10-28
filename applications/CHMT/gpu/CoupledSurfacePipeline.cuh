#ifndef CHMT_COUPLED_SURFACE_PIPELINE_CUH
#define CHMT_COUPLED_SURFACE_PIPELINE_CUH
#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
// Geometry is uploaded per moving microstep or once per static window.
// The wall program and gas driving states remain resident throughout a window.
struct CoupledSurfaceInput {
    SurfacePacketIdentity identity{};
    Real area=0,gasDistance=0,sweptVolume=0;
    Vec3 normal{};
};
struct CoupledSurfaceRecord {
    CoupledSurfaceSample sample{};
    GasWallResult rates{};
    GasQ raw{};
    ExchangePacket primary{},pore{};
    GasWallMatchingSample matching{};
    Real radiationEnergy=0;
    unsigned radiationReceiver=0;
    int failed=0,wallTransportCode=0;
    ugkwp::gaswall::WallStatus wallStatus{};
};
// Only fields consumed by host material/history/output cross at a microstep
// endpoint. Raw fluxes and unweighted stage packets remain device-only.
struct CoupledSurfaceDownload {
    GasPrimitive bulk{};GasGradient gradient{};Vec3 traction{};
    ExchangePacket primary{},pore{};
    Real radiationEnergy=0;unsigned radiationReceiver=0;
    int failed=0;
};
// Boundary-layer-only output has its own optional transfer. Low-Re interfaces
// neither allocate nor copy nonlinear wall solver diagnostic arrays.
struct CoupledWallDownload {
    GasWallMatchingSample matching{};
    ugkwp::gaswall::WallOutput<Real,Ns> layer{};
    int transportCode=0;
    ugkwp::gaswall::WallStatus status{};
};
__global__ void packCoupledSurfaceKernel(const CoupledSurfaceRecord* records,int count,
    CoupledSurfaceDownload* output,CoupledWallDownload* wallOutput){
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    const auto& in=records[slot];auto& out=output[slot];
    out.bulk=in.sample.primitive;out.gradient=in.sample.gradient;out.traction=in.rates.traction;
    out.primary=in.primary;out.pore=in.pore;
    out.radiationEnergy=in.radiationEnergy;out.radiationReceiver=in.radiationReceiver;
    out.failed=in.failed;
    if(wallOutput){auto& wall=wallOutput[slot];
        wall.matching=in.matching;wall.layer=in.rates.layer;
        wall.transportCode=in.wallTransportCode;wall.status=in.wallStatus;
    }
}
__global__ void applyCoupledSurfaceKernel(SharedGasDeviceView* state,const int* faces,int count,
    const CoupledSurfaceInput* supplied,const WallFaceSample* wallProgram,GasWallProgramPosition position,
    const PhysicsConfig* physics,Real dt,bool firstStage,std::uint64_t sequence,std::uint64_t geometryVersion,
    CoupledSurfaceRecord* records){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    auto& v=*state;auto& record=records[slot];
    if(firstStage)record=CoupledSurfaceRecord{};
    if(record.failed)return;
    const int face=faces[slot];const auto& geometry=supplied[slot];const auto w=sampleResidentGasWallFace(wallProgram,count,position,slot);
    const auto& p=*physics;
    record.sample=readCoupledSurfaceSample(v,face,p.singleGasSpecies);
    GasWallInput input;
    input.bulk=record.sample.primitive;input.gradient=record.sample.gradient;
    input.temperature=w.temperature;input.gasDistance=geometry.gasDistance;input.area=geometry.area;
    input.gasArea=v.magSf[face];input.dt=dt;input.normal=geometry.normal;input.velocity=w.velocity;
    input.primaryKind=w.primaryKind;input.normalSpeed=w.normalVelocity;input.sweptVolume=geometry.sweptVolume;
    Real massFlux=0;
    for(int s=0;s<Ns;++s){input.speciesRate[s]=w.speciesRate[s];input.poreRate[s]=w.poreRate[s];input.poreSweepRate[s]=w.poreSweepRate[s];
        massFlux+=(w.speciesRate[s]+w.poreRate[s])*(input.area/input.gasArea);}
    if(v.gasBoundaryLayer.enabled){
        const int wallSlot=v.gasBoundaryLayer.faceSlot[face];
        if(wallSlot<0){record.failed=1;return;}
        if(!v.gasBoundaryLayer.exchange[wallSlot].ready){
            record.failed=2;record.wallTransportCode=v.gasBoundaryLayer.status[wallSlot];
            record.wallStatus=v.gasBoundaryLayerModel.status[wallSlot];return;
        }
        const auto& prepared=v.gasBoundaryLayerModel.input[wallSlot];
        input.normal={-v.Sfx[face]/input.gasArea,-v.Sfy[face]/input.gasArea,-v.Sfz[face]/input.gasArea};
        input.preparedLayer=&v.gasBoundaryLayerModel.output[wallSlot];
        input.wallContext.matchingPressure=prepared.pressure;input.wallContext.matching=prepared.matching;
        input.preparedMatching=&prepared.matching;input.preparedMatchingPressure=&prepared.pressure;
        const auto& layer=*input.preparedLayer;const Vec3 outward=-input.normal;
        const Vec3 traceVelocity={layer.traceVelocity[0],layer.traceVelocity[1],layer.traceVelocity[2]};
        const Vec3 traction={layer.traction[0],layer.traction[1],layer.traction[2]};
        input.wallContext.mechanicalPressure=dot(record.sample.flux.momentum,outward)/input.gasArea
            +massFlux*dot(traceVelocity,outward)-dot(traction,outward);
        record.matching.pressure=prepared.pressure;record.matching.state=prepared.matching;
        record.matching.mechanicalPressure=input.wallContext.mechanicalPressure;
    }
    for(int c=0;c<Nc;++c)input.condensedRate[c]=w.condensedRate[c];
    auto identity=geometry.identity;identity.step=sequence;identity.geometry=geometryVersion;
    if(!evaluateGasWall(input,identity,p,record.rates)){record.failed=1;return;}
    // Output diagnostics retain the original prepared gas-wall model result,
    // including when the packet closure needed a distinct local evaluation.
    if(input.preparedLayer)record.rates.layer=*input.preparedLayer;
    const auto a=packetDelta(record.rates.primary).gas,c=packetDelta(record.rates.pore).gas;
    record.raw=(a+c)*(-1/dt);const auto& flux=record.raw;
    v.gasPhiRho[face]=flux.mass;v.gasPhiRhoUx[face]=flux.momentum.x;v.gasPhiRhoUy[face]=flux.momentum.y;
    v.gasPhiRhoUz[face]=flux.momentum.z;v.gasPhiRhoE[face]=flux.energy;
    if(v.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)
        for(int s=0;s<Ns;++s)v.gasSpecies.flux[s*v.nFaces+face]=flux.species[s];
    if(firstStage){
        record.primary=scaledIntervalPacket(record.rates.primary,0);record.primary.consumerMask=ConsumeGas;
        record.pore=scaledIntervalPacket(record.rates.pore,0);record.pore.consumerMask=ConsumeGas;
    }
}
__global__ void accountCoupledSurfaceKernel(const SharedGasDeviceView* state,const int* faces,int count,
    const CoupledSurfaceInput* supplied,const WallFaceSample* wallProgram,GasWallProgramPosition position,
    const PhysicsConfig* physics,Real stageDt,Real ledgerDt,CoupledSurfaceRecord* records){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    auto& record=records[slot];if(record.failed)return;
    const auto& raw=record.raw;const auto now=readCoupledFaceFlux(*state,faces[slot],physics->singleGasSpecies);
    Real denominator=raw.energy,numerator=now.energy;
    if(absValue(raw.mass)>absValue(denominator)){denominator=raw.mass;numerator=now.mass;}
    for(int s=0;s<Ns;++s)if(absValue(raw.species[s])>absValue(denominator)){denominator=raw.species[s];numerator=now.species[s];}
    const Real scale=denominator==0?1:numerator/denominator;
    auto primary=scaledIntervalPacket(record.rates.primary,(ledgerDt/stageDt)*scale);primary.consumerMask=ConsumeGas;
    auto pore=scaledIntervalPacket(record.rates.pore,(ledgerDt/stageDt)*scale);pore.consumerMask=ConsumeGas;
    addIntervalPacket(record.primary,primary);addIntervalPacket(record.pore,pore);
    const auto& in=supplied[slot];const auto wall=sampleResidentGasWallFace(wallProgram,count,position,slot);
    record.radiationEnergy+=ledgerDt*wall.radiationFlux*in.area;
    record.radiationReceiver=wall.primaryKind==ExchangeKind::GasFilm?ConsumeFilm:ConsumeSolid;
}
}
#endif
