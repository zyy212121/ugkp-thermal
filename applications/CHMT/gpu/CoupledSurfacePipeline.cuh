#ifndef CHMT_COUPLED_SURFACE_PIPELINE_CUH
#define CHMT_COUPLED_SURFACE_PIPELINE_CUH
#include "gpu/GasWallMath.H"
#include "../../../common/gasTransport/WallHeatExposure.H"
#include "gpu/GasWindowProgram.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
// Geometry is uploaded per moving microstep or once per static window.
// The wall program and gas driving states remain resident throughout a window.
struct CoupledSurfaceInput {
    SurfacePacketIdentity identity{};
    Real area=0,gasDistance=0,sweptVolume=0;
    Vec3 normal{},particleMaterialVelocity{};
};
struct CoupledSurfaceRecord {
    CoupledSurfaceSample sample{};
    GasWallResult rates{};
    WallFaceSample acceptedWall{};
    GasQ raw{};
    ExchangePacket primary{},pore{};
    GasWallMatchingSample matching{};
    Real radiationEnergy=0,particleHeatAtBegin=0;
    unsigned radiationReceiver=0;
    int failed=0,wallTransportCode=0;
    ugkwp::gaswall::WallStatus wallStatus{};
};
// Only fields consumed by host material/history/output cross at a microstep
// endpoint. Raw fluxes and unweighted stage packets remain device-only.
struct CoupledSurfaceDownload {
    GasPrimitive bulk{};GasGradient gradient{};Vec3 traction{};
    WallFaceSample acceptedWall{};
    ExchangePacket primary{},pore{};
    Real radiationEnergy=0,particleWallEnergy=0;unsigned radiationReceiver=0;
    int failed=0;
};
// Boundary-layer-only output has its own optional transfer. Low-Re interfaces
// neither allocate nor copy nonlinear wall solver diagnostic arrays.
struct CoupledWallDownload {
    GasWallMatchingSample matching{};
    ugkwp::gaswall::WallExchangeOutput<Real,Ns> layer{};
    int transportCode=0;
    ugkwp::gaswall::WallStatus status{};
};
__global__ void packCoupledSurfaceKernel(const CoupledSurfaceRecord* records,int count,
    const int* faces,ugkp::thermal::ParticleWallExchangeView particle,
    CoupledSurfaceDownload* output,CoupledWallDownload* wallOutput){
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    const auto& in=records[slot];auto& out=output[slot];
    out.bulk=in.sample.primitive;out.gradient=in.sample.gradient;out.traction=in.rates.traction;
    out.primary=in.primary;out.pore=in.pore;out.acceptedWall=in.acceptedWall;
    out.radiationEnergy=in.radiationEnergy;out.radiationReceiver=in.radiationReceiver;
    out.failed=in.failed;
    out.particleWallEnergy=particle.depositedEnergy
        ?particle.depositedEnergy[faces[slot]]+particle.reflectedEnergy[faces[slot]]-in.particleHeatAtBegin:0;
    if(wallOutput){auto& wall=wallOutput[slot];
        wall.matching=in.matching;wall.layer=in.rates.layer;
        wall.transportCode=in.wallTransportCode;wall.status=in.wallStatus;
    }
}
__global__ void publishParticleMaterialSurfaceKernel(int count,const int* faces,
    const CoupledSurfaceInput* geometry,const WallFaceSample* program,GasWallProgramPosition position,
    double* temperature,double* ux,double* uy,double* uz,double* effusivity){
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    const int f=faces[slot];const auto wall=sampleResidentGasWallFace(program,count,position,slot);
    temperature[f]=wall.temperature;effusivity[f]=wall.materialEffusivity;const auto velocity=geometry[slot].particleMaterialVelocity;
    ux[f]=velocity.x;uy[f]=velocity.y;uz[f]=velocity.z;
}
// The gas stage owns the final reactant budget. Proposals may outlive the
// inventory from which the material predictor constructed them. Sum all faces
// of a donor cell before choosing a common extent; no face can spend the same
// inventory twice. Incoming flux is deliberately not borrowed in this stage.
__global__ void boundSurfaceProgressKernel(const SharedGasDeviceView* state,
    const int* cells,int cellCount,const int* faceSlot,const CoupledSurfaceInput* geometry,const WallFaceSample* program,
    int count,GasWallProgramPosition position,Real dt,bool transportReady,Real* progress){
    const auto& v=*state;const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=cellCount)return;const int cell=cells[index];
    Real demand[Ns]{},other[Ns]{},demandMass=0,otherMass=0;
    const Real phase=v.gasVolumeFraction?v.gasVolumeFraction[cell]:1;
    const Real volume=v.gasGeometry.enabled?v.gasGeometry.oldVolume[cell]:v.V[cell];
    for(int j=0;j<v.cellPlaneCount[cell];++j){
        const int f=v.cellFaceId[v.cellPlaneStart[cell]+j];const int slot=faceSlot[f];
        if(slot>=0){
            const auto w=sampleResidentGasWallFace(program,count,position,slot);
            Real primaryMass=0,darcyMass=0,sweepMass=0;
            for(int k=0;k<Ns;++k){
                primaryMass+=w.speciesRate[k];darcyMass+=w.poreRate[k]-w.poreSweepRate[k];sweepMass+=w.poreSweepRate[k];
                // Primary stoichiometry and Darcy exchange share an extent.
                // Geometric pore sweep is fixed by the executed trajectory.
                demand[k]+=geometry[slot].area*(maxValue(0,-w.speciesRate[k])
                    +maxValue(0,-(w.poreRate[k]-w.poreSweepRate[k])));
                other[k]+=geometry[slot].area*maxValue(0,-w.poreSweepRate[k]);
            }
            demandMass+=geometry[slot].area*(maxValue(0,-primaryMass)+maxValue(0,-darcyMass));
            otherMass+=geometry[slot].area*maxValue(0,-sweepMass);
        }else if(transportReady){
            const Real sign=v.faceOwner[f]==cell?1:-1;
            otherMass+=phase*maxValue(0,sign*v.gasPhiRho[f]);
            for(int k=0;k<Ns;++k)other[k]+=phase*maxValue(0,sign*v.gasSpecies.flux[k*v.nFaces+f]);
        }
    }
    Real factor=1;
    for(int k=0;k<Ns;++k)if(demand[k]>0&&dt>0){
        const Real inventory=phase*v.gasSpecies.rho[k*v.nCells+cell]*volume;
        // A rounding reserve only, not a time-step or concentration floor.
        const Real available=maxValue(0,inventory*(1-64*2.2204460492503131e-16)-dt*other[k]);
        factor=minValue(factor,maxValue(0,minValue(1,available/(dt*demand[k]))));
    }
    if(demandMass>0&&dt>0){
        const Real inventory=phase*maxValue(0,v.rho[cell]-v.rhoMin)*volume;
        const Real available=maxValue(0,inventory*(1-64*2.2204460492503131e-16)-dt*otherMass);
        factor=minValue(factor,maxValue(0,minValue(1,available/(dt*demandMass))));
    }
    progress[cell]=factor;
}
__global__ void prepareResidentSurfaceKernel(SharedGasDeviceView* state,const int* faces,
    int count,const CoupledSurfaceInput* geometry,const WallFaceSample* program,
    GasWallProgramPosition position,const Real* progress,WallFaceSample* bounded,unsigned char* changed=nullptr){
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;
    auto& v=*state;const int face=faces[slot];
    auto w=sampleResidentGasWallFace(program,count,position,slot);
    Real factor=progress?progress[v.faceOwner[face]]:1;
    if(v.gasBoundaryLayer.enabled&&v.gasSpecies.mechanism.reactionCount==0){
        const auto& model=v.gasBoundaryLayerModel;const int wallSlot=v.gasBoundaryLayer.faceSlot[face];
        if(wallSlot>=0&&model.matchingOffsets){
            for(int k=0;k<Ns;++k)if(w.speciesRate[k]+w.poreRate[k]<0){
                Real donor=0;
                for(int j=model.matchingOffsets[wallSlot];j<model.matchingOffsets[wallSlot+1];++j)
                    donor+=model.matchingWeights[j]*v.gasSpecies.rho[k*v.nCells+model.matchingCells[j]];
                if(donor==0)factor=0;
            }
        }
    }
    if(factor!=1){
        for(int k=0;k<Ns;++k){w.speciesRate[k]*=factor;
            w.poreRate[k]=w.poreSweepRate[k]+factor*(w.poreRate[k]-w.poreSweepRate[k]);}
        for(int k=0;k<Nc;++k)w.condensedRate[k]*=factor;
    }
    bounded[slot]=w;
    if(v.gasBoundaryLayer.enabled){
        const int wallSlot=v.gasBoundaryLayer.faceSlot[face];if(wallSlot<0)return;
        auto& in=v.gasBoundaryLayerModel.input[wallSlot];
        bool dirty=in.temperature!=w.temperature;
        in.temperature=w.temperature;const Vec3 n={in.normal[0],in.normal[1],in.normal[2]};
        const auto velocity=w.velocity-n*dot(w.velocity,n)+n*w.normalVelocity;
        dirty=dirty||in.velocity[0]!=velocity.x||in.velocity[1]!=velocity.y||in.velocity[2]!=velocity.z;
        in.velocity[0]=velocity.x;in.velocity[1]=velocity.y;in.velocity[2]=velocity.z;
        const Real ratio=geometry[slot].area/v.magSf[face];
        for(int k=0;k<Ns;++k){const Real flux=(w.speciesRate[k]+w.poreRate[k])*ratio;
            dirty=dirty||in.massFlux[k]!=flux;in.massFlux[k]=flux;}
        if(changed)changed[wallSlot]=dirty;
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
    if(firstStage&&v.particleWall.depositedEnergy)
        record.particleHeatAtBegin=v.particleWall.depositedEnergy[face]+v.particleWall.reflectedEnergy[face];
    record.acceptedWall=w;
    record.sample=readCoupledSurfaceSample(v,face,p.singleGasSpecies);
    GasWallInput input;
    input.bulk=record.sample.primitive;input.gradient=record.sample.gradient;
    input.temperature=w.temperature;input.gasDistance=geometry.gasDistance;input.area=geometry.area;
    input.gasArea=v.magSf[face];input.dt=dt;input.normal=geometry.normal;input.velocity=w.velocity;
    if(v.particleWall.representedContactArea&&!ugkwp::wallHeatExposedFraction(input.gasArea,
        v.particleWall.representedContactArea[face],v.particleWall.contactAreaScale[face],input.exposedGasHeatFraction)){
        record.failed=1;return;}

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
    record.raw=(a+c)*(-1/dt);
    // The packet remains a physical transfer. Intrinsic gas densities receive
    // that transfer divided by the accepted gas phase fraction exactly once.
    const Real phase=v.gasVolumeFraction?v.gasVolumeFraction[v.faceOwner[face]]:1;
    const auto flux=record.raw*(1/phase);
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
    const Real phase=state->gasVolumeFraction?state->gasVolumeFraction[state->faceOwner[faces[slot]]]:1;
    const Real scale=denominator==0?1:numerator*phase/denominator;
    auto primary=scaledIntervalPacket(record.rates.primary,(ledgerDt/stageDt)*scale);primary.consumerMask=ConsumeGas;
    auto pore=scaledIntervalPacket(record.rates.pore,(ledgerDt/stageDt)*scale);pore.consumerMask=ConsumeGas;
    addIntervalPacket(record.primary,primary);addIntervalPacket(record.pore,pore);
    // Pair instantaneous accepted rates with this stage's matching state; the
    // RK-integrated packets remain the sole material inventory transfers.
    for(int s=0;s<Ns;++s){record.acceptedWall.speciesRate[s]*=scale;
        record.acceptedWall.poreRate[s]*=scale;record.acceptedWall.poreSweepRate[s]*=scale;}
    for(int c=0;c<Nc;++c)record.acceptedWall.condensedRate[c]*=scale;
    const auto& in=supplied[slot];const auto wall=sampleResidentGasWallFace(wallProgram,count,position,slot);
    record.radiationEnergy+=ledgerDt*wall.radiationFlux*in.area;
    record.radiationReceiver=wall.primaryKind==ExchangeKind::GasFilm?ConsumeFilm:ConsumeSolid;
}
}
#endif
