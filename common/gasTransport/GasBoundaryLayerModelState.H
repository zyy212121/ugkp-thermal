#pragma once
#include "GasBoundaryLayerEvaluation.H"
#include "GasGeometryValidation.H"
#include "GasBoundaryLayerWorkspace.H"
#include "../gasWall/WallModel.H"

namespace ugkwp
{
// Borrowed model input, geometry stencil and scratch. No independent inventory.
// Geometry arrays are rebuilt only when the mesh geometry version changes.
template<class Real,int Ns,int MaxNodes=48> struct GasBoundaryLayerModelState
{
    gaswall::WallModelConfig<Real> config;
    bool useSuppliedWallState=false; // material adapters own Tw, absolute velocity and total Js
    gaswall::WallInput<Real,Ns>* input=nullptr;
    gaswall::WallOutput<Real,Ns>* output=nullptr;
    void* workspace=nullptr;
    int workspaceCapacity=MaxNodes; // allocated capacity bucket: 24, 48 or 128
    gaswall::WallStatus* status=nullptr;
    int workspaceCount=0; // bounded pool, never one large scratch block per wall by default
    const int* faces=nullptr;
    const int* matchingOffsets=nullptr;
    const int* matchingCells=nullptr;
    const Real* matchingWeights=nullptr;
};
template<class State,class=void> struct GasBoundaryLayerModelCapability:std::false_type{};
template<class State> struct GasBoundaryLayerModelCapability<State,
    std::void_t<decltype(std::declval<State>().gasBoundaryLayerModel)>>:std::true_type{};

// Keep the three large capacity specializations out of the sparse kernel's
// inlining graph. Host callers retain ordinary inline semantics.
#if defined(__CUDA_ARCH__)
#define UGKWP_WALL_WORKSPACE_CALL __noinline__
#else
#define UGKWP_WALL_WORKSPACE_CALL inline
#endif
template<class Real,int Ns,int Capacity>
UGKWP_GAS_HD UGKWP_WALL_WORKSPACE_CALL bool evaluateGasBoundaryLayerWorkspace(
    const gaswall::WallInput<Real,Ns>& input,const gaswall::WallModelConfig<Real>& config,
    gaswall::WallWorkspace<Real,Ns,Capacity>& workspace,gaswall::WallOutput<Real,Ns>& output,
    gaswall::WallStatus& status)
{return gaswall::evaluateWallModel(input,config,workspace,output,status);}
#undef UGKWP_WALL_WORKSPACE_CALL

template<class State> UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerSlot(State& state,int slot,int workspaceSlot=0)
{
    using Real=typename std::remove_pointer<decltype(state.rho)>::type;
    constexpr int Ns=GasStateTraits<State>::speciesCount;
    auto& wall=state.gasBoundaryLayer;
    auto& model=state.gasBoundaryLayerModel;
    if(!wall.enabled || slot<0 || slot>=wall.count || !wall.exchange || !wall.status)return false;
    wall.exchange[slot].ready=false;
    wall.status[slot]=int(GasTransportCode::InvalidStorage);
    if(!model.faces)return false;
    const int face=model.faces[slot];
    auto fail=[&](GasTransportCode code)
    {
        wall.status[slot]=int(code);
        gasRecordFaceFailure(state,face,code);
        return false;
    };
    const bool analytic=model.config.model==gaswall::BoundaryLayerModel::ConstantTransport;
    if((!analytic && (workspaceSlot<0 || workspaceSlot>=model.workspaceCount || !model.workspace))
        || !model.input || !model.output || !model.status || !model.matchingOffsets
        || !model.matchingCells || !model.matchingWeights || !wall.sst || !wall.speciesFlux)
        return fail(GasTransportCode::InvalidStorage);
    if(face<0 || face>=state.nFaces || state.faceOwner[face]<0 || state.faceOwner[face]>=state.nCells
        || state.faceNeighbour[face]>=0 || state.riemannBoundaryKind[face]!=2)
        return fail(GasTransportCode::InvalidGeometry);
    if(!mixtureGasActive(state) || (!model.useSuppliedWallState && !state.riemannBoundaryTFix[face]) || !state.gasSpecies.diffusivity)
        return fail(GasTransportCode::UnsupportedConfiguration);
    auto& in=model.input[slot];
    if(!model.useSuppliedWallState)
    {
    in.temperature=state.riemannBoundaryT[face];
    in.velocity[0]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUx[face]:Real(0);
    in.velocity[1]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUy[face]:Real(0);
    in.velocity[2]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUz[face]:Real(0);
    }
    in.matching={};in.pressure=0;
    const int begin=model.matchingOffsets[slot],end=model.matchingOffsets[slot+1];
    if(begin<0 || end<=begin || end-begin>32)return fail(GasTransportCode::InvalidGeometry);
    Real sumWeight=0;
    for(int item=begin;item<end;++item)
    {
        const int cell=model.matchingCells[item];
        const Real weight=model.matchingWeights[item];
        if(cell<0 || cell>=state.nCells || gasBoundaryLayerOwnerSlot(state,cell)>=0
            || !gasFinite(weight) || !(weight>0) || !gasFinite(state.rho[cell]) || !(state.rho[cell]>0)
            || gasCellFailure(state,cell)!=0)return fail(GasTransportCode::InvalidGeometry);
        sumWeight+=weight;
        in.pressure+=weight*state.p[cell];
        in.matching.temperature+=weight*state.Tgas[cell];
        in.matching.velocity[0]+=weight*state.Ux[cell];
        in.matching.velocity[1]+=weight*state.Uy[cell];
        in.matching.velocity[2]+=weight*state.Uz[cell];
        if(model.config.enableSst)
        {
            in.matching.k+=weight*state.k[cell];
            in.matching.omega+=weight*state.omega[cell];
        }
        for(int species=0;species<Ns;++species)
            in.matching.massFraction[species]+=weight*state.gasSpecies.rho[species*state.nCells+cell]/state.rho[cell];
    }
    if(gaswall::detail::abs(sumWeight-Real(1))>Real(128)*std::numeric_limits<Real>::epsilon())
        return fail(GasTransportCode::InvalidGeometry);
    in.model.thermo=state.gasSpecies.thermo;
    in.model.mechanism=state.gasSpecies.mechanism;
    in.model.mode=state.gasSpecies.mode;
    in.model.viscosity=state.gasMu;
    Real matchingCp=0;
    for(int species=0;species<Ns;++species)matchingCp+=in.matching.massFraction[species]
        *speciesCp(species,in.matching.temperature,in.model.thermo);
    in.model.conductivity=gasHasDirectConductivity(state)?state.gasThermalConductivity
        :state.gasMu*matchingCp/state.gasPrClamped;
    in.model.molecularPrandtl=gasHasDirectConductivity(state)?Real(0):state.gasPrClamped;
    in.model.sstCoefficients=state.sstCoefficients;
    for(int species=0;species<Ns;++species)in.model.diffusivity[species]=state.gasSpecies.diffusivity[species];
    const int owner=state.faceOwner[face];
    Real volume=state.V[owner],normalSweepRate=0;
    if constexpr(GasGeometryCapability<State>::value)
        if(gasMovingGeometry(state))
        {
            GasFaceFrame<Real> frame;
            if(!gasGeometryFaceFrame(state,face,state.gasGeometry.interval,frame) || !state.gasGeometry.oldVolume)
                return fail(GasTransportCode::InvalidGeometry);
            volume=state.gasGeometry.oldVolume[owner];
            normalSweepRate=frame.wx*in.normal[0]+frame.wy*in.normal[1]+frame.wz*in.normal[2];
        }
    if(!gasFinite(volume) || !(volume>0) || gaswall::detail::abs(volume-in.quadrature.volume)
        >Real(256)*std::numeric_limits<Real>::epsilon()*volume)return fail(GasTransportCode::InvalidGeometry);
    bool evaluated=false;
    if(analytic)evaluated=gaswall::evaluateConstantTransportWall(in,model.config,model.output[slot],model.status[slot]);
    else
    {
#define WALL_EVALUATE_CAPACITY(capacity) case capacity: evaluated=evaluateGasBoundaryLayerWorkspace(in,model.config,static_cast<gaswall::WallWorkspace<Real,Ns,capacity>*>(model.workspace)[workspaceSlot],model.output[slot],model.status[slot]);break
        switch(model.workspaceCapacity)
        {
            WALL_EVALUATE_CAPACITY(24);
            WALL_EVALUATE_CAPACITY(48);
            WALL_EVALUATE_CAPACITY(128);
            default:return fail(GasTransportCode::InvalidStorage);
        }
#undef WALL_EVALUATE_CAPACITY
    }
    if(!evaluated)
        return fail(model.status[slot].code==gaswall::WallCode::Unsupported
            ?GasTransportCode::UnsupportedConfiguration:GasTransportCode::BoundaryLayerFailure);
    if(!publishGasBoundaryLayerOutput(state,slot,in,model.output[slot],state.magSf[face],normalSweepRate))
        return fail(GasTransportCode::NonFiniteState);
    return true;
}
}
