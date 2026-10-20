#pragma once
#include "GasBoundaryLayerEvaluation.H"
#include "GasWallMatching.H"
#include "GasGeometryValidation.H"
#include "GasBoundaryLayerWorkspace.H"
#include "../gasWall/WallModel.H"
#if defined(__CUDACC__)
#include "../gasWall/WallCudaTeam.cuh"
#endif

namespace ugkwp
{
// Borrowed model input, geometry stencil and scratch. No independent inventory.
// Geometry arrays are rebuilt only when the mesh geometry version changes.
template<class Real,int Ns,int MaxNodes=48> struct GasBoundaryLayerModelState
{
    gaswall::WallModelConfig<Real> config;
    bool volumeSstClosure=false; // gas FV adapter; material/interface adapters retain their own closure
    bool useSuppliedWallState=false; // material adapters own Tw, absolute velocity and total Js
    gaswall::WallInput<Real,Ns>* input=nullptr;
    gaswall::WallOutput<Real,Ns>* output=nullptr;
    void* workspace=nullptr;
    int workspaceCapacity=MaxNodes; // allocated capacity bucket: 24, 48 or 128
    gaswall::WallStatus* status=nullptr;
    Real* profile=nullptr; // enabled-at-startup rows: y,U[3],T,k,omega,Y[Ns]
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
// Type dispatch keeps scalar and borrowed-precision entrypoints independent of
// the cooperative FP64 implementation without adding a C++17-only branch.
template<class Real,int Ns,int Capacity,class Team>
UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerModelWithTeam(
    const gaswall::WallInput<Real,Ns>& input,const gaswall::WallModelConfig<Real>& config,
    gaswall::WallWorkspace<Real,Ns,Capacity>& workspace,gaswall::WallOutput<Real,Ns>& output,
    gaswall::WallStatus& status,const Team& team,std::true_type)
{
    bool evaluated=false;
    if(team.leader())evaluated=gaswall::evaluateWallModel(input,config,workspace,output,status);
    return team.broadcast(evaluated);
}
template<class Real,int Ns,int Capacity,class Team>
UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerModelWithTeam(
    const gaswall::WallInput<Real,Ns>& input,const gaswall::WallModelConfig<Real>& config,
    gaswall::WallWorkspace<Real,Ns,Capacity>& workspace,gaswall::WallOutput<Real,Ns>& output,
    gaswall::WallStatus& status,const Team& team,std::false_type)
{
    if(config.model==gaswall::BoundaryLayerModel::ReactingSst)
        return gaswall::evaluateReactingWallLayerImpl(input,config,workspace,output,status,team);
    // Preserve the central router for analytic, unsupported and future models.
    return evaluateGasBoundaryLayerModelWithTeam(input,config,workspace,output,status,team,std::true_type{});
}
template<bool Capture=false,class Real,int Ns,int Capacity,class Team>
UGKWP_GAS_HD UGKWP_WALL_WORKSPACE_CALL bool evaluateGasBoundaryLayerWorkspace(
    const gaswall::WallInput<Real,Ns>& input,const gaswall::WallModelConfig<Real>& config,
    gaswall::WallWorkspace<Real,Ns,Capacity>& workspace,gaswall::WallOutput<Real,Ns>& output,
    gaswall::WallStatus& status,Real* profile,const Team& team)
{
    using ScalarExecution=std::integral_constant<bool,
        std::is_same<Team,gaswall::detail::SerialWallTeam>::value
        || !std::is_same<Real,gaswall::WallReal>::value>;
    const bool evaluated=evaluateGasBoundaryLayerModelWithTeam(input,config,workspace,output,status,team,ScalarExecution{});
    if constexpr(Capture)
    {
        if(evaluated && team.leader())
        {
            gaswall::detail::LayerContext<Real,Ns> context;
            const bool available=gaswall::detail::makeContext(input,config,context);
            for(int node=0;node<config.nodes;++node)
            {
                gaswall::WallProfileNode<Real,Ns> p;
                const bool decoded=available && gaswall::reactingWallProfileNode(input,config,workspace,node,context,p);
                Real* row=profile+node*(7+Ns);
                row[0]=p.y;for(int d=0;d<3;++d)row[1+d]=p.velocity[d];
                row[4]=p.temperature;row[5]=p.k;row[6]=p.omega;
                for(int species=0;species<Ns;++species)row[7+species]=p.massFraction[species];
                // Diagnostic decoding cannot veto a physical solve. Missing values
                // remain explicitly unavailable instead of becoming invented zeros.
                if(!decoded)for(int col=0;col<7+Ns;++col)row[col]=std::numeric_limits<Real>::quiet_NaN();
            }
        }
    }
    team.sync();
    return evaluated;
}
// Preserve the original callable six-argument entrypoint, including function pointers.
template<bool Capture=false,class Real,int Ns,int Capacity>
UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerWorkspace(
    const gaswall::WallInput<Real,Ns>& input,const gaswall::WallModelConfig<Real>& config,
    gaswall::WallWorkspace<Real,Ns,Capacity>& workspace,gaswall::WallOutput<Real,Ns>& output,
    gaswall::WallStatus& status,Real* profile=nullptr)
{
    return evaluateGasBoundaryLayerWorkspace<Capture>(input,config,workspace,output,status,profile,
        gaswall::detail::SerialWallTeam{});
}
#undef UGKWP_WALL_WORKSPACE_CALL

template<class State,class Real> UGKWP_GAS_HD inline bool prepareGasBoundaryLayerInput(State& state,int slot,int workspaceSlot,Real& normalSweepRate)
{
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
    if(face<0 || face>=state.nFaces)return fail(GasTransportCode::InvalidStorage);
    const bool analytic=model.config.model==gaswall::BoundaryLayerModel::ConstantTransport;
    if((!analytic && (workspaceSlot<0 || workspaceSlot>=model.workspaceCount || !model.workspace))
        || !model.input || !model.output || !model.status || !model.matchingOffsets
        || !model.matchingCells || !model.matchingWeights || !wall.sst || !wall.speciesFlux
        || !state.rho || !state.rhoE || !state.rhoUx || !state.rhoUy || !state.rhoUz || !state.Tgas
        || !state.gasSpecies.rho || !state.magSf
        || (model.config.enableSst&&(!state.rhoK||!state.rhoOmega)))
        return fail(GasTransportCode::InvalidStorage);
    if(!state.gasSpecies.diffusivity || (!model.useSuppliedWallState &&
        (!state.riemannBoundaryT || !state.riemannBoundaryUFix
         || (state.riemannBoundaryUFix[face] && (!state.riemannBoundaryUx
             || !state.riemannBoundaryUy || !state.riemannBoundaryUz)))))
        return fail(GasTransportCode::InvalidStorage);
    auto& in=model.input[slot];
    in.diagnostics=state.diagnostics;in.diagnosticEntity=face;
    // The BVP replaces only the actual owner-volume normal SST operator.
    in.volumeSstClosure=model.volumeSstClosure&&model.config.enableSst&&model.config.model==gaswall::BoundaryLayerModel::ReactingSst;
    in.integrateOwnerKSource=in.volumeSstClosure;
    if(in.volumeSstClosure&&(!in.quadrature.distance||!in.quadrature.volumeWeight||in.quadrature.count<=0||in.quadrature.volume<=0))
        return fail(GasTransportCode::InvalidStorage);
    for(int i=0;i<3;++i)for(int j=0;j<2;++j)in.sstTangentialGradient[i][j]=0;
    if(in.volumeSstClosure){
        if(!state.faceOwner)return fail(GasTransportCode::InvalidStorage);
        const int owner=state.faceOwner[face];
        if(owner<0||owner>=state.nCells||!state.gradUxX||!state.gradUxY||!state.gradUxZ
            ||!state.gradUyX||!state.gradUyY||!state.gradUyZ
            ||!state.gradUzX||!state.gradUzY||!state.gradUzZ)
            return fail(GasTransportCode::InvalidStorage);
        Real tangent[2][3];
        if(!gaswall::detail::wallTangentBasis(in.normal,tangent))return fail(GasTransportCode::InvalidGeometry);
        const Real gradient[3][3]={{state.gradUxX[owner],state.gradUxY[owner],state.gradUxZ[owner]},
            {state.gradUyX[owner],state.gradUyY[owner],state.gradUyZ[owner]},
            {state.gradUzX[owner],state.gradUzY[owner],state.gradUzZ[owner]}};
        for(int i=0;i<3;++i)for(int j=0;j<2;++j){
            const Real* row=i<2?tangent[i]:in.normal;
            Real value=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)value+=row[a]*gradient[a][b]*tangent[j][b];
            if(!gasFinite(value))return fail(GasTransportCode::NonFiniteState);
            in.sstTangentialGradient[i][j]=value;
        }
    }
    if(!model.useSuppliedWallState)
    {
    in.temperature=state.riemannBoundaryT[face];
    in.velocity[0]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUx[face]:Real(0);
    in.velocity[1]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUy[face]:Real(0);
    in.velocity[2]=state.riemannBoundaryUFix[face]?state.riemannBoundaryUz[face]:Real(0);
    }
    in.matching={};in.pressure=0;
    const int begin=model.matchingOffsets[slot],end=model.matchingOffsets[slot+1];
    if(begin<end && begin<0)return fail(GasTransportCode::InvalidStorage);
    for(int item=begin;item<end;++item)
        if(model.matchingCells[item]<0||model.matchingCells[item]>=state.nCells)
            return fail(GasTransportCode::InvalidStorage);
    if(!recoverWallMatching(state,model.matchingCells+begin,model.matchingWeights+begin,
        end-begin,model.config.enableSst,in.matching,in.pressure))
        return fail(GasTransportCode::InvalidThermodynamics);
    in.model.thermo=state.gasSpecies.thermo;
    in.model.thermo.diagnostics=state.diagnostics;
    in.model.mechanism=state.gasSpecies.mechanism;
    in.model.mode=state.gasSpecies.mode;
    in.model.viscosity=state.gasMu;
    if(!in.model.thermo.species||!in.model.thermo.coefficients)return fail(GasTransportCode::InvalidStorage);
    Real matchingCp=0;
    for(int species=0;species<Ns;++species){bool available=false;
        const Real cp=gaswall::detail::wallProperty<false>(species,in.matching.temperature,in.model.thermo,&available);
        if(!available)return fail(GasTransportCode::InvalidStorage);
        matchingCp+=in.matching.massFraction[species]*cp;}
    in.model.conductivity=gasHasDirectConductivity(state)?state.gasThermalConductivity
        :state.gasMu*matchingCp/state.gasPrClamped;
    in.model.molecularPrandtl=gasHasDirectConductivity(state)?Real(0):state.gasPrClamped;
    in.model.sstCoefficients=state.sstCoefficients;
    for(int species=0;species<Ns;++species)in.model.diffusivity[species]=state.gasSpecies.diffusivity[species];
    normalSweepRate=0;
    if constexpr(GasGeometryCapability<State>::value)
        if(gasMovingGeometry(state))
        {
            GasFaceFrame<Real> frame;
            bool frameAvailable=false;
            gasGeometryFaceFrame(state,face,state.gasGeometry.interval,frame,&frameAvailable);
            if(!frameAvailable)return fail(GasTransportCode::InvalidStorage);
            normalSweepRate=frame.wx*in.normal[0]+frame.wy*in.normal[1]+frame.wz*in.normal[2];
        }
    return true;
}

template<bool Capture=false,int Capacity=0,class State,class Team>
UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerSlot(State& state,int slot,int workspaceSlot,const Team& team)
{
    using Real=typename std::remove_pointer<decltype(state.rho)>::type;
    constexpr int Ns=GasStateTraits<State>::speciesCount;
    Real normalSweepRate=0;bool prepared=false;
    if(team.leader())prepared=prepareGasBoundaryLayerInput(state,slot,workspaceSlot,normalSweepRate);
    if(!team.broadcast(prepared))return false;
    auto& wall=state.gasBoundaryLayer;auto& model=state.gasBoundaryLayerModel;
    const int face=model.faces[slot];
    auto fail=[&](GasTransportCode code)
    {
        if(team.leader()){wall.status[slot]=int(code);gasRecordFaceFailure(state,face,code);}
        return false;
    };
    auto& in=model.input[slot];
    const bool analytic=model.config.model==gaswall::BoundaryLayerModel::ConstantTransport;
    bool evaluated=false;
    if(analytic)
    {
        if(team.leader())evaluated=gaswall::evaluateConstantTransportWall(in,model.config,model.output[slot],model.status[slot]);
        evaluated=team.broadcast(evaluated);
    }
    else if constexpr(Capacity>0)
    {
        evaluated=evaluateGasBoundaryLayerWorkspace<Capture>(in,model.config,
            static_cast<gaswall::WallWorkspace<Real,Ns,Capacity>*>(model.workspace)[workspaceSlot],
            model.output[slot],model.status[slot],
            Capture?model.profile+std::size_t(slot)*model.config.nodes*(7+Ns):nullptr,team);
    }
    else
    {
#define WALL_EVALUATE_CAPACITY(capacity) case capacity: evaluated=evaluateGasBoundaryLayerWorkspace<Capture>(in,model.config,static_cast<gaswall::WallWorkspace<Real,Ns,capacity>*>(model.workspace)[workspaceSlot],model.output[slot],model.status[slot],Capture?model.profile+std::size_t(slot)*model.config.nodes*(7+Ns):nullptr,team);break
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
        return fail(GasTransportCode::InvalidStorage);
    bool published=false;
    if(team.leader())published=publishGasBoundaryLayerOutput(state,slot,in,model.output[slot],state.magSf[face],normalSweepRate);
    if(!team.broadcast(published))return fail(GasTransportCode::InvalidStorage);
    return true;
}

// Keep the original scalar signature and defaults; cooperative calls are explicit.
template<bool Capture=false,class State>
UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerSlot(State& state,int slot,int workspaceSlot=0)
{
    return evaluateGasBoundaryLayerSlot<Capture>(state,slot,workspaceSlot,gaswall::detail::SerialWallTeam{});
}
}
