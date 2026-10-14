#include "gpu/Backend.H"
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/CourantReductionStorage.cuh"
#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "gpu/SingleGasContract.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalAudit.H"
#include "mesh/TrajectorySurface.H"
#include "../../../common/GpuGasOperatorDependencies.cuh"
#include "../../../common/GpuPrecisionTypes.H"
#include "../../../common/gasNumerics/CharacteristicMuscl.cuh"
#include "../../../common/gasNumerics/OpenFoamLimitedLinear.cuh"
#include "../../../common/gasNumerics/OpenFoamViscousFlux.cuh"
#include "../../../common/gasNumerics/OpenFoamWallFunctions.cuh"
#include "../../../common/gasNumerics/RiemannBoundaryState.cuh"
#include "../../../common/gasNumerics/RiemannGasFlux.cuh"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include "../../../common/gasNumerics/GpuLesAlgebra.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <memory>
#include <algorithm>
#include <vector>
// Diagnostic-only host observer for the actual native CHMT CUDA backend.
// Compile and run the real solver; this file contains no replacement kernels.
#include "coupling/IntervalContracts.H"
#include <cstdio>
namespace chmt { namespace fp64_clock_trace {
inline void value(const CouplingInterval& w, std::uint64_t micro,
                  const char* event, std::size_t index, const char* field, double x) {
    std::fprintf(stderr, "F003,%llu,%llu,%llu,%s,%zu,%s,%a\n",
        static_cast<unsigned long long>(w.identity.sequence),
        static_cast<unsigned long long>(w.identity.epoch),
        static_cast<unsigned long long>(micro),event,index,field,x);
}
inline void begin(const CouplingInterval& w,std::uint64_t micro,double stamp,double dt) {
    value(w,micro,"begin",0,"window_begin",w.begin);
    value(w,micro,"begin",0,"window_end",w.end);
    value(w,micro,"begin",0,"micro_time",stamp);
    value(w,micro,"begin",0,"micro_dt",dt);
}
inline void wall(const CouplingInterval& w,std::uint64_t micro,double argumentStamp,
                 double sampledStamp,double dt,const WallKnot& wall) {
    value(w,micro,"stage",0,"argument_stamp",argumentStamp);
    value(w,micro,"stage",0,"sampled_stamp",sampledStamp);
    value(w,micro,"stage",0,"packet_dt",dt);
    for(std::size_t i=0;i<wall.faces.size();++i) {
        const auto& q=wall.faces[i];
        value(w,micro,"wall",i,"temperature",q.temperature);
        value(w,micro,"wall",i,"normal_velocity",q.normalVelocity);
        value(w,micro,"wall",i,"solid_normal_velocity",q.solidNormalVelocity);
        value(w,micro,"wall",i,"velocity_x",q.velocity.x);
        value(w,micro,"wall",i,"velocity_y",q.velocity.y);
        value(w,micro,"wall",i,"velocity_z",q.velocity.z);
        value(w,micro,"wall",i,"radiation_flux",q.radiationFlux);
        for(int s=0;s<Ns;++s) {
            const std::size_t item=i*Ns+s;
            value(w,micro,"wall_species",item,"species_rate",q.speciesRate[s]);
            value(w,micro,"wall_species",item,"pore_rate",q.poreRate[s]);
            value(w,micro,"wall_species",item,"pore_sweep_rate",q.poreSweepRate[s]);
        }
    }
}
inline void accepted(const CouplingInterval& w,const GasIntervalRecord& r,const HostState& state) {
    const auto micro=r.microSequence;
    value(w,micro,"accepted",0,"begin",r.begin);
    value(w,micro,"accepted",0,"end",r.end);
    value(w,micro,"accepted",0,"gas_trace_time",r.gasTraceTime);
    for(std::size_t c=0;c<state.gas.size();++c) {
        const auto& q=state.gas[c];
        value(w,micro,"gas",c,"mass",q.mass);
        value(w,micro,"gas",c,"energy",q.energy);
        value(w,micro,"gas",c,"momentum_x",q.momentum.x);
        value(w,micro,"gas",c,"momentum_y",q.momentum.y);
        value(w,micro,"gas",c,"momentum_z",q.momentum.z);
        for(int s=0;s<Ns;++s)value(w,micro,"gas_species",c*Ns+s,"mass",q.species[s]);
    }
    for(std::size_t i=0;i<r.packets.size();++i) {
        const auto& p=r.packets[i];
        value(w,micro,"packet",i,"face",double(p.face));
        value(w,micro,"packet",i,"stage",double(p.stage));
        value(w,micro,"packet",i,"kind",double(int(p.kind)));
        value(w,micro,"packet",i,"mass",p.mass);
        value(w,micro,"packet",i,"energy",p.energy);
        value(w,micro,"packet",i,"conductive",p.conductive);
        value(w,micro,"packet",i,"advective",p.advective);
        value(w,micro,"packet",i,"pressure_work",p.pressureWork);
        value(w,micro,"packet",i,"viscous_work",p.viscousWork);
        value(w,micro,"packet",i,"momentum_x",p.momentum.x);
        value(w,micro,"packet",i,"momentum_y",p.momentum.y);
        value(w,micro,"packet",i,"momentum_z",p.momentum.z);
    }
    std::fflush(stderr);
}
} }

namespace chmt {
struct Backend {
    ModelConfig model;GasExecutionOptions options;HostState accepted,trial,microBase;
    HostMesh endpointGas,endpointSolid;SurfaceMesh endpointSurface;HostStageGeometry gasStage,solidStage;
    SharedGasDeviceStorage storage;WallProgram program;IntervalHistory history;MaterialDonorReserve reserve;
    CoupledCourantStorage courantReductionStorage;
    bool poisoned=false;std::string error;
    GasIntervalRecord record;std::vector<GasPrimitive> stageBulk;std::vector<GasGradient> stageGradient;std::vector<GasWallResult> faceRates;std::vector<GasQ> rawInterface;
    Real microDt=0,microTime=0,stageTime=0,stagePacketDt=0;
    WindowTime elapsed{};Real nextGasSeed=0;bool microGeometryPrepared=false;int gasStageIndex=0;Budget trialBudget{};
    std::vector<ugkwp::ChemistryAudit<Real,Ns>> firstChemistry,secondChemistry;
    int nCells=0,nFaces=0,fixedCellBlockThreads=128,fixedFaceBlockThreads=128;
    int hostTurbulenceModel=0,hostGasFluxScheme=1,hostGasTimeIntegrator=1,hasPeriodicFaces=0;
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
    ugkwp::GasGeometryState<Real> gasGeometry;
    ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;
    std::vector<ugkwp::gaswall::WallOutput<Real,Ns>> preparedWallLayers;
    std::vector<GasWallMatchingSample> preparedMatching;
    Real preparedWallTime=0,preparedWallDt=0;bool preparedWallReady=false,terminalWallFailure=false;
    SharedGasDeviceView* deviceState=nullptr;cudaStream_t gasCaptureStream=nullptr;
};
inline void bindSharedGasStorage(Backend& b){
    const auto& view=b.storage.hostView();
    b.preparedWallReady=false;
    b.gasSpecies=view.gasSpecies;b.gasBoundaryLayer=view.gasBoundaryLayer;b.gasGeometry=view.gasGeometry;
    b.deviceState=b.storage.deviceView();b.nCells=view.nCells;b.nFaces=view.nFaces;
}
inline WindowTime coupledStageTime(const Backend& b){return b.elapsed+(b.gasStageIndex==0?0:b.gasStageIndex==1?b.microDt:.5*b.microDt);}
bool restoreGasSnapshot(Backend&,const HostState&,std::string&);
int prepareCoupledGeometry(Backend*);
int finishCoupledGeometry(Backend*);
int prepareCoupledBoundaryLayer(Backend*,Real,Real);
int applyCoupledFaces(Backend*,Real,Real);
int accountCoupledFaces(Backend*,Real);
int acceptSstAudit(Backend*);
namespace transport {
using std::isfinite;
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) static_cast<double>(x)
#define GPU_OPERATOR_TINY(x) x
constexpr double OfSmall=2.22044604925031308085e-16,OfVSmall=2.22507385850720138309e-308,OfGreat=1/OfSmall;
struct GasPrimDevice {double rho,ux,uy,uz,p,T;};
#include "../../../common/operators/clampMin.cuh"
#include "../../../common/operators/clampRange.cuh"
#include "../../../common/operators/linearScheduledValueDevice.cuh"
#include "../../../common/operators/makeGasPrimDevice.cuh"
#include "../../../common/operators/riemannFacePrimitiveForGradient.cuh"
#include "../../../common/operators/computeGasPrimitiveGradientsKernel.cuh"
#include "../../../common/operators/computeSstGradientsKernel.cuh"
#include "../../../common/operators/sstVelocityInvariants.cuh"
#include "../../../common/operators/computeGasHllcAdcSensorKernel.cuh"
#include "../../../common/operators/updateBarthLimiter.cuh"
#include "../../../common/operators/computeGasGradientLimiterKernel.cuh"
#include "../../../common/operators/computeGasEddyViscosityKernel.cuh"
#include "../../../common/operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "../../../common/operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "../../../common/operators/gasFaceSubgridTransportProperties.cuh"
#include "../../../common/operators/computeRiemannGasFaceFluxDevice.cuh"
#include "../../../common/operators/computeGasInternalFaceFluxKernel.cuh"
#include "../../../common/operators/computeGasFluxPositivityScaleKernel.cuh"
#include "../../../common/operators/computeSstFaceFluxKernel.cuh"
#include "../../../common/operators/evaluateGasBoundaryLayerKernel.cuh"
// Only a stale prepared packet needs this refresh. The normal stage already
// evaluated these exact face expressions before its material callback.
template<class State>
__global__ void refreshCoupledWallFluxKernel(State* state){
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=state->gasBoundaryLayer.count)return;
    const int face=state->gasBoundaryLayerModel.faces[slot];
    computeRiemannGasFaceFluxDevice<false>(*state,face,state->gasPhiRho[face],
        state->gasPhiRhoUx[face],state->gasPhiRhoUy[face],state->gasPhiRhoUz[face],state->gasPhiRhoE[face]);
}
} // transport
} // chmt
#include "../../../common/GpuGasHostPolicy.cuh"
namespace chmt { namespace transport {
thread_local std::string lastError;
inline void setLastError(const char* text,cudaError_t code){lastError=std::string(text)+": "+cudaGetErrorString(code);}
inline void setLastErrorText(const char* text){lastError=text;}
struct GasHostPolicy:GasHostWithWallEnergy<Real,Real,&accountCoupledFaces> {
    static int prepareBoundaryLayer(Backend* s,Real dt,Real time){return prepareCoupledBoundaryLayer(s,dt,time);}
    static int applyFaceSources(Backend* s,Real dt,Real time){return applyCoupledFaces(s,dt,time);}
};
#include "../../../common/GpuGasAdvance.cuh"
#include "../../../common/GpuGasCourant.cuh"
struct CoupledTrialPolicy {
    static int begin(Backend* b){
        fp64_clock_trace::begin(b->program.interval,b->history.records().size()+1,b->microTime,b->microDt);
        b->firstChemistry.clear();b->secondChemistry.clear();
        b->preparedWallReady=false;b->terminalWallFailure=false;ugkwp::invalidateGasBoundaryLayerPreparation(*b);
        if(!b->microGeometryPrepared)b->microBase=b->trial;b->gasStageIndex=0;b->trialBudget=b->trial.budget;b->record={};b->record.microSequence=b->history.records().size()+1;
        b->record.begin=b->microTime;b->record.actualDt=b->microDt;b->record.localBegin=b->elapsed;b->record.localEnd=b->elapsed+b->microDt;b->record.end=windowAbsolute(b->program.interval.begin,b->record.localEnd);b->record.gasGeometry=b->trial.gasMesh.geometryVersion;b->record.solidGeometry=b->trial.solidMesh.geometryVersion;
        if(!b->storage.clearTrialStatus())return 1;
        if(!b->microGeometryPrepared&&prepareCoupledGeometry(b)!=0)return 1;
        b->record.gasGeometry=b->endpointGas.geometryVersion;b->record.solidGeometry=b->endpointSolid.geometryVersion;
        b->microGeometryPrepared=false;
        return 0;
    }
    static const Real* stageVolumes(Backend* b,bool after){const auto& g=b->storage.hostView().gasGeometry;return g.enabled?(after?g.newVolume:g.oldVolume):b->storage.hostView().V;}
    static Real targetMaxCo(Backend* b){return b->model.physics.cfl;}
    static int captureChemistryAudit(Backend* b,bool after){
        auto& audit=after?b->secondChemistry:b->firstChemistry;
        if(b->model.physics.gasMode!=ugkwp::GasMode::MixtureChemistry){audit.clear();return 0;}
        return b->storage.read(b->storage.hostView().gasSpecies.chemistryAudit,b->nCells,audit)?0:1;
    }
    static int validateTimeStep(Backend*,Real){return 0;} // Direct CHMT path does not request native refinement.
    static int applySources(Backend* b,Real){return finishCoupledGeometry(b); }
    static int validate(Backend* b){
        if(!b->storage.checkStatus(b->error))return 1;
        return 0;
    }
    static int commit(Backend* b){
        if(!b->storage.downloadState(b->trial,b->error))return 1;
        if(b->model.physics.gasMode==ugkwp::GasMode::MixtureChemistry){
            b->trial.gasChemistryAudit.resize(b->nCells);
            for(const auto* half:{&b->firstChemistry,&b->secondChemistry}){
                if(b->nCells>0&&half->empty()){b->error="missing chemical half audit";return 1;}
                for(int c=0;c<b->nCells;++c){auto& dst=b->trial.gasChemistryAudit[c];const auto& src=(*half)[c];
                    for(int s=0;s<Ns;++s)dst.speciesMassChange[s]+=src.speciesMassChange[s];
                    dst.massResidual+=src.massResidual;dst.energyResidual+=src.energyResidual;dst.maximumElementResidual=maxValue(dst.maximumElementResidual,src.maximumElementResidual);
                    dst.maximumLocalError=maxValue(dst.maximumLocalError,src.maximumLocalError);dst.integratedTime+=src.integratedTime;
                    // Same-width counter arithmetic, with defined wrap instead of signed UB.
                    const auto add=[](int a,int b){const unsigned bits=unsigned(a)+unsigned(b);return bits<=unsigned(INT_MAX)?int(bits):-1-int(UINT_MAX-bits);};
                    dst.acceptedSteps=add(dst.acceptedSteps,src.acceptedSteps);dst.rejectedSteps=add(dst.rejectedSteps,src.rejectedSteps);dst.nonlinearIterations=add(dst.nonlinearIterations,src.nonlinearIterations);
                }
            }
        }
        if(acceptSstAudit(b)!=0)return 1;
        if(b->gasBoundaryLayer.enabled){b->trial.gasWallDiagnostics=b->preparedWallLayers;b->trial.gasWallDiagnosticTime=b->stageTime;}
        b->trial.time=b->record.end;b->trial.budget=b->trialBudget;return 0;
    }
    static void rollback(Backend* b){
        std::string rollbackError;
        if(!restoreGasSnapshot(*b,b->microBase,rollbackError)){
            b->error+="; rollback failed: "+rollbackError;
        }
    }
};
}}
namespace chmt {
namespace {
bool readBulk(Backend& b,std::vector<GasPrimitive>& primitive,std::vector<GasGradient>& gradient){
    const auto& v=b.storage.hostView();const std::size_t n=b.nCells;std::vector<Real> rho,T,p,ux,uy,uz,sound,Y;
    const bool mixture=v.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy;
    if(!b.storage.read(v.rho,n,rho)||!b.storage.read(v.Tgas,n,T)||!b.storage.read(v.p,n,p)||!b.storage.read(v.Ux,n,ux)||!b.storage.read(v.Uy,n,uy)||!b.storage.read(v.Uz,n,uz)||(mixture&&(!b.storage.read(v.gasSpecies.soundSpeed,n,sound)||!b.storage.read(v.gasSpecies.rho,Ns*n,Y))))return false;
    primitive.resize(n);gradient.resize(n);
    for(std::size_t c=0;c<n;++c){auto& w=primitive[c];w.rho=rho[c];w.temperature=T[c];w.pressure=p[c];w.velocity={ux[c],uy[c],uz[c]};w.soundSpeed=mixture?sound[c]:std::sqrt(v.gammaGas*p[c]/rho[c]);for(int s=0;s<Ns;++s)w.Y[s]=mixture?Y[s*n+c]/rho[c]:(s==b.model.physics.singleGasSpecies?1:0);}
    const Real* components[]={v.gradRhoX,v.gradRhoY,v.gradRhoZ,v.gradTX,v.gradTY,v.gradTZ,v.gradPx,v.gradPy,v.gradPz,v.gradUxX,v.gradUxY,v.gradUxZ,v.gradUyX,v.gradUyY,v.gradUyZ,v.gradUzX,v.gradUzY,v.gradUzZ};
    for(int j=0;j<18;++j){std::vector<Real> a;if(!b.storage.read(components[j],n,a))return false;for(std::size_t c=0;c<n;++c){Vec3* field=j<3?&gradient[c].rho:j<6?&gradient[c].temperature:j<9?&gradient[c].pressure:&gradient[c].velocity[(j-9)/3];if(j%3==0)field->x=a[c];else if(j%3==1)field->y=a[c];else field->z=a[c];}}
    const Real* species[]={v.gasSpecies.gradX,v.gasSpecies.gradY,v.gasSpecies.gradZ};
    for(int j=0;mixture&&j<3;++j){std::vector<Real>a;if(!b.storage.read(species[j],Ns*n,a))return false;for(int s=0;s<Ns;++s)for(std::size_t c=0;c<n;++c){auto& field=gradient[c].Y[s];if(j==0)field.x=a[s*n+c];else if(j==1)field.y=a[s*n+c];else field.z=a[s*n+c];}}
    return true;
}
bool fluxes(Backend& b,std::vector<GasQ>& values,bool writing){
    auto& v=b.storage.hostView();const int nf=b.nFaces;Real* field[]={v.gasPhiRho,v.gasPhiRhoUx,v.gasPhiRhoUy,v.gasPhiRhoUz,v.gasPhiRhoE};
    if(!writing)values.assign(nf,GasQ{});
    for(int j=0;j<5;++j){std::vector<Real>a(nf);if(writing){for(int f=0;f<nf;++f){const auto&q=values[f];a[f]=j==0?q.mass:j==1?q.momentum.x:j==2?q.momentum.y:j==3?q.momentum.z:q.energy;}if(!b.storage.write(field[j],a))return false;}
        else {if(!b.storage.read(field[j],nf,a))return false;for(int f=0;f<nf;++f){auto&q=values[f];if(j==0)q.mass=a[f];else if(j==1)q.momentum.x=a[f];else if(j==2)q.momentum.y=a[f];else if(j==3)q.momentum.z=a[f];else q.energy=a[f];}}}
    if(v.gasSpecies.mode==ugkwp::GasMode::SingleLegacy){
        if(!writing)for(auto& q:values)q.species[b.model.physics.singleGasSpecies]=q.mass;
        return true;
    }
    std::vector<Real>a(Ns*nf);if(writing){for(int s=0;s<Ns;++s)for(int f=0;f<nf;++f)a[s*nf+f]=values[f].species[s];return b.storage.write(v.gasSpecies.flux,a);}
    if(!b.storage.read(v.gasSpecies.flux,Ns*nf,a))return false;for(int s=0;s<Ns;++s)for(int f=0;f<nf;++f)values[f].species[s]=a[s*nf+f];return true;
}
void addGasBudget(Budget& budget,const ExchangePacket& packet){const auto delta=packetDelta(packet).gas;budget.exchangeMass[GasParticipant]+=delta.mass;budget.exchangeEnergy[GasParticipant]+=delta.energy;budget.exchangeMomentum[GasParticipant]+=delta.momentum;for(int s=0;s<Ns;++s)budget.exchangeSpecies[GasParticipant][s]+=delta.species[s];}
}
int prepareCoupledGeometry(Backend* b){
    const auto& base=b->microBase;const bool moving=b->model.physics.meshMotion.policy!=MeshMotionPolicy::Static;
    b->endpointGas=base.gasMesh;b->endpointSolid=base.solidMesh;b->endpointSurface=base.surface;
    std::vector<Real> gasSweep(base.gasMesh.owner.size(),0),solidSweep(base.solidMesh.owner.size(),0);
    if(moving){
        WallKnot endpoint;if(!sampleGasWallProgramLocal(b->program,b->elapsed+b->microDt,endpoint,b->error))return 1;
        const auto& tolerance=b->model.physics.tolerances;
        auto stage=[&](const HostMesh& old,const std::vector<Vec3>& points,HostMesh& next,std::vector<Real>& sweep){
            return makeStageGeometry(old,points,b->microDt,next,sweep,b->error,tolerance)
                ||detail::completeStageGeometry(old,points,b->microDt,next,sweep,b->error,tolerance);
        };
        if(!stage(base.gasMesh,endpoint.gasPoints,b->endpointGas,gasSweep)
            ||!stage(base.solidMesh,endpoint.solidPoints,b->endpointSolid,solidSweep))return 1;
        if(!rebuildTrajectorySurface(base,b->endpointGas,b->endpointSolid,b->microDt,b->endpointSurface,b->error,tolerance)
            &&!detail::completeTrajectorySurface(base.gasMesh,base.solidMesh,base.surface,b->endpointGas,b->endpointSolid,
                b->microDt,b->endpointSurface,b->error,tolerance))return 1;
    }
    auto trace=[&](const HostMesh& old,const HostMesh& end,const std::vector<Real>& sweep,HostStageGeometry& out){
        out.interval=b->microDt;out.geometryVersion=end.geometryVersion;out.topologyHash=end.topologyHash;
        out.oldVolume=old.volumes;out.newVolume=end.volumes;out.evaluationVolume=old.volumes;out.sweptVolume=sweep;
        out.areaVector=old.areaVectors;out.cellCentre=old.cellCentres;out.faceCentre=old.faceCentres;out.oldPoints=old.points;out.newPoints=end.points;
    };
    trace(base.gasMesh,b->endpointGas,gasSweep,b->gasStage);trace(base.solidMesh,b->endpointSolid,solidSweep,b->solidStage);
    b->record.gasGeometry=b->endpointGas.geometryVersion;b->record.solidGeometry=b->endpointSolid.geometryVersion;
    if(moving){if(!b->storage.bindStageGeometry(b->gasStage)){b->error=b->storage.error();return 1;}}
    else if(!b->storage.clearGeometry()){b->error=b->storage.error();return 1;}
    b->gasGeometry=b->storage.hostView().gasGeometry;return 0;
}
int finishCoupledGeometry(Backend* b){
    b->trial.gasMesh=b->endpointGas;b->trial.solidMesh=b->endpointSolid;b->trial.surface=b->endpointSurface;
    if(!b->storage.uploadGeometry(b->trial.gasMesh,false)||!b->storage.refreshView()){b->error=b->storage.error();return 1;}
    bindSharedGasStorage(*b);
    return 0;
}
int prepareCoupledBoundaryLayer(Backend* b,Real dt,Real stageTime){
    if(!b->gasBoundaryLayer.enabled)return 0;
    b->preparedWallReady=false;
    WallKnot wall;if(!sampleGasWallProgramLocal(b->program,coupledStageTime(*b),wall,b->error,false))return 1;
    if(!b->storage.prepareWallBoundary(wall,b->trial.surface,b->error))return 1;
    const auto& model=b->storage.hostView().gasBoundaryLayerModel;
    const int workers=model.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
        ?b->gasBoundaryLayer.count:model.workspaceCount;
    transport::evaluateGasBoundaryLayerKernel<<<(workers+31)/32,32,0,b->gasCaptureStream>>>(b->deviceState);
    const auto error=cudaGetLastError();if(error!=cudaSuccess){b->error=std::string("boundaryLayer CUDA launch: ")+cudaGetErrorString(error);return 1;}
    // Existing split material packets are host-owned. Only sparse wall outputs
    // and their matching snapshots cross back once per prepared gas stage.
    if(!b->storage.downloadWallOutputs(b->preparedWallLayers,b->preparedMatching)){b->terminalWallFailure=b->storage.wallModelFailed();b->error=b->storage.error();return 1;}
    b->preparedWallTime=stageTime;b->preparedWallDt=dt;b->preparedWallReady=true;return 0;
}
int applyCoupledFaces(Backend* b,Real dt,Real stageTime){
    if(b->gasBoundaryLayer.enabled && (!b->preparedWallReady
        || b->preparedWallTime!=stageTime || b->preparedWallDt!=dt)){
        if(prepareCoupledBoundaryLayer(b,dt,stageTime)!=0)return 1;
        const int count=b->gasBoundaryLayer.count;
        transport::refreshCoupledWallFluxKernel<<<(count+31)/32,32,0,b->gasCaptureStream>>>(b->deviceState);
        const auto code=cudaGetLastError();
        if(code!=cudaSuccess){b->error=std::string("refresh current wall face flux: ")+cudaGetErrorString(code);return 1;}
    }
    b->stageTime=windowAbsolute(b->program.interval.begin,coupledStageTime(*b));b->stagePacketDt=dt;WallKnot wall;if(!sampleGasWallProgramLocal(b->program,coupledStageTime(*b),wall,b->error,false))return 1;
    fp64_clock_trace::wall(b->program.interval,b->record.microSequence,stageTime,b->stageTime,dt,wall);
    std::vector<GasPrimitive> primitive;std::vector<GasGradient> gradient;std::vector<GasQ> flux;
    if(!readBulk(*b,primitive,gradient)||!fluxes(*b,flux,false)){b->error=b->storage.error();return 1;}
    const auto& surface=b->trial.surface;b->faceRates.resize(surface.area.size());b->rawInterface.resize(surface.area.size());
    for(std::size_t i=0;i<surface.area.size();++i){const int f=surface.gasFace[i],cell=b->trial.gasMesh.owner[f];const auto& w=wall.faces[i];GasWallInput input;
        input.bulk=primitive[cell];input.gradient=gradient[cell];input.temperature=w.temperature;input.gasDistance=surface.gasDistance[i];input.area=surface.area[i];input.gasArea=mag(b->trial.gasMesh.areaVectors[f]);input.dt=dt;input.normal=surface.normal[i];input.velocity=w.velocity;input.primaryKind=w.primaryKind;
        input.normalSpeed=w.normalVelocity;input.sweptVolume=-b->gasStage.sweptVolume[f];
        Real massFlux=0;for(int s=0;s<Ns;++s){input.speciesRate[s]=w.speciesRate[s];input.poreRate[s]=w.poreRate[s];input.poreSweepRate[s]=w.poreSweepRate[s];
            massFlux+=(w.speciesRate[s]+w.poreRate[s])*(input.area/input.gasArea);}
        if(b->gasBoundaryLayer.enabled){
            if(b->preparedWallLayers.empty()||b->preparedMatching.empty()){
                b->error="boundaryLayer prepared output backing missing";return 1;}
            input.normal=-b->trial.gasMesh.areaVectors[f]/input.gasArea;
            input.preparedLayer=&b->preparedWallLayers[i];input.wallContext.matchingPressure=b->preparedMatching[i].pressure;
            input.wallContext.matching=b->preparedMatching[i].state;
            input.preparedMatching=&b->preparedMatching[i].state;
            input.preparedMatchingPressure=&b->preparedMatching[i].pressure;
            // The common face operator has already reconstructed the local
            // mechanical pressure. Recover that exact pressure from its flux,
            // removing physical mass advection and normal viscous stress.
            // n_out=-input.normal and Fmom/A=-m*u_trace+p*n_out+traction_in.
            const auto& layer=*input.preparedLayer;const Vec3 outward=-input.normal;
            const Vec3 traceVelocity={layer.traceVelocity[0],layer.traceVelocity[1],layer.traceVelocity[2]};
            const Vec3 traction={layer.traction[0],layer.traction[1],layer.traction[2]};
            input.wallContext.mechanicalPressure=dot(flux[f].momentum,outward)/input.gasArea
                +massFlux*dot(traceVelocity,outward)-dot(traction,outward);
            b->preparedMatching[i].mechanicalPressure=input.wallContext.mechanicalPressure;
        }
        for(int c=0;c<Nc;++c)input.condensedRate[c]=w.condensedRate[c];
        SurfacePacketIdentity identity;identity.step=b->record.microSequence;identity.geometry=b->record.gasGeometry;identity.face=surface.persistentId[i];identity.stage=1;identity.gasCell=cell;identity.solidCell=surface.solidCell[i];identity.filmFace=int(i);
        if(!evaluateGasWall(input,identity,b->model.physics,b->faceRates[i])){b->error="coupled interface trace/packet failed";return 1;}
        const auto a=packetDelta(b->faceRates[i].primary).gas,c=packetDelta(b->faceRates[i].pore).gas;
        b->rawInterface[i]=(a+c)*(-1/dt);flux[f]=b->rawInterface[i];
    }
    b->stageBulk=std::move(primitive);b->stageGradient=std::move(gradient);
    return fluxes(*b,flux,true)?0:1;
}
int accountCoupledFaces(Backend* b,Real ledgerDt){
    std::vector<GasQ> accepted;if(!fluxes(*b,accepted,false)){b->error=b->storage.error();return 1;}
    std::fprintf(stderr,"LEDGER_STAGE,%llu,%llu,%d,%a,%a,%a\n",static_cast<unsigned long long>(b->program.interval.identity.sequence),static_cast<unsigned long long>(b->record.microSequence),b->gasStageIndex,ledgerDt,b->microDt,b->stagePacketDt);
    for(int face=0;face<b->nFaces;++face){const auto& q=accepted[face];std::fprintf(stderr,"LEDGER_FINAL_FLUX,%llu,%llu,%d,%d,%d,%d,%a,%a,%a,%a,%a",static_cast<unsigned long long>(b->program.interval.identity.sequence),static_cast<unsigned long long>(b->record.microSequence),b->gasStageIndex,face,b->trial.gasMesh.neighbour[face],int(b->trial.gasMesh.boundaryKind[face]),q.mass,q.energy,q.momentum.x,q.momentum.y,q.momentum.z);for(int species=0;species<Ns;++species)std::fprintf(stderr,",%a",q.species[species]);std::fprintf(stderr,"\n");}
    const auto& surface=b->trial.surface;std::vector<bool> interface(b->nFaces,false);
    if(b->record.packets.empty()){
        for(const auto& rates:b->faceRates){for(const auto* packet:{&rates.primary,&rates.pore}){auto p=scaledIntervalPacket(*packet,0);p.consumerMask=ConsumeGas;b->record.packets.push_back(p);}}
        b->record.radiationEnergy.assign(surface.area.size(),0);b->record.radiationReceiver.resize(surface.area.size());
    }
    WallKnot wall;if(!sampleGasWallProgramLocal(b->program,coupledStageTime(*b),wall,b->error,false))return 1;
    for(std::size_t i=0;i<surface.area.size();++i){const int f=surface.gasFace[i];interface[f]=true;const auto& raw=b->rawInterface[i];const auto& now=accepted[f];
        Real denominator=raw.energy,numerator=now.energy;if(absValue(raw.mass)>absValue(denominator)){denominator=raw.mass;numerator=now.mass;}
        for(int s=0;s<Ns;++s)if(absValue(raw.species[s])>absValue(denominator)){denominator=raw.species[s];numerator=now.species[s];}
        const Real scale=denominator==0?1:numerator/denominator;
        for(int channel=0;channel<2;++channel){const auto& rate=channel?b->faceRates[i].pore:b->faceRates[i].primary;auto packet=scaledIntervalPacket(rate,(ledgerDt/b->stagePacketDt)*scale);packet.consumerMask=ConsumeGas;
            std::fprintf(stderr,"LEDGER_PAIRED,%llu,%llu,%d,%zu,%d,%d,%a,%a,%a,%a,%a,%a,%a\n",static_cast<unsigned long long>(b->program.interval.identity.sequence),static_cast<unsigned long long>(b->record.microSequence),b->gasStageIndex,i,channel,int(packet.kind),scale,ledgerDt/b->stagePacketDt,packet.mass,packet.energy,packet.momentum.x,packet.momentum.y,packet.momentum.z);
            addIntervalPacket(b->record.packets[2*i+channel],packet);addGasBudget(b->trialBudget,packet);}
        b->record.radiationEnergy[i]+=ledgerDt*wall.faces[i].radiationFlux*surface.area[i];b->record.radiationReceiver[i]=wall.faces[i].primaryKind==ExchangeKind::GasFilm?ConsumeFilm:ConsumeSolid;
    }
    for(int f=0;f<b->nFaces;++f)if(b->trial.gasMesh.neighbour[f]<0&&!interface[f]&&b->trial.gasMesh.boundaryKind[f]!=BoundaryKind::Periodic){const auto q=accepted[f]*ledgerDt;b->trialBudget.boundaryMass+=q.mass;b->trialBudget.boundaryEnergy+=q.energy;b->trialBudget.boundaryMomentum+=q.momentum;
        for(int s=0;s<Ns;++s){b->trialBudget.boundarySpecies[s]+=q.species[s];for(int e=0;e<b->model.physics.nElements;++e)b->trialBudget.boundaryElements[e]+=q.species[s]*b->model.physics.species[s].element[e];}}
    ++b->gasStageIndex;return 0;
}
int acceptSstAudit(Backend* b){
    if(b->hostTurbulenceModel!=3)return 0;
    const auto& audit=b->storage.hostView().gasSstAudit;const Real* fields[]={audit.transportK,audit.transportOmega,audit.sourceK,audit.sourceOmega,audit.constraintK,audit.constraintOmega};
    std::array<std::vector<Real>,6> values;
    for(int j=0;j<6;++j)if(!b->storage.read(fields[j],b->nCells,values[j])){b->error=b->storage.error();return 1;}
    GasSstIntegralAudit totals;Real* sum[]={&totals.transportK,&totals.transportOmega,&totals.sourceK,&totals.sourceOmega,&totals.constraintK,&totals.constraintOmega};
    if(b->nCells>0&&b->trial.sst.empty()){b->error="SST audit/state size mismatch";return 1;}
    Real inventory=0;
    for(int c=0;c<b->nCells;++c){
        for(int j=0;j<6;++j)*sum[j]+=values[j][c];
        const auto& next=b->trial.sst[c];
        inventory+=next.rhoK;
    }
    auto& dst=b->trial.gasSstAudit;
    dst.transportK+=totals.transportK;dst.transportOmega+=totals.transportOmega;
    dst.sourceK+=totals.sourceK;dst.sourceOmega+=totals.sourceOmega;
    dst.constraintK+=totals.constraintK;dst.constraintOmega+=totals.constraintOmega;
    b->trialBudget.turbulenceInventory=inventory;
    b->trialBudget.turbulenceBoundaryFlux-=totals.transportK;
    b->trialBudget.turbulenceOmegaConstraint+=totals.constraintOmega;
    return 0;
}
bool restoreGasSnapshot(Backend& b,const HostState& snapshot,std::string& error){
    b.storage.beginOperation();
    b.microGeometryPrepared=false;b.preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(b);
    if(b.gasBoundaryLayer.enabled&&b.model.physics.meshMotion.policy!=MeshMotionPolicy::Static)b.storage.invalidateWallGeometry();
    if(!b.storage.uploadGeometry(snapshot.gasMesh,false)||!b.storage.clearGeometry()
        ||!b.storage.uploadState(snapshot,error)){
        b.poisoned=b.storage.runtimeFailure();if(error.empty())error=b.storage.error();
        if(error.empty())error="failed to restore common gas geometry/state";return false;
    }
    bindSharedGasStorage(b);b.trial=snapshot;error.clear();return true;
}
Backend* createBackend(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
 const ugkwp::GasMechanismConfiguration& mechanism,const GasExecutionOptions& options,const HostState& state,std::string& error){
    if(mag(model.physics.gravity)!=0){error="coupled gas gravity source/work ledger is not enabled";return nullptr;}
    if(options.turbulenceModel!=(model.physics.enableSst?3:0))
        std::fprintf(stderr,"CHMT requested physics SST=%d execution turbulence=%d; effective SST=%d\n",
            int(model.physics.enableSst),options.turbulenceModel,int(options.turbulenceModel==3));
    ugkwp::GasCapabilityRequest request;request.mode=gas.mode;request.fluxScheme=options.fluxScheme;request.reconstruction=options.reconstruction;request.limiter=options.limiter;request.timeIntegrator=options.timeIntegrator;request.turbulenceModel=options.turbulenceModel;request.sstWallTreatment=int(model.physics.wallModel.family);request.boundaryLayerModel=int(model.physics.wallModel.model);request.movingGeometry=model.physics.meshMotion.policy!=MeshMotionPolicy::Static;request.particleCoupling=model.physics.enableParticles;
    const auto supported=ugkwp::validateGasCapabilities(request);if(!supported){error=supported.message;return nullptr;}
    if(gas.mode!=ugkwp::GasMode::SingleLegacy
        && ((options.fluxScheme!=1&&options.fluxScheme!=2)||(options.reconstruction!=0&&options.reconstruction!=1)))
        std::fprintf(stderr,"CHMT mixture requested flux=%d reconstruction=%d; effective flux=%d reconstruction=%d (existing fallback)\n",
            options.fluxScheme,options.reconstruction,options.fluxScheme==2?2:1,options.reconstruction==1?1:0);
    std::unique_ptr<Backend> b(new Backend);b->model=model;b->options=options;b->accepted=b->trial=state;
    b->model.physics.enableSst=options.turbulenceModel==3;
    if(!b->storage.configure(b->model,gas,mechanism,state,error))return nullptr;
    auto& v=b->storage.hostView();v.gasFluxScheme=options.fluxScheme;v.gasReconstruction=options.reconstruction;v.gasLimiter=options.limiter;v.turbulenceModel=options.turbulenceModel;v.sstConfigured=options.turbulenceModel==3;
    if(!b->storage.refreshView()){error=b->storage.error();return nullptr;}
    bindSharedGasStorage(*b);b->hostTurbulenceModel=options.turbulenceModel;b->hostGasFluxScheme=options.fluxScheme;b->hostGasTimeIntegrator=options.timeIntegrator;
    for(auto kind:state.gasMesh.boundaryKind)if(kind==BoundaryKind::Periodic)b->hasPeriodicFaces=1;
    error.clear();return b.release();
}
void destroyBackend(Backend* backend)noexcept{delete backend;}
bool beginGasWindow(Backend& b,const WallProgram& program,std::string& error){
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    b.trial=b.accepted;b.program=program;
    if(!b.history.begin(program.interval,error)||!b.reserve.reset(b.accepted.solid,b.accepted.film,b.accepted.time,error))return false;
    if(!program.donorPlan.knots.empty()&&!b.reserve.setPlan(program.donorPlan,error))return false;
    if(!restoreGasSnapshot(b,b.accepted,error))return false;b.elapsed=WindowTime{};b.nextGasSeed=b.accepted.nextDt>0?b.accepted.nextDt:b.accepted.lastAcceptedDt;return true;
}
// Same maximum operation and Thrust reduction semantics as the native owner.
struct CoupledCourantMaximum {
    __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
struct CoupledCourantValue {double value;bool unavailable;unsigned char initializedTail[7]{};};
struct CoupledCourantRead {
    const double* field;const int* status;
    __host__ __device__ CoupledCourantValue operator()(int cell)const{
        return {field[cell],status&&status[cell]==int(ugkwp::GasTransportCode::InvalidStorage)};
    }
};
struct CoupledCourantReduce {
    __host__ __device__ CoupledCourantValue operator()(CoupledCourantValue a,CoupledCourantValue b)const{
        return {CoupledCourantMaximum{}(a.value,b.value),a.unavailable||b.unavailable};
    }
};
static void r3CoSeed(Backend& b,const char* position,Real referenceDt){
    const auto& v=b.storage.hostView();std::vector<Real> temperature,pressure;
    if(!b.storage.read(v.Tgas,b.nCells,temperature)||!b.storage.read(v.p,b.nCells,pressure)){std::fprintf(stderr,"R3SEED_READ_ERROR,%s,%s\n",position,b.storage.error());return;}
    std::fprintf(stderr,"R3SEED_META,%s,%llu,%zu,%a,%a,%a,%a\n",position,(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,referenceDt,b.trial.nextDt,b.trial.lastAcceptedDt,b.microTime);
    for(int c=0;c<b.nCells;++c){const auto& q=b.trial.gas[c];std::fprintf(stderr,"R3SEED_CELL,%s,%llu,%zu,%d,%a,%a,%a,%a,%a,%a,%a,%a\n",position,(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,c,temperature[c],pressure[c],q.mass,q.energy,q.momentum.x,q.momentum.y,q.momentum.z,b.trial.gasMesh.volumes[c]);}
}
static bool currentGasCourant(Backend& b,Real referenceDt,transport::GasCourantMaxima& result){
    const auto& v=b.storage.hostView();
    struct View {
        int nCells,nFaces,hostTurbulenceModel;SharedGasDeviceView* deviceState;
        Real *gasFluxPositivityScale,*gasDiffusionNumber,*sstSourceNumber;
    } view{b.nCells,b.nFaces,b.hostTurbulenceModel,b.deviceState,v.gasFluxPositivityScale,v.gasDiffusionNumber,v.sstSourceNumber};
    auto copy=[&](Real* to,const Real* from){const auto code=cudaMemcpy(to,from,b.nCells*sizeof(Real),cudaMemcpyDeviceToDevice);if(code!=cudaSuccess){transport::setLastError("current Co SST scratch copy",code);return false;}return true;};
    const bool sst=b.hostTurbulenceModel==3;
    if(sst&&b.gasBoundaryLayer.enabled&&!b.storage.wallMassPublished()){
        WallKnot wall;if(!sampleGasWallProgramLocal(b.program,b.elapsed,wall,b.error,false)
            ||!b.storage.prepareWallBoundary(wall,b.trial.surface,b.error))return false;
    }
    if(sst&&(!copy(v.rhoKInitial,v.rhoK)||!copy(v.rhoOmegaInitial,v.rhoOmega)))return false;
    const auto maximum=[&](const double* field,int count,cudaStream_t stream,double& value){
        try {
            if(field==v.gasFluxPositivityScale){
                // The existing first maximum traversal also reports actual
                // unavailable inputs. No separate status transfer/scan/sync.
                const auto first=thrust::make_counting_iterator(0);
                const auto reduced=thrust::transform_reduce(thrust::cuda::par(b.courantReductionStorage).on(stream),first,first+count,
                    CoupledCourantRead{field,v.gasSpecies.cellStatus},
                    CoupledCourantValue{-std::numeric_limits<double>::infinity(),false},CoupledCourantReduce{});
                if(reduced.unavailable){transport::setLastErrorText("current Co consumed input unavailable");return 1;}
                value=reduced.value;
            }else value=thrust::reduce(thrust::cuda::par(b.courantReductionStorage).on(stream),thrust::device_pointer_cast(field),thrust::device_pointer_cast(field)+count,-std::numeric_limits<double>::infinity(),CoupledCourantMaximum{});
        }
        catch(const std::exception& e){transport::setLastErrorText(e.what());return 1;}
        const auto code=cudaGetLastError();if(code!=cudaSuccess){transport::setLastError("gas Courant maximum reduction",code);return 1;}return 0;
    };
    r3CoSeed(b,"pre",referenceDt);
    const int status=transport::computeGasCurrentCourant(&view,referenceDt,b.model.physics.cfl,b.microTime,b.fixedCellBlockThreads,b.fixedFaceBlockThreads,b.gasCaptureStream,result,maximum);
    r3CoSeed(b,"post",referenceDt);
    // Initial scratch is not a live RK snapshot until advanceGasTrial saves it.
    bool restored=true;if(sst){restored=copy(v.rhoK,v.rhoKInitial);restored=copy(v.rhoOmega,v.rhoOmegaInitial)&&restored;}
    return status==0&&restored;
}
bool advanceGasMicrostep(Backend& b,Real cap,GasMicroReport& report,std::string& error){
    report={};report.recoverable=false;
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    if(b.history.complete()||b.elapsed==intervalDuration(b.program.interval)){
        report.time=b.trial.time;report.localTime=b.elapsed;report.nextGasDt=b.nextGasSeed;
        report.microSequence=b.history.records().size();error.clear();return true;
    }
    b.microTime=windowAbsolute(b.program.interval.begin,b.elapsed);
    const Real remaining=boundedWindowDuration(nextGasWallKnotLocal(b.program,b.elapsed),b.elapsed);
    Real referenceDt=minValue(cap,minValue(b.model.physics.maxDt,remaining));
    if(b.nextGasSeed>0)referenceDt=minValue(referenceDt,b.nextGasSeed);
    if(!(referenceDt>0)||b.elapsed+referenceDt==b.elapsed)referenceDt=remaining;
    b.microBase=b.trial;b.microDt=referenceDt;b.error.clear();transport::lastError.clear();
    if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}
    transport::GasCourantMaxima co;
    if(!currentGasCourant(b,referenceDt,co)){
        error=b.error.empty()?transport::lastError:b.error;std::string restoreError;if(!restoreGasSnapshot(b,b.microBase,restoreError))error+="; "+restoreError;return false;
    }
    const Real growthBase=b.trial.lastAcceptedDt>0?b.trial.lastAcceptedDt:referenceDt;
    const Real selectedDt=gasCoMaxStep(referenceDt,b.model.physics.cfl,co.maximum,growthBase);
    // An exhausted numerical split still executes the real requested interval.
    // No clock is advanced until that physical trial and commit have completed.
    const Real dt=selectedDt>0 && b.elapsed+selectedDt!=b.elapsed?selectedDt:referenceDt;
    std::fprintf(stderr,"R3DT,%llu,%zu,%a,%a,%a,%a\n",(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,referenceDt,selectedDt,dt,co.maximum);
    if(dt!=referenceDt){b.microDt=dt;if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}}
    b.microGeometryPrepared=true;
    if(transport::advanceGasTrial<transport::CoupledTrialPolicy>(&b,dt,b.microTime)!=0){
        error=b.error.empty()?transport::lastError:b.error;return false;
    }
    b.record.gasTraceTime=b.stageTime;
    for(std::size_t f=0;f<b.trial.surface.area.size();++f){const int cell=b.trial.gasMesh.owner[b.trial.surface.gasFace[f]];
        b.record.gasTrace.push_back(b.stageBulk[cell]);b.record.gasGradient.push_back(b.stageGradient[cell]);b.record.gasTraction.push_back(b.faceRates[f].traction);}
    if(b.gasBoundaryLayer.enabled)b.record.gasWallMatching=b.preparedMatching;
    MaterialDonorReserve::Transaction reservation;
    if(!b.reserve.prepare(b.record,reservation,b.error)){transport::CoupledTrialPolicy::rollback(&b);error=b.error;return false;}
    if(!b.history.appendAccepted(b.record,error)){transport::CoupledTrialPolicy::rollback(&b);if(b.poisoned)error=b.error;return false;}
    fp64_clock_trace::accepted(b.program.interval,b.record,b.trial);
    b.reserve.commitPrepared(std::move(reservation));b.elapsed=b.record.localEnd;
    b.trial.gasStages={{b.gasStage,b.gasStage}};b.trial.solidStages={{b.solidStage,b.solidStage}};
    b.nextGasSeed=minValue(b.model.physics.maxDt,1.05*dt);b.trial.nextDt=b.nextGasSeed;b.trial.lastAcceptedDt=dt;
    report.acceptedDt=dt;report.nextGasDt=b.nextGasSeed;report.time=b.trial.time;report.localTime=b.elapsed;report.microSequence=b.record.microSequence;error.clear();return true;
}
bool gasWindowHistory(const Backend& b,IntervalHistory& out,std::string& error){out=b.history;error.clear();return true;}
bool gasWindowStageGeometry(const Backend& b,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error){gas=b.trial.gasStages;solid=b.trial.solidStages;error.clear();return true;}
bool downloadGasWindow(const Backend& b,HostState& out,std::string& error){out=b.trial;error.clear();return true;}
bool rollbackGasWindow(Backend& b,std::string& error){
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    if(!restoreGasSnapshot(b,b.accepted,error))return false;
    return true;
}
bool commitGasWindow(Backend& b,const HostState& candidate,std::string& error){
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    // Prepare all throwing host copies before touching device/publication state.
    HostState newAccepted=candidate,newTrial=candidate;
    if(!b.storage.uploadState(candidate,error))return false;
    b.accepted=std::move(newAccepted);b.trial=std::move(newTrial);error.clear();return true;
}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
}
