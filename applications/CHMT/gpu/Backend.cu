#include "gpu/Backend.H"
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "gpu/SingleGasContract.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalAudit.H"
#include "mesh/TrajectorySurface.H"
#include "../../../common/gasTransport/GasCapabilities.H"
#include "../../../common/gasTransport/GasGeometryValidation.H"
#include "../../../common/gasTransport/SpeciesDiffusion.H"
#include "../../../common/GpuPrecisionTypes.H"
#include "../../../common/gasNumerics/CharacteristicMuscl.cuh"
#include "../../../common/gasNumerics/OpenFoamLimitedLinear.cuh"
#include "../../../common/gasNumerics/OpenFoamViscousFlux.cuh"
#include "../../../common/gasNumerics/OpenFoamWallFunctions.cuh"
#include "../../../common/gasNumerics/RiemannBoundaryState.cuh"
#include "../../../common/gasNumerics/RiemannGasFlux.cuh"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include "../../../common/gasNumerics/GpuLesAlgebra.cuh"
#include <memory>
#include <algorithm>
#include <vector>
namespace chmt {
struct Backend {
    ModelConfig model;GasExecutionOptions options;HostState accepted,trial,microBase;
    HostMesh endpointGas,endpointSolid;SurfaceMesh endpointSurface;HostStageGeometry gasStage,solidStage;
    SharedGasDeviceStorage storage;WallProgram program;IntervalHistory history;MaterialDonorReserve reserve;
    GasWindowLimits limits;bool pending=false,poisoned=false;std::string error;
    GasIntervalRecord record;std::vector<GasPrimitive> stageBulk;std::vector<GasWallResult> faceRates;std::vector<GasQ> rawInterface;
    Real microDt=0,microTime=0,stageTime=0,stagePacketDt=0;Budget trialBudget{};
    std::vector<ugkwp::ChemistryAudit<Real,Ns>> firstChemistry,secondChemistry;
    int nCells=0,nFaces=0,fixedCellBlockThreads=128,fixedFaceBlockThreads=128;
    int hostTurbulenceModel=0,hostGasFluxScheme=1,hostGasTimeIntegrator=1,hasPeriodicFaces=0;
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
    ugkwp::GasGeometryState<Real> gasGeometry;
    SharedGasDeviceView* deviceState=nullptr;cudaStream_t gasCaptureStream=nullptr;
};
bool restoreGasSnapshot(Backend&,const HostState&,std::string&);
int prepareCoupledGeometry(Backend*);
int finishCoupledGeometry(Backend*);
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
} // transport
} // chmt
// Chemistry header declares its own ugkwp namespace and shared kernel.
#include "../../../common/operators/advanceGasChemistryKernel.cuh"
#include "../../../common/GpuGasHostPolicy.cuh"
namespace chmt { namespace transport {
thread_local std::string lastError;
inline void setLastError(const char* text,cudaError_t code){lastError=std::string(text)+": "+cudaGetErrorString(code);}
inline void setLastErrorText(const char* text){lastError=text;}
struct GasHostPolicy:GasHostWithWallEnergy<Real,Real,&accountCoupledFaces> {
    static int applyFaceSources(Backend* s,Real dt,Real time){return applyCoupledFaces(s,dt,time);}
};
#include "../../../common/GpuGasAdvance.cuh"
struct CoupledTrialPolicy {
    static int begin(Backend* b){
        b->microBase=b->trial;b->trialBudget=b->trial.budget;b->record={};b->record.microSequence=b->history.records().size()+1;
        b->record.begin=b->microTime;b->record.end=b->microTime+b->microDt;b->record.gasGeometry=b->trial.gasMesh.geometryVersion;b->record.solidGeometry=b->trial.solidMesh.geometryVersion;
        if(!b->storage.clearTrialStatus()||prepareCoupledGeometry(b)!=0)return 1;
        return 0;
    }
    static const Real* stageVolumes(Backend* b,bool after){const auto& g=b->storage.hostView().gasGeometry;return g.enabled?(after?g.newVolume:g.oldVolume):b->storage.hostView().V;}
    static Real targetMaxCo(Backend* b){return b->model.physics.cfl;}
    static int captureChemistryAudit(Backend* b,bool after){
        auto& audit=after?b->secondChemistry:b->firstChemistry;
        if(b->model.physics.gasMode!=ugkwp::GasMode::MixtureChemistry){audit.clear();return 0;}
        return b->storage.read(b->storage.hostView().gasSpecies.chemistryAudit,b->nCells,audit)?0:1;
    }
    static int validateTimeStep(Backend* b,Real){
        std::vector<Real> convective,diffusion;const auto& v=b->storage.hostView();
        if(!b->storage.read(v.gasFluxPositivityScale,b->nCells,convective)||!b->storage.read(v.gasDiffusionNumber,b->nCells,diffusion)){b->error=b->storage.error();return 1;}
        if(b->hostTurbulenceModel==3){
            std::vector<Real> sst;if(!b->storage.read(v.sstSourceNumber,b->nCells,sst)){b->error=b->storage.error();return 1;}
            for(Real value:sst)if(!finite(value)||value>b->model.physics.cfl){b->error="shared SST stability limit exceeded";return 1;}
        }
        for(int c=0;c<b->nCells;++c)if(!finite(convective[c])||!finite(diffusion[c])||maxValue(convective[c],diffusion[c])>b->model.physics.cfl){b->error="shared gas stability limit exceeded";return 1;}
        return 0;
    }
    static int applySources(Backend* b,Real){return finishCoupledGeometry(b); }
    static int validate(Backend* b){
        if(!b->storage.checkStatus(b->error)||!b->storage.downloadState(b->trial,b->error))return 1;
        for(std::size_t c=0;c<b->trial.gas.size();++c){GasPrimitive w;if(!recoverGas(b->trial.gas[c],b->trial.gasMesh.volumes[c],b->model.physics,w)){b->error="invalid accepted common gas candidate";return 1;}}
        return 0;
    }
    static int commit(Backend* b){
        if(b->model.physics.gasMode==ugkwp::GasMode::MixtureChemistry){
            b->trial.gasChemistryAudit.resize(b->nCells);
            for(const auto* half:{&b->firstChemistry,&b->secondChemistry}){
                if(half->size()!=std::size_t(b->nCells)){b->error="missing chemical half audit";return 1;}
                for(int c=0;c<b->nCells;++c){auto& dst=b->trial.gasChemistryAudit[c];const auto& src=(*half)[c];
                    if(dst.acceptedSteps>INT_MAX-src.acceptedSteps||dst.rejectedSteps>INT_MAX-src.rejectedSteps||dst.nonlinearIterations>INT_MAX-src.nonlinearIterations){b->error="accepted chemistry counter overflow";return 1;}
                    for(int s=0;s<Ns;++s)dst.speciesMassChange[s]+=src.speciesMassChange[s];
                    dst.massResidual+=src.massResidual;dst.energyResidual+=src.energyResidual;dst.maximumElementResidual=maxValue(dst.maximumElementResidual,src.maximumElementResidual);
                    dst.maximumLocalError=maxValue(dst.maximumLocalError,src.maximumLocalError);dst.integratedTime+=src.integratedTime;
                    dst.acceptedSteps+=src.acceptedSteps;dst.rejectedSteps+=src.rejectedSteps;dst.nonlinearIterations+=src.nonlinearIterations;
                }
            }
        }
        if(acceptSstAudit(b)!=0)return 1;
        b->trial.time=b->record.end;b->trial.budget=b->trialBudget;return 0;
    }
    static void rollback(Backend* b){
        std::string rollbackError;
        if(!restoreGasSnapshot(*b,b->microBase,rollbackError)){
            b->poisoned=true;b->error+="; rollback failed: "+rollbackError;
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
        if(b->hostGasTimeIntegrator!=1){b->error="moving coupled geometry requires verified common Euler metrics";return 1;}
        WallKnot endpoint;if(!sampleGasWallProgram(b->program,b->microTime+b->microDt,endpoint,b->error))return 1;
        if(!makeStageGeometry(base.gasMesh,endpoint.gasPoints,b->microDt,b->endpointGas,gasSweep,b->error,b->model.physics.tolerances)
            ||!makeStageGeometry(base.solidMesh,endpoint.solidPoints,b->microDt,b->endpointSolid,solidSweep,b->error,b->model.physics.tolerances)
            ||!rebuildTrajectorySurface(base,b->endpointGas,b->endpointSolid,b->microDt,b->endpointSurface,b->error,b->model.physics.tolerances))return 1;
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
    if(!b->storage.uploadGeometry(b->trial.gasMesh)||!b->storage.refreshView()){b->error=b->storage.error();return 1;}
    return 0;
}
int applyCoupledFaces(Backend* b,Real dt,Real stageTime){
    b->stageTime=stageTime;b->stagePacketDt=dt;WallKnot wall;if(!sampleGasWallProgram(b->program,stageTime,wall,b->error))return 1;
    std::vector<GasPrimitive> primitive;std::vector<GasGradient> gradient;std::vector<GasQ> flux;
    if(!readBulk(*b,primitive,gradient)||!fluxes(*b,flux,false)){b->error=b->storage.error();return 1;}
    const auto& surface=b->trial.surface;b->faceRates.resize(surface.area.size());b->rawInterface.resize(surface.area.size());
    for(std::size_t i=0;i<surface.area.size();++i){const int f=surface.gasFace[i],cell=b->trial.gasMesh.owner[f];const auto& w=wall.faces[i];GasWallInput input;
        input.bulk=primitive[cell];input.gradient=gradient[cell];input.temperature=w.temperature;input.gasDistance=surface.gasDistance[i];input.area=surface.area[i];input.gasArea=mag(b->trial.gasMesh.areaVectors[f]);input.dt=dt;input.normal=surface.normal[i];input.velocity=w.velocity;input.primaryKind=w.primaryKind;
        input.normalSpeed=w.normalVelocity;input.sweptVolume=-b->gasStage.sweptVolume[f];
        if(!b->gasGeometry.enabled&&(w.normalVelocity!=0||w.solidNormalVelocity!=0)){b->error="wall motion supplied to static gas geometry";return 1;}
        for(int s=0;s<Ns;++s){input.speciesRate[s]=w.speciesRate[s];input.poreRate[s]=w.poreRate[s];input.poreSweepRate[s]=w.poreSweepRate[s];}
        for(int c=0;c<Nc;++c)input.condensedRate[c]=w.condensedRate[c];
        SurfacePacketIdentity identity;identity.step=b->record.microSequence;identity.geometry=b->record.gasGeometry;identity.face=surface.persistentId[i];identity.stage=1;identity.gasCell=cell;identity.solidCell=surface.solidCell[i];identity.filmFace=int(i);
        if(!evaluateGasWall(input,identity,b->model.physics,b->faceRates[i])){b->error="coupled interface trace/packet failed";return 1;}
        const auto a=packetDelta(b->faceRates[i].primary).gas,c=packetDelta(b->faceRates[i].pore).gas;
        b->rawInterface[i]=(a+c)*(-1/dt);flux[f]=b->rawInterface[i];
    }
    // Coupled channels keep separate donors even when their net face flux is
    // zero. Incoming pore/Darcy/sweep material cannot finance a simultaneous
    // explicit gas withdrawal. Common transport still owns flux limiting.
    std::vector<Real> withdrawal(std::size_t(b->nCells)*Ns,0);
    for(std::size_t i=0;i<surface.area.size();++i){const int c=b->trial.gasMesh.owner[surface.gasFace[i]];
        for(int k=0;k<Ns;++k)withdrawal[k*b->nCells+c]+=gasPacketSpeciesWithdrawal(b->faceRates[i].primary,k)+gasPacketSpeciesWithdrawal(b->faceRates[i].pore,k);
    }
    for(int c=0;c<b->nCells;++c)for(int k=0;k<Ns;++k){
        const Real available=primitive[c].rho*primitive[c].Y[k]*b->gasStage.oldVolume[c];
        const Real tolerance=b->model.physics.tolerances.absoluteMass+b->model.physics.tolerances.relativeMass*available;
        if(!finite(withdrawal[k*b->nCells+c])||withdrawal[k*b->nCells+c]>available+tolerance){b->error="gross coupled gas donor inventory exceeded";return 1;}
    }
    b->stageBulk=std::move(primitive);
    return fluxes(*b,flux,true)?0:1;
}
int accountCoupledFaces(Backend* b,Real ledgerDt){
    std::vector<GasQ> accepted;if(!fluxes(*b,accepted,false)){b->error=b->storage.error();return 1;}
    const auto& surface=b->trial.surface;std::vector<bool> interface(b->nFaces,false);
    std::vector<Real> gross(std::size_t(b->nCells)*Ns,0);
    if(b->record.packets.empty()){
        for(const auto& rates:b->faceRates){for(const auto* packet:{&rates.primary,&rates.pore}){auto p=scaledIntervalPacket(*packet,0);p.consumerMask=ConsumeGas;b->record.packets.push_back(p);}}
        b->record.radiationEnergy.assign(surface.area.size(),0);b->record.radiationReceiver.resize(surface.area.size());
    }
    WallKnot wall;if(!sampleGasWallProgram(b->program,b->stageTime,wall,b->error,false))return 1;
    for(std::size_t i=0;i<surface.area.size();++i){const int f=surface.gasFace[i];interface[f]=true;const auto& raw=b->rawInterface[i];const auto& now=accepted[f];
        Real denominator=raw.energy,numerator=now.energy;if(absValue(raw.mass)>absValue(denominator)){denominator=raw.mass;numerator=now.mass;}
        for(int s=0;s<Ns;++s)if(absValue(raw.species[s])>absValue(denominator)){denominator=raw.species[s];numerator=now.species[s];}
        Real scale=denominator==0?1:numerator/denominator;if(!finite(scale)||scale<0||scale>1+1e-12){b->error="inconsistent common interface positivity scale";return 1;}
        const int owner=b->trial.gasMesh.owner[f];
        for(int k=0;k<Ns;++k)gross[k*b->nCells+owner]+=scale*(gasPacketSpeciesWithdrawal(b->faceRates[i].primary,k)+gasPacketSpeciesWithdrawal(b->faceRates[i].pore,k));
        for(int channel=0;channel<2;++channel){const auto& rate=channel?b->faceRates[i].pore:b->faceRates[i].primary;auto packet=scaledIntervalPacket(rate,(ledgerDt/b->stagePacketDt)*scale);packet.consumerMask=ConsumeGas;
            addIntervalPacket(b->record.packets[2*i+channel],packet);addGasBudget(b->trialBudget,packet);}
        b->record.radiationEnergy[i]+=ledgerDt*wall.faces[i].radiationFlux*surface.area[i];b->record.radiationReceiver[i]=wall.faces[i].primaryKind==ExchangeKind::GasFilm?ConsumeFilm:ConsumeSolid;
    }
    for(int f=0;f<b->nFaces;++f)if(!interface[f]){
        const int owner=b->trial.gasMesh.owner[f],neighbour=b->trial.gasMesh.neighbour[f];
        for(int k=0;k<Ns;++k){const Real transfer=b->stagePacketDt*accepted[f].species[k];
            gross[k*b->nCells+owner]+=maxValue(0,transfer);
            if(neighbour>=0)gross[k*b->nCells+neighbour]+=maxValue(0,-transfer);
        }
    }
    for(int c=0;c<b->nCells;++c)for(int k=0;k<Ns;++k){
        const Real available=b->stageBulk[c].rho*b->stageBulk[c].Y[k]*b->gasStage.oldVolume[c];
        const Real tolerance=b->model.physics.tolerances.absoluteMass+b->model.physics.tolerances.relativeMass*available;
        if(!finite(gross[k*b->nCells+c])||gross[k*b->nCells+c]>available+tolerance){b->error="combined transport and gross coupled gas donor inventory exceeded";return 1;}
    }
    for(int f=0;f<b->nFaces;++f)if(b->trial.gasMesh.neighbour[f]<0&&!interface[f]&&b->trial.gasMesh.boundaryKind[f]!=BoundaryKind::Periodic){const auto q=accepted[f]*ledgerDt;b->trialBudget.boundaryMass+=q.mass;b->trialBudget.boundaryEnergy+=q.energy;b->trialBudget.boundaryMomentum+=q.momentum;
        for(int s=0;s<Ns;++s){b->trialBudget.boundarySpecies[s]+=q.species[s];for(int e=0;e<b->model.physics.nElements;++e)b->trialBudget.boundaryElements[e]+=q.species[s]*b->model.physics.species[s].element[e];}}
    return 0;
}
int acceptSstAudit(Backend* b){
    if(b->hostTurbulenceModel!=3)return 0;
    const auto& audit=b->storage.hostView().gasSstAudit;const Real* fields[]={audit.transportK,audit.transportOmega,audit.sourceK,audit.sourceOmega,audit.constraintK,audit.constraintOmega};
    std::array<std::vector<Real>,6> values;
    for(int j=0;j<6;++j)if(!b->storage.read(fields[j],b->nCells,values[j])){b->error=b->storage.error();return 1;}
    GasSstIntegralAudit totals;Real* sum[]={&totals.transportK,&totals.transportOmega,&totals.sourceK,&totals.sourceOmega,&totals.constraintK,&totals.constraintOmega};
    if(b->trial.sst.size()!=std::size_t(b->nCells)||b->microBase.sst.size()!=b->trial.sst.size()){b->error="SST audit/state size mismatch";return 1;}
    Real inventory=0;
    for(int c=0;c<b->nCells;++c){
        for(int j=0;j<6;++j){if(!finite(values[j][c])){b->error="nonfinite accepted common SST audit";return 1;}*sum[j]+=values[j][c];}
        const auto& old=b->microBase.sst[c];const auto& next=b->trial.sst[c];
        const Real expectedK=values[0][c]+values[2][c]+values[4][c],expectedOmega=values[1][c]+values[3][c]+values[5][c];
        const Real eps=256*std::numeric_limits<Real>::epsilon();
        if(!closeEnough(next.rhoK-old.rhoK,expectedK,eps*maxValue(1,maxValue(absValue(old.rhoK),absValue(next.rhoK))),eps)
            ||!closeEnough(next.rhoOmega-old.rhoOmega,expectedOmega,eps*maxValue(1,maxValue(absValue(old.rhoOmega),absValue(next.rhoOmega))),eps)){
            b->error="common SST integrated source/transport/constraint audit does not close";return 1;}
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
    if(!b.storage.uploadGeometry(snapshot.gasMesh)||!b.storage.clearGeometry()
        ||!b.storage.uploadState(snapshot,error)){
        b.poisoned=true;if(error.empty())error=b.storage.error();
        if(error.empty())error="failed to restore common gas geometry/state";return false;
    }
    b.gasGeometry={};b.trial=snapshot;error.clear();return true;
}
Backend* createBackend(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
 const ugkwp::GasMechanismConfiguration& mechanism,const GasExecutionOptions& options,const HostState& state,std::string& error){
    if(!validateCouplingEntry("Multirate",state,error)||!validateSingleGasInventory(model,state,error))return nullptr;
    if(mag(model.physics.gravity)!=0){error="coupled gas gravity source/work ledger is not enabled";return nullptr;}
    if(options.turbulenceModel!=(model.physics.enableSst?3:0)){error="coupled gas turbulence mode must match configured low-Re SST inventory";return nullptr;}
    if(gas.mode!=model.physics.gasMode){error="gas model/configuration mode mismatch";return nullptr;}
    ugkwp::GasCapabilityRequest request;request.mode=gas.mode;request.fluxScheme=options.fluxScheme;request.reconstruction=options.reconstruction;request.limiter=options.limiter;request.timeIntegrator=options.timeIntegrator;request.turbulenceModel=options.turbulenceModel;request.sstWallTreatment=0;request.movingGeometry=model.physics.meshMotion.policy!=MeshMotionPolicy::Static;request.particleCoupling=model.physics.enableParticles;
    const auto supported=ugkwp::validateGasCapabilities(request);if(!supported){error=supported.message;return nullptr;}
    std::unique_ptr<Backend> b(new Backend);b->model=model;b->options=options;b->accepted=b->trial=state;
    if(!b->storage.configure(model,gas,mechanism,state,error))return nullptr;
    auto& v=b->storage.hostView();v.gasFluxScheme=options.fluxScheme;v.gasReconstruction=options.reconstruction;v.gasLimiter=options.limiter;v.turbulenceModel=options.turbulenceModel;v.sstConfigured=options.turbulenceModel==3;
    if(!b->storage.refreshView()){error=b->storage.error();return nullptr;}
    b->gasSpecies=v.gasSpecies;b->deviceState=b->storage.deviceView();b->nCells=v.nCells;b->nFaces=v.nFaces;b->hostTurbulenceModel=options.turbulenceModel;b->hostGasFluxScheme=options.fluxScheme;b->hostGasTimeIntegrator=options.timeIntegrator;
    for(auto kind:state.gasMesh.boundaryKind)if(kind==BoundaryKind::Periodic)b->hasPeriodicFaces=1;
    error.clear();return b.release();
}
void destroyBackend(Backend* backend)noexcept{delete backend;}
bool beginGasWindow(Backend& b,const WallProgram& program,std::string& error,const GasWindowLimits& limits){
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    if(b.pending){error="gas coupling window already active";return false;}
    if(!validateGasWallProgram(program,b.accepted,error)||!validateSingleGasProgram(b.model,program,error))return false;
    b.trial=b.accepted;b.program=program;b.limits=limits;
    if(!b.history.begin(program.interval,error)||!b.reserve.reset(b.accepted.solid,b.accepted.film,b.accepted.time,error))return false;
    if(!program.donorPlan.knots.empty()&&!b.reserve.setPlan(program.donorPlan,error))return false;
    if(!restoreGasSnapshot(b,b.accepted,error))return false;b.pending=true;return true;
}
bool advanceGasMicrostep(Backend& b,Real cap,GasMicroReport& report,std::string& error){
    report={};if(b.poisoned||!b.pending||!finite(cap)||cap<=0||b.trial.time>=b.program.interval.end){error="invalid gas microstep request";report.recoverable=false;return false;}
    const std::size_t bytesPerRecord=sizeof(GasIntervalRecord)+b.trial.surface.area.size()*(2*sizeof(ExchangePacket)+sizeof(GasPrimitive)+sizeof(GasGradient)+sizeof(Vec3)+sizeof(Real)+sizeof(unsigned));
    if(bytesPerRecord>b.limits.maximumHistoryBytes||b.history.records().size()>=b.limits.maximumHistoryBytes/bytesPerRecord){error="coupled interval history byte bound reached";return false;}
    if(b.history.records().size()>=b.limits.maximumRecords){error="coupled interval history record bound reached";return false;}
    b.microTime=b.trial.time;Real dt=minValue(cap,minValue(b.model.physics.maxDt,nextGasWallKnot(b.program,b.microTime)-b.microTime));
    for(int retry=0;retry<=b.model.physics.tolerances.maxRetries;++retry){
        if(dt<b.model.physics.minDt||b.microTime+dt<=b.microTime){error="gas microstep minimum dt exhausted";return false;}
        b.microDt=dt;b.error.clear();transport::lastError.clear();
        if(transport::advanceGasTrial<transport::CoupledTrialPolicy>(&b,dt,b.microTime)!=0){
            if(b.poisoned){error=b.error;report.recoverable=false;return false;}
            ++report.rejectedTrials;dt*=.5;continue;
        }
        std::vector<GasPrimitive> primitive;std::vector<GasGradient> gradient;
        if(!readBulk(b,primitive,gradient)){error=b.storage.error();transport::CoupledTrialPolicy::rollback(&b);if(b.poisoned){error=b.error;report.recoverable=false;}return false;}
        b.record.gasTrace.clear();b.record.gasGradient.clear();b.record.gasTraction.clear();
        for(std::size_t f=0;f<b.trial.surface.area.size();++f){const int cell=b.trial.gasMesh.owner[b.trial.surface.gasFace[f]];b.record.gasTrace.push_back(primitive[cell]);b.record.gasGradient.push_back(gradient[cell]);b.record.gasTraction.push_back(b.faceRates[f].traction);}
        MaterialDonorReserve::Transaction reservation;
        if(!b.reserve.prepare(b.record,reservation,b.error)){
            transport::CoupledTrialPolicy::rollback(&b);
            if(b.poisoned){error=b.error;report.recoverable=false;return false;}
            ++report.rejectedTrials;dt*=.5;continue;
        }
        if(b.microBase.rejectedSteps>std::numeric_limits<std::uint64_t>::max()-std::uint64_t(report.rejectedTrials)){
            error="gas rejected-attempt counter overflow";report.recoverable=false;transport::CoupledTrialPolicy::rollback(&b);return false;
        }
        if(!b.history.appendAccepted(b.record,error)){transport::CoupledTrialPolicy::rollback(&b);if(b.poisoned){error=b.error;report.recoverable=false;}return false;}
        b.reserve.commitPrepared(std::move(reservation));
        b.trial.rejectedSteps=b.microBase.rejectedSteps+std::uint64_t(report.rejectedTrials);
        // Compatibility trace slots store the same certified Euler stage;
        // they do not introduce a second gas substage or midpoint update.
        b.trial.gasStages={{b.gasStage,b.gasStage}};b.trial.solidStages={{b.solidStage,b.solidStage}};
        b.trial.nextDt=dt;b.trial.lastAcceptedDt=dt;
        report.acceptedDt=dt;report.nextGasDt=dt;report.time=b.trial.time;report.microSequence=b.record.microSequence;error.clear();return true;
    }
    error=b.error.empty()?transport::lastError:b.error;if(error.empty())error="common gas trial retries exhausted";return false;
}
bool gasWindowHistory(const Backend& b,IntervalHistory& out,std::string& error){if(!b.pending){error="no active gas window";return false;}out=b.history;error.clear();return true;}
bool gasWindowStageGeometry(const Backend& b,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error){gas=b.trial.gasStages;solid=b.trial.solidStages;error.clear();return true;}
bool downloadGasWindow(const Backend& b,HostState& out,std::string& error){if(!b.pending){error="no active gas window";return false;}out=b.trial;error.clear();return true;}
bool rollbackGasWindow(Backend& b,std::string& error){
    if(b.poisoned){error="gas backend is unusable after failed device rollback";return false;}
    if(!restoreGasSnapshot(b,b.accepted,error))return false;
    b.pending=false;return true;
}
bool commitGasWindow(Backend& b,const HostState& candidate,std::string& error){if(b.poisoned||!b.pending||!b.history.complete()||candidate.time!=b.program.interval.end){error="coupled commit is not synchronized";return false;}
    if(!checkSynchronizedInterval(candidate,b.history,error)||!validateSingleGasInventory(b.model,candidate,error))return false;
    if(candidate.gas.size()!=b.trial.gas.size()||candidate.gasMesh.volumes!=b.trial.gasMesh.volumes
        ||candidate.gasMesh.topologyHash!=b.trial.gasMesh.topologyHash
        ||candidate.gasMesh.geometryVersion!=b.trial.gasMesh.geometryVersion){error="material commit changed gas endpoint geometry";return false;}
    for(std::size_t c=0;c<candidate.gas.size();++c){const auto& a=candidate.gas[c];const auto& z=b.trial.gas[c];
        if(a.mass!=z.mass||a.energy!=z.energy||a.momentum.x!=z.momentum.x||a.momentum.y!=z.momentum.y||a.momentum.z!=z.momentum.z){error="material commit changed accepted gas endpoint";return false;}
        for(int k=0;k<Ns;++k)if(a.species[k]!=z.species[k]){error="material commit changed accepted gas species";return false;}
    }
    if(candidate.sst.size()!=b.trial.sst.size()){error="material commit changed SST endpoint size";return false;}
    for(std::size_t c=0;c<candidate.sst.size();++c)
        if(candidate.sst[c].rhoK!=b.trial.sst[c].rhoK||candidate.sst[c].rhoOmega!=b.trial.sst[c].rhoOmega){error="material commit changed SST endpoint";return false;}
    // Prepare all throwing host copies before touching device/publication state.
    HostState newAccepted=candidate,newTrial=candidate;
    if(!b.storage.uploadState(candidate,error))return false;
    b.accepted=std::move(newAccepted);b.trial=std::move(newTrial);b.pending=false;error.clear();return true;
}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
}
