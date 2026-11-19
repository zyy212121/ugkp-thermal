#ifndef CHMT_GPU_MATERIALBACKEND_H
#define CHMT_GPU_MATERIALBACKEND_H
#include "../../../common/gpu/ScaledThermalSum.H"
#include "materials/MaterialTransport.H"
#include "coupling/IntervalContracts.H"
#include "gpu/MaterialThermalBackend.H"
#include "ablation/CpuSurfaceInterface.H"
#include "../../../common/DiagnosticsHost.H"
#include "../../../common/gpu/ExecutionGraph.H"
#include <future>
#include "../../../common/GpuPrecisionTypes.H"
#include <memory>
#include "gpu/MaterialLocalAudit.H"
#include "gpu/MaterialLocalState.H"
namespace ugkp { namespace gpu { class ImplicitDiffusion;class SparseConjugateGradient; }}
namespace chmt {
struct CpuMaterialControls;
struct ResidentMaterialDriveHistory;
class CoupledHistoryArchive;
struct StreamedMaterialProgress;
struct MaterialBackend;
MaterialBackend* createMaterialBackend(std::string&);
void destroyMaterialBackend(MaterialBackend*) noexcept;
void setMaterialDiagnosticsSink(MaterialBackend&,const ugkp::diagnostics::HostSink&);



bool beginMaterialWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,std::string&,bool collectDiagnostics=true);




struct ResidentMaterialState;
struct MaterialLocalProgress;
struct MaterialLocalDeviceView {
    const ResidentMaterialState* state=nullptr;
    const MaterialLocalProgress* report=nullptr;
    void* producerStream=nullptr;
    const StreamedMaterialProgress* storage=nullptr;


    const int* windowStage=nullptr;
    const ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    const MaterialWindowCommit* commit=nullptr;
};
struct MaterialLocalReport {
    std::uint64_t acceptedSteps=0;
    int failurePhase=0,failureCode=0,entity=-1;
    bool complete=false;Real time=0;MaterialWindowCommit commit{};
};


bool preparePendingResidentMaterialLocalWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,
    const ResidentMaterialDriveHistory&,const IntervalRadiationDeviceView&,const WallProgram&,
    const CpuMaterialControls&,std::string&);
bool serviceResidentMaterialLocalCapacity(MaterialBackend&,std::string&);


bool completePendingMaterialHistory(MaterialBackend&,const IntervalHistory&,
    const std::shared_ptr<CoupledHistoryArchive>&,std::string&);
bool releaseUnusedMaterialLocalWindow(MaterialBackend&,std::string&);
bool prepareResidentMaterialLocalWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,
    const IntervalHistory&,const ResidentMaterialDriveHistory&,const WallProgram&,
    const CpuMaterialControls&,std::string&);
bool prepareStreamedMaterialLocalWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,
    const IntervalHistory&,const std::shared_ptr<CoupledHistoryArchive>&,const WallProgram&,
    const CpuMaterialControls&,std::string&);


bool supplyResidentMaterialHistory(MaterialBackend&,std::size_t frame,std::string&);
ugkp::gpu::ExecutionGraphNode appendResidentMaterialLocalWindow(MaterialBackend&,
    ugkp::gpu::ExecutionGraph,ugkp::gpu::ExecutionGraphNode);
bool enqueueResidentMaterialLocalWindow(MaterialBackend&,std::string&);
void acceptResidentMaterialLocalCompletion(MaterialBackend&,const std::shared_future<void>&);
void acceptResidentMaterialLocalExecution(MaterialBackend&,void*);
MaterialLocalDeviceView residentMaterialLocalDeviceView(const MaterialBackend&);
bool collectResidentMaterialLocalWindow(MaterialBackend&,MaterialLocalReport&,std::vector<FilmAux>&,std::string&);


bool prepareMaterialLocalConduction(MaterialBackend&,ugkp::gpu::ImplicitDiffusion&,
    ugkp::gpu::SparseConjugateGradient&,const std::vector<int>& externalBoundary,Real dt,
    const CpuMaterialControls&,Real normalizationSmall,Real singularSmall,std::string&);
bool collectResidentMaterialInventory(MaterialBackend&,std::vector<SolidQ>&,std::string&);
bool collectResidentMaterialLocalAudit(MaterialBackend&,std::vector<MaterialLocalAudit>&,std::string&);


bool prepareResidentMaterialEndpointSurface(MaterialBackend&,const HostState&,std::string&);
bool bindResidentMaterialEndpointGasFraction(MaterialBackend&,const GpuReal*,void*,std::string&);
bool collectResidentMaterialEndpointSurface(MaterialBackend&,CpuSurfaceResult&,bool& published,std::string&);

bool beginResidentMaterialHistory(MaterialBackend&,const ResidentMaterialDriveHistory*,
    const IntervalHistory&,std::string&);
bool prepareResidentMaterialConduction(MaterialBackend&,const HostMesh&,
    const std::vector<Real>& nativeVolumes,std::vector<MaterialThermalInput>&,
    std::string&,bool& numericalFailure,bool publishInput=true);
bool publishResidentMaterialConductionInput(MaterialBackend&,std::vector<MaterialThermalInput>&,std::string&);
bool evaluateResidentMaterialConduction(MaterialBackend&,const std::vector<Real>& temperature,
    Real dt,std::vector<MaterialThermalCoefficients>&,std::string&,
    const Real* deviceTemperature=nullptr,void* temperatureStream=nullptr,bool download=true,bool startupDiagnostics=false);
struct MaterialConductionDeviceFields {
    const Real* capacity=nullptr;
    const Real* source=nullptr;
    const Real* conductivity=nullptr;
    void* readyStream=nullptr;
    const Real* initialTemperature=nullptr;
    std::size_t cells=0;
};
MaterialConductionDeviceFields residentMaterialConductionFields(MaterialBackend&);
struct MaterialConductionAudit {
    ugkp::gpu::ScaledThermalSum residualEnergy,inventory,stableChange,boundaryEnergy;
    Real change=0,invalidTemperature=0;
    int coefficientError=0,invalidCell=-1;
};
bool auditResidentMaterialConduction(MaterialBackend&,const Real* temperature,const Real* previous,
    const Real* powerResidual,const Real* boundaryFlux,const std::vector<int>& externalBoundary,Real dt,
    void* producerStream,MaterialConductionAudit&,std::string&);


bool solveResidentMaterialConduction(MaterialBackend&,ugkp::gpu::ImplicitDiffusion&,ugkp::gpu::SparseConjugateGradient&,
    const std::vector<int>& externalBoundary,Real dt,const CpuMaterialControls&,Real normalizationSmall,Real singularSmall,
    std::string&);
bool finishResidentMaterialConduction(MaterialBackend&,const std::vector<Real>& temperature,
    std::vector<SolidQ>& hostMirror,std::string&,bool& numericalFailure,
    const Real* deviceTemperature=nullptr,void* temperatureStream=nullptr);
bool evaluateResidentMaterialTransport(MaterialBackend&,const HostMesh&,
    const std::vector<Real>& sweeps,Real interval,Real cfl,Real& dtLimit,
    std::string&,bool& numericalFailure);
bool evaluateResidentMaterialSurface(MaterialBackend&,const HostState& geometry,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    Real dt,bool useTransportRates,bool candidateState,CpuSurfaceResult&,
    std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,
    const std::vector<GasWallClosureContext>* contexts=nullptr,
    bool* terminalWallFailure=nullptr,
    const std::vector<WallFaceSample>* acceptedRates=nullptr,int localAudit=-1);


struct MaterialSurfaceRequest {
    const std::vector<GasPrimitive>* bulk=nullptr;
    const std::vector<GasGradient>* gradients=nullptr;
    const std::vector<GasWallClosureContext>* contexts=nullptr;
    const std::vector<WallFaceSample>* acceptedRates=nullptr;
};
bool evaluateResidentMaterialSurfaces(MaterialBackend&,const HostState&,
    const std::vector<MaterialSurfaceRequest>&,Real dt,bool useTransportRates,
    bool candidateState,std::vector<CpuSurfaceResult>&,std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,bool* terminalWallFailure=nullptr,int localAudit=-1);



bool advanceResidentMaterialSources(MaterialBackend&,const HostMesh& endpointMesh,
    const SurfaceMesh&,const std::vector<ExchangePacket>&,
    const std::vector<Real>& radiation,const std::vector<unsigned>& radiationReceiver,
    Real dt,const CpuMaterialControls&,bool finalStep,
    std::vector<SolidQ>& hostMirror,std::vector<SolidQ>& cumulativeDonor,
    std::string&,bool& numericalFailure,const std::vector<int>* observedCells=nullptr,
    const IntervalSliceMetadata* selectedSlice=nullptr);
bool commitResidentMaterialStep(MaterialBackend&,Real time,std::string&);

void commitResidentMaterialStep(MaterialBackend&,Real time);
}
#endif
