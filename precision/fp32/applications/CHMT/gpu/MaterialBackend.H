#ifndef CHMT_GPU_MATERIALBACKEND_H
#define CHMT_GPU_MATERIALBACKEND_H
#include "../../../common/gpu/ScaledThermalSum.H"
#include "materials/MaterialTransport.H"
#include "gpu/MaterialThermalBackend.H"
#include "ablation/CpuSurfaceInterface.H"
#include "../../../common/DiagnosticsHost.H"
namespace ugkp { namespace gpu { class ImplicitDiffusion;class SparseConjugateGradient; }}
namespace chmt {
struct CpuMaterialControls;
struct MaterialBackend;
MaterialBackend* createMaterialBackend(std::string&);
void destroyMaterialBackend(MaterialBackend*) noexcept;
void setMaterialDiagnosticsSink(MaterialBackend&,const ugkp::diagnostics::HostSink&);
// Accepted material inventories remain resident across windows. A supplied
// state differing from the last full publication is explicitly re-imported;
// unchanged window handoffs retain the device inventory and workspaces.
bool beginMaterialWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,std::string&);
bool prepareResidentMaterialConduction(MaterialBackend&,const HostMesh&,
    const std::vector<Real>& nativeVolumes,std::vector<MaterialThermalInput>&,
    std::string&,bool& numericalFailure);
bool evaluateResidentMaterialConduction(MaterialBackend&,const std::vector<Real>& temperature,
    Real dt,std::vector<MaterialThermalCoefficients>&,std::string&,
    const ConductionReal* deviceTemperature=nullptr,void* temperatureStream=nullptr,bool download=true);
struct MaterialConductionDeviceFields {
    const ConductionReal* capacity=nullptr;
    const ConductionReal* source=nullptr;
    const ConductionReal* conductivity=nullptr;
    void* readyStream=nullptr;
    const ConductionReal* initialTemperature=nullptr;
};
MaterialConductionDeviceFields residentMaterialConductionFields(MaterialBackend&);
struct MaterialConductionAudit {
    ugkp::gpu::ScaledThermalSum residualEnergy,inventory,stableChange,boundaryEnergy;
    ConductionReal change=0,invalidTemperature=0;
    int coefficientError=0,invalidCell=-1;
};
bool auditResidentMaterialConduction(MaterialBackend&,const ConductionReal* temperature,const ConductionReal* previous,
    const ConductionReal* powerResidual,const ConductionReal* boundaryFlux,const std::vector<int>& externalBoundary,Real dt,
    void* producerStream,MaterialConductionAudit&,std::string&);
// The configured local discretization, physical error budget and nonlinear
// stopping rule execute as one nested device graph; publication is at its end.
bool solveResidentMaterialConduction(MaterialBackend&,ugkp::gpu::ImplicitDiffusion&,ugkp::gpu::SparseConjugateGradient&,
    const std::vector<int>& externalBoundary,Real dt,const CpuMaterialControls&,Real normalizationSmall,Real singularSmall,
    std::string&);
bool finishResidentMaterialConduction(MaterialBackend&,const std::vector<Real>& temperature,
    std::vector<SolidQ>& hostMirror,std::string&,bool& numericalFailure,
    const ConductionReal* deviceTemperature=nullptr,void* temperatureStream=nullptr);
bool evaluateResidentMaterialTransport(MaterialBackend&,const HostMesh&,
    const std::vector<Real>& sweeps,Real interval,PhysicalReal cfl,Real& dtLimit,
    std::string&,bool& numericalFailure);
bool evaluateResidentMaterialSurface(MaterialBackend&,const HostState& geometry,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    Real dt,bool useTransportRates,bool candidateState,CpuSurfaceResult&,
    std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,
    const std::vector<GasWallClosureContext>* contexts=nullptr,
    bool* terminalWallFailure=nullptr,
    const std::vector<WallFaceSample>* acceptedRates=nullptr);
// Independent history samples share the same resident material state. Batching
// changes only execution scheduling; each face retains its constitutive solve.
struct MaterialSurfaceRequest {
    const std::vector<GasPrimitive>* bulk=nullptr;
    const std::vector<GasGradient>* gradients=nullptr;
    const std::vector<GasWallClosureContext>* contexts=nullptr;
    const std::vector<WallFaceSample>* acceptedRates=nullptr;
};
bool evaluateResidentMaterialSurfaces(MaterialBackend&,const HostState&,
    const std::vector<MaterialSurfaceRequest>&,Real dt,bool useTransportRates,
    bool candidateState,std::vector<CpuSurfaceResult>&,std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,bool* terminalWallFailure=nullptr);
// The optional observation selects only material cells read by the geometry
// owner. In that mode hostMirror is packed in observedCells order; the full
// authoritative inventory stays resident through local source steps.
bool advanceResidentMaterialSources(MaterialBackend&,const HostMesh& endpointMesh,
    const SurfaceMesh&,const std::vector<ExchangePacket>&,
    const std::vector<EnergyReal>& radiation,const std::vector<unsigned>& radiationReceiver,
    Real dt,const CpuMaterialControls&,bool finalStep,
    std::vector<SolidQ>& hostMirror,std::vector<SolidQ>& cumulativeDonor,
    std::string&,bool& numericalFailure,const std::vector<int>* observedCells=nullptr);
void commitResidentMaterialStep(MaterialBackend&,Real time);
}
#endif
