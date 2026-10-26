#ifndef CHMT_GPU_MATERIALBACKEND_H
#define CHMT_GPU_MATERIALBACKEND_H
#include "materials/MaterialTransport.H"
#include "ablation/CpuSurfaceInterface.H"
#include "../../../common/DiagnosticsHost.H"
namespace chmt {
struct CpuMaterialControls;
struct MaterialBackend;
MaterialBackend* createMaterialBackend(std::string&);
void destroyMaterialBackend(MaterialBackend*) noexcept;
void setMaterialDiagnosticsSink(MaterialBackend&,const ugkp::diagnostics::HostSink&);
// Upload the accepted inventory once. Subsequent source steps retain both
// current/candidate inventories, transport work and donor changes on device.
bool beginMaterialWindow(MaterialBackend&,const HostState&,const PhysicsConfig&,std::string&);
bool uploadMaterialConductionEndpoint(MaterialBackend&,const HostState&,std::string&);
bool evaluateResidentMaterialTransport(MaterialBackend&,const HostMesh&,
    const std::vector<Real>& sweeps,Real interval,Real cfl,Real& dtLimit,
    std::string&,bool& numericalFailure);
bool evaluateResidentMaterialSurface(MaterialBackend&,const HostState& geometry,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    Real dt,bool useTransportRates,bool candidateState,CpuSurfaceResult&,
    std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,
    const std::vector<GasWallClosureContext>* contexts=nullptr,
    bool* terminalWallFailure=nullptr);
// Independent history samples share the same resident material state. Batching
// changes only execution scheduling; each face retains its constitutive solve.
struct MaterialSurfaceRequest {
    const std::vector<GasPrimitive>* bulk=nullptr;
    const std::vector<GasGradient>* gradients=nullptr;
    const std::vector<GasWallClosureContext>* contexts=nullptr;
};
bool evaluateResidentMaterialSurfaces(MaterialBackend&,const HostState&,
    const std::vector<MaterialSurfaceRequest>&,Real dt,bool useTransportRates,
    bool candidateState,std::vector<CpuSurfaceResult>&,std::string&,bool& numericalFailure,
    const std::vector<Real>* sideVolume=nullptr,bool* terminalWallFailure=nullptr);
// The host mirror is retained for the not-yet-migrated geometry and implicit
// conduction owners; it never feeds the next reaction/transport stage back.
bool advanceResidentMaterialSources(MaterialBackend&,const HostMesh& endpointMesh,
    const SurfaceMesh&,const std::vector<ExchangePacket>&,
    const std::vector<Real>& radiation,const std::vector<unsigned>& radiationReceiver,
    Real dt,const CpuMaterialControls&,bool finalStep,
    std::vector<SolidQ>& hostMirror,std::vector<SolidQ>& cumulativeDonor,
    std::string&,bool& numericalFailure);
void commitResidentMaterialStep(MaterialBackend&,Real time);
}
#endif
