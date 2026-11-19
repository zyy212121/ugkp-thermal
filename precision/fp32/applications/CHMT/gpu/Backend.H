#ifndef CHMT_GPU_BACKEND_H
#define CHMT_GPU_BACKEND_H
#include "../../../common/gpu/RadiationResidentView.H"
#include "core/HostState.H"
#include "../../../common/gpu/ParticleRadiationResponse.H"
#include "../../../common/DiagnosticsHost.H"
#include "coupling/IntervalContracts.H"
#include "../../../common/gasTransport/GasModelIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "gpu/ResidentMaterialDrive.H"
#include "gpu/MaterialLocalState.H"
#include <memory>
#include "../../../common/gpu/ExecutionGraph.H"
namespace ugkp { namespace gpu {class DiscreteOrdinateRadiation;class ParticleRadiationMaterial;}}
namespace chmt {

struct Backend;
struct MaterialBackend;
class CoupledHistoryArchive;
struct GasExecutionOptions {int fluxScheme=1,reconstruction=0,limiter=0,timeIntegrator=1,turbulenceModel=0;
    int courantUpdateInterval=1;double maximumTimeStepGrowth=1.05;
    bool adjustTimeStep=true;double fixedDeltaT=0;};
Backend* createBackend(const ModelConfig&,const ugkwp::GasModelConfiguration&,
    const ugkwp::GasMechanismConfiguration&,const GasExecutionOptions&,const HostState&,std::string&);
void destroyBackend(Backend*) noexcept;
void setDiagnosticsSink(Backend&,const ugkp::diagnostics::HostSink&);
struct GasMicroReport {
    Real acceptedDt=0,nextGasDt=0,time=0;WindowTime localTime{};
    std::uint64_t microSequence=0;
    int rejectedTrials=0;
    bool recoverable=true;
};
bool beginGasWindow(Backend&,const WallProgram&,std::string&);
bool advanceGasMicrostep(Backend&,Real,GasMicroReport&,std::string&);


struct GasWindowProgress;
struct GasWindowDeviceView {const GasWindowProgress* progress=nullptr;void* producerStream=nullptr;};
bool prepareGasWindow(Backend&,Real,std::string&);
ugkp::gpu::ExecutionGraphNode appendGasWindowBatch(Backend&,ugkp::gpu::ExecutionGraph,ugkp::gpu::ExecutionGraphNode);
GasWindowDeviceView gasWindowDeviceView(const Backend&);
bool acceptGasWindowExecution(Backend&,void* producerStream,std::string&);
bool prepareGasWindowRadiationClock(Backend&,double elapsed,const double*& deviceElapsed,std::string&);
bool bindGasWindowRadiation(Backend&,ugkp::gpu::DiscreteOrdinateRadiation&,ugkp::gpu::ParticleRadiationMaterial&,std::string&);
ResidentMaterialDriveHistory pendingGasWindowDriveHistory(const Backend&);
bool bindGasWindowMaterial(Backend&,MaterialBackend&,std::string&);
bool enqueuePreparedGasWindow(Backend&,std::string&);
bool enqueueGasWindow(Backend&,Real, std::string&);
bool collectGasWindow(Backend&,GasMicroReport&,std::string&);

const IntervalHistory& gasWindowHistory(const Backend&);
std::shared_ptr<CoupledHistoryArchive> gasWindowHistoryArchive(const Backend&);
ResidentMaterialDriveHistory residentGasWindowDriveHistory(const Backend&);
IntervalHistory releaseGasWindowHistory(Backend&);
bool gasWindowStageGeometry(Backend&,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string&);
bool downloadGasWindow(Backend&,HostState&,std::string&);
bool publishGasWindow(Backend&,HostState&,std::string&);
bool publishAcceptedState(Backend&,HostState&,std::string&);

bool commitGasWindow(Backend&,HostState&,std::string&,bool preservePublishedEndpoint=false);
bool acceptCoupledWindow(Backend&,HostState&,const MaterialWindowCommit&,Real externalHeat,Real nextDt,std::string&);
bool downloadState(const Backend&,HostState&,std::string&);
struct ParticleRadiationState {ugkp::gpu::RadiationParticleDeviceView resident{};std::vector<EnergyReal> mass,temperature,diameter,occupiedArea;ugkp::gpu::ParticleRadiationResponseView response{};};
bool downloadRadiationState(Backend&,ParticleRadiationState&,std::string&);
bool borrowRadiationState(Backend&,ParticleRadiationState&,std::string&);
bool applyResidentRadiationEnergy(Backend&,const ugkp::gpu::RadiationMaterialDeviceView&,ugkp::gpu::RadiationAcceptedEnergyDeviceView&,HostState&,std::string&);
bool applyRadiationEnergy(Backend&,const std::vector<EnergyReal>&,std::vector<EnergyReal>&,HostState&,std::string&);
void setWindowRadiationEnergy(Backend&,const std::vector<EnergyReal>&);
bool setWindowRadiationEnergy(Backend&,const ugkp::gpu::RadiationMaterialDeviceView&,std::string&);

}
#endif
