#ifndef CHMT_GPU_BACKEND_H
#define CHMT_GPU_BACKEND_H
#include "../../../common/gpu/RadiationResidentView.H"
#include "core/HostState.H"
#include "../../../common/gpu/ParticleRadiationResponse.H"
#include "../../../common/DiagnosticsHost.H"
#include "coupling/IntervalContracts.H"
#include "../../../common/gasTransport/GasModelIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
namespace chmt {
// Gas-window storage/scheduling boundary. All gas numerical stages are common.
struct Backend;
struct GasExecutionOptions {int fluxScheme=1,reconstruction=0,limiter=0,timeIntegrator=1,turbulenceModel=0;
    int courantUpdateInterval=1;double maximumTimeStepGrowth=1.05;};
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
const IntervalHistory& gasWindowHistory(const Backend&);
std::vector<GasIntervalRecord> releaseGasWindowRecords(Backend&); // after material consumers finish
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string&);
bool downloadGasWindow(Backend&,HostState&,std::string&);
bool commitGasWindow(Backend&,HostState&,std::string&);
bool downloadState(const Backend&,HostState&,std::string&);
struct ParticleRadiationState {ugkp::gpu::RadiationParticleDeviceView resident{};std::vector<Real> mass,temperature,diameter,occupiedArea;ugkp::gpu::ParticleRadiationResponseView response{};};
bool downloadRadiationState(Backend&,ParticleRadiationState&,std::string&);
bool borrowRadiationState(Backend&,ParticleRadiationState&,std::string&);
bool applyResidentRadiationEnergy(Backend&,const ugkp::gpu::RadiationMaterialDeviceView&,std::vector<Real>&,HostState&,std::string&);
bool applyRadiationEnergy(Backend&,const std::vector<Real>&,std::vector<Real>&,HostState&,std::string&);
void setWindowRadiationEnergy(Backend&,const std::vector<Real>&);

}
#endif
