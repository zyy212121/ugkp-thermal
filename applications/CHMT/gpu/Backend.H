#ifndef CHMT_GPU_BACKEND_H
#define CHMT_GPU_BACKEND_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
#include "../../../common/gasTransport/GasModelIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
namespace chmt {
// Gas-window storage/scheduling boundary. All gas numerical stages are common.
struct Backend;
struct GasExecutionOptions {int fluxScheme=1,reconstruction=0,limiter=0,timeIntegrator=1,turbulenceModel=0;};
Backend* createBackend(const ModelConfig&,const ugkwp::GasModelConfiguration&,
    const ugkwp::GasMechanismConfiguration&,const GasExecutionOptions&,const HostState&,std::string&);
void destroyBackend(Backend*) noexcept;
struct GasMicroReport {
    Real acceptedDt=0,nextGasDt=0,time=0;
    std::uint64_t microSequence=0;
    int rejectedTrials=0;
    bool recoverable=true;
};
struct GasWindowLimits {
    std::size_t maximumRecords=16384;
    std::size_t maximumHistoryBytes=256u*1024u*1024u;
};
bool beginGasWindow(Backend&,const WallProgram&,std::string&,const GasWindowLimits& = GasWindowLimits{});
bool advanceGasMicrostep(Backend&,Real,GasMicroReport&,std::string&);
bool gasWindowHistory(const Backend&,IntervalHistory&,std::string&);
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string&);
bool downloadGasWindow(const Backend&,HostState&,std::string&);
bool rollbackGasWindow(Backend&,std::string&);
bool commitGasWindow(Backend&,const HostState&,std::string&);
bool downloadState(const Backend&,HostState&,std::string&);
}
#endif
