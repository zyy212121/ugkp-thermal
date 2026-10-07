#ifndef CHMT_GPU_BACKEND_H
#define CHMT_GPU_BACKEND_H
#include "core/HostState.H"
#include "mesh/Motion.H"
#include "DeviceViews.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
struct Backend;
bool remapGas1D(const std::vector<Real>& oldEdges,const std::vector<GasQ>& oldQ,
    const std::vector<Real>& newEdges,std::vector<GasQ>& newQ,std::string& error);
Backend* createBackend(const ModelConfig&,const HostState&,std::string& error);
// Legacy all-explicit reference path. The multirate controller uses the window API.
bool advance(Backend&,double requestedDt,StepReport&,std::string& error);
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
bool beginGasWindow(Backend&,const WallProgram&,std::string& error,const GasWindowLimits& limits=GasWindowLimits{});
bool advanceGasMicrostep(Backend&,Real gasMaxDt,GasMicroReport&,std::string& error);
bool gasWindowHistory(const Backend&,IntervalHistory&,std::string& error);
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error);
bool downloadGasWindow(const Backend&,HostState&,std::string& error);
bool rollbackGasWindow(Backend&,std::string& error);
// Caller supplies the jointly validated material candidate. No gas delta is reapplied.
bool commitGasWindow(Backend&,const HostState&,std::string& error);
bool downloadState(const Backend&,HostState&,std::string& error);
bool restoreState(Backend&,const HostState&,std::string& error);
void destroyBackend(Backend*) noexcept;
std::string backendBuildInfo();
bool writeCheckpoint(const std::string& directory,const ModelConfig&,const HostState&,std::string& error);
bool readCheckpoint(const std::string& directory,const ModelConfig&,HostState&,std::string& error);

int launchGasRhs(GasView,GeometryView,PacketView,const PhysicsConfig&,double stageDt,void* stream);
int launchGasUpdate(const GasQ* base,GasView candidate,const GeometryView&,double dt,DeviceStatus*,void* stream);
int launchConservativeRemap1D(const double* oldEdges,int nOld,const GasQ* oldQ,const double* newEdges,int nNew,GasQ* newQ,DeviceStatus*,void* stream);
int launchMaterialRhs(SolidView,GeometryView,PacketView,const PhysicsConfig&,double stageDt,DeviceStatus*,void* stream);
int launchFilmRhs(FilmView,SurfaceView,PacketView,const PhysicsConfig&,double stageDt,DeviceStatus*,void* stream);
int launchInterfacePackets(GasView,SolidView,FilmView,GeometryView,SurfaceView,PacketView,const PhysicsConfig&,double stageDt,std::uint64_t step,int stage,DeviceStatus*,void* stream);
int launchResolvedNormalThermalRhs(NormalThermalView,const PhysicsConfig&,double stageDt,DeviceStatus*,void* stream);
} // namespace chmt
#endif
