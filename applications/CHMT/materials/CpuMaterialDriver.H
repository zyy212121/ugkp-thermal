#ifndef CHMT_MATERIALS_CPUMATERIALDRIVER_H
#define CHMT_MATERIALS_CPUMATERIALDRIVER_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
#include "coupling/WallProgramComparison.H"
#include <memory>
#include "gpu/MaterialLocalState.H"
#include "../../../common/gpu/SparseConjugateGradient.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
namespace Foam {class fvMesh;}
namespace chmt {
struct ResidentMaterialDriveHistory;
struct MaterialBackend;
class CoupledHistoryArchive;
enum class MaterialFailureOrigin { None,ConductionNumerical,ReactionNumerical,FilmNumerical,SurfaceNumerical,GeometryNumerical,TransportNumerical,OwnerNumerical,LocalSourceNumerical };
struct CpuMaterialControls {
    Real maxSubstep=0,transportCfl=.4,reactionCfl=.1,reactionMaxSubstep=0;
    Real driveRelativeTolerance=.05,driveSourceFraction=.05,driveTractionScale=1;
    int driveSourceDiagnosticSamples=16;


    bool retainWallProgram=true;
    bool retainDonorHistory=true;
    int maxSubsteps=100000,nonlinearMaxIterations=50;
    int linearMaxIterations=300;
    Real linearEnergyBudgetFraction=Real(.5);
    Real nonlinearRelativeTolerance=1e-10,energyAbsoluteTolerance=1e-10,energyRelativeTolerance=1e-10;
};
struct CpuMaterialReport {
    MaterialFailureOrigin failureOrigin=MaterialFailureOrigin::None;
    std::uint64_t materialSteps=0,sourceAuditSamples=0,sourceAuditPopulation=0;
    bool recoverable=false,geometryCorrection=false;Real eventTime=0,eventDuration=0;
    MaterialDonorPlan donorPlan{};
    WallProgramComparison wallComparison{};
    bool streamedWallComparison=false;MaterialWindowCommit deviceCommit{};
};



class CpuMaterialDriver {
    struct Implementation;std::unique_ptr<Implementation> data_;
    bool predictWallImpl(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport*,const PhysicsConfig&)const;
    bool advanceCandidateImpl(const HostState&,const HostState&,const IntervalHistory&,
        const CpuMaterialControls&,HostState&,WallProgram&,CpuMaterialReport&,std::string&,const WallProgram*,const PhysicsConfig&,const ResidentMaterialDriveHistory*,const std::shared_ptr<CoupledHistoryArchive>&);
public:
    CpuMaterialDriver(const ModelConfig&,Foam::fvMesh* actualSolidMesh);
    CpuMaterialDriver(const ModelConfig&,Foam::fvMesh* actualSolidMesh,const ugkwp::GasMechanismConfiguration* mechanism);
    ~CpuMaterialDriver();
    CpuMaterialDriver(const CpuMaterialDriver&)=delete;
    CpuMaterialDriver& operator=(const CpuMaterialDriver&)=delete;
    bool predictWall(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport* diagnostics=nullptr)const;
    bool prepareParentWindow(const HostState&,const WallProgram&,const ResidentMaterialDriveHistory&,
        const IntervalRadiationDeviceView&,const CpuMaterialControls&,MaterialBackend*&,std::string&);
    bool advanceCandidate(const HostState&,const HostState& geometryEndpoint,
        const IntervalHistory&,const CpuMaterialControls&,HostState& candidate,
        WallProgram& corrected,CpuMaterialReport&,std::string&,const WallProgram* executedProgram=nullptr,
        const ResidentMaterialDriveHistory* residentHistory=nullptr,const std::shared_ptr<CoupledHistoryArchive>& historyArchive={});
};
}
#endif
