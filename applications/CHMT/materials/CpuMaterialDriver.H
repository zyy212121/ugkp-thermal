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
    Real maxSubstep=0,reactionMaxSubstep=0;
    PhysicalReal transportCfl=PhysicalReal(.4),reactionCfl=PhysicalReal(.1);
    PhysicalReal driveRelativeTolerance=PhysicalReal(.05),driveSourceFraction=PhysicalReal(.05),driveTractionScale=1;
    int driveSourceDiagnosticSamples=16;


    bool retainWallProgram=true;
    bool retainDonorHistory=true;
    int maxSubsteps=100000,nonlinearMaxIterations=50;
    int linearMaxIterations=300;
    PhysicalReal linearEnergyBudgetFraction=PhysicalReal(.5);
    PhysicalReal nonlinearRelativeTolerance=PhysicalReal(1e-10),energyAbsoluteTolerance=PhysicalReal(1e-10),
        energyRelativeTolerance=PhysicalReal(1e-10);
};
struct CpuMaterialReport {
    MaterialFailureOrigin failureOrigin=MaterialFailureOrigin::None;
    std::uint64_t materialSteps=0,sourceAuditSamples=0,sourceAuditPopulation=0;
    int implicitIterations=0,geometrySolves=0,materialLinearSolves=0,materialLinearIterations=0;
    bool implicitConverged=false;
    Real massResidual=0,thermalResidual=0,recessionVolume=0,gclResidual=0;
    bool recoverable=false,geometryCorrection=false;Real eventTime=0,eventDuration=0;
    MaterialDonorPlan donorPlan{};
    WallProgramComparison wallComparison{};
    bool streamedWallComparison=false;MaterialWindowCommit deviceCommit{};
    std::uint64_t exchangeFrames=0;
};



class CpuMaterialDriver {
    struct Implementation;std::unique_ptr<Implementation> data_;
    bool predictWallImpl(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport*,const PhysicsConfig&,bool holdRecession)const;
    bool advanceCandidateImpl(const HostState&,const HostState&,const IntervalHistory&,
        const CpuMaterialControls&,HostState&,WallProgram&,CpuMaterialReport&,std::string&,const WallProgram*,const PhysicsConfig&,const ResidentMaterialDriveHistory*,const std::shared_ptr<CoupledHistoryArchive>&);
public:
    CpuMaterialDriver(const ModelConfig&,Foam::fvMesh* actualSolidMesh);
    CpuMaterialDriver(const ModelConfig&,Foam::fvMesh* actualSolidMesh,const ugkwp::GasMechanismConfiguration* mechanism);
    ~CpuMaterialDriver();
    CpuMaterialDriver(const CpuMaterialDriver&)=delete;
    CpuMaterialDriver& operator=(const CpuMaterialDriver&)=delete;
    bool predictWall(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport* diagnostics=nullptr)const;
    bool predictWindowWall(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport*,bool holdRecession)const;
    bool prepareImplicitParent(const HostState&,const CouplingInterval&,const CpuMaterialControls&,
        bool deferRadiation,MaterialBackend*&,bool& appendSolve,std::string&);
    bool finishImplicitParent(const HostState&,const HostState&,const CouplingInterval&,
        const CpuMaterialControls&,const IntervalRadiationDeviceView&,bool inParent,HostState&,CpuMaterialReport&,std::string&);
    bool commitImplicitInventory(const HostState&,std::string&);
    bool prepareParentWindow(const HostState&,const WallProgram&,const ResidentMaterialDriveHistory&,
        const IntervalRadiationDeviceView&,const CpuMaterialControls&,MaterialBackend*&,std::string&);
    bool advanceCandidate(const HostState&,const HostState& geometryEndpoint,
        const IntervalHistory&,const CpuMaterialControls&,HostState& candidate,
        WallProgram& corrected,CpuMaterialReport&,std::string&,const WallProgram* executedProgram=nullptr,
        const ResidentMaterialDriveHistory* residentHistory=nullptr,const std::shared_ptr<CoupledHistoryArchive>& historyArchive={});
};
}
#endif
