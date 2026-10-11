#ifndef CHMT_MATERIALS_CPUMATERIALDRIVER_H
#define CHMT_MATERIALS_CPUMATERIALDRIVER_H
#include "core/HostState.H"
#include "coupling/IntervalContracts.H"
#include <memory>
namespace Foam {class fvMesh;}
namespace chmt {
struct CpuMaterialControls {
    Real maxSubstep=0,transportCfl=.4,reactionCfl=.1,reactionMaxSubstep=0;
    Real driveRelativeTolerance=.05,driveSourceFraction=.05,driveTractionScale=1;
    int maxSubsteps=100000,nonlinearMaxIterations=50;
    Real nonlinearRelativeTolerance=1e-10,energyAbsoluteTolerance=1e-10,energyRelativeTolerance=1e-10;
};
struct CpuMaterialReport {
    std::uint64_t boundaryDriveSamples=0,materialRhsEvaluations=0;
    std::uint64_t materialSteps=0,reactionSteps=0,filmSteps=0,linearSolves=0;
    bool recoverable=true,geometryCorrection=false;Real eventTime=0,numericalMassRoundoff=0;
    int nonlinearIterations=0;Real energyResidual=0,minimumTemperature=std::numeric_limits<Real>::max();
    Budget budgetDelta{};MaterialDonorPlan donorPlan{};
    std::vector<Real> solidTargetSweeps{};
};
// Owns CPU numerical machinery, not a second copy of accepted inventories.
// Candidate solve never mutates the accepted native mesh: it builds an unregistered full-topology fvMesh for the real sparse solve.
// Successful output merges exact gasEndpoint inventories/budget with CPU changes once.
class CpuMaterialDriver {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    CpuMaterialDriver(const ModelConfig&,Foam::fvMesh* actualSolidMesh);
    ~CpuMaterialDriver();
    CpuMaterialDriver(const CpuMaterialDriver&)=delete;
    CpuMaterialDriver& operator=(const CpuMaterialDriver&)=delete;
    bool predictWall(const HostState&,const CouplingInterval&,WallProgram&,std::string&,CpuMaterialReport* diagnostics=nullptr)const;
    bool advanceCandidate(const HostState&,const HostState& geometryEndpoint,
        const IntervalHistory&,const CpuMaterialControls&,HostState& candidate,
        WallProgram& corrected,CpuMaterialReport&,std::string&,const WallProgram* executedProgram=nullptr);
};
}
#endif
