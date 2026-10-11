#ifndef CHMT_IO_MULTIRATEEVOLUTION_H
#define CHMT_IO_MULTIRATEEVOLUTION_H
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt {
struct Backend;
struct MicrostepGeometryReport {
    std::uint64_t microSequence=0;
    Real begin=0,end=0;
    std::array<HostStageGeometry,2> gas{},solid{};
};
struct WindowEvolutionReport {
    CouplingInterval interval{};
    std::uint64_t gasMicrosteps=0,acceptedGasMicrosteps=0,materialSolves=0,materialSubsteps=0,reactionSubsteps=0,filmSubsteps=0,linearSolves=0;
    std::uint64_t boundaryDriveSamples=0,materialRhsEvaluations=0;
    int windowIterations=0,windowRetries=0,gasRejectedTrials=0;
    Real nextGasDt=0,nextCouplingInterval=0;
    Real temperaturePredictionError=0,massPredictionError=0,geometryError=0,energyResidual=0;
    Real numericalMassRoundoff=0; // Committed CPU candidate's signed kg diagnostic, never an extra source.
    std::vector<MicrostepGeometryReport> geometry{};
};
// Advance at most one macro interval toward targetTime. The accepted state is
// immutable until gas and CPU material have both passed the common audit. A
// failed/replayed window never becomes output or restart authority.
bool advanceCoupledWindow(Backend&,CpuMaterialDriver&,const ModelConfig&,
    const CouplingControls&,const CpuMaterialControls&,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport&,std::string& error);
}
#endif
