#ifndef CHMT_CONFIGURATION_SOLVERCONTROLS_H
#define CHMT_CONFIGURATION_SOLVERCONTROLS_H
#include "core/Configuration.H"
#include "materials/CpuMaterialDriver.H"
#include "../../../common/FoamSolverScheme.H"
#include "../../../common/FoamParticleSolverControls.H"
#include "../../../common/FoamWallSolverControls.H"
#include "../../../common/gasNumerics/FoamGasScalarSolverControls.H"
namespace chmt { namespace io {
inline void readModelSolverControls(ModelConfig& model,const Foam::Time& time,
    const Foam::word& gasRegion,const Foam::word& solidRegion) {
    const auto& scheme=ugkp::solverControls::startupScheme(time);
    auto& p=model.physics;
    scheme.read("gasTimeStep","minDeltaT",p.minDt,"default",gasRegion);
    scheme.read("gasTimeStep","maxDeltaT",p.maxDt,"default",gasRegion);
    scheme.read("gasTimeStep","maxCo",p.cfl,"default",gasRegion);
    scheme.read("chmt.tolerances","absoluteMass",p.tolerances.absoluteMass,"default",solidRegion);
    scheme.read("chmt.tolerances","relativeMass",p.tolerances.relativeMass,"default",solidRegion);
    scheme.read("chmt.tolerances","absoluteEnergy",p.tolerances.absoluteEnergy,"default",solidRegion);
    scheme.read("chmt.tolerances","relativeEnergy",p.tolerances.relativeEnergy,"default",solidRegion);
    scheme.read("chmt.tolerances","absoluteTemperature",p.tolerances.absoluteTemperature,"default",solidRegion);
    scheme.read("chmt.tolerances","relativeTemperature",p.tolerances.relativeTemperature,"default",solidRegion);
    scheme.read("chmt.tolerances","absoluteGeometry",p.tolerances.absoluteGeometry,"default",solidRegion);
    scheme.read("chmt.tolerances","relativeGeometry",p.tolerances.relativeGeometry,"default",solidRegion);
    scheme.read("chmt.material.thermo","maximumIterations",p.numerics.thermoMaximumIterations,"default",solidRegion);
    scheme.read("chmt.material.reconstruction","maximumIterations",p.numerics.reconstructionMaximumIterations,"default",solidRegion);
    scheme.read("chmt.material.reconstruction","spatialOrder",p.spatialOrder,"default",solidRegion);
    Foam::word reconstruction("LimitedLinear");
    scheme.read("chmt.material.reconstruction","method",reconstruction,"default",solidRegion);
    p.reconstruction=reconstruction=="SmoothVerification"?ReconstructionMode::SmoothVerification:ReconstructionMode::LimitedLinear;
    auto& surface=p.numerics.surface;
    scheme.read("chmt.surface.composition","maximumIterations",surface.compositionMaximumIterations,"default",solidRegion);
    scheme.read("chmt.surface.composition","relaxation",surface.compositionRelaxation,"default",solidRegion);
    scheme.read("chmt.surface.composition","absoluteChangeTolerance",surface.compositionAbsoluteTolerance,"default",solidRegion);
    scheme.read("chmt.surface.composition","polynomialBisectionIterations",surface.polynomialBisectionIterations,"default",solidRegion);
    scheme.read("chmt.surface.composition","isolatedRootIterations",surface.isolatedRootIterations,"default",solidRegion);
    scheme.read("chmt.surface.composition","refinementIterations",surface.refinementIterations,"default",solidRegion);
    scheme.read("chmt.surface.composition","structuredNewtonIterations",surface.structuredNewtonIterations,"default",solidRegion);
    scheme.read("chmt.surface.composition","poleSearchIterations",surface.poleSearchIterations,"default",solidRegion);
    scheme.read("chmt.surface.thermal","maximumIterations",surface.thermalMaximumIterations,"default",solidRegion);
    scheme.read("chmt.surface.thermal","initialSubdivisions",surface.initialSubdivisions,"default",solidRegion);
    scheme.read("chmt.surface.thermal","maximumSubdivisions",surface.maximumSubdivisions,"default",solidRegion);
    scheme.read("chmt.surface.thermal","subdivisionGrowth",surface.subdivisionGrowth,"default",solidRegion);
    auto& motion=p.numerics.meshMotion;
    scheme.read("chmt.meshMotion","absoluteVolumeTolerance",motion.absoluteVolumeTolerance,"default",solidRegion);
    scheme.read("chmt.meshMotion","relativeVolumeTolerance",motion.relativeVolumeTolerance,"default",solidRegion);
    scheme.read("chmt.meshMotion","rankTolerance",motion.rankTolerance,"default",solidRegion);
    scheme.read("chmt.meshMotion","maximumIterations",motion.maximumIterations,"default",solidRegion);
    scheme.read("chmt.meshMotion","maximumLineSearch",motion.maximumLineSearch,"default",solidRegion);
    scheme.read("chmt.meshMotion","harmonicMaximumIterations",motion.harmonicMaximumIterations,"default",solidRegion);
    scheme.read("chmt.meshMotion","harmonicRelativeTolerance",motion.harmonicRelativeTolerance,"default",solidRegion);
    model.particleNumerics=ugkp::solverControls::readParticleNumericalControls(time,gasRegion);
    model.particlePhase.maxFaceWalkHops=ugkp::solverControls::readParticleTrackingMaxFaceWalkHops(time,gasRegion);
    // Preserve the existing particle checkpoint identity for a migrated same-value case.
    model.particlePhase.packingIterations=model.particleNumerics.packingIterations;
    scheme.read("coldWall1D","solidificationIterations",model.particleContact.nonlinearIterations,"default",gasRegion);
    ugkp::solverControls::readWallLayerSolverControls(p.wallModel,time,gasRegion);
    p.gasScalarSolvers=ugkp::gasNumerics::readGasScalarSolverControls<decltype(surface.compositionRelaxation)>(scheme,gasRegion);
}
inline CpuMaterialControls readMaterialSolverControls(const Foam::Time& time,const Foam::word& region) {
    CpuMaterialControls controls;
    const auto& scheme=ugkp::solverControls::startupScheme(time);
    scheme.read("chmt.material","maxSubstep",controls.maxSubstep,"default",region);
    scheme.read("chmt.material","transportCfl",controls.transportCfl,"default",region);
    scheme.read("chmt.material","reactionCfl",controls.reactionCfl,"default",region);
    scheme.read("chmt.material","reactionMaxSubstep",controls.reactionMaxSubstep,"default",region);
    scheme.read("chmt.material","driveRelativeTolerance",controls.driveRelativeTolerance,"default",region);
    scheme.read("chmt.material","driveSourceFraction",controls.driveSourceFraction,"default",region);
    scheme.read("chmt.material","driveTractionScale",controls.driveTractionScale,"default",region);
    scheme.read("chmt.material","maxSubsteps",controls.maxSubsteps,"default",region);
    scheme.read("chmt.material.conduction","nonlinearMaxIterations",controls.nonlinearMaxIterations,"default",region);
    scheme.read("chmt.material.conduction","nonlinearRelativeTolerance",controls.nonlinearRelativeTolerance,"default",region);
    scheme.read("chmt.material.conduction","energyAbsoluteTolerance",controls.energyAbsoluteTolerance,"default",region);
    scheme.read("chmt.material.conduction","energyRelativeTolerance",controls.energyRelativeTolerance,"default",region);
    scheme.read("chmt.material.conduction","linearMaxIterations",controls.linearMaxIterations,"default",region);
    scheme.read("chmt.material.conduction","linearEnergyBudgetFraction",controls.linearEnergyBudgetFraction,"default",region);
    return controls;
}
}}
#endif
