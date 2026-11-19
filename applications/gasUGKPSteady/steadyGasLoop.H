if(sharedGasModel.model().mode==ugkwp::GasMode::MixtureChemistry)
    Info<< "Reacting steady path uses local pseudo-time chemistry half-steps and "
        << "instantaneous transport-plus-reaction residuals. Retained split-step bias "
        << "may require reducing maxDeltaT; reacting steady validation is not part "
        << "of the frozen-mixture reference cases." << nl;


SteadyRestartState gasSteadyRestart;
const bool gasSteadyRestartLoaded=readSteadyRestartState
    ((runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),gasSteadyRestart);
const auto steadyControls=ugkp::solverControls::readGasSteadyControls(
    runTime,gasSteadyRestart.boundaryTime,mesh.name());
using SteadyReal=ugkp::solverControls::SteadySolverReal;
const SteadyReal steadyCourant=steadyControls.maxCo;
const scalar steadyMaxDt=steadyControls.maxDeltaT;
const scalar steadyBoundaryTime=steadyControls.boundaryTime;
const SteadyReal steadyRelativeTolerance=steadyControls.relativeTolerance;
const SteadyReal steadyAbsoluteTolerance=steadyControls.absoluteTolerance;
const label steadyCheckInterval=steadyControls.residualInterval;
const label steadyMinimumIterations=steadyControls.minimumIterations;
resident.configureSteady(steadyBoundaryTime);
Info<< "Steady local pseudo-time: Co=" << steadyCourant
    << ", fixed boundary time=" << steadyBoundaryTime
    << "; output time labels are iteration counters." << nl
    << "Spatial residual components: rho, rhoUx, rhoUy, rhoUz, rhoE, species, rhoK, rhoOmega." << nl;
scalarField initialSteadyResidual;
resident.steadyResidual(initialSteadyResidual);
Info<< "Initial steady spatial Linf residual " << initialSteadyResidual << nl;


const scalar referenceTime=steadyControls.referenceTime;
const SteadyReal referenceWeight=SteadyReal(referenceTime);


const SteadyReal emptyMaximum=std::numeric_limits<SteadyReal>::lowest();
SteadyReal densityMaximum=emptyMaximum,pressureMaximum=emptyMaximum,kineticMaximum=emptyMaximum;
SteadyReal momentumMaximum=emptyMaximum,kMaximum=emptyMaximum,omegaMaximum=emptyMaximum;
forAll(rho,cell)
{
    const SteadyReal density=SteadyReal(rho.primitiveField()[cell]);
    const SteadyReal pressure=SteadyReal(p.primitiveField()[cell]);
    const vector& momentum=rhoU.primitiveField()[cell];
    const SteadyReal mx=SteadyReal(momentum.x()),my=SteadyReal(momentum.y()),mz=SteadyReal(momentum.z());
    const SteadyReal momentumSquare=mx*mx+my*my+mz*mz;
    densityMaximum=max(densityMaximum,std::fabs(density));
    pressureMaximum=max(pressureMaximum,std::fabs(pressure));
    kineticMaximum=max(kineticMaximum,momentumSquare/density);
    momentumMaximum=max(momentumMaximum,std::sqrt(momentumSquare));
    kMaximum=max(kMaximum,std::fabs(density*SteadyReal(k.primitiveField()[cell])));
    omegaMaximum=max(omegaMaximum,std::fabs(density*SteadyReal(omega.primitiveField()[cell])));
}
reduce(densityMaximum,maxOp<SteadyReal>());
reduce(pressureMaximum,maxOp<SteadyReal>());
reduce(kineticMaximum,maxOp<SteadyReal>());
reduce(momentumMaximum,maxOp<SteadyReal>());
reduce(kMaximum,maxOp<SteadyReal>());
reduce(omegaMaximum,maxOp<SteadyReal>());
const SteadyReal densityScale=densityMaximum/referenceWeight;
const SteadyReal pressureScale=pressureMaximum/referenceWeight;
const SteadyReal energyScale=pressureScale+SteadyReal(0.5)*kineticMaximum/referenceWeight;
const SteadyReal momentumScale=max(momentumMaximum/referenceWeight,
    std::sqrt(densityScale*pressureScale));
List<SteadyReal> residualScale(initialSteadyResidual.size());
forAll(residualScale, component)
{
    SteadyReal stateScale=densityScale;
    if(component>=1 && component<=3)stateScale=momentumScale;
    if(component==4)stateScale=energyScale;
    if(component==5+ugkwp::compiledGasSpecies)stateScale=kMaximum/referenceWeight;
    if(component==6+ugkwp::compiledGasSpecies)stateScale=omegaMaximum/referenceWeight;
    residualScale[component]=max(SteadyReal(initialSteadyResidual[component]),stateScale);
}
Info<< "Steady restart metadata loaded=" << gasSteadyRestartLoaded
    << "; residual normalization restarts from the loaded state." << nl;
label steadyIteration=0;
bool steadyConverged=false;
ugkp::gpu::AdvanceState advanceInitial;
advanceInitial.clock=ugkp::gpu::AdvanceClock::pseudo;
advanceInitial.time=runTime.value();advanceInitial.dt=1;advanceInitial.clockDt=runTime.deltaTValue();
advanceInitial.steps=runTime.timeIndex();advanceInitial.initialSteps=advanceInitial.steps;
advanceInitial.acceptedSequence=advanceInitial.steps;
advanceInitial.boundaryFrozen=true;advanceInitial.boundaryTime=steadyBoundaryTime;
resident.configureAdvance(advanceInitial,true);
ugkp::gpu::SteadyAdvanceControls deviceSteadyControls;
deviceSteadyControls.courant=steadyCourant;deviceSteadyControls.maximumDt=steadyMaxDt;
deviceSteadyControls.relativeTolerance=steadyRelativeTolerance;deviceSteadyControls.absoluteTolerance=steadyAbsoluteTolerance;
deviceSteadyControls.residualInterval=steadyCheckInterval;deviceSteadyControls.minimumIterations=steadyMinimumIterations;
scalarField deviceResidualScale(residualScale.size());
forAll(deviceResidualScale,k)deviceResidualScale[k]=residualScale[k];
resident.configureSteadyAdvance(deviceSteadyControls,deviceResidualScale);
auto enqueueSteadyBoundary=[&]{
    auto request=ugkp::gpu::foamAdvanceBoundary(runTime,{},progressLogInterval,ugkp::gpu::AdvanceClock::pseudo);
    request.maximumSteps=std::min(request.maximumSteps,std::uint64_t(steadyCheckInterval-steadyIteration%steadyCheckInterval));
    resident.enqueueAdvance(request);
};
bool advancePending=false;
while (runTime.run())
{
    if(!advancePending)enqueueSteadyBoundary();
    const auto completed=resident.collectAdvance(&runTime);advancePending=false;
    ugkp::gpu::publishFoamAdvance(runTime,completed);
    steadyIteration=completed.steps-completed.initialSteps;
    const bool converged=completed.stop==ugkp::gpu::AdvanceStop::converged;
    if(completed.residualReady)
    {
        scalarField residual;resident.collectSteadyResidual(residual);
        Info<< "Steady iteration " << steadyIteration
            << " spatial Linf " << residual
            << " max relative " << completed.maximumRelative << nl;
    }
    if (ugkwp::logging::progressLogDue(runTime.timeIndex(), progressLogInterval))
        ugkwp::logging::writeProgressLog(runTime);
    if(runTime.writeTime() || converged || !runTime.run())
    {
        const bool force=converged || !runTime.run();
        ugkp::gpu::writeGasOutput(runTime,force,[&]{
            resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas);
            resident.downloadSharedGasSpecies(sharedGasModel,rho);
            resident.writeBoundaryLayerDiagnostics(boundaryLayerInput,runTime);
            ugkp::gpu::mirrorGasTurbulence(resident,runTime,gasTurbulenceModel,k,omega,nut);
            return []{};
        },[&]{enqueueSteadyBoundary();advancePending=true;},[&]{
            const SteadyRestartState checkpoint{steadyBoundaryTime,0,
                gasSteadyRestart.sampleCount+std::uint64_t(steadyIteration)};
            const bool clockWritten = writeSteadyRestartState(
                (runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),checkpoint);
            return clockWritten;
        });
    }
    if(converged)
    {
        steadyConverged=true;
        Info<< "Steady spatial residual converged after " << steadyIteration << " iterations." << nl;
        break;
    }
}

Info<< "Steady driver finished: criteriaReached=" << steadyConverged
    << "; output labels represent pseudo iterations, not physical time." << nl;
