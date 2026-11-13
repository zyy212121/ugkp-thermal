if(sharedGasModel.model().mode==ugkwp::GasMode::MixtureChemistry)
    Info<< "Reacting steady path uses local pseudo-time chemistry half-steps and "
        << "instantaneous transport-plus-reaction residuals. Retained split-step bias "
        << "may require reducing maxDeltaT; reacting steady validation is not part "
        << "of the frozen-mixture reference cases." << nl;
// All numerical operators remain in the resident gas pipeline. OpenFOAM's
// time directory labels count pseudo iterations, NOT elapsed physical time.
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
// Initial host mirrors supply fixed dimensional reference scales only; no
// active solution update or transport operator is evaluated on the host.
const scalar referenceTime=steadyControls.referenceTime;
const SteadyReal referenceWeight=SteadyReal(referenceTime);
// OpenFOAM host mirrors are an interface boundary. Compute reference norms in
// the active physical precision, including vector norms and global maxima.
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
while (runTime.run())
{
    resident.prepareSteadyStep(steadyCourant, steadyMaxDt);
    // Unit driver interval is scaled to dt_cell by the shared operator helper.
    resident.advanceOneStep(1, steadyBoundaryTime);
    runTime.setDeltaT(1);
    runTime++;
    ++steadyIteration;
    bool converged=false;
    if (steadyIteration % steadyCheckInterval == 0 || !runTime.run())
    {
        scalarField residual;
        resident.steadyResidual(residual);
        converged=steadyIteration>=steadyMinimumIterations;
        SteadyReal maximumRelative=0;
        forAll(residual, component)
        {
            const SteadyReal value=SteadyReal(residual[component]);
            const SteadyReal limit=steadyAbsoluteTolerance
                +steadyRelativeTolerance*residualScale[component];
            converged=converged && std::isfinite(value)
                && value<=limit;
            if(residualScale[component]>0)
                maximumRelative=max(maximumRelative,value/residualScale[component]);
        }
        Info<< "Steady iteration " << steadyIteration
            << " spatial Linf " << residual
            << " max relative " << maximumRelative << nl;
    }
    if (ugkwp::logging::progressLogDue(runTime.timeIndex(), progressLogInterval))
        ugkwp::logging::writeProgressLog(runTime);
    if(runTime.writeTime() || converged || !runTime.run())
    {
        ugkp::precision::ScopedRestartWritePrecision writePrecision;
        ugkp::precision::beginPrecisionWrite(runTime.timePath().c_str());
        resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas);
        resident.downloadSharedGasSpecies(sharedGasModel,rho);
        resident.writeBoundaryLayerDiagnostics(boundaryLayerInput,runTime);
        if(gasTurbulenceModel==3)resident.downloadSstToHostMirror(runTime,k,omega,nut);
        else resident.downloadNutToHostMirror(runTime,nut);
        const bool fieldsWritten = (converged || !runTime.run())
            ? runTime.writeNow() : runTime.write();
        const SteadyRestartState checkpoint{steadyBoundaryTime,0,
            gasSteadyRestart.sampleCount+std::uint64_t(steadyIteration)};
        const bool clockWritten = writeSteadyRestartState(
            (runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),checkpoint);
        if(!clockWritten)
            WarningInFunction<<"Could not write steady clock metadata; retain explicit boundaryTime on restart."<<nl;
        if(fieldsWritten && clockWritten)
            ugkp::precision::writeRestartPrecision(runTime.timePath().c_str());
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
