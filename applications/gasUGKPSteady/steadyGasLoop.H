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
const dictionary steadyControls = runTime.controlDict().subOrEmptyDict("steadyGas");
const scalar steadyCourant = steadyControls.lookupOrDefault<scalar>("maxCo", 0.5);
const scalar steadyMaxDt = steadyControls.lookupOrDefault<scalar>("maxDeltaT", GREAT);
const scalar steadyBoundaryTime = steadyControls.lookupOrDefault<scalar>("boundaryTime", gasSteadyRestart.boundaryTime);
const scalar steadyRelativeTolerance = steadyControls.lookupOrDefault<scalar>("relativeTolerance", 1e-8);
const scalar steadyAbsoluteTolerance = steadyControls.lookupOrDefault<scalar>("absoluteTolerance", 0);
const label steadyCheckInterval = max(label(1), steadyControls.lookupOrDefault<label>("residualInterval", 10));
const label steadyMinimumIterations = steadyControls.lookupOrDefault<label>("minimumIterations", 1);
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
const scalar referenceTime=steadyControls.lookupOrDefault<scalar>("referenceTime",1);
const scalar densityScale=gMax(mag(rho.primitiveField()))/referenceTime;
const scalar pressureScale=gMax(mag(p.primitiveField()))/referenceTime;
const scalar energyScale=pressureScale
    +0.5*gMax(magSqr(rhoU.primitiveField())/rho.primitiveField())/referenceTime;
const scalar momentumScale=max(gMax(mag(rhoU.primitiveField()))/referenceTime,
    std::sqrt(densityScale*pressureScale));
scalarField residualScale(initialSteadyResidual);
forAll(residualScale, component)
{
    scalar stateScale=densityScale;
    if(component>=1 && component<=3)stateScale=momentumScale;
    if(component==4)stateScale=energyScale;
    if(component==5+ugkwp::compiledGasSpecies)
        stateScale=gMax(mag(rho.primitiveField()*k.primitiveField()))/referenceTime;
    if(component==6+ugkwp::compiledGasSpecies)
        stateScale=gMax(mag(rho.primitiveField()*omega.primitiveField()))/referenceTime;
    residualScale[component]=max(initialSteadyResidual[component],stateScale);
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
        scalar maximumRelative=0;
        forAll(residual, component)
        {
            const scalar limit=steadyAbsoluteTolerance
                +steadyRelativeTolerance*residualScale[component];
            converged=converged && std::isfinite(residual[component])
                && residual[component]<=limit;
            if(residualScale[component]>0)
                maximumRelative=max(maximumRelative,residual[component]/residualScale[component]);
        }
        Info<< "Steady iteration " << steadyIteration
            << " spatial Linf " << residual
            << " max relative " << maximumRelative << nl;
    }
    if (ugkwp::logging::progressLogDue(runTime.timeIndex(), progressLogInterval))
        ugkwp::logging::writeProgressLog(runTime);
    if(runTime.writeTime() || converged || !runTime.run())
    {
        resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas);
        resident.downloadSharedGasSpecies(sharedGasModel,rho);
        resident.writeBoundaryLayerDiagnostics(boundaryLayerInput,runTime);
        if(gasTurbulenceModel==3)resident.downloadSstToHostMirror(runTime,k,omega,nut);
        else resident.downloadNutToHostMirror(runTime,nut);
        if(converged || !runTime.run())runTime.writeNow();
        else runTime.write();
        const SteadyRestartState checkpoint{steadyBoundaryTime,0,
            gasSteadyRestart.sampleCount+std::uint64_t(steadyIteration)};
        if(!writeSteadyRestartState((runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),checkpoint))
            WarningInFunction<<"Could not write steady clock metadata; retain explicit boundaryTime on restart."<<nl;
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
