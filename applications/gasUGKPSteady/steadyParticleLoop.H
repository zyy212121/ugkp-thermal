// Statistical steady state uses the complete original physical gas/particle
// step. Particle flight, collision, drag and heat exchange NEVER use gas LTS.
SteadyRestartState samplingRestart;
const bool samplingRestartLoaded=readSteadyRestartState
    ((runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),samplingRestart);
const dictionary samplingControls=runTime.controlDict().subOrEmptyDict("steadyGas");
const scalar originalControlDeltaT=runTime.controlDict().lookupOrDefault<scalar>("deltaT",runTime.deltaTValue());
const scalar samplingDeltaT=samplingControls.lookupOrDefault<scalar>("samplingDeltaT",
    samplingRestart.samplingDeltaT>0?samplingRestart.samplingDeltaT:originalControlDeltaT);
const scalar samplingBoundaryTime=samplingControls.lookupOrDefault<scalar>("boundaryTime",samplingRestart.boundaryTime);
const label windowSamples=max(label(1),samplingControls.lookupOrDefault<label>("windowSamples",200));
const label minimumWindows=max(label(2),samplingControls.lookupOrDefault<label>("minimumWindows",4));
const label requiredStableWindows=max(label(1),samplingControls.lookupOrDefault<label>("stableWindows",2));
const scalar meanTolerance=samplingControls.lookupOrDefault<scalar>("meanTolerance",1e-3);
const scalar inventoryTolerance=samplingControls.lookupOrDefault<scalar>("inventoryTolerance",1e-2);
const char* baseSamplingNames[14]={"rho","rhoUx","rhoUy","rhoUz","rhoE","rhoK","rhoOmega",
    "rhoSolid","rhoUsx","rhoUsy","rhoUsz","rhoEs","rhoHp","rhoDs"};
wordList samplingNames(GpuSteadySamplingReport::fields);
for(label component=0;component<14;++component)samplingNames[component]=baseSamplingNames[component];
for(label species=0;species<ugkwp::compiledGasSpecies;++species)
    samplingNames[14+species]="rhoY"+Foam::name(species);
resident.configureSteadySampling(samplingBoundaryTime);
resident.beginSamplingWindow();
Info<< "Particle statistical steady sampling: physical dt=" << samplingDeltaT
    << ", fixed external boundary time=" << samplingBoundaryTime
    << ", samples/window=" << windowSamples << nl
    << "Gas and particle operators share this physical step; no particle LTS acceleration." << nl
    << "Time directory labels count samples. Window RMS is reported, not used as a stationarity gate." << nl;
label samplesInWindow=0,windows=0,stableWindows=0,totalSamples=label(samplingRestart.sampleCount);
scalar samplingTime=samplingRestart.samplingTime,windowDuration=0;
Info<< "Steady restart metadata loaded=" << samplingRestartLoaded
    << "; samplingTime=" << samplingTime << "; samples=" << totalSamples
    << "; statistics windows start afresh (no convergence history reused)." << nl;
bool haveSamplingMean=false, samplingConverged=false;
while(runTime.run())
{
    resident.advanceOneStep(samplingDeltaT,samplingBoundaryTime);
    resident.accumulateSamplingWindow(samplingDeltaT);
    samplingTime+=samplingDeltaT;windowDuration+=samplingDeltaT;
    ++samplesInWindow;++totalSamples;
    runTime.setDeltaT(1);runTime++;
    bool converged=false;
    const bool windowComplete=samplesInWindow==windowSamples;
    if(windowComplete || !runTime.run())
    {
        const GpuSteadySamplingReport report=resident.finishSamplingWindow(windowDuration);
        ++windows;haveSamplingMean=true;
        bool stationary=windows>=minimumWindows && windowComplete;
        scalar maxChange=0,maxDrift=0;
        Info<< "Steady sampling window " << windows << " physicalDuration=" << windowDuration
            << " totalSamplingTime=" << samplingTime << nl;
        for(label component=0;component<GpuSteadySamplingReport::fields;++component)
        {
            const scalar scale=report.characteristicInventory[component];
            const scalar drift=mag(report.inventoryChange[component])/max(scale,scalar(1e-300));
            stationary=stationary && std::isfinite(report.meanChange[component])
                && std::isfinite(drift) && report.meanChange[component]<=meanTolerance
                && drift<=inventoryTolerance;
            maxChange=max(maxChange,report.meanChange[component]);maxDrift=max(maxDrift,drift);
            Info<< "  " << samplingNames[component]
                << " meanChange=" << report.meanChange[component]
                << " relativeRms=" << report.fluctuation[component]
                << " inventoryBegin=" << report.inventoryBegin[component]
                << " inventoryEnd=" << report.inventoryEnd[component]
                << " inventoryDrift=" << drift << nl;
        }
        Info<< "  gasIntrinsicFvBoundaryExchange (not porosity-weighted mixture flux):";
        for(label k=0;k<GpuSteadySamplingReport::gasFluxFields;++k)Info<< ' ' << report.gasBoundaryExchange[k];
        Info<< nl << "  particleInjected mass/momentumXYZ/mechanicalEnergy/materialHeat:";
        for(label k=0;k<6;++k)Info<< ' ' << report.particleInjection[k];
        Info<< nl << "  particleTrackingBoundaryExchange (outflow plus wall impulse/heat):";
        for(label k=0;k<6;++k)Info<< ' ' << report.particleTrackingExchange[k];
        Info<< nl << "  particleInventoryMinusBoundaryExchange (coupling/collision/internal effects):";
        const label particleBudgetFields[6]={7,8,9,10,11,12};
        for(label k=0;k<6;++k)Info<< ' ' << report.inventoryChange[particleBudgetFields[k]]
            -report.particleInjection[k]-report.particleTrackingExchange[k];
        Info<< nl;
        stableWindows=stationary?stableWindows+1:0;
        converged=stableWindows>=requiredStableWindows;
        Info<< "  windowStationary=" << stationary << " consecutive=" << stableWindows
            << " maxMeanChange=" << maxChange << " maxInventoryDrift=" << maxDrift << nl;
    }
    if(ugkwp::logging::progressLogDue(runTime.timeIndex(),progressLogInterval))
        ugkwp::logging::writeProgressLog(runTime);
    if(runTime.writeTime() || converged || !runTime.run())
    {
        resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas,epsS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);
        resident.downloadSharedGasSpecies(sharedGasModel,rho);
        if(gasTurbulenceModel==3)resident.downloadSstToHostMirror(runTime,k,omega,nut);
        else resident.downloadNutToHostMirror(runTime,nut);
        if(converged || !runTime.run())runTime.writeNow();else runTime.write();
        rhoHp.write(); // Export the actual downloaded material-heat moment.
        (void)resident.writeParticleRestartMirror(runTime);
        const SteadyRestartState checkpoint{samplingBoundaryTime,samplingTime,std::uint64_t(totalSamples),samplingDeltaT};
        if(!writeSteadyRestartState((runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),checkpoint))
            WarningInFunction<<"Could not write steady clock metadata; retain explicit boundaryTime on restart."<<nl;
        if(haveSamplingMean && (windowComplete || converged || !runTime.run()))
        {
            scalarField means;resident.downloadSamplingMean(means);
            std::ofstream meanOutput((runTime.path()/runTime.timeName()/"steadyWindowMean.csv").c_str());
            meanOutput.precision(17);meanOutput<<"cell";
            for(label component=0;component<GpuSteadySamplingReport::fields;++component)meanOutput<<','<<samplingNames[component];
            meanOutput<<'\n';
            forAll(rho,cell)
            {
                meanOutput<<cell;
                for(label component=0;component<GpuSteadySamplingReport::fields;++component)meanOutput<<','<<means[component*mesh.nCells()+cell];
                meanOutput<<'\n';
            }
        }
    }
    if(converged)
    {
        samplingConverged=true;
        Info<< "Particle statistical steady criteria reached after " << totalSamples
            << " physical samples; stochastic fluctuations remain." << nl;
        break;
    }
    if(windowComplete)
    {
        samplesInWindow=0;windowDuration=0;resident.beginSamplingWindow();
    }
}

Info<< "Particle steady driver finished: criteriaReached=" << samplingConverged
    << " physicalSamplingTime=" << samplingTime << " samples=" << totalSamples << nl;
