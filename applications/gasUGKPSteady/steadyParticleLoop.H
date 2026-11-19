

SteadyRestartState samplingRestart;
const bool samplingRestartLoaded=readSteadyRestartState
    ((runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),samplingRestart);
const scalar originalControlDeltaT=runTime.controlDict().lookupOrDefault<scalar>("deltaT",runTime.deltaTValue());
const auto samplingControls=ugkp::solverControls::readParticleSteadyControls(
    runTime,samplingRestart.samplingDeltaT>0?samplingRestart.samplingDeltaT:originalControlDeltaT,
    samplingRestart.boundaryTime,mesh.name());
const scalar samplingDeltaT=samplingControls.samplingDeltaT;
const scalar samplingBoundaryTime=samplingControls.boundaryTime;
const label windowSamples=samplingControls.windowSamples;
const label minimumWindows=samplingControls.minimumWindows;
const label requiredStableWindows=samplingControls.stableWindows;
using SteadyReal=ugkp::solverControls::SteadySolverReal;
const SteadyReal meanTolerance=samplingControls.meanTolerance;
const SteadyReal inventoryTolerance=samplingControls.inventoryTolerance;
const char* baseSamplingNames[14]={"rho","rhoUx","rhoUy","rhoUz","rhoE","rhoK","rhoOmega",
    "rhoSolid","rhoUsx","rhoUsy","rhoUsz","rhoEs","rhoHp","rhoDs"};
wordList samplingNames(GpuSteadySamplingReport::fields);
for(label component=0;component<14;++component)samplingNames[component]=baseSamplingNames[component];
for(label species=0;species<ugkwp::compiledGasSpecies;++species)
    samplingNames[14+species]="rhoY"+Foam::name(species);
resident.configureSteadySampling(samplingBoundaryTime,samplingRestart.sampleCount);
Info<< "Particle statistical steady sampling: physical dt=" << samplingDeltaT
    << ", fixed external boundary time=" << samplingBoundaryTime
    << ", samples/window=" << windowSamples << nl
    << "Gas and particle operators share this physical step; no particle LTS acceleration." << nl
    << "Time directory labels count samples. Window RMS is reported, not used as a stationarity gate." << nl;
label samplesInWindow=0,windows=0,stableWindows=0;
std::uint64_t totalSamples=samplingRestart.sampleCount;
scalar samplingTime=samplingRestart.samplingTime,windowDuration=0;
Info<< "Steady restart metadata loaded=" << samplingRestartLoaded
    << "; samplingTime=" << samplingTime << "; samples=" << totalSamples
    << "; statistics windows start afresh (no convergence history reused)." << nl;
bool haveSamplingMean=false, samplingConverged=false;
ugkp::gpu::AdvanceState advanceInitial;
advanceInitial.clock=ugkp::gpu::AdvanceClock::sampling;
advanceInitial.time=runTime.value();advanceInitial.dt=samplingDeltaT;advanceInitial.clockDt=runTime.deltaTValue();
advanceInitial.steps=runTime.timeIndex();advanceInitial.initialSteps=advanceInitial.steps;
advanceInitial.acceptedSequence=advanceInitial.steps;
advanceInitial.boundaryFrozen=true;advanceInitial.boundaryTime=samplingBoundaryTime;
advanceInitial.samplingTime=samplingTime;advanceInitial.positionSequence=totalSamples;
resident.configureAdvance(advanceInitial,false);
ugkp::gpu::SteadyAdvanceControls deviceSteadyControls;
deviceSteadyControls.windowSamples=windowSamples;deviceSteadyControls.minimumWindows=minimumWindows;
deviceSteadyControls.stableWindows=requiredStableWindows;
deviceSteadyControls.meanTolerance=meanTolerance;deviceSteadyControls.inventoryTolerance=inventoryTolerance;
resident.configureSteadyAdvance(deviceSteadyControls);
auto enqueueSteadyBoundary=[&]{
    auto request=ugkp::gpu::foamAdvanceBoundary(runTime,{},progressLogInterval,ugkp::gpu::AdvanceClock::sampling);
    request.maximumSteps=std::min(request.maximumSteps,std::uint64_t(windowSamples-samplesInWindow%windowSamples));
    resident.enqueueAdvance(request);
};
bool advancePending=false;
while(runTime.run())
{
    if(!advancePending)enqueueSteadyBoundary();
    const auto completed=resident.collectAdvance(&runTime);advancePending=false;
    ugkp::gpu::publishFoamAdvance(runTime,completed);
    samplesInWindow=completed.windowSamples;windows=completed.windows;stableWindows=completed.stableWindows;
    totalSamples=completed.positionSequence;samplingTime=completed.samplingTime;windowDuration=completed.windowDuration;
    const bool converged=completed.stop==ugkp::gpu::AdvanceStop::converged;
    const bool windowComplete=samplesInWindow==windowSamples;
    if(completed.windowReady)
    {
        const GpuSteadySamplingReport report=resident.collectSteadyWindow();
        haveSamplingMean=true;
        const bool stationary=completed.stationary;
        const scalar maxChange=completed.maximumMeanChange,maxDrift=completed.maximumInventoryDrift;
        Info<< "Steady sampling window " << windows << " physicalDuration=" << windowDuration
            << " totalSamplingTime=" << samplingTime << nl;
        for(label component=0;component<GpuSteadySamplingReport::fields;++component)
        {
            const SteadyReal scale=SteadyReal(report.characteristicInventory[component]);
            constexpr SteadyReal inventoryFloor=ugkp::precision::compiledBits==32
                ?std::numeric_limits<SteadyReal>::min():SteadyReal(1e-300);
            const SteadyReal drift=std::fabs(SteadyReal(report.inventoryChange[component]))
                /max(scale,inventoryFloor);
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
        for(label k=0;k<6;++k)Info<< ' ' << SteadyReal(report.inventoryChange[particleBudgetFields[k]])
            -SteadyReal(report.particleInjection[k])-SteadyReal(report.particleTrackingExchange[k]);
        Info<< nl;
        Info<< "  windowStationary=" << stationary << " consecutive=" << stableWindows
            << " maxMeanChange=" << maxChange << " maxInventoryDrift=" << maxDrift << nl;
    }
    if(ugkwp::logging::progressLogDue(runTime.timeIndex(),progressLogInterval))
        ugkwp::logging::writeProgressLog(runTime);
    if(runTime.writeTime() || converged || !runTime.run())
    {
        const bool force=converged || !runTime.run();
        scalarField means;
        const bool writeMean=haveSamplingMean && (windowComplete || force);
        ugkp::gpu::writeGasOutput(runTime,force,[&]{
            resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas,epsS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);
            resident.downloadSharedGasSpecies(sharedGasModel,rho);
            ugkp::gpu::mirrorGasTurbulence(resident,runTime,gasTurbulenceModel,k,omega,nut);
            if(writeMean)resident.downloadSamplingMean(means);
            return resident.prepareParticleRestartMirror(runTime);
        },[&]{enqueueSteadyBoundary();advancePending=true;},[&]{
            const bool heatWritten=rhoHp.write();
            const SteadyRestartState checkpoint{samplingBoundaryTime,samplingTime,std::uint64_t(totalSamples),samplingDeltaT};
            const bool clockWritten = writeSteadyRestartState(
                (runTime.path()/runTime.timeName()/"steadyState.dat").c_str(),checkpoint);
            return heatWritten && clockWritten;
        });
        if(writeMean)
        {
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
        samplesInWindow=0;windowDuration=0;
    }
}

Info<< "Particle steady driver finished: criteriaReached=" << samplingConverged
    << " physicalSamplingTime=" << samplingTime << " samples=" << totalSamples << nl;
