#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "materials/MaterialWindowExchange.H"
#include "coupling/IntervalAudit.H"
#include "DiagnosticsFoam.H"
#include <algorithm>
#include <exception>
#include <iostream>

namespace chmt {
bool advanceCoupledWindow(Backend& backend,CpuMaterialDriver& material,const ModelConfig&,
    const CouplingControls& controls,const CpuMaterialControls& materialControls,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport& report,std::string& error,WindowRadiation* radiation,bool publishHostState,bool outputBoundary) {
    report=WindowEvolutionReport{};
    try {
        const HostState& base=accepted;
        CouplingWindowController controller;
        if(!controller.begin(base.time,targetTime,base.commitSequence+1,controls,error))return false;
        const CouplingInterval interval=controller.interval();
        const bool radiate=radiation && radiation->due(interval,controls,base,
            outputBoundary || ugkp::gpu::reachedEventTime(interval.end,targetTime,interval.end-interval.begin));
        WallProgram program;
        CpuMaterialReport prediction;
        if(!material.predictWindowWall(base,interval,program,error,&prediction,true))return false;
        if(!beginGasWindow(backend,program,error))return false;
        Real gasTime=interval.begin;WindowTime gasElapsed{};





        if(!retainStageGeometry){
            if(radiate&&!radiation->prepareParent(backend,program,base,error))return false;
            if(!prepareGasWindow(backend,controls.gasMaxDt,error))return false;
            GasMicroReport completed;
            if(!enqueuePreparedGasWindow(backend,error)||!collectGasWindow(backend,completed,error))return false;
            report.nextGasDt=completed.nextGasDt;gasTime=completed.time;gasElapsed=completed.localTime;
            std::cout<<"CHMT_GAS window="<<interval.identity.sequence<<" microstep="<<completed.microSequence
                <<" time="<<completed.time<<" nextDt="<<completed.nextGasDt<<" execution=device_window"<<std::endl;
        } else while(gasElapsed<intervalDuration(interval)) {
            GasMicroReport micro;
            if(!advanceGasMicrostep(backend,controls.gasMaxDt,micro,error))return false;
            report.nextGasDt=micro.nextGasDt;
            if(micro.microSequence%1000==0)std::cout<<"CHMT_GAS window="<<interval.identity.sequence
                <<" microstep="<<micro.microSequence<<" time="<<micro.time<<" nextDt="<<micro.nextGasDt<<std::endl;
            if(retainStageGeometry) {
                MicrostepGeometryReport geometry;geometry.microSequence=micro.microSequence;
                geometry.begin=gasTime;geometry.end=micro.time;
                if(!gasWindowStageGeometry(backend,geometry.gas,geometry.solid,error))return false;
                report.geometry.push_back(std::move(geometry));
            }
            gasTime=micro.time;gasElapsed=micro.localTime;
        }


        HostState gasEndpoint;
        if(!(publishHostState?downloadGasWindow(backend,gasEndpoint,error):publishGasWindow(backend,gasEndpoint,error)))return false;
        const IntervalHistory& history=gasWindowHistory(backend);
        EnergyReal externalRadiationHeat=0;
        if(radiate&&!radiation->advance(backend,program,gasEndpoint,externalRadiationHeat,error))return false;
        MaterialWindowExchange exchange(base);
        if(!collectMaterialWindowExchange(backend,exchange,error))return false;
        HostState candidate;CpuMaterialReport cpu;
        ++report.materialSolves;
        if(!material.advanceImplicitWindow(base,gasEndpoint,interval,exchange,materialControls,candidate,cpu,error))return false;
        report.acceptedGasMicrosteps=history.records().size();
        if(!finalizeReleasedIntervalLedger(releaseGasWindowHistory(backend),candidate,error))return false;
        if(!acceptCoupledWindow(backend,candidate,cpu.deviceCommit,externalRadiationHeat,report.nextGasDt,error))return false;
        std::cout<<"CHMT_IMPLICIT_WINDOW window="<<interval.identity.sequence
            <<" materialSteps="<<cpu.materialSteps<<" nonlinearIterations="<<cpu.implicitIterations
            <<" converged="<<cpu.implicitConverged<<" geometrySolves="<<cpu.geometrySolves
            <<" exchangePackets="<<exchange.packets<<" massResidualRatio="<<cpu.massResidual
            <<" energyResidualRatio="<<cpu.thermalResidual<<" recessionVolume="<<cpu.recessionVolume
            <<" gclResidual="<<cpu.gclResidual<<std::endl;
        accepted=std::move(candidate);error.clear();return true;
    } catch(const Foam::error&) {


        throw;
    } catch(const std::exception& exception) {
        error=std::string("single-pass multirate operation failed: ")+exception.what();
        return false;
    }
}
}
