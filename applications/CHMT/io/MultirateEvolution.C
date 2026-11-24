#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "DiagnosticsFoam.H"
#include <algorithm>
#include <exception>
#include <iostream>

namespace chmt {
bool advanceCoupledWindow(Backend& backend,CpuMaterialDriver& material,const ModelConfig&,
    const CouplingControls& controls,const CpuMaterialControls& materialControls,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport& report,std::string& error,WindowRadiation* radiation,bool publishHostState) {
    report=WindowEvolutionReport{};
    try {
        const HostState& base=accepted;
        CouplingWindowController controller;
        if(!controller.begin(base.time,targetTime,base.commitSequence+1,controls,error))return false;
        const CouplingInterval interval=controller.interval();
        const bool radiate=radiation && radiation->due(interval,controls,base,
            ugkp::gpu::reachedEventTime(interval.end,targetTime,interval.end-interval.begin));
        WallProgram program;
        CpuMaterialReport prediction;
        if(!material.predictWindowWall(base,interval,program,error,&prediction,true))return false;
        if(!beginGasWindow(backend,program,error))return false;
        Real gasTime=interval.begin;WindowTime gasElapsed{};
        bool materialInParent=false;

        if(!retainStageGeometry){
            if(radiate&&!radiation->prepareParent(backend,program,base,error))return false;
            if(!prepareGasWindow(backend,controls.gasMaxDt,error,false))return false;
        }
        MaterialBackend* bound=nullptr;
        const bool deferMaterial=retainStageGeometry||(radiate&&!gasWindowHasResidentRadiation(backend));
        if(!material.prepareImplicitParent(base,interval,materialControls,deferMaterial,bound,materialInParent,error)
            ||(bound&&!bindImplicitGasWindowMaterial(backend,*bound,materialInParent,error)))return false;
        if(!retainStageGeometry){
            GasMicroReport completed;
            if(!enqueuePreparedGasWindow(backend,error)||!collectGasWindow(backend,completed,error))return false;
            report.acceptedGasMicrosteps=completed.microSequence;
            report.nextGasDt=completed.nextGasDt;gasTime=completed.time;gasElapsed=completed.localTime;
            std::cout<<"CHMT_GAS window="<<interval.identity.sequence<<" microstep="<<completed.microSequence
                <<" time="<<completed.time<<" nextDt="<<completed.nextGasDt<<" execution=device_window"<<std::endl;
        } else while(gasElapsed<intervalDuration(interval)) {
            GasMicroReport micro;
            if(!advanceGasMicrostep(backend,controls.gasMaxDt,micro,error))return false;
            report.acceptedGasMicrosteps=micro.microSequence;
            report.nextGasDt=micro.nextGasDt;
            if(micro.microSequence%1000==0)std::cout<<"CHMT_GAS window="<<interval.identity.sequence
                <<" microstep="<<micro.microSequence<<" time="<<micro.time<<" nextDt="<<micro.nextGasDt<<std::endl;
            MicrostepGeometryReport geometry;geometry.microSequence=micro.microSequence;
            geometry.begin=gasTime;geometry.end=micro.time;
            if(!gasWindowStageGeometry(backend,geometry.gas,geometry.solid,error))return false;
            report.geometry.push_back(std::move(geometry));
            gasTime=micro.time;gasElapsed=micro.localTime;
        }


        HostState gasEndpoint;
        if(!(publishHostState?downloadGasWindow(backend,gasEndpoint,error):publishGasWindow(backend,gasEndpoint,error)))return false;
        const IntervalHistory& history=gasWindowHistory(backend);
        if(radiate){
            if(!radiation->advance(backend,program,gasEndpoint,error))return false;
            report.radiationInterval=gasEndpoint.radiationTime-base.radiationTime;
        }
        HostState candidate;CpuMaterialReport cpu;
        ++report.materialSolves;
        if(!material.finishImplicitParent(base,gasEndpoint,interval,materialControls,history.externalRadiationDevice(),
            materialInParent,candidate,cpu,error))return false;
        if(!base.solid.empty()&&cpu.exchangeFrames!=report.acceptedGasMicrosteps){
            error="implicit material window did not consume all accepted gas frames";return false;
        }
        releaseGasWindowHistory(backend);
        if(!acceptCoupledWindow(backend,candidate,cpu.deviceCommit,report.nextGasDt,error)
            ||!material.commitImplicitInventory(candidate,error))return false;
        std::cout<<"CHMT_IMPLICIT_WINDOW window="<<interval.identity.sequence
            <<" materialSteps="<<cpu.materialSteps<<" nonlinearIterations="<<cpu.implicitIterations
            <<" converged="<<cpu.implicitConverged<<" geometrySolves="<<cpu.geometrySolves
            <<" massResidualRatio="<<cpu.massResidual
            <<" energyResidualRatio="<<cpu.thermalResidual<<" recessionVolume="<<cpu.recessionVolume
            <<" gclResidual="<<cpu.gclResidual<<" exchangeFrames="<<cpu.exchangeFrames
            <<" linearSolves="<<cpu.materialLinearSolves<<" linearIterations="<<cpu.materialLinearIterations
            <<" materialExecution="<<(materialInParent?"parent_graph":"continuation_graph")
            <<std::endl;
        accepted=std::move(candidate);error.clear();return true;
    } catch(const Foam::error&) {


        throw;
    } catch(const std::exception& exception) {
        error=std::string("single-pass multirate operation failed: ")+exception.what();
        return false;
    }
}
}
