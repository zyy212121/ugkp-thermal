#include "materials/ConductionContinuation.H"
#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "coupling/IntervalAudit.H"
#include <algorithm>
#include <exception>
#include <limits>

namespace chmt {
namespace {
bool sameEndpointPoints(const HostMesh& a,const HostMesh& b) {
    if(a.topologyHash!=b.topologyHash||a.points.size()!=b.points.size())return false;
    for(std::size_t i=0;i<a.points.size();++i)
        if(a.points[i].x!=b.points[i].x||a.points[i].y!=b.points[i].y||a.points[i].z!=b.points[i].z)return false;
    return true;
}
}
bool advanceCoupledWindow(Backend& backend,CpuMaterialDriver& material,const ModelConfig& model,
    const CouplingControls& controls,const CpuMaterialControls& materialControls,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport& report,std::string& error) {
    report=WindowEvolutionReport{};
    bool active=false;
    auto discard=[&](const std::string& cause)->bool {
        std::string rollbackError;
        if(active){const bool restored=rollbackGasWindow(backend,rollbackError);active=false;
            if(!restored){error=cause+"; gas rollback failed: "+rollbackError;return false;}}
        error=cause;return true;
    };
    try {
        const HostState base=accepted;
        CouplingWindowController controller;
        if(!controller.begin(base.time,targetTime,base.commitSequence+1,controls,error))return false;
        WallProgram program;
        // Prediction has no gas trial to roll back. Only the CPU owner's typed
        // duration-related failures permit reducing H; text is diagnostic only.
        auto predict=[&](const std::string& context)->bool {
            for(;;) {
                CpuMaterialReport diagnostics;
                diagnostics.recoverable=false;
                std::string predictionError;
                if(material.predictWall(base,controller.interval(),program,predictionError,&diagnostics)) {
                    error.clear();return true;
                }
                const std::string cause=context+": "+predictionError;
                auto completePrediction=[&](){
                    if(diagnostics.failureOrigin==MaterialFailureOrigin::None){error=cause;return false;}
                    return detail::ConductionContinuation::predict(material,base,controller.interval(),program,error,&diagnostics);
                };
                if(!diagnostics.recoverable)return completePrediction();
                const auto& interval=controller.interval();
                const Real duration=diagnostics.eventDuration>0?diagnostics.eventDuration:diagnostics.eventTime>interval.begin&&diagnostics.eventTime<interval.end
                    ?diagnostics.eventTime-interval.begin:.5*(interval.end-interval.begin);
                std::string retryError;
                if(!controller.shorten(duration,retryError))return completePrediction();
            }
        };
        if(!predict("initial window predictor"))return false;
        auto shorten=[&](Real eventTime,const std::string& cause,Real eventDuration=0)->bool {
            if(!discard(cause))return false;
            const auto& interval=controller.interval();
            const Real duration=eventDuration>0?eventDuration:eventTime>interval.begin&&eventTime<interval.end
                ?eventTime-interval.begin:.5*(interval.end-interval.begin);
            std::string retryError;
            if(!controller.shorten(duration,retryError)){error=cause+"; "+retryError;return false;}
            return predict("shortened-window predictor after "+cause);
        };
        auto replay=[&](WallProgram corrected,const std::string& cause)->bool {
            if(!discard(cause))return false;
            std::string replayError;
            if(!controller.replay(replayError))return shorten(0,cause+"; "+replayError);
            corrected.interval=controller.interval();
            if(!corrected.donorPlan.knots.empty())corrected.donorPlan.interval=controller.interval();
            program=std::move(corrected);return true;
        };
        for(;;) {
            report.geometry.clear();
            const CouplingInterval interval=controller.interval();
            if(!beginGasWindow(backend,program,error))return false;
            active=true;
            Real gasTime=interval.begin,nextGasCap=controls.gasMaxDt;WindowTime gasElapsed{};
            bool retryWindow=false;
            while(gasElapsed<intervalDuration(interval)) {
                GasMicroReport micro;
                if(!advanceGasMicrostep(backend,nextGasCap,micro,error)) {
                    const std::string cause="gas microstep: "+error;
                    if(!micro.recoverable){discard(cause);return false;}
                    if(!shorten(0,cause))return false;
                    retryWindow=true;break;
                }
                report.nextGasDt=micro.nextGasDt;
                if(retainStageGeometry) {
                    MicrostepGeometryReport geometry;geometry.microSequence=micro.microSequence;
                    geometry.begin=gasTime;geometry.end=micro.time;
                    if(!gasWindowStageGeometry(backend,geometry.gas,geometry.solid,error)){
                        discard("gas microstage geometry: "+error);return false;}
                    report.geometry.push_back(std::move(geometry));
                }
                gasTime=micro.time;gasElapsed=micro.localTime;nextGasCap=std::min(controls.gasMaxDt,micro.nextGasDt);
            }
            if(retryWindow)continue;
            IntervalHistory history;
            HostState gasEndpoint;
            if(!gasWindowHistory(backend,history,error)
                ||!downloadGasWindow(backend,gasEndpoint,error)) {
                discard("completed gas window unavailable: "+error);return false;
            }
            HostState candidate;
            WallProgram corrected;
            CpuMaterialReport cpu;
            ++report.materialSolves;
            bool terminalPublication=false;
            bool materialSolved=material.advanceCandidate(base,gasEndpoint,history,
                materialControls,candidate,corrected,cpu,error,&program);
            if(!materialSolved&&cpu.failureOrigin!=MaterialFailureOrigin::None
                &&(!cpu.recoverable||controller.retries()>=controls.maxWindowRetries)) {


                ++report.materialSolves;
                materialSolved=detail::ConductionContinuation::complete(material,base,gasEndpoint,history,
                    materialControls,candidate,corrected,cpu,error,&program);
                if(!materialSolved){discard("terminal material producer: "+error);return false;}
                terminalPublication=true;
            }
            if(!materialSolved) {
                const std::string cause="CPU material candidate: "+error;
                if(!cpu.recoverable){discard(cause);return false;}
                if(cpu.geometryCorrection&&!corrected.knots.empty()) {
                    if(!replay(std::move(corrected),cause))return false;
                } else if(!shorten(cpu.eventTime,cause,cpu.eventDuration))return false;
                continue;
            }
            WallProgramComparison comparison;
            if(!compareWallPrograms(program,corrected,model.physics.tolerances,comparison,error)) {
                discard("invalid corrected wall program: "+error);return false;
            }
            WindowAssessment assessment;
            assessment.regimeConsistent=comparison.regimeMatches;
            assessment.geometryMatches=comparison.geometryError<=1
                &&sameEndpointPoints(candidate.gasMesh,gasEndpoint.gasMesh)
                &&sameEndpointPoints(candidate.solidMesh,gasEndpoint.solidMesh);
            assessment.predictionError=std::max(comparison.thermalError,comparison.massPredictionError);
            const auto decision=controller.assess(assessment);
            if(!terminalPublication&&decision==WindowDecision::Replay
                &&(controller.iterations()<controls.maxWindowIterations||controller.retries()<controls.maxWindowRetries)) {
                if(!replay(std::move(corrected),"coupling waveform/regime/geometry requires replay"))return false;
                continue;
            }


            if(!finalizeIntervalLedger(history,candidate,error)) {
                discard("macro synchronization ledger: "+error);return false;
            }
            candidate.acceptedSteps=base.acceptedSteps+1;
            candidate.commitSequence=base.commitSequence+1;
            candidate.lastAcceptedDt=interval.end-interval.begin;
            candidate.nextDt=report.nextGasDt; // Compatibility slot is gas-only here.
            if(!commitGasWindow(backend,candidate,error)) {
                discard("atomic gas/material commit: "+error);return false;
            }
            active=false;
            report.acceptedGasMicrosteps=history.records().size();
            accepted=std::move(candidate);
            error.clear();return true;
        }
    } catch(const std::exception& exception) {
        discard(std::string("multirate transaction preserved synchronized base: ")+exception.what());
        return false;
    }
}
}
