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
        if(accepted.acceptedSteps==std::numeric_limits<std::uint64_t>::max()
            ||accepted.commitSequence==std::numeric_limits<std::uint64_t>::max())
            return intervalFailure(error,"macro commit counter overflow");
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
                if(!diagnostics.recoverable){error=cause;return false;}
                if(!finite(diagnostics.eventTime)||diagnostics.eventTime<0) {
                    error=cause+"; invalid typed predictor event time";return false;
                }
                const auto& interval=controller.interval();
                const Real duration=diagnostics.eventTime>interval.begin&&diagnostics.eventTime<interval.end
                    ?diagnostics.eventTime-interval.begin:.5*(interval.end-interval.begin);
                std::string retryError;
                if(!controller.shorten(duration,retryError)){error=cause+"; "+retryError;return false;}
                report.windowRetries=controller.retries();
            }
        };
        if(!predict("initial window predictor"))return false;
        auto shorten=[&](Real eventTime,const std::string& cause)->bool {
            if(!discard(cause))return false;
            const auto& interval=controller.interval();
            const Real duration=eventTime>interval.begin&&eventTime<interval.end
                ?eventTime-interval.begin:.5*(interval.end-interval.begin);
            std::string retryError;
            if(!controller.shorten(duration,retryError)){error=cause+"; "+retryError;return false;}
            report.windowRetries=controller.retries();
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
            ++report.windowIterations;
            report.geometry.clear();
            const CouplingInterval interval=controller.interval();
            if(program.interval.begin!=interval.begin||program.interval.end!=interval.end
                ||program.interval.identity.sequence!=interval.identity.sequence
                ||program.interval.identity.epoch!=interval.identity.epoch)
                return intervalFailure(error,"CPU predictor did not preserve coupling interval identity");
            if(!beginGasWindow(backend,program,error))return false;
            active=true;
            Real gasTime=interval.begin;
            bool retryWindow=false;
            while(gasTime<interval.end) {
                GasMicroReport micro;
                if(!advanceGasMicrostep(backend,controls.gasMaxDt,micro,error)) {
                    const std::string cause="gas microstep: "+error;
                    if(!micro.recoverable){discard(cause);return false;}
                    if(!shorten(0,cause))return false;
                    retryWindow=true;break;
                }
                if(!finite(micro.time)||!finite(micro.acceptedDt)||micro.acceptedDt<=0
                    ||micro.time<=gasTime||micro.time>interval.end
                    ||micro.acceptedDt>controls.gasMaxDt*(1+1e-12)
                    ||!finite(micro.nextGasDt)||micro.nextGasDt<=0) {
                    discard("gas window returned invalid microstep clocks");return false;
                }
                ++report.gasMicrosteps;
                report.gasRejectedTrials+=micro.rejectedTrials;
                report.nextGasDt=micro.nextGasDt;
                if(retainStageGeometry) {
                    MicrostepGeometryReport geometry;geometry.microSequence=micro.microSequence;
                    geometry.begin=gasTime;geometry.end=micro.time;
                    if(!gasWindowStageGeometry(backend,geometry.gas,geometry.solid,error)){
                        discard("gas microstage geometry: "+error);return false;}
                    report.geometry.push_back(std::move(geometry));
                }
                gasTime=micro.time;
            }
            if(retryWindow)continue;
            IntervalHistory history;
            HostState gasEndpoint;
            if(!gasWindowHistory(backend,history,error)||!history.complete()
                ||!downloadGasWindow(backend,gasEndpoint,error)) {
                discard("completed gas window unavailable: "+error);return false;
            }
            HostState candidate;
            WallProgram corrected;
            CpuMaterialReport cpu;
            ++report.materialSolves;
            const bool materialSolved=material.advanceCandidate(base,gasEndpoint,history,
                materialControls,candidate,corrected,cpu,error,&program);
            report.materialSubsteps+=cpu.materialSteps;report.reactionSubsteps+=cpu.reactionSteps;
            report.filmSubsteps+=cpu.filmSteps;report.linearSolves+=cpu.linearSolves;
            report.boundaryDriveSamples+=cpu.boundaryDriveSamples;
            report.materialRhsEvaluations+=cpu.materialRhsEvaluations;
            if(!materialSolved) {
                const std::string cause="CPU material candidate: "+error;
                if(!cpu.recoverable){discard(cause);return false;}
                if(cpu.geometryCorrection&&!corrected.knots.empty()) {
                    if(!replay(std::move(corrected),cause))return false;
                } else if(!shorten(cpu.eventTime,cause))return false;
                continue;
            }
            if(!finite(cpu.numericalMassRoundoff)) {
                discard("CPU material normalization diagnostic is nonfinite");return false;
            }
            if(candidate.time!=interval.end) {
                discard("CPU material candidate is not at the exact gas endpoint");return false;
            }
            WallProgramComparison comparison;
            if(!compareWallPrograms(program,corrected,model.physics.tolerances,comparison,error)) {
                discard("invalid corrected wall program: "+error);return false;
            }
            report.temperaturePredictionError=comparison.thermalError;
            report.massPredictionError=comparison.massPredictionError;
            report.geometryError=comparison.geometryError;
            WindowAssessment assessment;
            assessment.regimeConsistent=comparison.regimeMatches;
            assessment.geometryMatches=comparison.geometryError<=1
                &&sameEndpointPoints(candidate.gasMesh,gasEndpoint.gasMesh)
                &&sameEndpointPoints(candidate.solidMesh,gasEndpoint.solidMesh);
            assessment.predictionError=std::max(comparison.thermalError,comparison.massPredictionError);
            std::string donorError,auditError;
            assessment.donorValid=validateMaterialDonorHistory(base,history,
                corrected.donorPlan.knots.empty()?nullptr:&corrected.donorPlan,donorError);
            WindowConservationAudit audit;
            assessment.conservative=auditCoupledCandidate(base,candidate,history,model.physics,audit,auditError);
            report.energyResidual=audit.energyResidual;
            // Each actual gas/material geometric trajectory was independently
            // validated by its owner; comparison prevents committing another one.
            assessment.gclValid=finite(candidate.budget.gclResidual);
            const auto decision=controller.assess(assessment);
            if(decision==WindowDecision::Replay) {
                if(!replay(std::move(corrected),"coupling waveform/regime/geometry requires replay"))return false;
                continue;
            }
            if(decision==WindowDecision::Shorten) {
                const std::string cause=!donorError.empty()?donorError:
                    !auditError.empty()?auditError:"coupling geometric/admissibility audit failed";
                if(!shorten(cpu.eventTime,cause))return false;
                continue;
            }
            if(decision==WindowDecision::Reject) {
                discard("nonrecoverable coupling assessment");return false;
            }
            if(!finalizeIntervalLedger(history,candidate,error)
                ||!checkSynchronizedInterval(candidate,history,error)) {
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
            report.interval=interval;report.windowRetries=controller.retries();
            report.acceptedGasMicrosteps=history.records().size();
            report.nextCouplingInterval=controls.interval;
            // Rejected/replayed candidates never contribute a physical roundoff
            // diagnostic to the committed window. Do not reapply this amount.
            report.numericalMassRoundoff=cpu.numericalMassRoundoff;
            accepted=std::move(candidate);
            error.clear();return true;
        }
    } catch(const std::exception& exception) {
        discard(std::string("multirate transaction preserved synchronized base: ")+exception.what());
        return false;
    }
}
}
