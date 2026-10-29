#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "coupling/IntervalAudit.H"
#include "DiagnosticsFoam.H"
#include <algorithm>
#include <exception>

namespace chmt {
bool advanceCoupledWindow(Backend& backend,CpuMaterialDriver& material,const ModelConfig& model,
    const CouplingControls& controls,const CpuMaterialControls& materialControls,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport& report,std::string& error) {
    report=WindowEvolutionReport{};
    try {
        const HostState& base=accepted;
        CouplingWindowController controller;
        if(!controller.begin(base.time,targetTime,base.commitSequence+1,controls,error))return false;
        const CouplingInterval interval=controller.interval();
        WallProgram program;
        CpuMaterialReport prediction;
        if(!material.predictWall(base,interval,program,error,&prediction))return false;
        if(!beginGasWindow(backend,program,error))return false;
        Real gasTime=interval.begin,nextGasCap=controls.gasMaxDt;WindowTime gasElapsed{};
        // Ordinary physical gas microsteps remain. A failed operation returns
        // directly to the frontend fatal boundary, without rollback or replay.
        while(gasElapsed<intervalDuration(interval)) {
            GasMicroReport micro;
            if(!advanceGasMicrostep(backend,nextGasCap,micro,error))return false;
            report.nextGasDt=micro.nextGasDt;
            if(retainStageGeometry) {
                MicrostepGeometryReport geometry;geometry.microSequence=micro.microSequence;
                geometry.begin=gasTime;geometry.end=micro.time;
                if(!gasWindowStageGeometry(backend,geometry.gas,geometry.solid,error))return false;
                report.geometry.push_back(std::move(geometry));
            }
            gasTime=micro.time;gasElapsed=micro.localTime;nextGasCap=std::min(controls.gasMaxDt,micro.nextGasDt);
        }
        // The backend owns the completed immutable history until the next
        // window begins. All material/ledger consumers finish before then.
        const IntervalHistory& history=gasWindowHistory(backend);
        HostState gasEndpoint;
        if(!downloadGasWindow(backend,gasEndpoint,error))return false;
        HostState candidate;WallProgram corrected;CpuMaterialReport cpu;
        ++report.materialSolves;
        if(!material.advanceCandidate(base,gasEndpoint,history,materialControls,candidate,corrected,cpu,error,&program))return false;
        WallProgramComparison comparison;
        if(!compareWallPrograms(program,corrected,model.physics.tolerances,comparison,error))return false;
        const Real residual=std::max(comparison.geometryError,std::max(comparison.thermalError,comparison.massPredictionError));
        if(!comparison.regimeMatches||residual>1) {
            using namespace ugkp::diagnostics;
            auto record=makeRecord(Code::AccuracyResidual,4001,4,0,EntityKind::None,-1,-1,residual,1,residual,Quantity::Accuracy);
            record.operationSequence=base.commitSequence+1;record.step=base.acceptedSteps;record.simulationTime=interval.end;
            Snapshot snapshot;snapshot.warnings.push_back({record,1,false});
            snapshot.sources.push_back({4001,"applications/CHMT/io/MultirateEvolution.C","advanceCoupledWindow",__LINE__,"single-pass wall waveform/regime mismatch; computed result retained"});
            dispatch(snapshot,makeFoamSink());
        }
        if(!finalizeIntervalLedger(history,candidate,error))return false;
        candidate.acceptedSteps=base.acceptedSteps+1;
        candidate.commitSequence=base.commitSequence+1;
        candidate.lastAcceptedDt=interval.end-interval.begin;
        candidate.nextDt=report.nextGasDt;
        if(!commitGasWindow(backend,candidate,error))return false;
        report.acceptedGasMicrosteps=history.records().size();
        accepted=std::move(candidate);error.clear();return true;
    } catch(const Foam::error&) {
        // Preserve the original GPU/OpenFOAM diagnostic for the application
        // boundary; Foam::error::what() alone discards its physical context.
        throw;
    } catch(const std::exception& exception) {
        error=std::string("single-pass multirate operation failed: ")+exception.what();
        return false;
    }
}
}
