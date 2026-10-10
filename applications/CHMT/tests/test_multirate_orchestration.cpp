// Host transaction unit test with scripted participant responses. This verifies
// orchestration/rollback only; it does not mock a CFD run or establish accuracy.
#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "gpu/GasWindowProgram.H"
#include <cassert>
#include <iostream>

namespace chmt {
struct Backend {
    HostState accepted{},trial{};
    WallProgram program{};
    IntervalHistory history{};
    bool pending=false;
    int begins=0,rollbacks=0,commits=0;
};
struct Script {
    int materialCalls=0,failMaterial=0,predictorCalls=0,rejectPredictionCall=0;
    bool failAll=false,movePrediction=false,badEnergy=false,permanent=false;
    bool failPredictor=false,permanentPredictor=false;
    Real eventTime=0,maxPredictionInterval=0,numericalMassRoundoff=0;
    WallProgram current{};
} script;
struct CpuMaterialDriver::Implementation {};
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig&,Foam::fvMesh*):data_(new Implementation){}
CpuMaterialDriver::~CpuMaterialDriver()=default;
bool CpuMaterialDriver::predictWall(const HostState& base,const CouplingInterval& interval,WallProgram& out,std::string& error,CpuMaterialReport* diagnostics)const {
    ++script.predictorCalls;
    if(script.failPredictor||script.predictorCalls==script.rejectPredictionCall
        ||(script.maxPredictionInterval>0&&interval.end-interval.begin>script.maxPredictionInterval)) {
        if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=!script.permanentPredictor;}
        error="scripted predictor failure";return false;
    }
    out=WallProgram{};out.interval=interval;out.surface=base.surface;
    WallKnot a,b;a.time=interval.begin;b.time=interval.end;
    a.gasPoints=b.gasPoints=base.gasMesh.points;a.solidPoints=b.solidPoints=base.solidMesh.points;
    out.knots={a,b};error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& endpoint,const IntervalHistory& history,
    const CpuMaterialControls&,HostState& candidate,WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram*) {
    assert(base.time==history.interval().begin);++script.materialCalls;
    if(script.failAll||script.materialCalls<=script.failMaterial){report.recoverable=!script.permanent;report.eventTime=script.eventTime;error="scripted material failure";return false;}
    candidate=endpoint;corrected=script.current;report.materialSteps=3;report.linearSolves=3;
    report.numericalMassRoundoff=script.numericalMassRoundoff*script.materialCalls;
    if(script.movePrediction&&script.materialCalls==1)corrected.knots.back().gasPoints[0].x+=.1;
    if(script.badEnergy)candidate.gas[0].energy+=1;
    error.clear();return true;
}
bool beginGasWindow(Backend& b,const WallProgram& p,std::string& error,const GasWindowLimits&) {
    assert(!b.pending);b.pending=true;++b.begins;b.program=p;b.trial=b.accepted;script.current=p;
    return b.history.begin(p.interval,error);
}
bool advanceGasMicrostep(Backend& b,Real maxDt,GasMicroReport& report,std::string& error) {
    assert(b.pending);const Real start=b.trial.time;const Real end=std::min(b.program.interval.end,start+maxDt);
    GasIntervalRecord record;record.microSequence=b.history.records().size()+1;record.begin=start;record.end=end;
    if(!b.history.appendAccepted(record,error))return false;
    b.trial.time=end;b.trial.nextDt=maxDt;b.trial.gasMesh.points=b.program.knots.back().gasPoints;
    report.acceptedDt=end-start;report.nextGasDt=maxDt;report.time=end;report.microSequence=record.microSequence;
    return true;
}
bool gasWindowHistory(const Backend& b,IntervalHistory& out,std::string& error){out=b.history;error.clear();return true;}
bool downloadGasWindow(const Backend& b,HostState& out,std::string& error){out=b.trial;error.clear();return true;}
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string& error){error.clear();return true;}
bool rollbackGasWindow(Backend& b,std::string& error){if(b.pending)++b.rollbacks;b.pending=false;error.clear();return true;}
bool commitGasWindow(Backend& b,const HostState& candidate,std::string& error){assert(b.pending);b.accepted=candidate;b.pending=false;++b.commits;error.clear();return true;}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
}

int main(){
    using namespace chmt;
    ModelConfig model;CpuMaterialControls material;
    CouplingControls controls;controls.interval=1;controls.minimumInterval=1e-6;controls.gasMaxDt=.01;controls.maxWindowRetries=2;
    HostState initial;GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=3;initial.gas.push_back(gas);initial.gasMesh.points.push_back({0,0,0});
    CpuMaterialDriver cpu(model,nullptr);std::string error;
    {
        script=Script{};Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.2&&accepted.acceptedSteps==1&&accepted.commitSequence==1);
        assert(script.materialCalls==1&&backend.begins==1&&backend.commits==1);
        assert(report.gasMicrosteps>=20&&report.interval.end-report.interval.begin==.2);
        assert(report.nextCouplingInterval==1&&report.nextGasDt==.01&&!backend.pending);
    }
    {
        script=Script{};script.failMaterial=1;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.1&&script.materialCalls==2&&backend.rollbacks==1&&backend.commits==1);
    }
    {
        script=Script{};script.movePrediction=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.2&&backend.begins==2&&backend.rollbacks==1&&backend.commits==1);
        assert(accepted.gasMesh.points[0].x==.1&&report.interval.identity.epoch==1);
    }
    {
        script=Script{};script.failAll=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&accepted.acceptedSteps==0&&backend.commits==0&&!backend.pending);
        assert(backend.begins==3&&backend.rollbacks==3);
    }
    {
        script=Script{};script.badEnergy=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.gas[0].energy==3&&accepted.time==0&&backend.commits==0&&!backend.pending);
    }
    {
        script=Script{};script.failMaterial=1;script.eventTime=.075;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.075&&backend.begins==2&&backend.rollbacks==1);
    }
    {
        script=Script{};script.failAll=true;script.permanent=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&backend.begins==1&&backend.rollbacks==1&&backend.commits==0);
    }
    {
        script=Script{};script.maxPredictionInterval=.1;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.1&&script.predictorCalls==2&&backend.begins==1&&backend.commits==1);
        assert(report.windowRetries==1&&report.interval.identity.epoch==1);
    }
    {
        script=Script{};script.failPredictor=true;script.permanentPredictor=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&accepted.gas[0].energy==3&&script.predictorCalls==1);
        assert(backend.begins==0&&backend.commits==0&&backend.rollbacks==0);
    }
    {
        script=Script{};script.failMaterial=1;script.rejectPredictionCall=2;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.05&&script.predictorCalls==3&&report.windowRetries==2);
        assert(backend.begins==2&&backend.rollbacks==1&&backend.commits==1);
    }
    {
        script=Script{};script.failPredictor=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&accepted.acceptedSteps==0&&accepted.commitSequence==0);
        assert(script.predictorCalls==3&&backend.begins==0&&backend.commits==0);
    }
    {
        script=Script{};script.failMaterial=1;script.rejectPredictionCall=2;script.permanentPredictor=true;
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&script.predictorCalls==2&&backend.begins==1&&backend.rollbacks==1&&backend.commits==0);
    }
    {
        script=Script{};script.failPredictor=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        CouplingControls bounded=controls;bounded.minimumInterval=.1;bounded.maxWindowRetries=20;
        assert(!advanceCoupledWindow(backend,cpu,model,bounded,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&script.predictorCalls==2&&backend.begins==0&&backend.commits==0);
    }
    {
        script=Script{};script.numericalMassRoundoff=-4e-18;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(report.numericalMassRoundoff==script.numericalMassRoundoff);
        assert(accepted.gas[0].mass==initial.gas[0].mass&&accepted.budget.boundaryMass==0);
    }
    {
        script=Script{};script.movePrediction=true;script.numericalMassRoundoff=4e-18;
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(script.materialCalls==2&&report.numericalMassRoundoff==2*script.numericalMassRoundoff);
        assert(accepted.gas[0].mass==initial.gas[0].mass&&accepted.budget.boundaryMass==0);
    }
    {
        script=Script{};script.numericalMassRoundoff=std::numeric_limits<Real>::quiet_NaN();
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&accepted.gas[0].mass==initial.gas[0].mass&&backend.commits==0);
    }
    std::cout<<"multirate orchestration host unit tests passed (no CFD/OF/CUDA execution)\n";
}
