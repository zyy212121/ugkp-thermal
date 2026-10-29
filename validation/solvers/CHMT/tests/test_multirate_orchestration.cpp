// Host transaction unit test with scripted participant responses. This verifies
// orchestration/rollback only; it does not mock a CFD run or establish accuracy.
#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "gpu/GasWindowProgram.H"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace chmt {
struct Backend {
    HostState accepted{},trial{};
    WallProgram program{};
    IntervalHistory history{};
    bool pending=false;
    bool failCommit=false,failRollback=false;
    int failInput=0;
    int begins=0,rollbacks=0,commits=0;
};
struct Script {
    int materialCalls=0,failMaterial=0,predictorCalls=0,rejectPredictionCall=0;
    bool failAll=false,movePrediction=false,badEnergy=false,permanent=false;
    bool failPredictor=false,permanentPredictor=false,conductionFailure=false,failTerminal=false;
    int terminalCalls=0,predictorTerminalCalls=0;Real terminalDt=0,predictorTerminalDt=0;
    bool numericalPredictor=false,failTerminalPredictor=false;
    MaterialFailureOrigin origin=MaterialFailureOrigin::None;
    Real eventTime=0,maxPredictionInterval=0,legacyAuditResidual=0;
    WallProgram current{};
} script;
struct CpuMaterialDriver::Implementation {};
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig&,Foam::fvMesh*):data_(new Implementation){}
CpuMaterialDriver::~CpuMaterialDriver()=default;
bool CpuMaterialDriver::predictWall(const HostState& base,const CouplingInterval& interval,WallProgram& out,std::string& error,CpuMaterialReport* diagnostics)const {
    ++script.predictorCalls;
    if(script.failPredictor||script.predictorCalls==script.rejectPredictionCall
        ||(script.maxPredictionInterval>0&&interval.end-interval.begin>script.maxPredictionInterval)) {
        if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=!script.permanentPredictor;diagnostics->failureOrigin=script.numericalPredictor?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;}
        error="scripted predictor failure";return false;
    }
    out=WallProgram{};out.interval=interval;out.surface=base.surface;
    WallKnot a,b;a.time=interval.begin;b.time=interval.end;
    a.gasPoints=b.gasPoints=base.gasMesh.points;a.solidPoints=b.solidPoints=base.solidMesh.points;
    out.knots={a,b};error.clear();return true;
}
bool CpuMaterialDriver::predictWallImpl(const HostState& base,const CouplingInterval& interval,WallProgram& out,std::string& error,CpuMaterialReport*,bool terminal)const {
    assert(terminal);++script.predictorTerminalCalls;script.predictorTerminalDt=interval.end-interval.begin;
    if(script.failTerminalPredictor){error="actual predictor input unavailable";return false;}
    out=WallProgram{};out.interval=interval;out.surface=base.surface;
    WallKnot first,last;first.time=interval.begin;last.time=interval.end;
    first.gasPoints=last.gasPoints=base.gasMesh.points;first.solidPoints=last.solidPoints=base.solidMesh.points;
    out.knots={first,last};error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& endpoint,const IntervalHistory& history,
    const CpuMaterialControls&,HostState& candidate,WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram*) {
    assert(base.time==history.interval().begin);++script.materialCalls;
    if(script.failAll||script.materialCalls<=script.failMaterial){report.failureOrigin=script.origin!=MaterialFailureOrigin::None?script.origin:script.conductionFailure?MaterialFailureOrigin::ConductionNumerical:MaterialFailureOrigin::None;report.recoverable=!script.permanent;report.eventTime=script.eventTime;error="scripted material failure";return false;}
    candidate=endpoint;corrected=script.current;report.materialSteps=3;
    candidate.budget.numericalEnergyResidual=script.legacyAuditResidual*script.materialCalls;
    if(script.movePrediction&&script.materialCalls==1)corrected.knots.back().gasPoints[0].x+=.1;
    if(script.badEnergy)candidate.gas[0].energy+=1;
    error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidateImpl(const HostState& base,const HostState& endpoint,const IntervalHistory& history,
    const CpuMaterialControls&,HostState& candidate,WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram*,bool terminal) {
    assert(terminal);++script.terminalCalls;script.terminalDt=history.interval().end-history.interval().begin;
    assert(endpoint.time==history.interval().end&&base.time==history.interval().begin);
    if(script.failTerminal){error="unfinished endpoint producer";return false;}
    candidate=endpoint;candidate.gas[0].energy+=2;corrected=script.current;report=CpuMaterialReport{};
    error.clear();return true;
}
bool beginGasWindow(Backend& b,const WallProgram& p,std::string& error) {
    assert(!b.pending);b.pending=true;++b.begins;b.program=p;b.trial=b.accepted;script.current=p;
    return b.history.begin(p.interval,error);
}
bool advanceGasMicrostep(Backend& b,Real maxDt,GasMicroReport& report,std::string& error) {
    assert(b.pending);const Real start=b.trial.time;const Real end=std::min(b.program.interval.end,start+maxDt);
    GasIntervalRecord record;record.microSequence=b.history.records().size()+1;record.begin=start;record.end=end;
    if(!b.history.appendAccepted(record,error))return false;
    b.trial.time=end;b.trial.nextDt=maxDt;b.trial.gasMesh.points=b.program.knots.back().gasPoints;
    report.acceptedDt=end-start;report.nextGasDt=maxDt;report.time=end;report.localTime=b.history.localEnd();report.microSequence=record.microSequence;
    return true;
}
const IntervalHistory& gasWindowHistory(const Backend& b){if(b.failInput==1)throw std::runtime_error("history access failed");return b.history;}
bool downloadGasWindow(Backend& b,HostState& out,std::string& error){if(b.failInput==2){error="endpoint transfer failed";return false;}out=b.trial;error.clear();return true;}
bool gasWindowStageGeometry(const Backend& b,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string& error){if(b.failInput==3){error="stage transfer failed";return false;}error.clear();return true;}
bool rollbackGasWindow(Backend& b,std::string& error){if(b.failRollback){error="actual rollback copy failure";return false;}if(b.pending)++b.rollbacks;b.pending=false;error.clear();return true;}
bool commitGasWindow(Backend& b,HostState& candidate,std::string& error){if(b.failCommit){error="actual commit copy failure";return false;}assert(b.pending);b.accepted=candidate;b.pending=false;++b.commits;error.clear();return true;}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
}

int main(){
    using namespace chmt;
    ModelConfig model;CpuMaterialControls material;
    CouplingControls controls;controls.interval=1;controls.gasMaxDt=.01;
    HostState initial;GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=3;initial.gas.push_back(gas);initial.gasMesh.points.push_back({0,0,0});
    CpuMaterialDriver cpu(model,nullptr);std::string error;
    {
        script=Script{};script.conductionFailure=true;script.failMaterial=1;
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(script.terminalCalls==0&&script.materialCalls==2&&accepted.time==.1&&backend.rollbacks==1);
    }
    for(int input=1;input<=3;++input){
        script=Script{};Backend backend;backend.accepted=initial;backend.failInput=input;
        HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,input==3,accepted,report,error));
        assert(script.materialCalls==0&&script.terminalCalls==0&&backend.commits==0&&accepted.time==initial.time&&!error.empty());
    }
    for(auto origin:{MaterialFailureOrigin::ConductionNumerical,MaterialFailureOrigin::ReactionNumerical,MaterialFailureOrigin::FilmNumerical,MaterialFailureOrigin::SurfaceNumerical,MaterialFailureOrigin::GeometryNumerical,MaterialFailureOrigin::TransportNumerical,MaterialFailureOrigin::LocalSourceNumerical,MaterialFailureOrigin::OwnerNumerical})
    for(int failure=0;failure<3;++failure){
        script=Script{};script.origin=origin;script.failAll=true;script.failTerminal=failure==1;
        Backend backend;backend.accepted=initial;backend.failCommit=failure==2;
        HostState accepted=initial;WindowEvolutionReport report;
        const bool ok=advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error);
        assert(ok==(failure==0));assert(script.materialCalls==3&&script.terminalCalls==1&&script.terminalDt==.05);
        assert(backend.rollbacks==(failure?3:2));
        if(ok){assert(accepted.time==.05&&accepted.gas[0].energy==5&&script.terminalCalls==1&&!backend.pending&&backend.commits==1);}
        else {assert(accepted.time==0&&backend.commits==0&&!error.empty());}
    }
    {
        script=Script{};script.conductionFailure=true;script.failAll=true;
        Backend backend;backend.accepted=initial;backend.failRollback=true;
        HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(script.terminalCalls==0&&script.materialCalls==1&&error.find("rollback")!=std::string::npos);
    }
    {
        script=Script{};Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.2&&accepted.acceptedSteps==1&&accepted.commitSequence==1);
        assert(script.materialCalls==1&&backend.begins==1&&backend.commits==1);
        assert(backend.history.records().size()>=20&&backend.program.interval.end-backend.program.interval.begin==.2);
        assert(accepted.lastAcceptedDt==.2&&report.nextGasDt==.01&&!backend.pending);
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
        assert(accepted.gasMesh.points[0].x==.1&&backend.program.interval.identity.epoch==1);
    }
    {
        script=Script{};script.failAll=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==0&&accepted.acceptedSteps==0&&backend.commits==0&&!backend.pending);
        assert(backend.begins==3&&backend.rollbacks==3);
    }
    {
        script=Script{};script.badEnergy=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));


        assert(accepted.gas[0].energy==4&&accepted.time==.2&&backend.commits==1&&!backend.pending);
        assert(script.materialCalls==1&&backend.rollbacks==0);
        assert(accepted.gas[0].energy!=initial.gas[0].energy);
        std::cout<<"nonconservative endpoint accepted without audit-only shortening\n";
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
        assert(script.predictorCalls==2&&backend.program.interval.identity.epoch==1&&script.predictorTerminalCalls==0);
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
        assert(accepted.time==.05&&script.predictorCalls==3&&backend.program.interval.identity.epoch==2);
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
    for(int failure=0;failure<2;++failure){
        script=Script{};script.failPredictor=true;script.numericalPredictor=true;script.failTerminalPredictor=failure;
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        const bool ok=advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error);
        assert(ok==!failure);assert(script.predictorCalls==3&&script.predictorTerminalCalls==1&&script.predictorTerminalDt==.05);
        assert(backend.begins==!failure&&backend.commits==!failure&&backend.rollbacks==0);
        assert(accepted.time==(failure?0:.05));
        if(ok){assert(backend.program.interval.end==.05&&backend.program.knots.back().time==.05);}
    }
    {


        script=Script{};script.legacyAuditResidual=-4e-18;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.budget.numericalEnergyResidual==script.legacyAuditResidual);
        assert(accepted.gas[0].mass==initial.gas[0].mass&&accepted.budget.boundaryMass==0);
    }
    {
        script=Script{};script.movePrediction=true;script.legacyAuditResidual=4e-18;
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(script.materialCalls==2&&accepted.budget.numericalEnergyResidual==2*script.legacyAuditResidual);
        assert(accepted.gas[0].mass==initial.gas[0].mass&&accepted.budget.boundaryMass==0);
    }
    {
        script=Script{};script.legacyAuditResidual=std::numeric_limits<Real>::quiet_NaN();
        Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
        assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,false,accepted,report,error));
        assert(accepted.time==.2&&accepted.gas[0].mass==initial.gas[0].mass&&backend.commits==1&&std::isnan(accepted.budget.numericalEnergyResidual));
    }
    std::cout<<"multirate orchestration host unit tests passed (no CFD/OF/CUDA execution)\n";
}
