

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
    int begins=0,rollbacks=0,commits=0,fullPublications=0,compactPublications=0,acceptedPublications=0;
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

bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& endpoint,const IntervalHistory& history,
    const CpuMaterialControls&,HostState& candidate,WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram*,const ResidentMaterialDriveHistory*) {
    assert(base.time==history.interval().begin);++script.materialCalls;
    if(script.failAll||script.materialCalls<=script.failMaterial){report.failureOrigin=script.origin!=MaterialFailureOrigin::None?script.origin:script.conductionFailure?MaterialFailureOrigin::ConductionNumerical:MaterialFailureOrigin::None;report.recoverable=!script.permanent;report.eventTime=script.eventTime;error="scripted material failure";return false;}
    candidate=endpoint;corrected=script.current;report.materialSteps=3;
    candidate.budget.numericalEnergyResidual=script.legacyAuditResidual*script.materialCalls;
    if(script.movePrediction&&script.materialCalls==1)corrected.knots.back().gasPoints[0].x+=.1;
    if(script.badEnergy)candidate.gas[0].energy+=1;
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
bool downloadGasWindow(Backend& b,HostState& out,std::string& error){++b.fullPublications;if(b.failInput==2){error="endpoint transfer failed";return false;}out=b.trial;error.clear();return true;}
bool gasWindowStageGeometry(Backend& b,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string& error){if(b.failInput==3){error="stage transfer failed";return false;}error.clear();return true;}
bool rollbackGasWindow(Backend& b,std::string& error){if(b.failRollback){error="actual rollback copy failure";return false;}if(b.pending)++b.rollbacks;b.pending=false;error.clear();return true;}
bool commitGasWindow(Backend& b,HostState& candidate,std::string& error,bool){if(b.failCommit){error="actual commit copy failure";return false;}assert(b.pending);b.accepted=candidate;b.pending=false;++b.commits;error.clear();return true;}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
bool publishGasWindow(Backend& b,HostState& out,std::string& error){++b.compactPublications;if(b.failInput==2){error="endpoint transfer failed";return false;}out=b.trial;error.clear();return true;}
bool publishAcceptedState(Backend& b,HostState& out,std::string& error){++b.acceptedPublications;out=b.accepted;error.clear();return true;}
ResidentMaterialDriveHistory residentGasWindowDriveHistory(const Backend&){return {};}
std::vector<GasIntervalRecord> releaseGasWindowRecords(Backend& b){return b.history.records();}

}


int main(){
 using namespace chmt;
 ModelConfig model;CpuMaterialControls material;CouplingControls controls;controls.clock.contactPeriod=.1;controls.gasMaxDt=.01;
 HostState initial;GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=3;initial.gas={gas};initial.gasMesh.points={{0,0,0}};
 CpuMaterialDriver cpu(model,nullptr);std::string error;
 for(bool full:{false,true}){
  script=Script{};Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;
  for(int i=0;i<3;++i)assert(advanceCoupledWindow(backend,cpu,model,controls,material,Real(.3),false,accepted,report,error,nullptr,full));
  assert(backend.commits==3&&script.materialCalls==3&&backend.rollbacks==0);
  assert(backend.fullPublications==(full?3:0)&&backend.compactPublications==(full?0:3));
  assert(publishAcceptedState(backend,accepted,error)&&backend.acceptedPublications==1);
  assert(accepted.acceptedSteps==3&&accepted.commitSequence==3&&accepted.gas[0].energy==3);
 }
 for(int failure:{1,2,3}){
  script=Script{};Backend backend;backend.accepted=initial;backend.failInput=failure;HostState accepted=initial;WindowEvolutionReport report;
  assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,failure==3,accepted,report,error,nullptr,false));
  assert(backend.commits==0&&accepted.time==initial.time&&!error.empty());
 }
 std::cout<<"multirate single-pass publication tests passed\n";
}
