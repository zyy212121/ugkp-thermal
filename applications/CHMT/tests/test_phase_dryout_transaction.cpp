// Promoted reviewed regression. Scripted gas/material ownership surrounds the
// actual IntervalCursor, macro controller, audit and CPU film update. No native CFD.
// Transaction/film component integration only. No CFD or OpenFOAM solver.
#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "gpu/GasWindowProgram.H"
#include "film/CpuFilmDriver.H"
#include "film/FilmMath.H"
#include <cassert>
#include <iostream>
namespace chmt {
struct Backend { HostState accepted,trial; WallProgram program; IntervalHistory history; bool pending=false; int begins=0,commits=0,rollbacks=0; };
WallProgram executed;PhysicsConfig physics;int filmPacketsAfterEvent=0;
struct CpuMaterialDriver::Implementation{};
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig&,Foam::fvMesh*):data_(new Implementation){}
CpuMaterialDriver::~CpuMaterialDriver()=default;
bool CpuMaterialDriver::predictWall(const HostState& base,const CouplingInterval& window,WallProgram& out,std::string& error,CpuMaterialReport*)const {
 out={};out.interval=window;out.surface=base.surface;WallKnot a,b;a.time=window.begin;b.time=window.end;
 WallFaceSample face;face.temperature=300;face.primaryKind=base.film[0].mass>0?ExchangeKind::GasFilm:ExchangeKind::GasSolid;a.faces={face};b.faces={face};out.knots={a,b};error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& endpoint,const IntervalHistory& history,const CpuMaterialControls&,HostState& out,WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram*) {
 out=endpoint;out.ledger.clear();std::vector<CpuFilmForcing> forcing(1);forcing[0].topWorkIncluded=forcing[0].bottomWorkIncluded=true;
 auto cursor=history.cursor();std::vector<ExchangePacket> packets;std::vector<Real> radiation;
 if(!cursor.takeThrough(history.interval().end,packets,radiation,error))return false;
 for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasFilm){const auto delta=packetDelta(packet);forcing[0].mass+=delta.film.mass;forcing[0].energy-=packet.energy;for(int s=0;s<Ns;++s)forcing[0].species[s]+=delta.film.species[s];}
 std::vector<CpuFilmDrive> drive(1);drive[0].pressure=100000;drive[0].hasCoupledNormalTrace=true;CpuFilmCandidate film;
 if(!advanceCpuFilmCandidate(base,physics,history.interval().end-history.interval().begin,forcing,drive,film,error))return false;
 out.film=film.film;out.filmAux=film.filmAux;
 out.budget.exchangeMass[FilmParticipant]+=film.report.budgetDelta.exchangeMass[FilmParticipant];out.budget.exchangeEnergy[FilmParticipant]+=film.report.budgetDelta.exchangeEnergy[FilmParticipant];
 for(int s=0;s<Ns;++s)out.budget.exchangeSpecies[FilmParticipant][s]+=film.report.budgetDelta.exchangeSpecies[FilmParticipant][s];
 out.budget.filmPressureVolume+=film.report.budgetDelta.filmPressureVolume;out.budget.filmKineticDefect+=film.report.budgetDelta.filmKineticDefect;
 corrected=executed;corrected.knots.back().faces[0].primaryKind=out.film[0].mass>0?ExchangeKind::GasFilm:ExchangeKind::GasSolid;
 // Match production CPU: old owner owns [begin,end); endpoint state seeds next window.
 corrected.knots.back().faces[0].primaryKind=corrected.knots.front().faces[0].primaryKind;
 report.materialSteps=1;report.filmSteps=1;return true;
}
bool beginGasWindow(Backend& b,const WallProgram& p,std::string& error,const GasWindowLimits&) {
 if(!validateGasWallProgram(p,b.accepted,error))return false;
 assert(!b.pending);b.pending=true;++b.begins;b.program=executed=p;b.trial=b.accepted;return b.history.begin(p.interval,error);
}
bool advanceGasMicrostep(Backend& b,Real dt,GasMicroReport& report,std::string& error) {
 const Real start=b.trial.time,end=std::min(start+dt,b.program.interval.end);GasIntervalRecord record;record.begin=start;record.end=end;record.microSequence=b.history.records().size()+1;
 ExchangePacket p;p.kind=b.program.knots.front().faces[0].primaryKind;p.step=record.microSequence;p.stage=1;p.gasCell=0;p.solidCell=0;p.filmFace=0;p.consumerMask=ConsumeGas;
 if(p.kind==ExchangeKind::GasFilm){if(b.accepted.time>=1)++filmPacketsAfterEvent;p.mass=(end-start)/(b.program.interval.end-b.program.interval.begin)*b.accepted.film[0].mass;p.species[0]=p.mass;p.energy=p.advective=600000*p.mass;}
 record.packets={p};if(!b.history.appendAccepted(record,error))return false;
 const auto delta=packetDelta(p);b.trial.gas[0]+=delta.gas;b.trial.budget.exchangeMass[GasParticipant]+=p.mass;b.trial.budget.exchangeEnergy[GasParticipant]+=p.energy;b.trial.budget.exchangeSpecies[GasParticipant][0]+=p.mass;b.trial.time=end;
 report.acceptedDt=end-start;report.nextGasDt=dt;report.time=end;report.microSequence=record.microSequence;return true;
}
bool gasWindowHistory(const Backend& b,IntervalHistory& h,std::string&){h=b.history;return true;}
bool downloadGasWindow(const Backend& b,HostState& h,std::string&){h=b.trial;return true;}
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>&,std::array<HostStageGeometry,2>&,std::string&){return true;}
bool rollbackGasWindow(Backend& b,std::string&){b.pending=false;++b.rollbacks;return true;}
bool commitGasWindow(Backend& b,const HostState& c,std::string&){b.accepted=c;b.pending=false;++b.commits;return true;}
}
int main(){using namespace chmt;ModelConfig model;auto& p=model.physics;p.enableFilm=true;p.liquid.rho=1000;p.liquid.cp0=2000;p.liquid.Tmin=100;p.liquid.Tmax=2000;p.liquidViscosity=1;physics=p;
 HostState state;state.surface.area={1};state.surface.oldArea={1};state.surface.normal={{0,0,1}};state.surface.gasFace={0};state.surface.solidFace={0};state.surface.solidCell={0};state.surface.persistentId={1};state.surface.baseVelocity={{0,0,0}};
 GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=1000000;state.gas={gas};state.solid.resize(1);state.solid[0].condensed[0]=1;state.solid[0].energy=100;
 FilmQ film;film.mass=film.species[0]=1;film.enthalpy=600100;state.film={film};state.filmAux.resize(1);assert(recoverFilm(film,1,100000,p,state.filmAux[0]));
 Backend backend;backend.accepted=state;CpuMaterialDriver cpu(model,nullptr);CouplingControls controls;controls.interval=1;controls.gasMaxDt=.25;controls.minimumInterval=1e-8;CpuMaterialControls local;WindowEvolutionReport report;std::string error;
 bool ok=advanceCoupledWindow(backend,cpu,model,controls,local,1,false,state,report,error);if(!ok){std::cerr<<error<<'\n';return 1;}
 assert(state.time==1&&state.film[0].mass==0&&state.film[0].enthalpy==0&&backend.commits==1&&backend.rollbacks==0);assert(state.ledger.size()==4&&state.ledger[0].kind==ExchangeKind::GasFilm);
 ok=advanceCoupledWindow(backend,cpu,model,controls,local,2,false,state,report,error);if(!ok){std::cerr<<error<<'\n';return 1;}
 assert(state.time==2&&backend.commits==2&&backend.rollbacks==0&&filmPacketsAfterEvent==0);assert(state.ledger.size()==4&&state.ledger[0].kind==ExchangeKind::GasSolid);
 std::cout<<"PASS: real film+cursor endpoint dryout through scripted transaction; old GasFilm owner commits once, next window GasSolid, no post-event GasFilm packet\n";
}
