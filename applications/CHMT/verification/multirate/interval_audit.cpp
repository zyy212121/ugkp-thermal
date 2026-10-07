#include "coupling/IntervalAudit.H"
#include <iostream>
#include <stdexcept>
using namespace chmt;
static void check(bool yes,const char* what){if(!yes)throw std::runtime_error(what);}
int main(){try{
 std::string e;PhysicsConfig p;HostState base,next;base.gas.resize(1);base.solid.resize(1);base.gas[0].mass=1;base.gas[0].species[0]=1;base.gas[0].energy=10;base.solid[0].condensed[0]=1;base.solid[0].energy=20;
 p.nElements=1;p.species[0].element[0]=p.condensed[0].element[0]=1;
 CouplingInterval window;window.end=1;IntervalHistory history;check(history.begin(window,e),"begin");GasIntervalRecord r;r.end=1;r.gasGeometry=2;
 ExchangePacket x;x.kind=ExchangeKind::GasSolid;x.stage=1;x.geometry=2;x.face=1;x.gasCell=x.solidCell=0;x.mass=x.species[0]=x.condensed[0]=.1;x.energy=x.conductive=7;x.consumerMask=ConsumeGas;r.packets={x};check(history.appendAccepted(r,e),"append");
 next=base;next.time=1;next.gas[0].mass+=.1;next.gas[0].species[0]+=.1;next.gas[0].energy+=7;next.solid[0].condensed[0]-=.1;next.solid[0].energy-=7;
 next.budget.exchangeMass[GasParticipant]=.1;next.budget.exchangeMass[SolidParticipant]=-.1;next.budget.exchangeEnergy[GasParticipant]=7;next.budget.exchangeEnergy[SolidParticipant]=-7;next.budget.exchangeSpecies[GasParticipant][0]=.1;next.budget.exchangeSpecies[SolidParticipant][0]=-.1;
 WindowConservationAudit audit;check(auditCoupledCandidate(base,next,history,p,audit,e),"valid paired candidate");
 const Budget correctBudget=next.budget;next.budget.exchangeMass[SolidParticipant]=0;next.budget.exchangeMass[FilmParticipant]=-.1;next.budget.exchangeEnergy[SolidParticipant]=0;next.budget.exchangeEnergy[FilmParticipant]=-7;next.budget.exchangeSpecies[SolidParticipant][0]=0;next.budget.exchangeSpecies[FilmParticipant][0]=-.1;
 check(!auditCoupledCandidate(base,next,history,p,audit,e),"wrong CPU owner attribution accepted");next.budget=correctBudget;next.solid[0].energy+=1;check(!auditCoupledCandidate(base,next,history,p,audit,e),"unpaired heat passed audit");next.solid[0].energy-=1;
 check(finalizeIntervalLedger(history,next,e),"finalize ledger");check(checkSynchronizedInterval(next,history,e),"synchronized guard");next.ledger[0].consumerMask=ConsumeGas;check(!checkSynchronizedInterval(next,history,e),"pending checkpoint allowed");
 WallProgram a,b;a.interval=b.interval=window;a.surface.area={1};b.surface=a.surface;WallKnot k;k.time=0;k.faces.resize(1);k.faces[0].temperature=300;k.gasPoints={{0,0,0}};a.knots={k,k};a.knots[1].time=1;b=a;b.knots[1].faces[0].temperature=310;
 WallProgramComparison difference;check(compareWallPrograms(a,b,p.tolerances,difference,e),"compare");check(difference.thermalError>1&&difference.geometryError==0&&difference.massPredictionError==0&&difference.regimeMatches,"error channels mixed");
 b=a;b.knots[1].gasPoints[0].x=.05;WallProgramComparison origin,translated;
 check(compareWallPrograms(a,b,p.tolerances,origin,e),"origin geometry comparison");
 for(auto* program:{&a,&b})for(auto& knot:program->knots)for(auto& point:knot.gasPoints)point.x+=1e9;
 check(compareWallPrograms(a,b,p.tolerances,translated,e),"translated geometry comparison");
 check(origin.geometryError>1&&translated.geometryError>1,"translation changed geometry mismatch acceptance");
 b=a;b.knots[1].gasPoints[0].x+=.0625;check(compareWallPrograms(a,b,p.tolerances,translated,e),"binary translated mismatch");
 for(auto* program:{&a,&b})for(auto& knot:program->knots)for(auto& point:knot.gasPoints)point.x-=1e9;
 check(compareWallPrograms(a,b,p.tolerances,origin,e),"binary origin mismatch");check(origin.geometryError==translated.geometryError,"geometry scale depends on origin");
 a.surface.area={4};b=a;b.knots[1].gasPoints[0].x+=1e-6;Tolerances volumeTolerance=p.tolerances;volumeTolerance.absoluteGeometry=8e-6;volumeTolerance.relativeGeometry=0;
 check(compareWallPrograms(a,b,volumeTolerance,difference,e)&&difference.geometryError<=1,"volume tolerance not converted through face area");
 b.knots[1].gasPoints[0].x+=2e-6;check(compareWallPrograms(a,b,volumeTolerance,difference,e)&&difference.geometryError>1,"face-area displacement tolerance too loose");
 b=a;b.knots.back().faces[0].primaryKind=ExchangeKind::GasFilm;
 check(compareWallPrograms(a,b,p.tolerances,difference,e)&&!difference.regimeMatches,"uncertified endpoint owner change accepted");
 // A located terminal phase event keeps the executed owner in THIS interval's
 // program; the next interval predictor derives its new owner from FilmQ.
 b.knots.back().faces[0].primaryKind=a.knots.back().faces[0].primaryKind;
 check(compareWallPrograms(a,b,p.tolerances,difference,e)&&difference.regimeMatches,"half-open executed-owner program rejected");
 CouplingInterval unvisitedWindow;unvisitedWindow.end=3;IntervalHistory unvisitedHistory;check(unvisitedHistory.begin(unvisitedWindow,e),"unvisited history begin");GasIntervalRecord emptyRecord;emptyRecord.end=3;emptyRecord.microSequence=1;check(unvisitedHistory.appendAccepted(emptyRecord,e),"unvisited complete history");
 MaterialDonorPlan unvisitedPlan;unvisitedPlan.interval=unvisitedWindow;for(int i=0;i<4;++i){MaterialReserveKnot knot;knot.time=i;knot.cumulativeSolid.resize(1);knot.cumulativeSolid[0].condensed[0]=i==1?-2:0;unvisitedPlan.knots.push_back(knot);}
 check(!validateMaterialDonorHistory(base,unvisitedHistory,&unvisitedPlan,e),"final history validator missed untouched internal deficit");
 std::cout<<"interval audits: PASS\n";return 0;
 }catch(const std::exception& x){std::cerr<<x.what()<<'\n';return 1;}}
