#include "restart/Checkpoint.H"
#include <cstdio>
#include <cmath>
#include <string>
using namespace chmt;
void num(double x){if(std::isfinite(x))std::printf("%.17g",x);else std::printf("\"%s\"",std::isnan(x)?"NaN":x<0?"-Infinity":"Infinity");}
template<class T>void arr(const T& x){std::printf("[");bool first=true;for(auto y:x){if(!first)std::printf(",");first=false;num(double(y));}std::printf("]");}
void vec(Vec3 x){double a[3]={x.x,x.y,x.z};arr(a);}
void mesh(const HostMesh&m){std::printf("{\"volumes\":");arr(m.volumes);std::printf(",\"cellFaceOffsets\":");arr(m.cellFaceOffsets);std::printf(",\"cellFaces\":");arr(m.cellFaces);std::printf(",\"cellFaceSigns\":");arr(m.cellFaceSigns);std::printf("}");}
void stage(const HostStageGeometry&s){std::printf("{\"interval\":");num(s.interval);std::printf(",\"oldVolume\":");arr(s.oldVolume);std::printf(",\"newVolume\":");arr(s.newVolume);std::printf(",\"evaluationVolume\":");arr(s.evaluationVolume);std::printf(",\"sweptVolume\":");arr(s.sweptVolume);std::printf("}");}
int main(int argc,char**argv){if(argc!=2)return 2;HostState s;ModelConfig model;std::string error;if(!readCheckpoint(argv[1],model,s,error)){std::fprintf(stderr,"READ_ERROR %s\n",error.c_str());return 3;}
 std::printf("{\"time\":");num(s.time);std::printf(",\"gas\":[");bool first=true;for(auto&q:s.gas){if(!first)std::printf(",");first=false;std::printf("{\"mass\":");num(q.mass);std::printf(",\"energy\":");num(q.energy);std::printf(",\"momentum\":");vec(q.momentum);std::printf(",\"species\":");arr(q.species);std::printf("}");}
 std::printf("],\"solid\":[");first=true;for(auto&q:s.solid){if(!first)std::printf(",");first=false;std::printf("{\"condensed\":");arr(q.condensed);std::printf(",\"pore\":");arr(q.pore);std::printf(",\"energy\":");num(q.energy);std::printf(",\"porosity\":");num(q.porosity);std::printf("}");}
 std::printf("],\"sst\":[");first=true;for(auto&q:s.sst){if(!first)std::printf(",");first=false;double a[2]={q.rhoK,q.rhoOmega};arr(a);}std::printf("]");
 std::printf(",\"gasSstAudit\":{");
#define AUD(x) std::printf("\"" #x "\":");num(s.gasSstAudit.x);
 AUD(transportK);std::printf(",");AUD(transportOmega);std::printf(",");AUD(sourceK);std::printf(",");AUD(sourceOmega);std::printf(",");AUD(constraintK);std::printf(",");AUD(constraintOmega);
#undef AUD
 std::printf("},\"physics\":{\"liquidRho\":");num(model.physics.liquid.rho);
 std::printf(",\"phaseCondensed\":%d,\"enableMelting\":%d,\"meltTemperature\":",model.physics.material.phaseCondensed,int(model.physics.material.enableMelting));num(model.physics.meltTemperature);
 std::printf(",\"liquidE0\":");num(model.physics.liquid.e0);std::printf(",\"liquidReferencePressure\":");num(model.physics.liquidReferencePressure);
 std::printf(",\"condensedRho\":[");for(int i=0;i<Nc;++i){if(i)std::printf(",");num(model.physics.condensed[i].rho);}std::printf("]}");
 std::printf(",\"solidMesh\":");mesh(s.solidMesh);
 std::printf(",\"surface\":{\"area\":");arr(s.surface.area);std::printf(",\"gasFace\":");arr(s.surface.gasFace);std::printf(",\"solidFace\":");arr(s.surface.solidFace);std::printf(",\"solidCell\":");arr(s.surface.solidCell);std::printf("}");
 std::printf(",\"gasMesh\":");mesh(s.gasMesh);std::printf(",\"gasStages\":[");stage(s.gasStages[0]);std::printf(",");stage(s.gasStages[1]);std::printf("],\"solidStages\":[");stage(s.solidStages[0]);std::printf(",");stage(s.solidStages[1]);std::printf("]");
 std::printf(",\"budget\":{");
#define B(x) std::printf("\"" #x "\":");num(s.budget.x);std::printf(",");
 B(boundaryMass);B(boundaryEnergy);B(supportWork);B(bodyWork);B(radiation);B(gclResidual);B(filmKineticDefect);B(filmPressureVolume);B(turbulenceOmegaConstraint);B(turbulenceInventory);B(turbulenceProduction);B(turbulenceDissipation);B(turbulenceBoundaryFlux);B(numericalEnergyResidual);
#undef B
 std::printf("\"exchangeSpecies\":[");for(int i=0;i<NParticipants;++i){if(i)std::printf(",");arr(s.budget.exchangeSpecies[i]);}std::printf("],\"boundarySpecies\":");arr(s.budget.boundarySpecies);std::printf(",");
 std::printf("\"exchangeMass\":");arr(s.budget.exchangeMass);std::printf(",\"exchangeEnergy\":");arr(s.budget.exchangeEnergy);std::printf(",\"boundaryMomentum\":");vec(s.budget.boundaryMomentum);std::printf(",\"supportImpulse\":");vec(s.budget.supportImpulse);std::printf(",\"exchangeMomentum\":[");for(int i=0;i<NParticipants;++i){if(i)std::printf(",");vec(s.budget.exchangeMomentum[i]);}std::printf("],\"consumedPackets\":%llu}",static_cast<unsigned long long>(s.budget.consumedPackets));
 std::printf(",\"film\":[");first=true;for(std::size_t i=0;i<s.film.size();++i){if(!first)std::printf(",");first=false;const auto&q=s.film[i];const auto&a=s.filmAux[i];std::printf("{\"mass\":");num(q.mass);std::printf(",\"enthalpy\":");num(q.enthalpy);std::printf(",\"species\":");arr(q.species);std::printf(",\"pressure\":");num(a.pressure);std::printf(",\"kineticEnergy\":");num(a.kineticEnergy);std::printf(",\"area\":");num(a.area);std::printf(",\"thickness\":");num(a.thickness);std::printf(",\"solidFront\":");num(a.solidFront);std::printf(",\"gasFront\":");num(a.gasFront);std::printf(",\"temperature\":");num(a.temperature);std::printf(",\"meanVelocity\":");vec(a.meanVelocity);std::printf(",\"normal\":");vec(a.normal);std::printf(",\"baseVelocity\":");vec(a.baseVelocity);std::printf(",\"normalVelocity\":");num(a.normalVelocity);std::printf(",\"solidNormalVelocity\":");num(a.solidNormalVelocity);std::printf("}");}std::printf("]");
 std::printf(",\"ledger\":[");first=true;for(const auto&q:s.ledger){if(!first)std::printf(",");first=false;std::printf("{\"kind\":%d,\"stage\":%d,\"consumerMask\":%u,\"step\":%llu,\"face\":%llu,\"mass\":",int(q.kind),q.stage,q.consumerMask,static_cast<unsigned long long>(q.step),static_cast<unsigned long long>(q.face));num(q.mass);std::printf(",\"energy\":");num(q.energy);std::printf(",\"momentum\":");vec(q.momentum);std::printf(",\"species\":");arr(q.species);std::printf(",\"condensed\":");arr(q.condensed);std::printf(",\"pore\":");arr(q.pore);std::printf(",\"conductive\":");num(q.conductive);std::printf(",\"advective\":");num(q.advective);std::printf(",\"pressureWork\":");num(q.pressureWork);std::printf(",\"viscousWork\":");num(q.viscousWork);std::printf(",\"radiation\":");num(q.radiation);std::printf(",\"liquidKineticAdvection\":");num(q.liquidKineticAdvection);std::printf("}");}std::printf("]");
 std::printf(",\"chemistryAudit\":[");first=true;for(auto&q:s.gasChemistryAudit){if(!first)std::printf(",");first=false;std::printf("{\"integratedTime\":");num(q.integratedTime);std::printf(",\"speciesMassChange\":");arr(q.speciesMassChange);std::printf(",\"massResidual\":");num(q.massResidual);std::printf(",\"energyResidual\":");num(q.energyResidual);std::printf(",\"acceptedSteps\":%d,\"rejectedSteps\":%d}",q.acceptedSteps,q.rejectedSteps);}std::printf("]}\n");
}
