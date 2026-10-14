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
 std::printf("],\"solid\":[");first=true;for(auto&q:s.solid){if(!first)std::printf(",");first=false;std::printf("{\"condensed\":");arr(q.condensed);std::printf(",\"pore\":");arr(q.pore);std::printf(",\"energy\":");num(q.energy);std::printf("}");}
 std::printf("],\"sst\":[");first=true;for(auto&q:s.sst){if(!first)std::printf(",");first=false;double a[2]={q.rhoK,q.rhoOmega};arr(a);}std::printf("]");
 std::printf(",\"gasSstAudit\":{");
#define AUD(x) std::printf("\"" #x "\":");num(s.gasSstAudit.x);
 AUD(transportK);std::printf(",");AUD(transportOmega);std::printf(",");AUD(sourceK);std::printf(",");AUD(sourceOmega);std::printf(",");AUD(constraintK);std::printf(",");AUD(constraintOmega);
#undef AUD
 std::printf("},\"gasMesh\":");mesh(s.gasMesh);std::printf(",\"gasStages\":[");stage(s.gasStages[0]);std::printf(",");stage(s.gasStages[1]);std::printf("],\"solidStages\":[");stage(s.solidStages[0]);std::printf(",");stage(s.solidStages[1]);std::printf("]");
 std::printf(",\"budget\":{");
#define B(x) std::printf("\"" #x "\":");num(s.budget.x);std::printf(",");
 B(boundaryMass);B(boundaryEnergy);B(supportWork);B(bodyWork);B(radiation);B(gclResidual);B(filmKineticDefect);B(filmPressureVolume);B(turbulenceOmegaConstraint);B(turbulenceInventory);B(turbulenceProduction);B(turbulenceDissipation);B(turbulenceBoundaryFlux);B(numericalEnergyResidual);
#undef B
 std::printf("\"exchangeMass\":");arr(s.budget.exchangeMass);std::printf(",\"exchangeEnergy\":");arr(s.budget.exchangeEnergy);std::printf(",\"boundaryMomentum\":");vec(s.budget.boundaryMomentum);std::printf(",\"supportImpulse\":");vec(s.budget.supportImpulse);std::printf(",\"exchangeMomentum\":[");for(int i=0;i<NParticipants;++i){if(i)std::printf(",");vec(s.budget.exchangeMomentum[i]);}std::printf("],\"consumedPackets\":%llu}",static_cast<unsigned long long>(s.budget.consumedPackets));
 std::printf(",\"chemistryAudit\":[");first=true;for(auto&q:s.gasChemistryAudit){if(!first)std::printf(",");first=false;std::printf("{\"integratedTime\":");num(q.integratedTime);std::printf(",\"speciesMassChange\":");arr(q.speciesMassChange);std::printf(",\"massResidual\":");num(q.massResidual);std::printf(",\"energyResidual\":");num(q.energyResidual);std::printf(",\"acceptedSteps\":%d,\"rejectedSteps\":%d}",q.acceptedSteps,q.rejectedSteps);}std::printf("]}\n");
}
