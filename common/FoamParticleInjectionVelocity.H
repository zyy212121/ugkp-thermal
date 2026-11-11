#pragma once
// constant/particleProperties (optional, m/s):
// particleTemperatureTransport false;
// particleGasHeatTransferModel none;
// particleInjectionVelocity (0 0.1 0);
// Omit particleInjectionVelocity to follow the gas boundary velocity.
// The override applies to particle sources selected by fixed epsilonS patches.
#include "IOdictionary.H"
#include "fvMesh.H"
#include "ParticleInjectionVelocity.H"
#include <cmath>
extern "C" int ugkwpGpuResidentStrictConfigureParticleInjectionVelocity
(void*, int, double, double, double);
namespace Foam {
struct ParticleInjectionVelocitySetting {
 bool specified=false;
 vector value=vector::zero;
 vector select(const vector& gas) const {return specified ? value : gas;}
};
inline ParticleInjectionVelocitySetting readParticleInjectionVelocity(const fvMesh& mesh) {
 ParticleInjectionVelocitySetting setting;
 if(!mesh.foundObject<IOdictionary>("particleProperties"))return setting;
 const auto& properties=mesh.lookupObject<IOdictionary>("particleProperties");
 setting.specified=properties.found("particleInjectionVelocity");
 if(setting.specified){
  setting.value=vector(properties.lookup("particleInjectionVelocity"));
  for(int i=0;i<3;++i)if(!std::isfinite(setting.value[i]))
   FatalErrorInFunction<<"particleInjectionVelocity must be a finite velocity vector in constant/particleProperties"<<exit(FatalError);
 }
 return setting;
}
}
