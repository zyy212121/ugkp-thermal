#pragma once






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
inline ParticleInjectionVelocitySetting readParticleInjectionVelocity(const dictionary& properties) {
 ParticleInjectionVelocitySetting setting;
 setting.specified=properties.found("particleInjectionVelocity");
 if(setting.specified){
  setting.value=vector(properties.lookup("particleInjectionVelocity"));
  for(int i=0;i<3;++i)if(!std::isfinite(setting.value[i]))
   FatalErrorInFunction<<"particleInjectionVelocity must be a finite velocity vector in constant/particleProperties"<<exit(FatalError);
 }
 return setting;
}

inline ParticleInjectionVelocitySetting readParticleInjectionVelocity(const fvMesh& mesh) {
 if(!mesh.foundObject<IOdictionary>("particleProperties"))return {};
 return readParticleInjectionVelocity(mesh.lookupObject<IOdictionary>("particleProperties"));
}
}
