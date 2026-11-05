#pragma once
#include "ParticleState.H"
#include "ParticlePhaseConfiguration.H"
#include "../GpuParticleCaloric.H"
#include <cmath>
namespace ugkp { namespace thermal {
// The attached state code identifies the lifecycle, not the caloric model.
// The selected face/contact model determines which stored profile is active.
template<class Vector>
double particleSnapshotThermalEnergy(const ParticleState<Vector>& p,int contactModel) {
    double h=ugkwp::ParticleCaloric{}.enthalpy(p.thermal.temperature);
    if(p.state&&contactModel==3){h=0;for(float value:p.thermal.coldNodeSpecificEnthalpy)h+=double(value)/8;}
    if(p.state&&contactModel==4){h=0;for(float value:p.thermal.cold2DNodeSpecificEnthalpy)h+=double(value)/64;}
    return p.mass*(h+p.thermal.specificEnthalpyResidual);
}
// Import of an uninitialized, free parcel. Attached profiles require their
// complete history and cannot be reconstructed from one total energy value.
template<class Vector>
bool initializeMobileParticleHistory(ParticleState<Vector>& p,const ParticlePhaseConfiguration& config) {
    if(p.thermal.initialized)return true;
    if(p.state!=0||!std::isfinite(p.mass)||!(p.mass>0)||!std::isfinite(p.energy))return false;
    const ugkwp::ParticleCaloric law;
    const double h=p.energy/p.mass,T=law.temperature(h);
    if(!std::isfinite(T)||T<config.minimumTemperature||T>config.maximumTemperature)return false;
    p.thermal.temperature=T;p.thermal.specificEnthalpyResidual=h-law.enthalpy(T);
    p.thermal.savedVelocity[0]=p.velocity.x;p.thermal.savedVelocity[1]=p.velocity.y;p.thermal.savedVelocity[2]=p.velocity.z;
    p.thermal.initialized=true;return true;
}
}}
