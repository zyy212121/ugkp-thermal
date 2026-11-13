#pragma once
namespace ugkp { namespace thermal {
// Numerical overrides are independent of the saved physical particle model.
struct ParticleNumericalControls {
    int finiteContactMaximumIterations=96;
    int packingIterations=20;
};
int setParticleNumericalControls(void* owner,const ParticleNumericalControls&);
}}
