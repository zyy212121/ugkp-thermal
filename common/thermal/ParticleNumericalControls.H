#pragma once
namespace ugkp { namespace thermal {

struct ParticleNumericalControls {
    int finiteContactMaximumIterations=96;
    int packingIterations=20;
};
int setParticleNumericalControls(void* owner,const ParticleNumericalControls&);
}}
