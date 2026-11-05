#pragma once
#include "ParticleState.H"
#include <vector>
namespace ugkp { namespace thermal {
// Window/checkpoint boundaries only. Internal staging has a bounded size;
// per-step execution continues to operate on the resident SoA arrays.
int uploadParticleState(void* owner,int count,const ParticleState<>* particles);
int downloadParticleState(void* owner,std::vector<ParticleState<>>& particles);
}}
