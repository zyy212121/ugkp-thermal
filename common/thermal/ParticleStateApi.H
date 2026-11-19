#pragma once
#include "ParticleState.H"
#include "../GpuPrecisionTypes.H"
#include <vector>
namespace ugkp { namespace thermal {


struct ParticleSnapshotShape {
    int count=0, sources=0;
    const int* sourceFaces=nullptr;
    const GpuReal* sourceResidualMass=nullptr;
};
ParticleSnapshotShape particleSnapshotShape(void* owner,void* producerStream);
void packParticleSnapshot(void* owner,int count,ParticleState<>* deviceOutput,void* producerStream);
int uploadParticleState(void* owner,int count,const ParticleState<>* particles);
int downloadParticleState(void* owner,std::vector<ParticleState<>>& particles);
}}
