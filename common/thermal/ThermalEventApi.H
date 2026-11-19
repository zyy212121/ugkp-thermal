#pragma once
#include "../gpu/ExecutionGraph.H"
#include "../gpu/ThermalEventState.H"
#include "../gpu/ThermalInterfaceDeviceView.H"
#include "../gpu/RadiationResidentView.H"
namespace ugkp { namespace thermal {

int prepareThermalEvent(void*,const gpu::ThermalInterfaceStateDeviceView&);

int prepareThermalParticleEnergy(void*);
int enqueueThermalResponse(void*,void* stream);
int enqueueThermalParticleEnergy(void*,const gpu::RadiationMaterialDeviceView&,void* stream);
int enqueueThermalWallState(void*,const gpu::ThermalInterfaceStateDeviceView&,void* stream);
int enqueueThermalInterface(void*,const gpu::ThermalEventState*,
    const gpu::RadiationMaterialDeviceView*,void* stream);
int enqueueThermalLedgerReset(void*,void* stream);
gpu::ExecutionGraphNode appendThermalRadiationInputs(void*,gpu::ExecutionGraph,
    gpu::ExecutionGraphNode,void* stream);
gpu::ExecutionGraphNode appendThermalParticleMoments(void*,gpu::ExecutionGraph,
    gpu::ExecutionGraphNode,void* stream);
}}
