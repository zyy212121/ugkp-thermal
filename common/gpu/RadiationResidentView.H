#ifndef UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#define UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#include "RadiationPrecision.H"
namespace ugkp { namespace diagnostics { struct DeviceContext; } namespace gpu {
// Borrowed buffers remain valid until the next particle advance/aggregation.
// The DO owner waits on producerStream before reading them. Radiation and
// the shared production state use the selected radiation precision.
struct RadiationParticleDeviceView {
    bool enabled=false;
    int cells=0,boundaryOffset=0,boundaryFaces=0;
    const RadiationReal *mass=nullptr,*temperatureMass=nullptr,*diameterMass=nullptr;
    const RadiationReal *boundaryTemperature=nullptr,*contactArea=nullptr,*contactScale=nullptr;
    RadiationReal materialDensity=0,minimumTemperature=0;
    void* producerStream=nullptr;
    const RadiationReal* boundaryEmissivity=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
// Asynchronous DO state, owned by the radiation workspace. Consumers wait on
// producerStream and return ownership after their final read.
struct RadiationDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const RadiationReal *epsilon=nullptr,*mass=nullptr,*temperature=nullptr,*diameter=nullptr;
    const RadiationReal *incident=nullptr,*power=nullptr,*volume=nullptr,*boundaryArea=nullptr,*boundaryFlux=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
// Actual extensive particle heat after bounded enthalpy publication. The
// owner retains the buffer until the consumer returns its stream dependency.
struct RadiationAcceptedEnergyDeviceView {
    int cells=0;
    const RadiationReal* energy=nullptr; // Null denotes an absent particle phase.
    void* producerStream=nullptr;
};
struct RadiationMaterialDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const RadiationReal *temperature=nullptr,*energy=nullptr,*boundaryEnergy=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
}}
#endif
