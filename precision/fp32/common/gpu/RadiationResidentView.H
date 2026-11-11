#ifndef UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#define UGKP_GPU_RADIATION_RESIDENT_VIEW_H
namespace ugkp { namespace diagnostics { struct DeviceContext; } namespace gpu {
// Borrowed buffers remain valid until the next particle advance/aggregation.
// The DO owner waits on producerStream before reading them. Radiation and
// aggregated energy channels retain FP64; boundary primitives may be FP32.
struct RadiationParticleDeviceView {
    bool enabled=false;
    int cells=0,boundaryOffset=0,boundaryFaces=0;
    const double *mass=nullptr,*temperatureMass=nullptr,*diameterMass=nullptr;
    const double *boundaryTemperature=nullptr,*contactArea=nullptr,*contactScale=nullptr;
    double materialDensity=0,minimumTemperature=0;
    void* producerStream=nullptr;
    const double* boundaryEmissivity=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
    const float* boundaryTemperatureFp32=nullptr;
};
// Asynchronous DO state, owned by the radiation workspace. Consumers wait on
// producerStream and return ownership after their final read.
struct RadiationDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const double *epsilon=nullptr,*mass=nullptr,*temperature=nullptr,*diameter=nullptr;
    const double *incident=nullptr,*power=nullptr,*volume=nullptr,*boundaryArea=nullptr,*boundaryFlux=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
// Actual extensive particle heat after bounded enthalpy publication. The
// owner retains the buffer until the consumer returns its stream dependency.
struct RadiationAcceptedEnergyDeviceView {
    int cells=0;
    const double* energy=nullptr; // Null denotes an absent particle phase.
    void* producerStream=nullptr;
};
struct RadiationMaterialDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const double *temperature=nullptr,*energy=nullptr,*boundaryEnergy=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};
}}
#endif
