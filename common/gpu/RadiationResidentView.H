#ifndef UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#define UGKP_GPU_RADIATION_RESIDENT_VIEW_H
namespace ugkp { namespace gpu {
// Borrowed buffers remain valid until the next particle advance/aggregation.
// The DO owner waits on producerStream before reading them. Radiation and
// the shared production state both retain their FP64 representation.
struct RadiationParticleDeviceView {
    bool enabled=false;
    int cells=0,boundaryOffset=0,boundaryFaces=0;
    const double *mass=nullptr,*temperatureMass=nullptr,*diameterMass=nullptr;
    const double *boundaryTemperature=nullptr,*contactArea=nullptr,*contactScale=nullptr;
    double materialDensity=0,minimumTemperature=0;
    void* producerStream=nullptr;
};
// Completed DO state, owned by the radiation workspace. Consumers never need
// to upload the terminal diagnostic copies of these fields.
struct RadiationDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const double *epsilon=nullptr,*mass=nullptr,*temperature=nullptr,*diameter=nullptr;
    const double *incident=nullptr,*power=nullptr,*volume=nullptr,*boundaryArea=nullptr,*boundaryFlux=nullptr;
    void* producerStream=nullptr;
};
struct RadiationMaterialDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const double *temperature=nullptr,*energy=nullptr,*boundaryEnergy=nullptr;
    void* producerStream=nullptr;
};
}}
#endif
