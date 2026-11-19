#ifndef UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#define UGKP_GPU_RADIATION_RESIDENT_VIEW_H
#include "RadiationPrecision.H"
namespace ugkp { namespace diagnostics { struct DeviceContext; } namespace gpu {






struct RadiationThermalCouplingDeviceView {
    const RadiationReal *trialMeanTemperature=nullptr;
    const RadiationReal *linearizedEmission=nullptr,*thermalFeedback=nullptr;
    const RadiationReal *materialRelativeResidual=nullptr;
    RadiationReal materialRelativeTolerance=RadiationReal(1e-10);
    void* producerStream=nullptr;

    unsigned long long graphOwner=0,graphRevision=0;
};
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


struct RadiationDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const RadiationReal *epsilon=nullptr,*mass=nullptr,*temperature=nullptr,*diameter=nullptr;
    const RadiationReal *incident=nullptr,*power=nullptr,*volume=nullptr,*boundaryArea=nullptr,*boundaryFlux=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
};


struct RadiationAcceptedEnergyDeviceView {
    int cells=0;
    const RadiationReal* energy=nullptr;
    void* producerStream=nullptr;
};
struct RadiationMaterialDeviceView {
    bool enabled=false;
    int cells=0,boundaryFaces=0;
    const RadiationReal *temperature=nullptr,*energy=nullptr,*boundaryEnergy=nullptr;
    void* producerStream=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;
    const RadiationReal* publicationFraction=nullptr;
};
}}
#endif
