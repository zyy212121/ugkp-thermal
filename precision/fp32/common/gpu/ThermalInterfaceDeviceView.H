#ifndef UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#define UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#include "../GpuPrecisionTypes.H"
namespace ugkp { namespace gpu {
// Borrowed interface fields use the configured mapper's compact solid-face
// order. The producer owns storage through candidate evaluation and commit;
// consumers order their stream against producerStream before reading it.
struct ThermalInterfaceEnergyDeviceView {
    bool enabled=false;
    int faces=0;
    const GpuWallEnergy* energy=nullptr; // J, positive into the solid over the interval.
    void* producerStream=nullptr;
    const double* interval=nullptr; // Resident duration of the completed exchange.
};
struct ThermalInterfaceStateDeviceView {
    bool enabled=false;
    int faces=0;
    const GpuReal* temperature=nullptr; // Surface temperature, not owner-cell T.
    const GpuReal* effusivity=nullptr;
    const int* contactEnabled=nullptr;
    void* producerStream=nullptr;
    const GpuReal* emissivity=nullptr;
};
struct ThermalSolidContactDeviceView {
    int faces=0;
    const GpuReal* temperature=nullptr;
    const GpuReal* conductivity=nullptr;
    void* producerStream=nullptr;
};
}}
#endif
