#ifndef UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#define UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#include "../GpuPrecisionTypes.H"
namespace ugkp { namespace gpu {



struct ThermalInterfaceEnergyDeviceView {
    bool enabled=false;
    int faces=0;
    const GpuWallEnergy* energy=nullptr;
    void* producerStream=nullptr;
    const double* interval=nullptr;
};
struct ThermalInterfaceStateDeviceView {
    bool enabled=false;
    int faces=0;
    const GpuReal* temperature=nullptr;
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
