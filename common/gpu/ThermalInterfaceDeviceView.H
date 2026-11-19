#ifndef UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#define UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
namespace ugkp { namespace gpu {



struct ThermalInterfaceEnergyDeviceView {
    bool enabled=false;
    int faces=0;
    const double* energy=nullptr;
    void* producerStream=nullptr;
    const double* interval=nullptr;
};
struct ThermalInterfaceStateDeviceView {
    bool enabled=false;
    int faces=0;
    const double* temperature=nullptr;
    const double* effusivity=nullptr;
    const int* contactEnabled=nullptr;
    void* producerStream=nullptr;
    const double* emissivity=nullptr;
};
struct ThermalSolidContactDeviceView {
    int faces=0;
    const double* temperature=nullptr;
    const double* conductivity=nullptr;
    void* producerStream=nullptr;
};
}}
#endif
