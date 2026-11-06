#ifndef UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
#define UGKP_GPU_THERMAL_INTERFACE_DEVICE_VIEW_H
namespace ugkp { namespace gpu {
// Borrowed interface fields use the configured mapper's compact solid-face
// order. The producer owns storage through candidate evaluation and commit;
// consumers order their stream against producerStream before reading it.
struct ThermalInterfaceEnergyDeviceView {
    bool enabled=false;
    int faces=0;
    const double* energy=nullptr; // J, positive into the solid over the interval.
    void* producerStream=nullptr;
};
struct ThermalInterfaceStateDeviceView {
    bool enabled=false;
    int faces=0;
    const double* temperature=nullptr; // Surface temperature, not owner-cell T.
    const double* effusivity=nullptr;
    const int* contactEnabled=nullptr;
    void* producerStream=nullptr;
};
struct ThermalSolidContactDeviceView {
    int faces=0;
    const double* temperature=nullptr;
    const double* conductivity=nullptr;
    void* producerStream=nullptr;
};
}}
#endif
