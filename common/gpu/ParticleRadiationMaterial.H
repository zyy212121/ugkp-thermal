#ifndef UGKP_GPU_PARTICLE_RADIATION_MATERIAL_H
#define UGKP_GPU_PARTICLE_RADIATION_MATERIAL_H
#include "MieOpticalProperties.H"
#include "ParticleRadiationResponse.H"
#include "RadiationResidentView.H"
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct ParticleRadiationMaterialInput {
    std::vector<double> epsilon,mass,temperature,diameter,incident,volume,boundaryArea,boundaryFlux;
    std::vector<unsigned char> coupledBoundary;
    ParticleRadiationResponseView response{};
    RadiationDeviceView resident{};
    double heatFactor=0,elapsed=0,minimumTemperature=0,maximumTemperature=0,activeEpsilonMinimum=0;
};
struct ParticleRadiationMaterialResult {
    std::vector<double> temperature,energy,boundaryEnergy;
    double escapedEnergy=0,radiativeEnergyMagnitude=0,acceptedParticleEnergy=0;
    RadiationMaterialDeviceView resident{};
};
class ParticleRadiationMaterial {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit ParticleRadiationMaterial(const MieOpticalTable&);
    ~ParticleRadiationMaterial();
    ParticleRadiationMaterialResult solve(const ParticleRadiationMaterialInput&,bool exportFields=true);
    ParticleRadiationMaterialResult collect();
    // Reconcile the already computed wall contribution with actual bounded
    // particle heat. The same conservative scaling stays entirely on device.
    void settleAcceptedEnergy(const RadiationAcceptedEnergyDeviceView&);
    ParticleRadiationMaterialResult collectBoundaryEnergy();
};
}}
#endif
