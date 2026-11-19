#ifndef UGKP_GPU_PARTICLE_RADIATION_MATERIAL_H
#define UGKP_GPU_PARTICLE_RADIATION_MATERIAL_H
#include "RadiationPrecision.H"
#include "ParticleRadiationControls.H"
#include "MieOpticalProperties.H"
#include "ParticleRadiationResponse.H"
#include "RadiationResidentView.H"
#include "DiscreteOrdinateRadiation.H"
#include <memory>
#include <future>
#include <vector>
namespace ugkp { namespace gpu {
struct ParticleRadiationMaterialInput {
    std::vector<double> epsilon,mass,temperature,diameter,incident,volume,boundaryArea,boundaryFlux;
    std::vector<unsigned char> coupledBoundary;
    ParticleRadiationResponseView response{};
    RadiationDeviceView resident{};
    RadiationParticleDeviceView particleResident{};
    RadiationReal heatFactor=0;
    double elapsed=0;
    RadiationReal minimumTemperature=0,maximumTemperature=0,activeEpsilonMinimum=0;
};
struct ParticleRadiationMaterialResult {
    std::vector<RadiationReal> temperature,energy,boundaryEnergy;
    RadiationReal escapedEnergy=0;
    RadiationMaterialDeviceView resident{};
};
class ParticleRadiationMaterial {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit ParticleRadiationMaterial(const MieOpticalTable&,const ParticleRadiationSolveControls& controls={});
    ~ParticleRadiationMaterial();
    RadiationThermalCouplingDeviceView beginAnchoredWindow(const ParticleRadiationMaterialInput&);
    RadiationThermalCouplingDeviceView prepareParentWindow(const ParticleRadiationMaterialInput&,const double* deviceElapsed);
    ExecutionGraphNode appendWindowStartTo(ExecutionGraph,ExecutionGraphNode dependency=nullptr);
    ExecutionGraphNode appendWindowFinishTo(ExecutionGraph,ExecutionGraphNode,const RadiationDeviceView&);
    RadiationMaterialDeviceView deviceView()const noexcept;
    void acceptExternalCompletion(const std::shared_future<void>&);
    void acceptExternalExecution(void* producerStream);
    void awaitCompletion()const;
    void* producerStream()const noexcept;
    ExecutionGraphNode appendAnchoredIteration(ExecutionGraph,ExecutionGraphNode,void*,const RadiationThermalIterationDeviceView&);
    ParticleRadiationMaterialResult finishAnchoredWindow(const RadiationDeviceView&,bool exportFields=true);
    ParticleRadiationMaterialResult solve(const ParticleRadiationMaterialInput&,bool exportFields=true);
    ParticleRadiationMaterialResult collect();
    ParticleRadiationMaterialResult collectBoundaryEnergy();
    RadiationReal collectEscapedEnergy();
};
}}
#endif
