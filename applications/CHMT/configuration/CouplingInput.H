#include "GpuSchedulingConfiguration.H"
#include "FoamParticleInjectionVelocity.H"
#ifndef CHMT_CONFIGURATION_COUPLINGINPUT_H
#define CHMT_CONFIGURATION_COUPLINGINPUT_H
#include "configuration/MultirateIO.H"
#include "configuration/SharedPhasePhysics.H"
#include "configuration/SolverControls.H"
#include "../../../common/FoamGasSolverControls.H"
#include "io/MeshIO.H"
#include "gpu/StateValidation.H"
#include "../../../common/gasTransport/GasNumericsIO.H"
#include "../../../common/FoamGasNumericsControls.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "OStringStream.H"
#include "uniformDimensionedFields.H"
#include "../../../common/gasTransport/FoamMolecularTransportIO.H"
#include <memory>
#include "../../../common/thermal/ThermalCouplingSchedule.H"
namespace chmt { namespace io {
struct CoupledCaseInput {
    ugkwp::GasModelConfiguration gasModel;
    ugkwp::GasMechanismConfiguration gasMechanism;
    Foam::GasNumericsConfiguration gasNumerics;
    ModelConfig model;
    HostState state;
    CouplingControls coupling;
    ugkp::thermal::CouplingSchedule thermalSchedule{0,1,false};
    CpuMaterialControls material;
    std::unique_ptr<Foam::fvMesh> gasMesh,solidMesh;
};

inline void readCoupledCase(Foam::Time& time,const Foam::dictionary& d,CoupledCaseInput& out) {
    CoupledCaseInput candidate;
    Foam::IOdictionary gasProperties(Foam::IOobject("gasModelProperties",time.constant(),time,
        Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
    Foam::IOdictionary solidProperties(Foam::IOobject("solidRegionProperties",time.constant(),time,
        Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE,false));
    const auto& sharedCoupling=solidProperties.subDict("solidThermalCoupling");
    const Foam::word gasRegion(text(sharedCoupling,"gasRegion",Foam::polyMesh::defaultRegion));
    const Foam::word solidRegion(text(sharedCoupling,"solidRegion"));
    Foam::dictionary surface;
    surface.set("patch",Foam::word(text(sharedCoupling,"solidPatch")));
    surface.set("gasPatch",Foam::word(text(sharedCoupling,"fluidPatch")));
    Foam::dictionary interfaceInput(d);
    interfaceInput.set("surface",surface);
    ugkp::solverControls::applyGasSolverControlOverrides(gasProperties,time,gasRegion);
    Foam::OStringStream expanded;expanded.precision(std::numeric_limits<double>::max_digits10);
    gasProperties.Foam::dictionary::write(expanded,false);
    candidate.gasModel=ugkwp::parseGasModelProperties(expanded.str());
    ugkp::solverControls::reportGasSolverControls(candidate.gasModel,time,gasRegion);
    if(candidate.gasModel.mode!=ugkwp::GasMode::MixtureFrozen)
        throw std::runtime_error("CHMT requires frozen multicomponent gas; volumetric chemistry and single-component modes are not application capabilities. Surface reactions and pyrolysis remain available.");

    if(candidate.gasModel.mode==ugkwp::GasMode::SingleLegacy){
        Foam::IOdictionary metadata(Foam::IOobject("materialGasProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        Foam::OStringStream canonical;canonical.precision(std::numeric_limits<double>::max_digits10);
        metadata.Foam::dictionary::write(canonical,false);
        candidate.gasModel=ugkwp::parseGasModelProperties(canonical.str());
            candidate.gasModel.mode=ugkwp::GasMode::SingleLegacy;
    }
    candidate.model=readModel(d,candidate.gasModel);
    Foam::uniformDimensionedVectorField gravityField
    (
        Foam::IOobject("g",time.constant(),time,Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false),
        Foam::dimensionedVector("g",Foam::dimAcceleration,Foam::vector::zero)
    );
    using GravityScalar=decltype(candidate.model.physics.gravity.x);
    candidate.model.physics.gravity={
        GravityScalar(gravityField.value().x()),
        GravityScalar(gravityField.value().y()),
        GravityScalar(gravityField.value().z())};
    Foam::IOdictionary fluidProperties(Foam::IOobject("fluidProperties",time.constant(),time,
        Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE,false));
    const auto& fluidMixture=fluidProperties.subDict("mixture");
    const auto& fluidTransport=fluidMixture.subDict("transport");
    candidate.model.physics.gasViscosity=readNumber(fluidTransport,"mu",0);
    candidate.model.physics.gasConductivity=readNumber(fluidTransport,"kappa",0);

    // Scheduling is read exactly once, before CUDA state and graph creation.
    Foam::IOdictionary scheduling(Foam::IOobject("schedulingProperties",time.constant(),time,
        Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE,false));
    const auto plan=Foam::readGpuSchedulingConfiguration(scheduling);
    Foam::IOdictionary momentum(Foam::IOobject("momentumTransport",time.constant(),time,
        Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false));
    Foam::IOdictionary particlePhysics(Foam::IOobject("particleProperties",time.constant(),time,
        Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false));
    readSharedPhasePhysics(candidate.model,momentum,particlePhysics,!plan.pureGasOnly);

    auto& particlePlan=candidate.model.particlePhase;
    particlePlan.capacity=plan.particleCapacity;
    particlePlan.maxFaceWalkHops=plan.maxFaceWalkHops;
    particlePlan.csrCellLocalPath=plan.csrCellLocalPath;
    particlePlan.heavyReductionMode=int(plan.heavyReductionMode);
    particlePlan.heavyAutoInterval=plan.heavyReductionAutoInterval;
    particlePlan.particleBlockThreads=plan.particleBlockThreads;
    particlePlan.reductionBlockThreads=plan.reductionBlockThreads;
    particlePlan.warpAggregatedBinning=plan.warpAggregatedBinning;

    readModelSolverControls(candidate.model,time,gasRegion,solidRegion);
    {
        auto names=candidate.gasModel.speciesNames;
        if(candidate.model.physics.gasMode==ugkwp::GasMode::SingleLegacy)
            names={candidate.model.speciesNames[candidate.model.physics.singleGasSpecies]};
        const auto transport=Foam::readFoamMolecularTransport(time,time,
            candidate.model.physics.gasMode==ugkwp::GasMode::SingleLegacy?std::vector<std::string>{}:names);
        if(candidate.model.physics.gasMode==ugkwp::GasMode::SingleLegacy&&!transport.speciesNames.empty()&&transport.speciesNames!=names)
            throw std::runtime_error("molecular transport species must match the active legacy gas");
        auto masses=candidate.gasModel.species;
        if(candidate.model.physics.gasMode==ugkwp::GasMode::SingleLegacy)
            masses={masses[candidate.model.physics.singleGasSpecies]};
        ugkwp::validateTransportMolarMasses(transport,masses);
        auto& p=candidate.model.physics;
        p.gasViscosityModel=transport.model;p.gasTransportFingerprint=transport.hash;
        p.gasViscositySpecies=int(transport.species.size());
        p.gasTransportCoefficients=transport.coefficients;
        p.gasPowerLawEnabled=!transport.diffusion.empty();
        for(std::size_t j=0;j<transport.diffusion.size();++j)p.gasPowerLaw[j]=transport.diffusion[j];
        if(p.gasViscositySpecies>Ns)throw std::runtime_error("transport species exceed CHMT species count");
        for(int s=0;s<p.gasViscositySpecies;++s)p.gasSutherland[s]=transport.species[s];
        for(std::size_t i=0;i<transport.massRatioQuarter.size();++i){
            p.gasMassRatioQuarter[i]=transport.massRatioQuarter[i];p.gasWilkeDenominator[i]=transport.wilkeDenominator[i];
        }
    }
    if(candidate.gasModel.mode==ugkwp::GasMode::MixtureChemistry){
        const auto mechanism=ugkwp::resolveGasMechanismPath(candidate.gasModel,std::string(time.path()/time.constant()));
        candidate.gasMechanism=ugkwp::readGasMechanismProperties<Ns>(mechanism,
            candidate.gasModel.thermoView<Ns>(),candidate.gasModel.phase);
        candidate.model.physics.gasMechanismFingerprint=candidate.gasMechanism.mechanismHash;
        candidate.model.gasMechanismSourceSha256=candidate.gasMechanism.sourceSha256;
    }
    candidate.gasMesh.reset(new Foam::fvMesh(Foam::IOobject(gasRegion,time.timeName(),time,Foam::IOobject::MUST_READ)));
    candidate.solidMesh.reset(new Foam::fvMesh(Foam::IOobject(solidRegion,time.timeName(),time,Foam::IOobject::MUST_READ)));
    candidate.gasNumerics=Foam::readGasNumericsConfiguration(ugkp::solverControls::readGasNumericsDictionary(time,gasRegion));
    auto& state=candidate.state;state.time=time.value();state.radiationTime=state.time;
    state.gasMesh=importMesh(*candidate.gasMesh,solidProperties);
    state.solidMesh=importMesh(*candidate.solidMesh,solidProperties.found("solidBoundaries")?solidProperties.subDict("solidBoundaries"):solidProperties);
    Foam::volScalarField inletSolidFraction(Foam::IOobject("epsilonS",time.timeName(),*candidate.gasMesh,
        Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE),*candidate.gasMesh,
        Foam::dimensionedScalar("epsilonS",Foam::dimless,0));
    importGasFields(*candidate.gasMesh,candidate.model,state);
    importSolidFields(*candidate.solidMesh,candidate.model,state);
    auto importedSurface=importSurface(*candidate.solidMesh,state.solidMesh,surface);
    state.surface=importedSurface.mesh;initializeDryInterfaceAux(state);
    attachCoupledSurface(*candidate.gasMesh,interfaceInput,state);
    if(candidate.model.physics.enableParticles){
        Foam::IOdictionary particleProperties(Foam::IOobject("particleProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        auto& config=candidate.model.particleInlet;
        const auto& boundary=*state.gasMesh.gasBoundary;
        for(std::size_t face=0;face<boundary.faces.size();++face)
            if(boundary.faces[face].geometry==0&&boundary.faces[face].solidMode>=0
                &&face>=std::size_t(candidate.gasMesh->nInternalFaces()))
                config.faces.push_back(int(face));
        config.temperature=readNumber(particleProperties,"gpuResidentInjectionTp",candidate.model.particlePhase.minimumTemperature);
        config.theta=readNumber(particleProperties,"gpuResidentInjectionTheta",0);
        const auto velocity=Foam::readParticleInjectionVelocity(particleProperties);
        config.velocitySpecified=velocity.specified;
        config.velocity[0]=velocity.value.x();config.velocity[1]=velocity.value.y();config.velocity[2]=velocity.value.z();
        state.particleSourceFaces=config.faces;state.particleSourceResidualMass.assign(config.faces.size(),0);
    }
    importParticles(particlePhysics,candidate.model,state);
    std::string error;
    require(validateRuntimeState(candidate.model,state,error),error);
    const Real dt=time.controlDict().lookup<Foam::scalar>("deltaT");
    candidate.state.nextDt=dt;
    candidate.thermalSchedule=ugkp::thermal::readCouplingSchedule(sharedCoupling,time);
    candidate.coupling=readCouplingControls(time,dt,candidate.thermalSchedule.interval);
    Foam::IOdictionary radiationProperties(Foam::IOobject("radiationProperties",time.constant(),time,
        Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false));
    ugkp::thermal::readRadiationSchedule(candidate.thermalSchedule,radiationProperties,time);
    candidate.material=readMaterialSolverControls(time,solidRegion);
    candidate.coupling.gasMaxDt=candidate.model.physics.maxDt;
    candidate.coupling.materialMaxSubstep=candidate.material.maxSubstep;
    out=std::move(candidate);
}
}}
#endif
