#ifndef CHMT_CONFIGURATION_COUPLINGINPUT_H
#define CHMT_CONFIGURATION_COUPLINGINPUT_H
#include "configuration/MultirateIO.H"
#include "configuration/SolverControls.H"
#include "../../../common/FoamGasSolverControls.H"
#include "io/MeshIO.H"
#include "gpu/StateValidation.H"
#include "../../../common/gasTransport/GasNumericsIO.H"
#include "../../../common/FoamGasNumericsControls.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "OStringStream.H"
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
    const Foam::word gasRegion(text(d,"gasRegion",Foam::polyMesh::defaultRegion));
    const Foam::word solidRegion(text(d,"solidRegion"));
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
    candidate.gasMesh.reset(new Foam::fvMesh(Foam::IOobject(text(d,"gasRegion",Foam::polyMesh::defaultRegion),time.timeName(),time,Foam::IOobject::MUST_READ)));
    candidate.solidMesh.reset(new Foam::fvMesh(Foam::IOobject(text(d,"solidRegion"),time.timeName(),time,Foam::IOobject::MUST_READ)));
    candidate.gasNumerics=Foam::readGasNumericsConfiguration(ugkp::solverControls::readGasNumericsDictionary(time,gasRegion));
    auto& state=candidate.state;state.time=time.value();state.radiationTime=state.time;
    state.gasMesh=importMesh(*candidate.gasMesh,d);
    state.solidMesh=importMesh(*candidate.solidMesh,d.found("solidBoundaries")?d.subDict("solidBoundaries"):d);
    importGasFields(*candidate.gasMesh,candidate.model,state);
    importSolidFields(*candidate.solidMesh,candidate.model,state);
    auto surface=importSurface(*candidate.solidMesh,state.solidMesh,d.subDict("surface"));
    state.surface=surface.mesh;initializeDryInterfaceAux(state);
    attachCoupledSurface(*candidate.gasMesh,d,state);
    if(candidate.model.physics.enableParticles&&d.subDict("particle").found("inlet")){
        const auto& inlet=d.subDict("particle").subDict("inlet");
        auto& config=candidate.model.particleInlet;
        config.faces=selectedPatchFaces(*candidate.gasMesh,inlet,"patches","patch");
        config.temperature=readPhysical(inlet,"temperature");config.theta=readPhysical(inlet,"theta",0);
        require(finite(config.temperature)&&config.temperature>=candidate.model.particlePhase.minimumTemperature
            &&config.temperature<=candidate.model.particlePhase.maximumTemperature&&finite(config.theta)&&config.theta>=0,
            "invalid particle inlet temperature/theta");
        if(inlet.found("volumeFractionTable")){
            const Foam::List<Foam::Tuple2<Foam::scalar,Foam::scalar>> rows(inlet.lookup("volumeFractionTable"));
            for(const auto& row:rows){config.times.push_back(row.first());config.volumeFraction.push_back(PhysicalReal(row.second()));}
        }else{config.times={0};config.volumeFraction={readPhysical(inlet,"volumeFraction")};}
        require(!config.times.empty(),"empty particle inlet volume-fraction table");
        for(std::size_t i=0;i<config.times.size();++i)
            require(finite(config.times[i])&&(!i||config.times[i]>config.times[i-1])
                &&finite(config.volumeFraction[i])&&config.volumeFraction[i]>=0&&config.volumeFraction[i]<1,
                "invalid particle inlet volume-fraction table");
        config.velocitySpecified=inlet.found("velocity");const auto u=readPhysicalVector(inlet,"velocity");
        require(finite(u),"invalid particle inlet velocity");
        config.velocity[0]=u.x;config.velocity[1]=u.y;config.velocity[2]=u.z;
        state.particleSourceFaces=config.faces;state.particleSourceResidualMass.assign(config.faces.size(),0);
    }
    importParticles(d,candidate.model,state);
    std::string error;
    require(validateRuntimeState(candidate.model,state,error),error);
    const Real dt=time.controlDict().lookup<Foam::scalar>("deltaT");
    candidate.state.nextDt=dt;
    if(text(d,"couplingSchedule","multirate")=="solidThermalCoupling"){
        Foam::IOdictionary solidProperties(Foam::IOobject("solidRegionProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        candidate.thermalSchedule=ugkp::thermal::readCouplingSchedule(solidProperties.subDict("solidThermalCoupling"),time);
        Foam::IOdictionary radiationProperties(Foam::IOobject("radiationProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        ugkp::thermal::readRadiationSchedule(candidate.thermalSchedule,radiationProperties,time);
        require(finite(candidate.thermalSchedule.interval)&&candidate.thermalSchedule.interval>0
            &&(!candidate.thermalSchedule.radiationEnabled||candidate.thermalSchedule.radiationFrequency>0),"invalid solid thermal coupling schedule");
    }else require(text(d,"couplingSchedule","multirate")=="multirate","unknown couplingSchedule");
    if(candidate.thermalSchedule.radiationEnabled)candidate.model.physics.enableRadiation=false; // DO owns radiation; no parallel ambient-grey source
    candidate.coupling=readCouplingControls(time,dt,candidate.thermalSchedule.interval);
    candidate.material=readMaterialSolverControls(time,solidRegion);
    candidate.coupling.gasMaxDt=candidate.model.physics.maxDt;
    candidate.coupling.materialMaxSubstep=candidate.material.maxSubstep;
    out=std::move(candidate);
}
}}
#endif
