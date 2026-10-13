#ifndef CHMT_CONFIGURATION_COUPLINGINPUT_H
#define CHMT_CONFIGURATION_COUPLINGINPUT_H
#include "configuration/MultirateIO.H"
#include "configuration/InitialConditions.H"
#include "gpu/StateValidation.H"
#include "../../../common/gasTransport/GasNumericsIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "OStringStream.H"
#include <memory>
namespace chmt { namespace io {
struct CoupledCaseInput {
    ugkwp::GasModelConfiguration gasModel;
    ugkwp::GasMechanismConfiguration gasMechanism;
    Foam::GasNumericsConfiguration gasNumerics;
    ModelConfig model;
    HostState state;
    CouplingControls coupling;
    CpuMaterialControls material;
    std::unique_ptr<Foam::fvMesh> gasMesh,solidMesh;
};

inline void readCoupledCase(Foam::Time& time,const Foam::dictionary& d,CoupledCaseInput& out) {
    CoupledCaseInput candidate;
    Foam::IOdictionary gasProperties(Foam::IOobject("gasModelProperties",time.constant(),time,
        Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
    Foam::OStringStream expanded;expanded.precision(std::numeric_limits<double>::max_digits10);
    gasProperties.Foam::dictionary::write(expanded,false);
    candidate.gasModel=ugkwp::parseGasModelProperties(expanded.str());
    if(candidate.gasModel.mode==ugkwp::GasMode::SingleLegacy){
        Foam::IOdictionary metadata(Foam::IOobject("materialGasProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        Foam::OStringStream canonical;canonical.precision(std::numeric_limits<double>::max_digits10);
        metadata.Foam::dictionary::write(canonical,false);
        candidate.gasModel=ugkwp::parseGasModelProperties(canonical.str());
            candidate.gasModel.mode=ugkwp::GasMode::SingleLegacy;
    }
    candidate.model=readModel(d,candidate.gasModel);
    if(candidate.gasModel.mode==ugkwp::GasMode::MixtureChemistry){
        const auto mechanism=ugkwp::resolveGasMechanismPath(candidate.gasModel,std::string(time.path()/time.constant()));
        candidate.gasMechanism=ugkwp::readGasMechanismProperties<Ns>(mechanism,
            candidate.gasModel.thermoView<Ns>(),candidate.gasModel.phase);
        candidate.model.physics.gasMechanismFingerprint=candidate.gasMechanism.mechanismHash;
        candidate.model.gasMechanismSourceSha256=candidate.gasMechanism.sourceSha256;
    }
    candidate.gasMesh.reset(new Foam::fvMesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ)));
    candidate.solidMesh.reset(new Foam::fvMesh(Foam::IOobject(text(d,"solidRegion"),time.timeName(),time,Foam::IOobject::MUST_READ)));
    const auto& solution=candidate.gasMesh->solution().dict();
    candidate.gasNumerics=Foam::readGasNumericsConfiguration(candidate.gasMesh->schemes().dict(),
        solution.found("UGKP")?&solution.subDict("UGKP"):nullptr);
    auto& state=candidate.state;state.time=time.value();
    state.gasMesh=importMesh(*candidate.gasMesh,d);
    state.solidMesh=importMesh(*candidate.solidMesh,d.found("solidBoundaries")?d.subDict("solidBoundaries"):d);
    importGasFields(*candidate.gasMesh,candidate.model,state);
    importSolidFields(*candidate.solidMesh,candidate.model,state);
    auto surface=importSurface(*candidate.solidMesh,state.solidMesh,d.subDict("surface"));
    if(candidate.model.physics.enableFilm)initialFilm(d,candidate.model,surface,state);
    else {state.surface=surface.mesh;initializeDryInterfaceAux(state);}
    attachCoupledSurface(*candidate.gasMesh,d,state);
    std::string error;
    require(validateRuntimeState(candidate.model,state,error),error);
    const Real dt=time.controlDict().lookup<Foam::scalar>("deltaT");
    candidate.coupling=readCouplingControls(d,dt);
    candidate.material=readCpuMaterialControls(d,candidate.coupling);
    out=std::move(candidate);
}
}}
#endif
