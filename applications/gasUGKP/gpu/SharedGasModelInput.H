#ifndef UGKWP_SHARED_GAS_MODEL_INPUT_H
#define UGKWP_SHARED_GAS_MODEL_INPUT_H

#include "fvCFD.H"
#include "GpuBackendApi.H"
#include "gasTransport/GasBuildConfig.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/GasMechanismIO.H"
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace Foam
{
// OpenFOAM I/O and storage adapter only. EOS, calorics, model parsing and
// mechanism validation have common owners; this class performs no evolution.
class SharedGasModelInput
{
    ugkwp::GasModelConfiguration model_;
    ugkwp::GasMechanismConfiguration mechanism_;
    std::string modelText_;
    std::string mechanismText_;
    PtrList<volScalarField> massFractions_;
    PtrList<volScalarField> speciesDensity_;
    autoPtr<IOdictionary> restartIdentity_;
    const fvMesh& mesh_;

    static void ensureRestartPrecision()
    {

        IOstream::defaultPrecision(max(IOstream::defaultPrecision(),
            static_cast<unsigned>(std::numeric_limits<scalar>::max_digits10)));
    }
    static void checkResult(int result,const char* operation)
    {
        if(result) FatalErrorInFunction<<operation<<": "<<ugkwpGpuResidentStrictLastError()<<exit(FatalError);
    }
    ugkwpGpuIpc::GasSpeciesIdentityV1 identity() const
    {
        return {1,static_cast<std::uint32_t>(ugkwp::compiledGasSpecies),model_.speciesOrderHash,
                model_.thermoHash,mechanism_.mechanismHash};
    }
    void pointState(scalar pressure,scalar temperature,const vector& velocity,const double* y,
                    scalar& density,scalar& energy) const
    {
        const auto thermo=model_.thermoView<ugkwp::compiledGasSpecies>();
        double rho;
        const auto densityStatus=ugkwp::mixtureDensityFromPressureTemperature(
            double(pressure),double(temperature),y,thermo,rho);


        if(densityStatus!=ugkwp::ThermoStatus::Success)
            FatalErrorInFunction<<"Mixture density expression unavailable; status="
                <<static_cast<int>(densityStatus)<<exit(FatalError);
        double internal;
        bool evaluated=false;
        const auto energyStatus=ugkwp::evaluateMixtureInternalEnergy(
            y,double(temperature),thermo,internal,&evaluated);
        if(energyStatus!=ugkwp::ThermoStatus::Success || !evaluated)
            FatalErrorInFunction<<"Mixture energy expression unavailable; status="
                <<static_cast<int>(energyStatus)<<exit(FatalError);
        const scalar total=rho*(internal+0.5*magSqr(velocity));
        if(!std::isfinite(total))
            FatalErrorInFunction<<"Nonfinite mixture total energy: "<<total<<exit(FatalError);
        density=rho;energy=total;
    }
    void initialiseIdentity(const Time& time,const fvMesh& mesh)
    {

        restartIdentity_.reset(new IOdictionary(IOobject("gasModelIdentity",time.timeName(),mesh,
            IOobject::NO_READ,IOobject::AUTO_WRITE)));
        auto& dict=restartIdentity_();
        dict.set("schemaVersion",label(1));
        dict.set("speciesCount",label(ugkwp::compiledGasSpecies));
        dict.set("mode",label(model_.mode));
        dict.set("speciesOrderHash",string(std::to_string(model_.speciesOrderHash)));
        dict.set("thermoHash",string(std::to_string(model_.thermoHash)));
        dict.set("mechanismHash",string(std::to_string(mechanism_.mechanismHash)));
    }
public:
    SharedGasModelInput(const Time& time,const fvMesh& mesh):mesh_(mesh)
    {
        const fileName modelPath=time.path()/time.constant()/"gasModelProperties";
        if(!isFile(modelPath)) return;
        try
        {
            IOdictionary dictionary(IOobject("gasModelProperties",time.constant(),mesh,
                IOobject::MUST_READ,IOobject::NO_WRITE,false));
            OStringStream output;output.precision(std::numeric_limits<double>::max_digits10);
            static_cast<const Foam::dictionary&>(dictionary).write(output,false);
            modelText_=output.str();
            model_=ugkwp::parseGasModelProperties(modelText_);
            if(!active()) return;
            if(model_.speciesNames.size()<ugkwp::compiledGasSpecies
                || model_.species.size()<ugkwp::compiledGasSpecies)
                throw std::runtime_error("missing species storage for compiled gas model");
            const auto thermo=model_.thermoView<ugkwp::compiledGasSpecies>();
            if(model_.mode==ugkwp::GasMode::MixtureChemistry)
            {
                const std::string mechanismPath=ugkwp::resolveGasMechanismPath(model_,(time.path()/time.constant()).c_str());
                std::ifstream input(mechanismPath);
                if(!input) throw std::runtime_error("cannot read required gas mechanism "+mechanismPath);
                mechanismText_.assign(std::istreambuf_iterator<char>(input),{});
                if(input.bad()) throw std::runtime_error("failed reading gas mechanism "+mechanismPath);
                mechanism_=ugkwp::parseGasMechanismProperties<ugkwp::compiledGasSpecies>(mechanismText_,thermo,model_.phase,model_.speciesNames);
            }
            initialiseIdentity(time,mesh);
        }
        catch(const std::exception& error)
        {FatalErrorInFunction<<error.what()<<exit(FatalError);}
    }
    bool active() const {return model_.mode!=ugkwp::GasMode::SingleLegacy;}
    const ugkwp::GasModelConfiguration& model() const {return model_;}
    void validateNumerics(label flux,label reconstruction,label limiter,label integrator,label turbulence,bool particles,label wallTreatment=0,label boundaryLayerModel=1) const
    {
        if(!active() && wallTreatment!=2) return;
        ugkwp::GasCapabilityRequest request;
        request.mode=model_.mode;request.fluxScheme=flux;request.reconstruction=reconstruction;
        request.limiter=limiter;request.timeIntegrator=integrator;request.turbulenceModel=turbulence;
        request.sstWallTreatment=wallTreatment;
        request.boundaryLayerModel=boundaryLayerModel;
        request.particleCoupling=particles;
        const auto result=ugkwp::validateGasCapabilities(request);
        if(!result) FatalErrorInFunction<<"Unsupported shared gas model: "<<result.message<<exit(FatalError);
    }
    void initialiseFields(const Time& time,const fvMesh& mesh,volScalarField& rho,volVectorField& rhoU,
                          volScalarField& rhoE,volVectorField& U,volScalarField& p,volScalarField& temperature)
    {
        if(!active()) return;
        ensureRestartPrecision();
        massFractions_.setSize(ugkwp::compiledGasSpecies);
        forAll(massFractions_,species)
        {
            const word name("Y_"+model_.speciesNames[species]);
            massFractions_.set(species,new volScalarField(IOobject(name,time.timeName(),mesh,
                IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh));
            massFractions_[species].correctBoundaryConditions();
        }
        U.correctBoundaryConditions();p.correctBoundaryConditions();temperature.correctBoundaryConditions();
        forAll(rho,cell)
        {
            double y[ugkwp::compiledGasSpecies];
            forAll(massFractions_,species) y[species]=massFractions_[species][cell];
            scalar density=0,energy=0;pointState(p[cell],temperature[cell],U[cell],y,density,energy);
            rho[cell]=density;rhoU[cell]=density*U[cell];rhoE[cell]=energy;
        }
        forAll(mesh.boundary(),patch)
        {
            forAll(rho.boundaryField()[patch],face)
            {
                double y[ugkwp::compiledGasSpecies];
                forAll(massFractions_,species) y[species]=massFractions_[species].boundaryField()[patch][face];
                scalar density=0,energy=0;
                pointState(p.boundaryField()[patch][face],temperature.boundaryField()[patch][face],
                    U.boundaryField()[patch][face],y,density,energy);
                rho.boundaryFieldRef()[patch][face]=density;
                rhoU.boundaryFieldRef()[patch][face]=density*U.boundaryField()[patch][face];
                rhoE.boundaryFieldRef()[patch][face]=energy;
            }
        }
        speciesDensity_.setSize(ugkwp::compiledGasSpecies);
        forAll(speciesDensity_,species)
        {
            const word name("rhoY_"+model_.speciesNames[species]);
            speciesDensity_.set(species,new volScalarField(IOobject(name,time.timeName(),mesh,
                IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),rho*massFractions_[species]));

        }
    }
    void configureResident(void* handle) const
    {
        if(!active()) return;
        const ugkwpGpuIpc::GasModelConfigureArgsV1 config{1,static_cast<std::uint32_t>(model_.mode),
            ugkwp::compiledGasSpecies,0,model_.speciesOrderHash,model_.thermoHash,mechanism_.mechanismHash,
            modelText_.size(),mechanismText_.size()};
        checkResult(ugkwpGpuResidentStrictConfigureGasModelV1(handle,&config,modelText_.data(),mechanismText_.data()),"configure shared gas model");
        Info<<"Shared gas model: api=1 Ns="<<ugkwp::compiledGasSpecies<<" mode="<<label(model_.mode)
            <<" speciesOrderHash="<<word(std::to_string(model_.speciesOrderHash))
            <<" thermoHash="<<word(std::to_string(model_.thermoHash))
            <<" mechanismHash="<<word(std::to_string(mechanism_.mechanismHash))<<nl;
    }
    void uploadInitialSpecies(void* handle) const
    {
        if(!active()) return;
        const auto id=identity();
        std::vector<double> density(std::size_t(mesh_.nCells())*ugkwp::compiledGasSpecies);
        forAll(speciesDensity_,species) forAll(speciesDensity_[species],cell)
            density[std::size_t(species)*mesh_.nCells()+cell]=speciesDensity_[species][cell];
        checkResult(ugkwpGpuResidentStrictUploadSpeciesV1(handle,&id,density.data()),"upload initial shared species density");
        std::vector<double> y(std::size_t(mesh_.nFaces())*ugkwp::compiledGasSpecies);
        std::vector<int> fixed(mesh_.nFaces(),0);
        for(label face=0;face<mesh_.nFaces();++face) forAll(massFractions_,species)
            y[std::size_t(species)*mesh_.nFaces()+face]=massFractions_[species][mesh_.faceOwner()[face]];
        forAll(mesh_.boundary(),patch)
        {
            bool patchFixed=massFractions_[0].boundaryField()[patch].fixesValue();
            forAll(massFractions_,species)
            {
                const auto& field=massFractions_[species].boundaryField()[patch];
                const word type=field.type();
                if(!field.fixesValue() && type!="zeroGradient" && type!="cyclic" && type!="empty"
                   && type!="symmetry" && type!="symmetryPlane")
                    FatalErrorInFunction<<"Unsupported mixture composition boundary type "<<type<<exit(FatalError);
                if(field.fixesValue()!=patchFixed)
                    FatalErrorInFunction<<"All species must share the composition boundary operator on patch "<<mesh_.boundary()[patch].name()<<exit(FatalError);
                forAll(field,localFace)
                {
                    const label face=mesh_.boundary()[patch].start()+localFace;
                    fixed[face]=patchFixed?1:0;
                    y[std::size_t(species)*mesh_.nFaces()+face]=field[localFace];
                }
            }
        }
        checkResult(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(handle,&id,fixed.data(),y.data()),"upload shared species boundary composition");
    }
    void downloadSpecies(void* handle,const volScalarField& rho)
    {
        if(!active()) return;
        // controlDict may have changed since construction.
        ensureRestartPrecision();
        const auto id=identity();
        std::vector<double> values(std::size_t(mesh_.nCells())*ugkwp::compiledGasSpecies);
        checkResult(ugkwpGpuResidentStrictDownloadSpeciesV1(handle,&id,values.data()),"download accepted shared species");
        forAll(speciesDensity_,species)
        {
            forAll(rho,cell)
            {
                const double value=values[std::size_t(species)*mesh_.nCells()+cell];
                speciesDensity_[species][cell]=value;
                massFractions_[species][cell]=value/rho[cell];
            }
            massFractions_[species].correctBoundaryConditions();
            forAll(mesh_.boundary(),patch) forAll(speciesDensity_[species].boundaryField()[patch],face)
                speciesDensity_[species].boundaryFieldRef()[patch][face]=
                    rho.boundaryField()[patch][face]*massFractions_[species].boundaryField()[patch][face];
        }
    }
};
} // namespace Foam
#endif
