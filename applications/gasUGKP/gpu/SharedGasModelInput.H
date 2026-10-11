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

    static void rejectMixtureCheckpointForLegacy(const Time& time)
    {
        if(isFile(time.timePath()/"gasModelIdentity"))
            FatalErrorInFunction<<"A mixture checkpoint requires constant/gasModelProperties "
                <<"with the same non-legacy model identity"<<exit(FatalError);
    }
    static void ensureRestartPrecision()
    {
        // Redundant conservative and primitive fields must round-trip within
        // the same strict thermo/composition checks used at initialization.
        IOstream::defaultPrecision(max(IOstream::defaultPrecision(),
            static_cast<unsigned>(std::numeric_limits<scalar>::max_digits10)));
    }
    static void checkResult(int result,const char* operation)
    {
        if(result) FatalErrorInFunction<<operation<<": "<<ugkwpGpuResidentStrictLastError()<<exit(FatalError);
    }
    ugkwpGpuIpc::GasSpeciesIdentityV1 identity() const
    {
        return {1,static_cast<std::uint32_t>(model_.speciesNames.size()),model_.speciesOrderHash,
                model_.thermoHash,mechanism_.mechanismHash};
    }
    static bool closeScalar(scalar actual,scalar expected)
    {
        return std::isfinite(actual) && mag(actual-expected)<=1e-10*max(mag(expected),scalar(1e-20));
    }
    void pointState(scalar pressure,scalar temperature,const vector& velocity,const double* y,
                    scalar& density,scalar& energy) const
    {
        const auto thermo=model_.thermoView<ugkwp::compiledGasSpecies>();
        double rho=0;
        if(ugkwp::mixtureDensityFromPressureTemperature(double(pressure),double(temperature),y,thermo,rho)
           !=ugkwp::ThermoStatus::Success)
            FatalErrorInFunction<<"Invalid mixture p/T/Y state or temperature outside common thermo range"<<exit(FatalError);
        const double internal=ugkwp::mixtureEnergy(y,double(temperature),thermo);
        const scalar total=rho*(internal+0.5*magSqr(velocity));
        if(!std::isfinite(total)) FatalErrorInFunction<<"Nonfinite formation-inclusive mixture energy"<<exit(FatalError);
        density=rho;energy=total;
    }
    void initialiseIdentity(const Time& time,const fvMesh& mesh)
    {
        IOobject header("gasModelIdentity",time.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE,false);
        const bool present=header.headerOk();
        if(time.value()>0 && !present)
            FatalErrorInFunction<<"Mixture restart requires gasModelIdentity at time "<<time.timeName()<<exit(FatalError);
        restartIdentity_.reset(new IOdictionary(IOobject("gasModelIdentity",time.timeName(),mesh,
            IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE)));
        auto& dict=restartIdentity_();
        if(present)
        {
            string speciesHash,thermoHash,mechanismHash;
            dict.lookup("speciesOrderHash")>>speciesHash;
            dict.lookup("thermoHash")>>thermoHash;
            dict.lookup("mechanismHash")>>mechanismHash;
            if(dict.lookupOrDefault<label>("schemaVersion",0)!=1
               || dict.lookupOrDefault<label>("speciesCount",0)!=ugkwp::compiledGasSpecies
               || dict.lookupOrDefault<label>("mode",-1)!=static_cast<label>(model_.mode)
               || std::string(speciesHash.c_str())!=std::to_string(model_.speciesOrderHash)
               || std::string(thermoHash.c_str())!=std::to_string(model_.thermoHash)
               || std::string(mechanismHash.c_str())!=std::to_string(mechanism_.mechanismHash))
                FatalErrorInFunction<<"Mixture restart gas model identity differs from constant/gasModelProperties"<<exit(FatalError);
        }
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
        if(!isFile(modelPath)){rejectMixtureCheckpointForLegacy(time);return;}
        try
        {
            IOdictionary dictionary(IOobject("gasModelProperties",time.constant(),mesh,
                IOobject::MUST_READ,IOobject::NO_WRITE,false));
            OStringStream output;output.precision(std::numeric_limits<double>::max_digits10);
            static_cast<const Foam::dictionary&>(dictionary).write(output,false);
            modelText_=output.str();
            model_=ugkwp::parseGasModelProperties(modelText_);
            if(!active()){rejectMixtureCheckpointForLegacy(time);return;}
            const auto thermo=model_.thermoView<ugkwp::compiledGasSpecies>();
            if(model_.mode==ugkwp::GasMode::MixtureChemistry)
            {
                const std::string mechanismPath=ugkwp::resolveGasMechanismPath(model_,(time.path()/time.constant()).c_str());
                std::ifstream input(mechanismPath);
                if(!input) throw std::runtime_error("cannot read required gas mechanism "+mechanismPath);
                mechanismText_.assign(std::istreambuf_iterator<char>(input),{});
                if(input.bad()) throw std::runtime_error("failed reading gas mechanism "+mechanismPath);
                mechanism_=ugkwp::parseGasMechanismProperties<ugkwp::compiledGasSpecies>(mechanismText_,thermo,model_.phase);
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
        const bool suppliedRho=isFile(time.timePath()/"rho");
        const bool suppliedMomentum=isFile(time.timePath()/"rhoU");
        const bool suppliedEnergy=isFile(time.timePath()/"rhoE");
        massFractions_.setSize(ugkwp::compiledGasSpecies);
        forAll(massFractions_,species)
        {
            const word name("Y_"+model_.speciesNames[species]);
            massFractions_.set(species,new volScalarField(IOobject(name,time.timeName(),mesh,
                IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh));
            if(massFractions_[species].dimensions()!=dimless)
                FatalErrorInFunction<<name<<" must be dimensionless mass fraction"<<exit(FatalError);
            massFractions_[species].correctBoundaryConditions();
        }
        U.correctBoundaryConditions();p.correctBoundaryConditions();temperature.correctBoundaryConditions();
        forAll(rho,cell)
        {
            double y[ugkwp::compiledGasSpecies];
            forAll(massFractions_,species) y[species]=massFractions_[species][cell];
            scalar density=0,energy=0;pointState(p[cell],temperature[cell],U[cell],y,density,energy);
            if(suppliedRho && !closeScalar(rho[cell],density))
                FatalErrorInFunction<<"Initial rho is inconsistent with p/T/Y in cell "<<cell<<exit(FatalError);
            if(suppliedEnergy && !closeScalar(rhoE[cell],energy))
                FatalErrorInFunction<<"Initial rhoE is inconsistent with formation-inclusive thermo in cell "<<cell<<exit(FatalError);
            if(suppliedMomentum && mag(rhoU[cell]-density*U[cell])>1e-10*max(mag(density*U[cell]),scalar(1e-20)))
                FatalErrorInFunction<<"Initial rhoU is inconsistent with rho/U in cell "<<cell<<exit(FatalError);
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
            const bool supplied=isFile(time.timePath()/name);
            speciesDensity_.set(species,new volScalarField(IOobject(name,time.timeName(),mesh,
                IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),rho*massFractions_[species]));
            if(speciesDensity_[species].dimensions()!=dimDensity)
                FatalErrorInFunction<<name<<" must have density dimensions"<<exit(FatalError);
            if(supplied) forAll(rho,cell)
                if(!closeScalar(speciesDensity_[species][cell],rho[cell]*massFractions_[species][cell]))
                    FatalErrorInFunction<<name<<" is inconsistent with rho/Y in cell "<<cell<<exit(FatalError);
        }
    }
    void configureResident(void* handle) const
    {
        if(!active()) return;
        ugkwpGpuIpc::GasModelCapabilitiesV1 capabilities{};
        checkResult(ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1(handle,&capabilities),"query shared gas model capabilities");
        if(capabilities.compiledSpecies!=ugkwp::compiledGasSpecies
           || !(capabilities.modeMask&(1U<<static_cast<unsigned>(model_.mode))))
            FatalErrorInFunction<<"GPU backend does not support the requested compiled Ns/mode"<<exit(FatalError);
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
                if(type!="fixedValue" && type!="zeroGradient" && type!="cyclic" && type!="empty"
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
                if(!std::isfinite(value) || value<0 || !(rho[cell]>0))
                    FatalErrorInFunction<<"Invalid accepted species output"<<exit(FatalError);
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
