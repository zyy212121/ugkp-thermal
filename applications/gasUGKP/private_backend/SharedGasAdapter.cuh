#pragma once
// Host adapter for the native resident backend. No transport/thermo/chemistry
// formula is implemented here: model parsing and capability gates are common.
inline ugkwp::GasCapabilityRequest sharedGasCapabilityRequest
(const DeviceState* state,ugkwp::GasMode mode,int wallTreatment)
{
    ugkwp::GasCapabilityRequest request;
    request.mode=mode;request.fluxScheme=state->hostGasFluxScheme;
    request.reconstruction=state->hostGasReconstruction;request.limiter=state->hostGasLimiter;
    request.timeIntegrator=state->hostGasTimeIntegrator;request.turbulenceModel=state->hostTurbulenceModel;
    request.sstWallTreatment=wallTreatment;
    request.particleCoupling=state->particleCapacity!=0;
    return request;
}
inline bool sharedGasIdentityMatches
(
    const DeviceState* state,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity
)
{
    return state && !state->gasModelPoisoned && identity && identity->version==ugkwpGpuIpc::gasModelApiVersion
        && identity->speciesCount==ugkwp::compiledGasSpecies
        && state->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy
        && identity->speciesOrderHash==state->gasSpecies.thermo.speciesOrderHash
        && identity->thermoHash==state->gasSpecies.thermo.thermoHash
        && identity->mechanismHash==state->gasSpecies.mechanism.mechanismHash;
}
inline bool sharedGasViewRollbackFailed(DeviceState* state)
{
    if(!syncSharedGasSpecies(state)) return false;
    // The device may still hold addresses from the rejected view. Forbid all
    // further resident operations and retain those allocations until teardown.
    state->gasModelPoisoned=true;
    setLastErrorText("shared gas device view rollback failed; resident must be destroyed");
    return true;
}
extern "C" int ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1
(void* handle,ugkwpGpuIpc::GasModelCapabilitiesV1* capabilities)
{
    if(validateState(asState(handle),"gas model query") || !capabilities) return 1;
    std::uint32_t modes=1;
    ugkwp::GasCapabilityRequest request;
    request.mode=ugkwp::GasMode::MixtureFrozen;
    if(ugkwp::validateGasCapabilities(request)) modes|=2;
    request.mode=ugkwp::GasMode::MixtureChemistry;
    if(ugkwp::validateGasCapabilities(request)) modes|=4;
    *capabilities={ugkwpGpuIpc::gasModelApiVersion,ugkwp::compiledGasSpecies,modes,0};
    return 0;
}
extern "C" int ugkwpGpuResidentStrictConfigureGasModelV1
(void* handle,const ugkwpGpuIpc::GasModelConfigureArgsV1* args,
 const char* modelText,const char* mechanismText)
{
    DeviceState* state=asState(handle);
    std::uint64_t bytes=0;
    if(validateState(state,"gas model configuration") || !args || !modelText
       || !ugkwpGpuIpc::gasModelPayloadBytes(*args,ugkwp::compiledGasSpecies,bytes)
       || (args->mechanismBytes && !mechanismText))
    {setLastErrorText("invalid shared gas model payload");return 1;}
    if(state->gasInitialFieldsUploaded || state->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)
    {setLastErrorText("gas model must be configured once before initial state upload");return 1;}
    ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies> candidate;
    try
    {
        const auto model=ugkwp::parseGasModelProperties(std::string(modelText,args->modelBytes));
        const auto hostThermo=model.thermoView<ugkwp::compiledGasSpecies>();
        if(static_cast<unsigned>(model.mode)!=args->mode || model.speciesOrderHash!=args->speciesOrderHash
           || model.thermoHash!=args->thermoHash)
            throw std::runtime_error("configured gas model identity differs from parsed canonical model");
        const auto request=sharedGasCapabilityRequest(state,model.mode,state->sstWallTreatment);
        const auto capability=ugkwp::validateGasCapabilities(request);
        if(!capability) throw std::runtime_error(capability.message);
        ugkwp::GasMechanismConfiguration mechanism;
        if(model.mode==ugkwp::GasMode::MixtureChemistry)
        {
            mechanism=ugkwp::parseGasMechanismProperties<ugkwp::compiledGasSpecies>
                (std::string(mechanismText,args->mechanismBytes),hostThermo,model.phase);
            if(mechanism.mechanismHash!=args->mechanismHash)
                throw std::runtime_error("configured mechanism identity differs from canonical model");
        }
        const std::size_t cells=static_cast<std::size_t>(state->nCells),faces=static_cast<std::size_t>(state->nFaces);
        const std::size_t species=ugkwp::compiledGasSpecies;
        int rc=0;
#define ALLOC_SHARED(Name,Count) rc|=allocateSharedGasZero(candidate.Name,(Count))
        ALLOC_SHARED(rho,species*cells); ALLOC_SHARED(initial,species*cells);
        ALLOC_SHARED(flux,species*faces); ALLOC_SHARED(gradX,species*cells);
        ALLOC_SHARED(gradY,species*cells); ALLOC_SHARED(gradZ,species*cells);
        ALLOC_SHARED(limiter,cells); ALLOC_SHARED(boundaryMassFraction,species*faces);
        ALLOC_SHARED(compositionBoundaryFixed,faces); ALLOC_SHARED(positivityScale,species*cells);
        ALLOC_SHARED(soundSpeed,cells); ALLOC_SHARED(heatCapacity,cells);
        ALLOC_SHARED(gasConstant,cells); ALLOC_SHARED(cellStatus,cells); ALLOC_SHARED(faceStatus,faces);
        if(model.mode==ugkwp::GasMode::MixtureChemistry)
        {ALLOC_SHARED(chemistryAudit,cells);ALLOC_SHARED(chemistryStatus,cells);}
#undef ALLOC_SHARED
        // Copy only canonical immutable metadata into separate device allocations.
        candidate.thermo=hostThermo;
        candidate.thermo.species=nullptr;candidate.thermo.coefficients=nullptr;candidate.thermo.elementComposition=nullptr;
        rc|=uploadSharedGasTable(candidate.thermo.species,model.species.data(),model.species.size());
        rc|=uploadSharedGasTable(candidate.thermo.coefficients,model.coefficients.data(),model.coefficients.size());
        rc|=uploadSharedGasTable(candidate.thermo.elementComposition,model.elementComposition.data(),model.elementComposition.size());
        if(model.diffusionModel==ugkwp::GasDiffusionModel::Constant)
            rc|=uploadSharedGasTable(candidate.diffusivity,model.diffusionCoefficients.data(),species);
        candidate.turbulentSchmidt=model.turbulentSchmidt;
        candidate.chemistryControls=model.chemistryControls;
        candidate.thermoControls=model.chemistryControls.thermo;
        if(model.mode==ugkwp::GasMode::MixtureChemistry)
        {
            candidate.mechanism=mechanism.mechanismView<ugkwp::compiledGasSpecies>();
            candidate.mechanism.reactions=nullptr;candidate.mechanism.reactants=nullptr;candidate.mechanism.products=nullptr;
            candidate.mechanism.efficiencies=nullptr;candidate.mechanism.stoichiometricBasis=nullptr;
            rc|=uploadSharedGasTable(candidate.mechanism.reactions,mechanism.reactions.data(),mechanism.reactions.size());
            rc|=uploadSharedGasTable(candidate.mechanism.reactants,mechanism.reactants.data(),mechanism.reactants.size());
            rc|=uploadSharedGasTable(candidate.mechanism.products,mechanism.products.data(),mechanism.products.size());
            rc|=uploadSharedGasTable(candidate.mechanism.efficiencies,mechanism.efficiencies.data(),mechanism.efficiencies.size());
            rc|=uploadSharedGasTable(candidate.mechanism.stoichiometricBasis,mechanism.stoichiometricBasis.data(),mechanism.stoichiometricBasis.size());
        }
        if(rc) {releaseSharedGasSpecies(candidate);return 1;}
        candidate.mode=model.mode;
        state->gasSpecies=candidate;
        if(syncSharedGasSpecies(state))
        {
            state->gasSpecies={};
            if(sharedGasViewRollbackFailed(state)) state->gasRejectedView=candidate;
            else releaseSharedGasSpecies(candidate);
            return 1;
        }
        return 0;
    }
    catch(const std::exception& error)
    {releaseSharedGasSpecies(candidate);setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictUploadSpeciesV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,const double* values)
{
    DeviceState* state=asState(handle);
    if(!sharedGasIdentityMatches(state,identity) || !values || !state->gasInitialFieldsUploaded || state->gasSpeciesUploaded)
    {setLastErrorText("initial species upload requires matching identity and unpopulated species storage");return 1;}
    // This is initial/restart validation only. Evolved fields are never repaired
    // or transferred to the host inside a transport/chemistry stage.
    std::vector<double> density(state->nCells);
    if(copyToHost(density.data(),state->rho,density.size(),"validate initial mixture density")) return 1;
    for(int cell=0;cell<state->nCells;++cell)
    {
        double sum=0;
        for(int species=0;species<ugkwp::compiledGasSpecies;++species)
        {
            const double value=values[std::size_t(species)*state->nCells+cell];
            if(!std::isfinite(value) || value<0){setLastErrorText("invalid initial species inventory");return 1;}
            sum+=value;
        }
        if(!std::isfinite(density[cell]) || density[cell]<=0
           || std::abs(sum-density[cell])>state->gasSpecies.densityClosureTolerance*density[cell])
        {setLastErrorText("initial species density does not close to total density");return 1;}
    }
    // A failed transfer must not leave a partially populated resident field.
    // Stage the complete input before publishing a new borrowed device view.
    double* staged=nullptr;
    const std::size_t count=std::size_t(state->nCells)*ugkwp::compiledGasSpecies;
    if(allocateSharedGasZero(staged,count)
       ||copyToDevice(staged,values,count,"upload initial species density"))
    {releaseSharedGasPointer(staged);return 1;}
    double* accepted=state->gasSpecies.rho;
    state->gasSpecies.rho=staged;
    if(syncSharedGasSpecies(state))
    {
        state->gasSpecies.rho=accepted;
        if(sharedGasViewRollbackFailed(state)) state->gasRejectedView.rho=staged;
        else releaseSharedGasPointer(staged);
        return 1;
    }
    releaseSharedGasPointer(accepted);
    state->gasSpeciesUploaded=true;
    return 0;
}
extern "C" int ugkwpGpuResidentStrictDownloadSpeciesV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,double* values)
{
    DeviceState* state=asState(handle);
    if(!sharedGasIdentityMatches(state,identity) || !values || !state->gasSpeciesUploaded)
    {setLastErrorText("species download identity/state mismatch");return 1;}
    std::vector<double> staged(std::size_t(state->nCells)*ugkwp::compiledGasSpecies);
    if(copyToHost(staged.data(),state->gasSpecies.rho,staged.size(),"download accepted species density"))return 1;
    std::copy(staged.begin(),staged.end(),values);
    return 0;
}
extern "C" int ugkwpGpuResidentStrictUploadSpeciesBoundaryV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,const int* fixed,const double* values)
{
    DeviceState* state=asState(handle);
    if(!sharedGasIdentityMatches(state,identity) || !fixed || !values)
    {setLastErrorText("species boundary identity/state mismatch");return 1;}
    for(int face=0;face<state->nFaces;++face)
    {
        if(fixed[face]!=0 && fixed[face]!=1){setLastErrorText("invalid species composition boundary flag");return 1;}
        double sum=0;
        for(int species=0;species<ugkwp::compiledGasSpecies;++species)
        {
            const double value=values[std::size_t(species)*state->nFaces+face];
            if(!std::isfinite(value) || value<0 || value>1){setLastErrorText("invalid species boundary mass fraction");return 1;}
            sum+=value;
        }
        if(std::abs(sum-1)>state->gasSpecies.densityClosureTolerance)
        {setLastErrorText("species boundary composition must sum to one");return 1;}
    }
    int* stagedFixed=nullptr;
    double* stagedValues=nullptr;
    const std::size_t count=std::size_t(state->nFaces)*ugkwp::compiledGasSpecies;
    if(allocateSharedGasZero(stagedFixed,state->nFaces)
       ||allocateSharedGasZero(stagedValues,count)
       ||copyToDevice(stagedFixed,fixed,state->nFaces,"upload species boundary operators")
       ||copyToDevice(stagedValues,values,count,"upload species boundary composition"))
    {
        releaseSharedGasPointer(stagedFixed);releaseSharedGasPointer(stagedValues);
        return 1;
    }
    int* acceptedFixed=state->gasSpecies.compositionBoundaryFixed;
    double* acceptedValues=state->gasSpecies.boundaryMassFraction;
    state->gasSpecies.compositionBoundaryFixed=stagedFixed;
    state->gasSpecies.boundaryMassFraction=stagedValues;
    if(syncSharedGasSpecies(state))
    {
        state->gasSpecies.compositionBoundaryFixed=acceptedFixed;
        state->gasSpecies.boundaryMassFraction=acceptedValues;
        if(sharedGasViewRollbackFailed(state))
        {
            state->gasRejectedView.compositionBoundaryFixed=stagedFixed;
            state->gasRejectedView.boundaryMassFraction=stagedValues;
        }
        else
        {releaseSharedGasPointer(stagedFixed);releaseSharedGasPointer(stagedValues);}
        return 1;
    }
    releaseSharedGasPointer(acceptedFixed);releaseSharedGasPointer(acceptedValues);
    return 0;
}
