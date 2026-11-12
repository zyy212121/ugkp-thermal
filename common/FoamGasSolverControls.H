#pragma once
#include "FoamSolverScheme.H"
#include "GpuPrecisionTypes.H"
#include "gasTransport/GasModelIO.H"

namespace ugkp { namespace solverControls {

// The serialized model is the common host/device source of truth. Only live
// numerical keys are injected; physical model bounds and compatibility-only fields are not exposed.
inline void applyGasSolverControlOverrides(Foam::dictionary& model,
    const Foam::Time& time,const Foam::word& region=Foam::word::null)
{
    const auto& scheme=startupScheme(time);
    model.remove("solverControlOverrides"); // This transport block is owned by scheme, not by the model file.
    Foam::dictionary overrides,chemistry,thermo;
    const auto chemical=scheme.module("gasChemistry",region);
    const auto thermal=scheme.module("gasThermo",region);
    for(const char* key:{"relativeTolerance","absoluteMassFractionTolerance",
        "newtonTolerance","lineSearchReduction"})
        if(chemical.found(key))chemistry.set(key,chemical.lookupOrDefault<double>(key,0));
    for(const char* key:{"maximumNewtonIterations","maximumLineSearchIterations"})
        if(chemical.found(key))chemistry.set(key,chemical.lookupOrDefault<Foam::label>(key,0));
    for(const char* key:{"absoluteEnergyTolerance","relativeEnergyTolerance",
        "absoluteTemperatureTolerance","relativeTemperatureTolerance"})
        if(thermal.found(key))thermo.set(key,thermal.lookupOrDefault<double>(key,0));
    if(thermal.found("maximumIterations"))
        thermo.set("maximumIterations",thermal.lookupOrDefault<Foam::label>("maximumIterations",0));
    if(!chemistry.empty())overrides.set("chemistry",chemistry);
    if(!thermo.empty())overrides.set("thermo",thermo);
    if(!overrides.empty())model.set("solverControlOverrides",overrides);
}
inline void reportGasSolverControls(const ugkwp::GasModelConfiguration& model,
    const Foam::Time& time,const Foam::word& region=Foam::word::null)
{
    if(model.mode==ugkwp::GasMode::SingleLegacy)return;
    const auto& scheme=startupScheme(time);
    const auto thermal=scheme.module("gasThermo",region);
    const auto chemical=scheme.module("gasChemistry",region);
    const auto c=ugkwp::castChemistryControls<GpuReal>(model.chemistryControls);
#define UGKP_REPORT_THERMO(Key) scheme.report("gasThermo",#Key,c.thermo.Key,thermal.found(#Key),"precisionDefault",region)
    UGKP_REPORT_THERMO(absoluteEnergyTolerance);
    UGKP_REPORT_THERMO(relativeEnergyTolerance);
    UGKP_REPORT_THERMO(absoluteTemperatureTolerance);
    UGKP_REPORT_THERMO(relativeTemperatureTolerance);
    UGKP_REPORT_THERMO(maximumIterations);
#undef UGKP_REPORT_THERMO
    if(model.mode!=ugkwp::GasMode::MixtureChemistry)return;
#define UGKP_REPORT_CHEMISTRY(Key) scheme.report("gasChemistry",#Key,c.Key,chemical.found(#Key),"precisionDefault",region)
    UGKP_REPORT_CHEMISTRY(relativeTolerance);
    UGKP_REPORT_CHEMISTRY(absoluteMassFractionTolerance);
    UGKP_REPORT_CHEMISTRY(newtonTolerance);
    UGKP_REPORT_CHEMISTRY(maximumNewtonIterations);
    UGKP_REPORT_CHEMISTRY(maximumLineSearchIterations);
    UGKP_REPORT_CHEMISTRY(lineSearchReduction);
#undef UGKP_REPORT_CHEMISTRY
}
}}
