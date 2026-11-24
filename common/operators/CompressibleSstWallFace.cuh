#pragma once

#include "gasNumerics/CompressibleWallFunction.H"
#include "gasTransport/GasGeometryValidation.H"
#include "gasTransport/GasMolecularConductivity.cuh"
#include "gasWall/WallFamily.H"

template<class GasState>
__device__ CompressibleWallFunction::Result evaluateSstWallLaw
(
    const GasState& s, int f, int owner, GpuReal density, GpuReal viscosity,
    GpuReal cp, GpuReal conductivity, GpuReal thermalYPlus
)
{
    using namespace ugkp::diagnostics;
    const bool fixedTemperature = s.riemannBoundaryTFix[f] != 0;
    const bool fixedVelocity = s.riemannBoundaryUFix[f] != 0;
    const GpuReal area = s.magSf[f];
    if (!std::isfinite(area) || area <= GpuReal(0))
    {
        ugkwp::gasRecordFaceFailure(s, f, ugkwp::GasTransportCode::InvalidGeometry);
        return {};
    }
    const GpuReal nx = GpuReal(s.Sfx[f])/area;
    const GpuReal ny = GpuReal(s.Sfy[f])/area;
    const GpuReal nz = GpuReal(s.Sfz[f])/area;
    const GpuReal ux = s.Ux[owner] - (fixedVelocity ? s.riemannBoundaryUx[f] : GpuReal(0));
    const GpuReal uy = s.Uy[owner] - (fixedVelocity ? s.riemannBoundaryUy[f] : GpuReal(0));
    const GpuReal uz = s.Uz[owner] - (fixedVelocity ? s.riemannBoundaryUz[f] : GpuReal(0));
    const GpuReal normalSpeed = ux*nx + uy*ny + uz*nz;
    const GpuReal tx = ux - normalSpeed*nx;
    const GpuReal ty = uy - normalSpeed*ny;
    const GpuReal tz = uz - normalSpeed*nz;
    const GpuReal speed = sqrt(tx*tx + ty*ty + tz*tz);
    const auto law = CompressibleWallFunction::evaluate
    (
        {speed, GpuReal(s.deltaCoeffs[f]), density, viscosity, conductivity, cp,
            s.turbulentPrandtl, s.sstWallKappa, s.sstWallE,
            fixedTemperature ? s.riemannBoundaryT[f] : s.Tgas[owner], s.Tgas[owner],
            thermalYPlus, fixedTemperature},
        s.gasScalarSolvers.spaldingMaximumIterations, s.gasScalarSolvers.spaldingTolerance
    );
    if (!law.thermal.valid)
    {
        if (s.diagnostics)
            recordFatal(s.diagnostics, makeRecord(Code::InvalidPhysicalState, 2920, 32, 0,
                EntityKind::Face, f, -1, law.thermal.conductivity, 0, speed));
        ugkwp::gasRecordFaceFailure(s, f, ugkwp::GasTransportCode::InvalidThermodynamics);
    }
    else if (fabs(law.residual) > s.gasScalarSolvers.spaldingTolerance && s.diagnostics)
    {
        recordWarning(s.diagnostics, makeRecord(Code::AccuracyResidual, 2921, 32, 0,
            EntityKind::Face, f, -1, law.residual, s.gasScalarSolvers.spaldingTolerance,
            law.momentum.uTau, Quantity::Accuracy));
    }
    return law;
}

struct SstWallTransport
{
    GpuReal density{};
    GpuReal nut{};
    GpuReal dynamicViscosity{};
    GpuReal excessConductivity{};
    GpuReal heatFlux{};
    bool valid{};
};

template<class GasState>
__device__
#if defined(__CUDACC__)
__noinline__
#endif
SstWallTransport evaluateCompressibleSstWallFace(const GasState& s, int f, int own, GpuReal muWall)
{
    using namespace ugkp::diagnostics;
    const auto failure=[&](Code code,unsigned site,GpuReal value,
        unsigned quantity,int component=-1)
    {
        if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(code,site,26,0,
            EntityKind::Face,f,component,value,0,0,quantity));
        ugkwp::gasRecordFaceFailure(s,f,code==Code::MissingOperator
            ?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);
    };
    const GpuReal temperatures[]={
        s.riemannBoundaryTFix[f]!=0?s.riemannBoundaryT[f]:s.Tgas[own],s.Tgas[own]};
    for(int component=0;component<2;++component)
    {
        const auto value=temperatures[component];
        if(!ugkwp::gasFinite(value) || value<GpuReal(0)
            || (component==0 && value==GpuReal(0)))
        {
            failure(!ugkwp::gasFinite(value)?Code::NonFinite:
                (value<GpuReal(0)?Code::NegativeTemperature:Code::ZeroDenominator),
                2650,value,Quantity::Temperature,component);
            return {};
        }
    }
    const GpuReal denominators[]={s.rho[own],muWall,s.gasPrClamped,s.sstWallKappa};
    for(int component=0;component<4;++component)
        if(!ugkwp::gasFinite(denominators[component]) || denominators[component]==GpuReal(0))
        {
            failure(!ugkwp::gasFinite(denominators[component])?Code::NonFinite:Code::ZeroDenominator,
                2651,denominators[component],component==0?Quantity::Density:Quantity::Denominator,component);
            return {};
        }
    const GpuReal velocities[]={s.Ux[own],s.Uy[own],s.Uz[own],
        s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUx[f]:GpuReal(0),
        s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUy[f]:GpuReal(0),
        s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUz[f]:GpuReal(0)};
    for(int component=0;component<6;++component)
        if(!ugkwp::gasFinite(velocities[component]))
        {failure(Code::NonFinite,2652,velocities[component],Quantity::Velocity,component%3);return {};}
    const GpuReal transportInputs[]={muWall,s.p[own],s.deltaCoeffs[f],s.turbulentPrandtl,
        s.sstWallDistance[own],s.sstWallE};
    for(int component=0;component<6;++component)
        if(!ugkwp::gasFinite(transportInputs[component]))
        {failure(Code::NonFinite,2664,transportInputs[component],component==1?Quantity::Pressure:Quantity::Unspecified,component);return {};}
    const GpuReal density = riemannFacePrimitiveForGradient(s, own, f).rho;
    const GpuReal rhoSafe = clampMin(density, s.rhoMin);
    const GpuReal wallTemperature = s.riemannBoundaryTFix[f] != 0
      ? s.riemannBoundaryT[f] : s.Tgas[own];
    GpuReal wallCp=s.gasCp, wallPr=ugkwp::gasFaceMolecularPrandtl(s,f,own,-1);
    GpuReal thermalYPlus=s.sstThermalYPlus;
    const bool prescribedConductivity=ugkwp::gasHasDirectConductivity(s);
    GpuReal molecularBaseline=molecularGasConductivity(s);
    if((ugkwp::gasVariableViscosity(s)||ugkwp::gasVariablePrandtl(s))&&!prescribedConductivity)
        molecularBaseline=muWall*ugkwp::gasFaceMolecularHeatCapacity(s,f,own,-1)/wallPr;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {




            constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
            using TableReal=std::remove_cv_t<std::remove_pointer_t<decltype(s.gasSpecies.thermo.coefficients)>>;
            GpuReal sum=GpuReal(0),clippedSum=GpuReal(0);


            auto thermo=s.gasSpecies.thermo;thermo.diagnostics=nullptr;
            wallCp=GpuReal(0.0);
            for(int species=0;species<Ns;++species)
            {
                if(!ugkwp::gasThermoDetail::validMetadata(species,thermo))
                {failure(Code::MissingOperator,2653,GpuReal(thermo.coefficientCount),Quantity::Index,species);return {};}
                const auto& data=thermo.species[species];
                if(!ugkwp::gasFinite(data.molarMass) || data.molarMass==TableReal(0))
                {failure(!ugkwp::gasFinite(data.molarMass)?Code::NonFinite:Code::ZeroDenominator,
                    2654,GpuReal(data.molarMass),Quantity::Denominator,species);return {};}
                const GpuReal value=(ugkwp::gasVariableViscosity(s)||ugkwp::gasVariablePrandtl(s))
                    &&s.gasSpecies.compositionBoundaryFixed&&s.gasSpecies.compositionBoundaryFixed[f]
                    ?s.gasSpecies.boundaryMassFraction[species*s.nFaces+f]
                    :s.gasSpecies.rho[species*s.nCells+own]/s.rho[own];
                if(!ugkwp::gasFinite(value))
                {failure(Code::NonFinite,2655,value,Quantity::MassFraction,species);return {};}
                sum+=value;
                const GpuReal fraction=clampRange(value,GpuReal(0),GpuReal(1));
                clippedSum+=fraction;
                if(value!=fraction)recordWarning(s.diagnostics,makeRecord(Code::FiniteYClip,
                    2656,26,0,EntityKind::Face,f,species,value,0,fraction,Quantity::MassFraction));
                if(wallTemperature<data.minTemperature || wallTemperature>data.maxTemperature)
                    recordWarning(s.diagnostics,makeRecord(Code::AccuracyResidual,2657,26,0,
                        EntityKind::Face,f,species,wallTemperature,
                        wallTemperature<data.minTemperature?data.minTemperature:data.maxTemperature,
                        wallTemperature,Quantity::Temperature));
                ugkwp::gasThermoDetail::SpeciesValues<GpuReal> values{};
                bool propertyEvaluated=false;
                GpuReal nonfiniteProperty=ugkwp::gasThermoDetail::invalidValue<GpuReal>();
                const auto property=ugkwp::gasThermoDetail::evaluateSpecies<GpuReal,Ns,TableReal,false>
                    (species,wallTemperature,thermo,values,&propertyEvaluated,&nonfiniteProperty);
                if(!propertyEvaluated || property!=ugkwp::ThermoStatus::Success)
                {
                    const int segment=data.model==ugkwp::SpeciesThermoModel::NASA7 && wallTemperature>data.midTemperature?7:0;
                    GpuReal observed=property==ugkwp::ThermoStatus::InvalidModel
                        ?GpuReal(thermo.coefficientCount-data.coefficientOffset-segment)
                        :nonfiniteProperty;
                    const int count=data.model==ugkwp::SpeciesThermoModel::LinearCp?2:5;
                    for(int coefficient=0;coefficient<count && data.coefficientOffset+segment+coefficient<thermo.coefficientCount;++coefficient)
                        if(!ugkwp::gasFinite(thermo.coefficients[data.coefficientOffset+segment+coefficient]))
                        {observed=thermo.coefficients[data.coefficientOffset+segment+coefficient];break;}
                    failure(property==ugkwp::ThermoStatus::InvalidModel?Code::MissingOperator:Code::NonFinite,
                        2658,observed,property==ugkwp::ThermoStatus::InvalidModel?Quantity::Index:Quantity::HeatCapacity,species);
                    return {};
                }
                wallCp+=fraction*values.cp;
            }
            if(clippedSum==GpuReal(0))
            {failure(Code::ZeroDenominator,2659,clippedSum,Quantity::MassFraction);return {};}


            if(!ugkwp::gasFinite(sum) || fabs(sum-GpuReal(1))>s.gasSpecies.densityClosureTolerance)
                recordWarning(s.diagnostics,makeRecord(Code::ConservationResidual,2660,26,0,
                    EntityKind::Face,f,-1,sum,1,sum,Quantity::Conservation));
            if(!prescribedConductivity&&!ugkwp::gasVariableViscosity(s)&&!ugkwp::gasVariablePrandtl(s))
                molecularBaseline=muWall*s.gasSpecies.heatCapacity[own]/wallPr;
        }
    if(!ugkwp::gasFinite(wallPr)||wallPr==GpuReal(0))
    {failure(!ugkwp::gasFinite(wallPr)?Code::NonFinite:Code::ZeroDenominator,
        2669,wallPr,Quantity::Denominator);return {};}
    const GpuReal wallConductivity=prescribedConductivity
        ? molecularBaseline : muWall*wallCp/wallPr;
    if(prescribedConductivity)
    {
        if(!ugkwp::gasFinite(wallConductivity)||wallConductivity==GpuReal(0))
        {failure(!ugkwp::gasFinite(wallConductivity)?Code::NonFinite:Code::ZeroDenominator,
            2669,wallConductivity,Quantity::Denominator);return {};}
        wallPr=muWall*wallCp/wallConductivity;
    }
    if(prescribedConductivity||ugkwp::gasVariablePrandtl(s))
    {
        const GpuReal ratio=wallPr/clampMin(s.turbulentPrandtl,OfSmall);
        thermalYPlus=ugkpwall::jayatillekeThermalYPlus(ratio,s.sstWallKappa,s.sstWallE,s.diagnostics,f,
            s.gasScalarSolvers.thermalMaximumIterations,s.gasScalarSolvers.thermalRelativeTolerance);
    }
    const GpuReal gradient = (wallTemperature-s.Tgas[own])*s.deltaCoeffs[f];
    const GpuReal thermalInputs[]={wallConductivity,wallPr,thermalYPlus,
        gradient,molecularBaseline,wallCp};
    for(int component=0;component<6;++component)
        if(!ugkwp::gasFinite(thermalInputs[component]))
        {failure(Code::NonFinite,2665,thermalInputs[component],Quantity::Unspecified,component);return {};}
    const auto wall = evaluateSstWallLaw
        (s, f, own, rhoSafe, muWall, wallCp, wallConductivity, thermalYPlus);
    if (!wall.thermal.valid) return {};
    const GpuReal outputs[]={rhoSafe*wall.momentum.nut,
        wall.thermal.heatFlux,wall.thermal.conductivity-molecularBaseline};
    for(int component=0;component<3;++component)
        if(!ugkwp::gasFinite(outputs[component]))
        {failure(Code::NonFinite,2662,outputs[component],Quantity::Unspecified,component);return {};}
    return {density,wall.momentum.nut,outputs[0],outputs[2],outputs[1],true};
}
