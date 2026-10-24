#pragma once
#include "Diagnostics.H"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
// One operator implementation; scalar/time adapters are compile-time only.
template<class GasState>
__device__ void gasFaceSubgridTransportProperties
(
    const GasState& s,
    const int f,
    const int own,
    const int nei,
    const int boundaryKind,
    const GPU_OPERATOR_REAL rhoFace,
    GPU_OPERATOR_REAL& muTurbulent,
    GPU_OPERATOR_REAL& kTurbulent,
    GPU_OPERATOR_REAL& directWallHeatFlux,
    int& directWallHeatFluxActive,
    const bool nativeRefinement=false, bool* evaluated=nullptr
)
{
    using namespace ugkp::diagnostics;
    (void)nativeRefinement; // One physical evaluation; legacy trial selector is inert.
    if(evaluated)*evaluated=false;
    muTurbulent=kTurbulent=directWallHeatFlux=GPU_OPERATOR_R(0.0);
    directWallHeatFluxActive=0;
    const auto failure=[&](Code code,unsigned site,GPU_OPERATOR_REAL value,
        unsigned quantity,int component=-1)
    {
        if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(code,site,26,0,
            EntityKind::Face,f,component,value,0,0,quantity));
        ugkwp::gasRecordFaceFailure(s,f,code==Code::MissingOperator
            ?ugkwp::GasTransportCode::InvalidStorage:ugkwp::GasTransportCode::InvalidThermodynamics);
    };
    const auto publishTransport=[&](GPU_OPERATOR_REAL mu,GPU_OPERATOR_REAL conductivity)
    {
        const GPU_OPERATOR_REAL values[]={mu,conductivity};
        for(int component=0;component<2;++component)
            if(!ugkwp::gasFinite(values[component]))
            {failure(Code::NonFinite,2667,values[component],Quantity::Unspecified,component);return;}
        muTurbulent=mu;kTurbulent=conductivity;
        if(evaluated)*evaluated=true;
    };
    // Stability estimation also calls this helper for laminar walls. Never
    // evaluate a turbulent wall law (notably nu=0 Spalding) on that path.
    if(s.turbulenceModel==0)
    {
        muTurbulent=GPU_OPERATOR_R(0.0);kTurbulent=GPU_OPERATOR_R(0.0);
        directWallHeatFlux=GPU_OPERATOR_R(0.0);directWallHeatFluxActive=0;
        if(evaluated)*evaluated=true;
        return;
    }
    if(nei>=0)
    {
        GPU_OPERATOR_REAL cpOwner=s.gasCp,cpNeighbour=s.gasCp;
        if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
            if(ugkwp::mixtureGasActive(s))
            {
                cpOwner=s.gasSpecies.heatCapacity[own];
                cpNeighbour=s.gasSpecies.heatCapacity[nei];
            }
        const GPU_OPERATOR_REAL inputs[]={s.rho[own],s.rho[nei],s.nut[own],s.nut[nei],
            cpOwner,cpNeighbour,s.turbulentPrandtl,s.faceWeight[f]};
        for(int component=0;component<8;++component)
            if(!ugkwp::gasFinite(inputs[component]))
            {failure(Code::NonFinite,2668,inputs[component],Quantity::Unspecified,component);return;}
        const GPU_OPERATOR_REAL ownerWeight=clampRange(s.faceWeight[f],GPU_OPERATOR_R(0),GPU_OPERATOR_R(1));
        // The cell-wise timestep caller visits the same face from both cells.
        const GPU_OPERATOR_REAL w=own==s.faceOwner[f]?ownerWeight:GPU_OPERATOR_R(1)-ownerWeight;
        // Interpolate the coefficients multiplying the physical gradients.
        // Averaging nut first (or using reconstructed Riemann density) loses
        // rho/nut/Cp correlations and the location of a non-midpoint face.
        const GPU_OPERATOR_REAL muOwner=clampMin(s.rho[own],s.rhoMin)*fmax(s.nut[own],GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL muNeighbour=clampMin(s.rho[nei],s.rhoMin)*fmax(s.nut[nei],GPU_OPERATOR_R(0.0));
        publishTransport(w*muOwner+(GPU_OPERATOR_R(1)-w)*muNeighbour,
            (w*cpOwner*muOwner+(GPU_OPERATOR_R(1)-w)*cpNeighbour*muNeighbour)
            /clampMin(s.turbulentPrandtl,OfSmall));
        return;
    }
    GPU_OPERATOR_REAL turbulentCp=s.gasCp;
    if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
        if(ugkwp::mixtureGasActive(s))
        {
            const GPU_OPERATOR_REAL w=nei>=0?clampRange(s.faceWeight[f],GPU_OPERATOR_R(0.0),GPU_OPERATOR_R(1.0)):GPU_OPERATOR_R(1.0);
            turbulentCp=w*s.gasSpecies.heatCapacity[own]
                +(GPU_OPERATOR_R(1.0)-w)*s.gasSpecies.heatCapacity[nei>=0?nei:own];
        }
    directWallHeatFlux = GPU_OPERATOR_R(0.0);
    directWallHeatFluxActive = 0;
    GPU_OPERATOR_REAL nutFace = nei >= 0
      ? GPU_OPERATOR_R(0.5)*(s.nut[own] + s.nut[nei])
      : s.nut[own];

    if (nei < 0 && boundaryKind == 2)
    {
        if (s.turbulenceModel == 3 && s.sstWallTreatment != 1)
        {
            muTurbulent = GPU_OPERATOR_R(0.0);
            kTurbulent = GPU_OPERATOR_R(0.0);
            if(evaluated)*evaluated=true;
            return;
        }
        if(s.turbulenceModel==3)
        {
            const GPU_OPERATOR_REAL temperatures[]={
                s.riemannBoundaryTFix[f]!=0?s.riemannBoundaryT[f]:s.Tgas[own],s.Tgas[own]};
            for(int component=0;component<2;++component)
            {
                const auto value=temperatures[component];
                if(!ugkwp::gasFinite(value) || value<GPU_OPERATOR_R(0)
                    || (component==0 && value==GPU_OPERATOR_R(0)))
                {
                    failure(!ugkwp::gasFinite(value)?Code::NonFinite:
                        (value<GPU_OPERATOR_R(0)?Code::NegativeTemperature:Code::ZeroDenominator),
                        2650,value,Quantity::Temperature,component);
                    return;
                }
            }
            const GPU_OPERATOR_REAL denominators[]={s.rho[own],s.gasMu,s.gasPrClamped,s.sstWallKappa};
            for(int component=0;component<4;++component)
                if(!ugkwp::gasFinite(denominators[component]) || denominators[component]==GPU_OPERATOR_R(0))
                {
                    failure(!ugkwp::gasFinite(denominators[component])?Code::NonFinite:Code::ZeroDenominator,
                        2651,denominators[component],component==0?Quantity::Density:Quantity::Denominator,component);
                    return;
                }
        }
        const GPU_OPERATOR_REAL velocities[]={s.Ux[own],s.Uy[own],s.Uz[own],
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUx[f]:GPU_OPERATOR_R(0),
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUy[f]:GPU_OPERATOR_R(0),
            s.riemannBoundaryUFix[f]!=0?s.riemannBoundaryUz[f]:GPU_OPERATOR_R(0)};
        for(int component=0;component<6;++component)
            if(!ugkwp::gasFinite(velocities[component]))
            {failure(Code::NonFinite,2652,velocities[component],Quantity::Velocity,component%3);return;}
        const GPU_OPERATOR_REAL transportInputs[]={s.gasMu,s.p[own],s.deltaCoeffs[f],s.turbulentPrandtl,
            s.turbulenceModel==3?s.sstWallDistance[own]:GPU_OPERATOR_R(1),
            s.turbulenceModel==3?s.sstWallE:GPU_OPERATOR_R(9.8),
            s.turbulenceModel==3?GPU_OPERATOR_R(0):turbulentCp};
        for(int component=0;component<7;++component)
            if(!ugkwp::gasFinite(transportInputs[component]))
            {failure(Code::NonFinite,2664,transportInputs[component],component==1?Quantity::Pressure:Quantity::Unspecified,component);return;}
        const GPU_OPERATOR_REAL wallUx = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL wallUz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], GPU_OPERATOR_R(0.0)) : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL dux = s.Ux[own] - wallUx;
        const GPU_OPERATOR_REAL duy = s.Uy[own] - wallUy;
        const GPU_OPERATOR_REAL duz = s.Uz[own] - wallUz;
        const GPU_OPERATOR_REAL velocityDifference = sqrt
        (
            dux*dux + duy*duy + duz*duz
        );
        const GPU_OPERATOR_REAL wallDistance = s.turbulenceModel == 3
          ? clampMin(s.sstWallDistance[own], OfVSmall)
          : GPU_OPERATOR_R(1.0)/clampMin(s.deltaCoeffs[f], OfVSmall);
        const GPU_OPERATOR_REAL rhoSafe = clampMin
        (
            riemannFacePrimitiveForGradient(s, own, f).rho, s.rhoMin
        );
        const ugkpwall::SpaldingWallState wallState =
            ugkpwall::spaldingWallStateFromNormalGradient
            (
                velocityDifference,
                wallDistance,
                s.gasMu/rhoSafe,
                velocityDifference*clampMin(s.deltaCoeffs[f], GPU_OPERATOR_R(0.0)),
                s.turbulenceModel == 3 ? s.sstWallKappa : GPU_OPERATOR_R(0.41),
                s.turbulenceModel == 3 ? s.sstWallE : GPU_OPERATOR_R(9.8), s.diagnostics, f
            );
        const GPU_OPERATOR_REAL mechanicalOutputs[]={wallState.uTau,wallState.yPlus,wallState.nut};
        for(int component=0;component<3;++component)
            if(!ugkwp::gasFinite(mechanicalOutputs[component]))
            {failure(Code::NonFinite,2666,mechanicalOutputs[component],Quantity::Unspecified,component);return;}
        if (s.turbulenceModel == 3)
        {
            const GPU_OPERATOR_REAL wallTemperature = s.riemannBoundaryTFix[f] != 0
              ? s.riemannBoundaryT[f] : s.Tgas[own];
            GPU_OPERATOR_REAL wallRho = rhoSafe;
            if(!ugkwp::mixtureGasActive(s))
                wallRho=clampMin(s.p[own]/(s.Rgas*clampMin(wallTemperature,s.TgasMin)),s.rhoMin);
            GPU_OPERATOR_REAL wallCp=s.gasCp, wallPr=s.gasPrClamped;
            GPU_OPERATOR_REAL wallP=s.sstJayatillekeP, thermalYPlus=s.sstThermalYPlus;
            GPU_OPERATOR_REAL molecularBaseline=molecularGasConductivity(s);
            if constexpr (ugkwp::GasStateTraits<GasState>::speciesCount > 0)
                if(ugkwp::mixtureGasActive(s))
                {
                    // The impermeable wall inherits owner composition, but its
                    // thermodynamic properties are evaluated at wall temperature.
                    // No legacy bootstrap Cp/R participates in the shared
                    // mixture closure.
                    constexpr int Ns=ugkwp::GasStateTraits<GasState>::speciesCount;
                    using TableReal=typename std::remove_reference_t<decltype(s.gasSpecies)>::RealType;
                    GPU_OPERATOR_REAL Y[Ns],sum=GPU_OPERATOR_R(0),clippedSum=GPU_OPERATOR_R(0);
                    // Record failures at this face producer, with its operator and
                    // component, rather than letting a property-only context win.
                    auto thermo=s.gasSpecies.thermo;thermo.diagnostics=nullptr;
                    wallCp=GPU_OPERATOR_R(0.0);
                    for(int species=0;species<Ns;++species)
                    {
                        if(!ugkwp::gasThermoDetail::validMetadata(species,thermo))
                        {failure(Code::MissingOperator,2653,GPU_OPERATOR_REAL(thermo.coefficientCount),Quantity::Index,species);return;}
                        const auto& data=thermo.species[species];
                        if(!ugkwp::gasFinite(data.molarMass) || data.molarMass==TableReal(0))
                        {failure(!ugkwp::gasFinite(data.molarMass)?Code::NonFinite:Code::ZeroDenominator,
                            2654,GPU_OPERATOR_REAL(data.molarMass),Quantity::Denominator,species);return;}
                        const GPU_OPERATOR_REAL value=s.gasSpecies.rho[species*s.nCells+own]/s.rho[own];
                        if(!ugkwp::gasFinite(value))
                        {failure(Code::NonFinite,2655,value,Quantity::MassFraction,species);return;}
                        sum+=value;
                        Y[species]=clampRange(value,GPU_OPERATOR_R(0),GPU_OPERATOR_R(1));
                        clippedSum+=Y[species];
                        if(value!=Y[species])recordWarning(s.diagnostics,makeRecord(Code::FiniteYClip,
                            2656,26,0,EntityKind::Face,f,species,value,0,Y[species],Quantity::MassFraction));
                        if(wallTemperature<data.minTemperature || wallTemperature>data.maxTemperature)
                            recordWarning(s.diagnostics,makeRecord(Code::AccuracyResidual,2657,26,0,
                                EntityKind::Face,f,species,wallTemperature,
                                wallTemperature<data.minTemperature?data.minTemperature:data.maxTemperature,
                                wallTemperature,Quantity::Temperature));
                        ugkwp::gasThermoDetail::SpeciesValues<GPU_OPERATOR_REAL> values{};
                        bool propertyEvaluated=false;
                        GPU_OPERATOR_REAL nonfiniteProperty=ugkwp::gasThermoDetail::invalidValue<GPU_OPERATOR_REAL>();
                        const auto property=ugkwp::gasThermoDetail::evaluateSpecies<GPU_OPERATOR_REAL,Ns,TableReal,false>
                            (species,wallTemperature,thermo,values,&propertyEvaluated,&nonfiniteProperty);
                        if(!propertyEvaluated || property!=ugkwp::ThermoStatus::Success)
                        {
                            const int segment=data.model==ugkwp::SpeciesThermoModel::NASA7 && wallTemperature>data.midTemperature?7:0;
                            GPU_OPERATOR_REAL observed=property==ugkwp::ThermoStatus::InvalidModel
                                ?GPU_OPERATOR_REAL(thermo.coefficientCount-data.coefficientOffset-segment)
                                :nonfiniteProperty;
                            const int count=data.model==ugkwp::SpeciesThermoModel::LinearCp?2:5;
                            for(int coefficient=0;coefficient<count && data.coefficientOffset+segment+coefficient<thermo.coefficientCount;++coefficient)
                                if(!ugkwp::gasFinite(thermo.coefficients[data.coefficientOffset+segment+coefficient]))
                                {observed=thermo.coefficients[data.coefficientOffset+segment+coefficient];break;}
                            failure(property==ugkwp::ThermoStatus::InvalidModel?Code::MissingOperator:Code::NonFinite,
                                2658,observed,property==ugkwp::ThermoStatus::InvalidModel?Quantity::Index:Quantity::HeatCapacity,species);
                            return;
                        }
                        wallCp+=Y[species]*values.cp;
                    }
                    if(clippedSum==GPU_OPERATOR_R(0))
                    {failure(Code::ZeroDenominator,2659,clippedSum,Quantity::MassFraction);return;}
                    // Raw fractions supply only the conservation audit. Physical
                    // properties above and below consume the finite clipped Y.
                    if(!ugkwp::gasFinite(sum) || fabs(sum-GPU_OPERATOR_R(1))>s.gasSpecies.densityClosureTolerance)
                        recordWarning(s.diagnostics,makeRecord(Code::ConservationResidual,2660,26,0,
                            EntityKind::Face,f,-1,sum,1,sum,Quantity::Conservation));
                    const GPU_OPERATOR_REAL R=ugkwp::mixtureGasConstant(Y,thermo);
                    GPU_OPERATOR_REAL molecular=s.gasMu*wallCp/wallPr;
                    if constexpr(ugkwp::GasDirectConductivity<GasState>::value)
                        if(ugkwp::gasHasDirectConductivity(s))molecular=s.gasThermalConductivity;
                    const GPU_OPERATOR_REAL thermodynamicDenominators[]={R*wallTemperature,
                        ugkwp::gasHasDirectConductivity(s)?molecular:GPU_OPERATOR_R(1)};
                    for(int component=0;component<2;++component)
                        if(!ugkwp::gasFinite(thermodynamicDenominators[component]) || thermodynamicDenominators[component]==GPU_OPERATOR_R(0))
                        {failure(!ugkwp::gasFinite(thermodynamicDenominators[component])?Code::NonFinite:Code::ZeroDenominator,
                            2669,thermodynamicDenominators[component],Quantity::Denominator,component);return;}
                    wallRho=s.p[own]/(R*wallTemperature);
                    const GPU_OPERATOR_REAL properties[]={wallCp,wallRho,molecular};
                    for(int component=0;component<3;++component)
                        if(!ugkwp::gasFinite(properties[component]))
                        {failure(Code::NonFinite,2661,properties[component],component==1?Quantity::Density:Quantity::Unspecified,component);return;}
                    if(ugkwp::gasHasDirectConductivity(s))
                    {
                        // Only direct conductivity makes Pr depend on wall Cp(T,Y).
                        // Constant-Pr transport reuses the configuration-stage cache.
                        wallPr=s.gasMu*wallCp/molecular;
                        const GPU_OPERATOR_REAL ratio=wallPr/clampMin(s.turbulentPrandtl,OfSmall);
                        wallP=ugkpwall::jayatillekeSmoothP(ratio);
                        thermalYPlus=ugkpwall::jayatillekeThermalYPlus(ratio,s.sstWallKappa,s.sstWallE,s.diagnostics,f);
                    }
                    // The face flux adds owner-mixture molecular conductivity.
                    // Return exactly its correction,
                    // even when Cp(Twall) differs from Cp(Towner).
                    if(!ugkwp::gasHasDirectConductivity(s))
                        molecularBaseline=s.gasMu*s.gasSpecies.heatCapacity[own]/s.gasPrClamped;
                }
            const GPU_OPERATOR_REAL gradient = (wallTemperature-s.Tgas[own])*s.deltaCoeffs[f];
            const GPU_OPERATOR_REAL thermalInputs[]={wallRho,wallCp,wallPr,wallP,thermalYPlus,
                s.k[own],s.sstWallCmu,gradient,molecularBaseline};
            for(int component=0;component<9;++component)
                if(!ugkwp::gasFinite(thermalInputs[component]))
                {failure(Code::NonFinite,2665,thermalInputs[component],component==0?Quantity::Density:Quantity::Unspecified,component);return;}
            const auto thermal = ugkpwall::sstJayatillekeThermalTransport
            (
                wallRho, wallCp, s.gasMu, wallPr,
                s.turbulentPrandtl, s.sstWallCmu, s.sstWallKappa,
                s.sstWallE, wallP, thermalYPlus,
                s.k[own], wallDistance, velocityDifference,
                sqrt(wallUx*wallUx + wallUy*wallUy + wallUz*wallUz),
                gradient,false
            );
            const GPU_OPERATOR_REAL outputs[]={rhoSafe*wallState.nut,
                thermal.heatFlux,thermal.conductivity-molecularBaseline};
            for(int component=0;component<3;++component)
                if(!ugkwp::gasFinite(outputs[component]))
                {failure(Code::NonFinite,2662,outputs[component],Quantity::Unspecified,component);return;}
            // This flag denotes failed arithmetic (including an unexposed
            // nonfinite intermediate), never a convergence residual.
            if(thermal.valid==0)
            {failure(Code::InvalidPhysicalState,2663,GPU_OPERATOR_REAL(thermal.valid),Quantity::Unspecified);return;}
            muTurbulent = rhoSafe*wallState.nut;
            directWallHeatFlux = thermal.heatFlux;
            directWallHeatFluxActive = s.riemannBoundaryTFix[f] != 0;
            kTurbulent = thermal.conductivity - molecularBaseline;
            if(evaluated)*evaluated=true;
            return;
        }
        const ugkpwall::WallSubgridTransport wallTransport =
            ugkpwall::wallSubgridTransport
            (
                rhoSafe,
                turbulentCp,
                s.turbulentPrandtl,
                wallState.nut
            );
        publishTransport(wallTransport.dynamicViscosity,wallTransport.thermalConductivity);
        return;
    }

    const GPU_OPERATOR_REAL bulkInputs[]={rhoFace,nutFace,turbulentCp,s.turbulentPrandtl};
    for(int component=0;component<4;++component)
        if(!ugkwp::gasFinite(bulkInputs[component]))
        {failure(Code::NonFinite,2668,bulkInputs[component],component==0?Quantity::Density:Quantity::Unspecified,component);return;}
    const GPU_OPERATOR_REAL muT = clampMin(rhoFace, s.rhoMin)*fmax(nutFace, GPU_OPERATOR_R(0.0));
    publishTransport(muT,turbulentCp*muT/clampMin(s.turbulentPrandtl, OfSmall));
}
