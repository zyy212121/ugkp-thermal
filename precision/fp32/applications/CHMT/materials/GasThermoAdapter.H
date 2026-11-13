#ifndef CHMT_MATERIALS_GASTHERMOADAPTER_H
#define CHMT_MATERIALS_GASTHERMOADAPTER_H
#include "core/Types.H"
#include "../../../common/gasTransport/MixtureThermo.H"
namespace chmt {
// Immutable application POD -> short-lived common borrowed view. Copies own
// their arrays; view() always points at the current object, never a prior copy.
template<int Count,class TableReal=EnergyReal>
struct GasThermoAdapter {
    ugkwp::SpeciesThermoData<TableReal> data[Count]{};
    TableReal coefficients[14*Count]{};
    CHMT_HD explicit GasThermoAdapter(const SpeciesThermo* input,bool =false) {
        for(int i=0;i<Count;++i) {
            const SpeciesThermo& s=input[i];auto& d=data[i];
            d.model=s.model;d.coefficientOffset=14*i;
            // Inactive historical CHMT tables can be entirely unconfigured.
            // Form invalid metadata without raising OpenFOAM's enabled FPE
            // traps; common validation still rejects it when it is active.
            const TableReal Ru=ugkwp::universalGasConstant<TableReal>();
            const TableReal minimumR=Ru/std::numeric_limits<TableReal>::max();



            d.molarMass=finite(s.R)&&s.R>minimumR ? Ru/TableReal(s.R) : TableReal(0);
            d.minTemperature=s.Tmin;d.midTemperature=s.Tmid;d.maxTemperature=s.Tmax;
            d.referencePressure=s.referencePressure;d.referenceTemperature=s.referenceTemperature;
            d.referenceEntropy=s.referenceEntropy;d.hasEntropyReference=s.hasEntropyReference;
            if(s.model==ugkwp::SpeciesThermoModel::NASA7)
                for(int j=0;j<14;++j)coefficients[14*i+j]=s.nasa7[j];
            else {coefficients[14*i]=s.cp0;coefficients[14*i+1]=s.cp1;coefficients[14*i+2]=s.e0;}
        }
    }
    CHMT_HD ugkwp::SpeciesThermoView<TableReal,Count> view(ugkp::diagnostics::DeviceContext* diagnostics=nullptr) const {
        ugkwp::SpeciesThermoView<TableReal,Count> result;
        result.species=data;result.coefficients=coefficients;result.coefficientCount=14*Count;result.diagnostics=diagnostics;
        return result;
    }
};



#define CHMT_SHARED_SPECIES_PROPERTY(Name) \
CHMT_HD inline EnergyReal Name(const SpeciesThermo& s,EnergyReal T) { \
    const GasThermoAdapter<1> thermo(&s);return ugkwp::Name(0,T,thermo.view()); \
}
CHMT_SHARED_SPECIES_PROPERTY(speciesCp)
CHMT_SHARED_SPECIES_PROPERTY(speciesCv)
CHMT_SHARED_SPECIES_PROPERTY(speciesE)
CHMT_SHARED_SPECIES_PROPERTY(speciesH)
#undef CHMT_SHARED_SPECIES_PROPERTY

// This callback joins independent condensed calorics with shared gas calorics.
// Its inactive-species behavior preserves historical CHMT thermal recovery;
// configuration/chemistry validation still checks every configured species.
struct MaterialEnergyEvaluator {
    ugkwp::SpeciesThermoView<EnergyReal,Ns> thermo;
    const PhysicalReal* gasMass;
    EnergyReal a0,a1,a2;
    const SpeciesThermo* supplied=nullptr;
    CHMT_HD ugkwp::ThermoStatus evaluate(EnergyReal T,EnergyReal& energy,EnergyReal& capacity) const {
        if(!finite(T))return ugkwp::ThermoStatus::NonFiniteInput;
        if(T<=0)return ugkwp::ThermoStatus::TemperatureOutOfRange;
        EnergyReal value=a0+T*(a1+EnergyReal(.5)*a2*T),derivative=a1+a2*T;
        for(int s=0;s<Ns;++s) {
            if((!finite(gasMass[s])||gasMass[s]<0))return ugkwp::ThermoStatus::InvalidComposition;
            if(gasMass[s]==0)continue;
            EnergyReal e,cv;
            e=ugkwp::speciesE(s,T,thermo);cv=ugkwp::speciesCv(s,T,thermo);
            if(!finite(e)||!finite(cv))return ugkwp::ThermoStatus::NonFiniteInput;
            value+=EnergyReal(gasMass[s])*e;derivative+=EnergyReal(gasMass[s])*cv;
        }
        if((!finite(value)||!finite(derivative)))return ugkwp::ThermoStatus::NonFiniteInput;
        if(!(derivative>0))return ugkwp::ThermoStatus::NonPositiveHeatCapacity;
        energy=value;capacity=derivative;return ugkwp::ThermoStatus::Success;
    }
    CHMT_HD ugkwp::ThermoStatus operator()(EnergyReal T,EnergyReal& energy,EnergyReal& capacity) const {
        return evaluate(T,energy,capacity);
    }
};


// Three-point Gauss integration is exact for NASA7's quartic cv on one
// temperature branch and for LinearCp. Formation constants never enter.
template<bool Terminal=false>
CHMT_HD inline EnergyReal gasCapacitySecantBranch(const SpeciesThermo& s,EnergyReal a,EnergyReal b) {
    const EnergyReal midpoint=EnergyReal(.5)*(a+b),offset=EnergyReal(.5)*(b-a)*std::sqrt(EnergyReal(3)/5);
    const GasThermoAdapter<1> thermo(&s);
    return (EnergyReal(5)*ugkwp::speciesCv(0,midpoint-offset,thermo.view())+EnergyReal(8)*ugkwp::speciesCv(0,midpoint,thermo.view())
        +EnergyReal(5)*ugkwp::speciesCv(0,midpoint+offset,thermo.view()))/18;
}
template<bool Terminal=false>
CHMT_HD inline EnergyReal speciesEnergySecant(const SpeciesThermo& s,EnergyReal a,EnergyReal b) {
    if(a==b)return speciesCv(s,a);
    if((!finite(a)||!finite(b)||a<=0||b<=0))return std::numeric_limits<EnergyReal>::quiet_NaN();
    if(a>b){const EnergyReal swap=a;a=b;b=swap;}
    if(s.model==ugkwp::SpeciesThermoModel::NASA7&&a<=EnergyReal(s.Tmid)&&b>EnergyReal(s.Tmid)) {
        // Evaluate both one-sided values with the shared evaluator. Source
        // tables may have small jumps; integrating cv alone would refit them.
        GasThermoAdapter<1> upper(&s);
        for(int j=0;j<7;++j)upper.coefficients[j]=s.nasa7[j+7];
        const EnergyReal upperEnergy=ugkwp::speciesE(0,EnergyReal(s.Tmid),upper.view());
        const EnergyReal jump=upperEnergy-speciesE(s,EnergyReal(s.Tmid));
        return ((EnergyReal(s.Tmid)-a)*gasCapacitySecantBranch<Terminal>(s,a,EnergyReal(s.Tmid))
            +(b-EnergyReal(s.Tmid))*gasCapacitySecantBranch<Terminal>(s,EnergyReal(s.Tmid),b)+jump)/(b-a);
    }
    return gasCapacitySecantBranch<Terminal>(s,a,b);
}
}
#endif
