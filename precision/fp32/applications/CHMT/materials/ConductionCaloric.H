#ifndef CHMT_MATERIALS_CONDUCTION_CALORIC_H
#define CHMT_MATERIALS_CONDUCTION_CALORIC_H
#include "materials/MaterialCaloric.H"
#include "../../../common/GpuPrecisionTypes.H"
namespace chmt { namespace materialConductionDetail {
using Real=GpuReal;
CHMT_HD inline bool finite(Real x){return std::isfinite(x);}
CHMT_HD inline Real undefinedValue(){return std::numeric_limits<Real>::quiet_NaN();}
CHMT_HD inline Real caloricIntegral(Real a0,Real a1,Real a2,Real T){return a0+T*(a1+Real(.5)*a2*T);}
struct SpeciesThermo {
    Real R=0,cp0=0,cp1=0,e0=0,Tmin=0,Tmax=0;
    ugkwp::SpeciesThermoModel model=ugkwp::SpeciesThermoModel::LinearCp;
    Real Tmid=0,nasa7[14]{},referencePressure=101325,referenceTemperature=0,referenceEntropy=0;
    bool hasEntropyReference=false;
    CHMT_HD SpeciesThermo& operator=(const chmt::SpeciesThermo& s){
        R=Real(s.R);cp0=Real(s.cp0);cp1=Real(s.cp1);e0=Real(s.e0);Tmin=Real(s.Tmin);Tmax=Real(s.Tmax);
        model=s.model;Tmid=Real(s.Tmid);for(int j=0;j<14;++j)nasa7[j]=Real(s.nasa7[j]);
        referencePressure=Real(s.referencePressure);referenceTemperature=Real(s.referenceTemperature);
        referenceEntropy=Real(s.referenceEntropy);hasEntropyReference=s.hasEntropyReference;return *this;
    }
};
// Immutable application POD -> short-lived common borrowed view. Copies own
// their arrays; view() always points at the current object, never a prior copy.
template<int Count>
struct GasThermoAdapter {
    ugkwp::SpeciesThermoData<Real> data[Count]{};
    Real coefficients[14*Count]{};
    CHMT_HD explicit GasThermoAdapter(const SpeciesThermo* input,bool =false) {
        for(int i=0;i<Count;++i) {
            const SpeciesThermo& s=input[i];auto& d=data[i];
            d.model=s.model;d.coefficientOffset=14*i;
            // Inactive historical CHMT tables can be entirely unconfigured.
            // Form invalid metadata without raising OpenFOAM's enabled FPE
            // traps; common validation still rejects it when it is active.
            const Real Ru=ugkwp::universalGasConstant<Real>();
            const Real minimumR=Ru/std::numeric_limits<Real>::max();



            d.molarMass=finite(s.R)&&s.R>minimumR ? Ru/s.R : Real(0);
            d.minTemperature=s.Tmin;d.midTemperature=s.Tmid;d.maxTemperature=s.Tmax;
            d.referencePressure=s.referencePressure;d.referenceTemperature=s.referenceTemperature;
            d.referenceEntropy=s.referenceEntropy;d.hasEntropyReference=s.hasEntropyReference;
            if(s.model==ugkwp::SpeciesThermoModel::NASA7)
                for(int j=0;j<14;++j)coefficients[14*i+j]=s.nasa7[j];
            else {coefficients[14*i]=s.cp0;coefficients[14*i+1]=s.cp1;coefficients[14*i+2]=s.e0;}
        }
    }
    CHMT_HD ugkwp::SpeciesThermoView<Real,Count> view(ugkp::diagnostics::DeviceContext* diagnostics=nullptr) const {
        ugkwp::SpeciesThermoView<Real,Count> result;
        result.species=data;result.coefficients=coefficients;result.coefficientCount=14*Count;result.diagnostics=diagnostics;
        return result;
    }
};



#define CHMT_SHARED_SPECIES_PROPERTY(Name) \
CHMT_HD inline Real Name(const SpeciesThermo& s,Real T) { \
    const GasThermoAdapter<1> thermo(&s);return ugkwp::Name(0,T,thermo.view()); \
}
CHMT_SHARED_SPECIES_PROPERTY(speciesCp)
CHMT_SHARED_SPECIES_PROPERTY(speciesCv)
CHMT_SHARED_SPECIES_PROPERTY(speciesE)
CHMT_SHARED_SPECIES_PROPERTY(speciesH)
#undef CHMT_SHARED_SPECIES_PROPERTY

// Three-point Gauss integration is exact for NASA7's quartic cv on one
// temperature branch and for LinearCp. Formation constants never enter.
template<bool Terminal=false>
CHMT_HD inline Real gasCapacitySecantBranch(const SpeciesThermo& s,Real a,Real b) {
    const Real midpoint=Real(.5)*(a+b),offset=Real(.5)*(b-a)*std::sqrt(Real(3)/5);
    const GasThermoAdapter<1> thermo(&s);
    return (Real(5)*ugkwp::speciesCv(0,midpoint-offset,thermo.view())+Real(8)*ugkwp::speciesCv(0,midpoint,thermo.view())
        +Real(5)*ugkwp::speciesCv(0,midpoint+offset,thermo.view()))/18;
}
template<bool Terminal=false>
CHMT_HD inline Real speciesEnergySecant(const SpeciesThermo& s,Real a,Real b) {
    if(a==b)return speciesCv(s,a);
    if((!finite(a)||!finite(b)||a<=0||b<=0))return undefinedValue();
    if(a>b){const Real swap=a;a=b;b=swap;}
    if(s.model==ugkwp::SpeciesThermoModel::NASA7&&a<=s.Tmid&&b>s.Tmid) {
        // Evaluate both one-sided values with the shared evaluator. Source
        // tables may have small jumps; integrating cv alone would refit them.
        GasThermoAdapter<1> upper(&s);
        for(int j=0;j<7;++j)upper.coefficients[j]=s.nasa7[j+7];
        const Real upperEnergy=ugkwp::speciesE(0,s.Tmid,upper.view());
        const Real jump=upperEnergy-speciesE(s,s.Tmid);
        return ((s.Tmid-a)*gasCapacitySecantBranch<Terminal>(s,a,s.Tmid)
            +(b-s.Tmid)*gasCapacitySecantBranch<Terminal>(s,s.Tmid,b)+jump)/(b-a);
    }
    return gasCapacitySecantBranch<Terminal>(s,a,b);
}
struct MaterialCaloric {
    Real a0=0,a1=0,a2=0,minimumTemperature=0,maximumTemperature=0;
    SpeciesThermo gas[Ns]{};Real pore[Ns]{};
    template<bool Terminal=false>
    CHMT_HD Real energy(Real T) const {
        Real result=caloricIntegral(a0,a1,a2,T);
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesE(gas[s],T);
        return result;
    }
    template<bool Terminal=false>
    CHMT_HD Real secant(Real a,Real b) const {
        Real result=a1+Real(.5)*a2*(a+b);
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesEnergySecant<Terminal>(gas[s],a,b);
        return result;
    }
    CHMT_HD Real capacity(Real T) const {
        Real result=a1+a2*T;
        for(int s=0;s<Ns;++s)if(pore[s]>0)result+=pore[s]*speciesCv(gas[s],T);
        return result;
    }
};

}}
namespace chmt {
using ConductionReal=GpuReal;
using MaterialConductionCaloric=materialConductionDetail::MaterialCaloric;
struct MaterialConductionInventory {
    ConductionReal condensed[Nc]{},pore[Ns]{};
    CHMT_HD MaterialConductionInventory& operator=(const SolidQ& q){
        for(int i=0;i<Nc;++i)condensed[i]=ConductionReal(q.condensed[i]);
        for(int i=0;i<Ns;++i)pore[i]=ConductionReal(q.pore[i]);return *this;
    }
};
// Keep the original material-caloric validation branches, while all
// coefficients and their accumulation are formed from FP32 inputs.
CHMT_HD inline MaterialCaloricCode makeMaterialConductionCaloric(const SolidQ& q,const PhysicsConfig& p,
    MaterialConductionCaloric& out,bool& numerical){
    numerical=true;MaterialConductionCaloric c;
    c.maximumTemperature=std::numeric_limits<ConductionReal>::max();ConductionReal mass=0;
    for(int i=0;i<Nc;++i){
        const PhysicalReal original=q.condensed[i];const auto& t=p.condensed[i];
        if(!finite(original)||original<0)return MaterialCaloricCode::CondensedInventory;
        if(original==0)continue;
        if(!validCondensed(t))return MaterialCaloricCode::CondensedTable;
        const ConductionReal m=ConductionReal(original);
        c.a0+=m*ConductionReal(t.e0);c.a1+=m*ConductionReal(t.cp0);c.a2+=m*ConductionReal(t.cp1);mass+=m;
        c.minimumTemperature=fmaxf(c.minimumTemperature,ConductionReal(t.Tmin));
        c.maximumTemperature=fminf(c.maximumTemperature,ConductionReal(t.Tmax));
    }
    for(int i=0;i<Ns;++i){
        const PhysicalReal original=q.pore[i];const auto& t=p.species[i];
        if(!finite(original)||original<0)return MaterialCaloricCode::PoreInventory;
        if(original==0)continue;
        if(!validSpecies(t)){
            numerical=t.model==ugkwp::SpeciesThermoModel::LinearCp||t.model==ugkwp::SpeciesThermoModel::NASA7;
            return MaterialCaloricCode::PoreTable;
        }
        const ConductionReal m=ConductionReal(original);c.gas[i]=t;c.pore[i]=m;mass+=m;
        c.minimumTemperature=fmaxf(c.minimumTemperature,ConductionReal(t.Tmin));
        c.maximumTemperature=fminf(c.maximumTemperature,ConductionReal(t.Tmax));
    }
    if(!(mass>0)||!std::isfinite(c.a0)||!std::isfinite(c.a1)||!std::isfinite(c.a2))return MaterialCaloricCode::State;
    out=c;return MaterialCaloricCode::Success;
}

}
#endif
