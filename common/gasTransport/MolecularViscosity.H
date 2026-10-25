#pragma once
#include "MixtureThermo.H"
namespace ugkwp {
enum class MolecularViscosityModel : unsigned char { Constant, SutherlandWilke };
template<class R> struct SutherlandSpecies {
    R muRef=0,TRef=0,S=0,molarMass=0; // Pa s, K, K, kg/mol
};
// Immutable, caller-owned tables. Upload once, then evaluate at local T,Y.
template<class R> struct MolecularViscosityView {
    MolecularViscosityModel model=MolecularViscosityModel::Constant;
    int count=0;
    const SutherlandSpecies<R>* species=nullptr;
    const R* massRatioQuarter=nullptr; // (Mj/Mi)^1/4, row-major
    const R* wilkeDenominator=nullptr; // sqrt(8*(1+Mi/Mj)), row-major
};
template<class R,class T> UGKWP_GAS_HD inline R sutherlandViscosity(
    R temperature,const SutherlandSpecies<T>& sp) noexcept {
    if(!gasFinite(temperature)||!(temperature>0))return gasThermoDetail::invalidValue<R>();
    const R ratio=temperature/R(sp.TRef);
    return R(sp.muRef)*ratio*::sqrt(ratio)*(R(sp.TRef)+R(sp.S))/(temperature+R(sp.S));
}
// mass can be Y, partial density or mass: Wilke's normalization cancels.
// Skip exact-zero components; a pure endpoint returns its pure viscosity.
template<int N,class R,class T> UGKWP_GAS_HD inline R molecularViscosity(
    const MolecularViscosityView<T>& model,R constant,R temperature,const R* mass) noexcept {
    if(model.model==MolecularViscosityModel::Constant)return constant;
    const R invalid=gasThermoDetail::invalidValue<R>();
    if(model.model!=MolecularViscosityModel::SutherlandWilke||!model.species)return invalid;
    if(model.count==1)return sutherlandViscosity(temperature,model.species[0]);
    if(model.count!=N||!mass||!model.massRatioQuarter||!model.wilkeDenominator)return invalid;
    R amount[N]{},mu[N]{},root[N]{};int active=0,last=-1;
    for(int i=0;i<N;++i){
        if(!gasFinite(mass[i])||mass[i]<0)return invalid;
        if(mass[i]==0)continue;
        amount[i]=mass[i]/R(model.species[i].molarMass);
        mu[i]=sutherlandViscosity(temperature,model.species[i]);
        if(!gasFinite(amount[i])||!gasFinite(mu[i])||!(mu[i]>0))return invalid;
        root[i]=::sqrt(mu[i]);++active;last=i;
    }
    if(!active)return invalid;
    if(active==1)return mu[last];
    R result=0;
    for(int i=0;i<N;++i)if(amount[i]>0){
        R denominator=amount[i];
        for(int j=0;j<N;++j)if(i!=j&&amount[j]>0){
            const R a=R(1)+(root[i]/root[j])*R(model.massRatioQuarter[i*N+j]);
            denominator+=amount[j]*a*a/R(model.wilkeDenominator[i*N+j]);
        }
        result+=amount[i]*mu[i]/denominator;
    }
    return result;
}
} // namespace ugkwp
