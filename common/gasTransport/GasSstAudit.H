#pragma once
#include <type_traits>
#include <utility>
#include "GasModel.H"
namespace ugkwp
{
// Optional trial-owned SST accounting, in integrated cell inventories, not
// densities. Positive transport is inward. Sources retain their signed net
// value; constraints include suppressed equations, floors and wall projection.
// Both totals and initial copies follow the same SSPRK algebra as the state.
template<class Real> struct GasSstAuditState
{
    bool enabled=false;
    Real *transportK=nullptr,*transportOmega=nullptr,*sourceK=nullptr,*sourceOmega=nullptr;
    Real *constraintK=nullptr,*constraintOmega=nullptr;
    Real *initialTransportK=nullptr,*initialTransportOmega=nullptr,*initialSourceK=nullptr,*initialSourceOmega=nullptr;
    Real *initialConstraintK=nullptr,*initialConstraintOmega=nullptr;
    Real* volume=nullptr; // explicit current stage inventory volume per cell
};
template<class State,class=void>struct GasSstAuditCapability:std::false_type{};
template<class State>struct GasSstAuditCapability<State,std::void_t<decltype(std::declval<State>().gasSstAudit)>>:std::true_type{};
template<class State>UGKWP_GAS_HD inline bool gasSstAuditEnabled(const State&s)
{
    if constexpr(GasSstAuditCapability<State>::value)return s.gasSstAudit.enabled;
    return false;
}
// Host adapters may keep the enabled view only in their owned device storage.
// In that case preserve the existing kernel's device-side enabled check.
// Native hosts exposing the view can skip a known-disabled audit launch.
template<class State> inline bool gasSstAuditPreparationRequired(const State& s)
{
    if constexpr(GasSstAuditCapability<State>::value)return s.gasSstAudit.enabled;
    return true;
}
template<class State>UGKWP_GAS_HD inline bool gasSstAuditComplete(const State&s)
{
    if constexpr(GasSstAuditCapability<State>::value)
    {
        const auto&a=s.gasSstAudit;
        return a.transportK&&a.transportOmega&&a.sourceK&&a.sourceOmega&&a.constraintK&&a.constraintOmega
            &&a.initialTransportK&&a.initialTransportOmega&&a.initialSourceK&&a.initialSourceOmega
            &&a.initialConstraintK&&a.initialConstraintOmega&&a.volume;
    }
    return false;
}
template<class State,class Real>UGKWP_GAS_HD inline void gasSstAuditConstraint(const State&s,int c,Real beforeK,Real beforeOmega)
{
    if constexpr(GasSstAuditCapability<State>::value)
        if(gasSstAuditEnabled(s)&&gasSstAuditComplete(s))
        {
            auto&a=s.gasSstAudit;
            a.constraintK[c]+=(s.rhoK[c]-beforeK)*a.volume[c];
            a.constraintOmega[c]+=(s.rhoOmega[c]-beforeOmega)*a.volume[c];
        }
}
template<class State,class Real,class Time>UGKWP_GAS_HD inline void gasSstAuditEuler
(const State&s,int c,Real beforeK,Real beforeOmega,Real oldVolume,Real newVolume,Time dt,Real fluxK,Real fluxOmega,Real sourceK,Real sourceOmega)
{
    if constexpr(GasSstAuditCapability<State>::value)
        if(gasSstAuditEnabled(s)&&gasSstAuditComplete(s))
        {
            auto&a=s.gasSstAudit;
            const Real transportK=dt*fluxK,transportOmega=dt*fluxOmega;
            const Real srcK=dt*oldVolume*sourceK,srcOmega=dt*oldVolume*sourceOmega;
            a.transportK[c]+=transportK;a.transportOmega[c]+=transportOmega;
            a.sourceK[c]+=srcK;a.sourceOmega[c]+=srcOmega;
            a.constraintK[c]+=s.rhoK[c]*newVolume-beforeK*oldVolume-transportK-srcK;
            a.constraintOmega[c]+=s.rhoOmega[c]*newVolume-beforeOmega*oldVolume-transportOmega-srcOmega;
            a.volume[c]=newVolume;
        }
}
template<class State>UGKWP_GAS_HD inline void saveGasSstAudit(const State&s,int c)
{
    if constexpr(GasSstAuditCapability<State>::value)
        if(gasSstAuditEnabled(s)&&gasSstAuditComplete(s))
        {
            auto&a=s.gasSstAudit;
#define UGKWP_SST_SAVE(field,initial) a.initial[c]=a.field[c];
            UGKWP_SST_SAVE(transportK,initialTransportK) UGKWP_SST_SAVE(transportOmega,initialTransportOmega)
            UGKWP_SST_SAVE(sourceK,initialSourceK) UGKWP_SST_SAVE(sourceOmega,initialSourceOmega)
            UGKWP_SST_SAVE(constraintK,initialConstraintK) UGKWP_SST_SAVE(constraintOmega,initialConstraintOmega)
#undef UGKWP_SST_SAVE
        }
}
template<class State,class Real>UGKWP_GAS_HD inline void blendGasSstAudit(const State&s,int c,Real initialWeight,Real stageWeight)
{
    if constexpr(GasSstAuditCapability<State>::value)
        if(gasSstAuditEnabled(s)&&gasSstAuditComplete(s))
        {
            auto&a=s.gasSstAudit;
#define UGKWP_SST_BLEND(field,initial) a.field[c]=initialWeight*a.initial[c]+stageWeight*a.field[c];
            UGKWP_SST_BLEND(transportK,initialTransportK) UGKWP_SST_BLEND(transportOmega,initialTransportOmega)
            UGKWP_SST_BLEND(sourceK,initialSourceK) UGKWP_SST_BLEND(sourceOmega,initialSourceOmega)
            UGKWP_SST_BLEND(constraintK,initialConstraintK) UGKWP_SST_BLEND(constraintOmega,initialConstraintOmega)
#undef UGKWP_SST_BLEND
        }
}
}
