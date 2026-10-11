#pragma once
#include <type_traits>
#include <utility>
#include "GasModel.H"
namespace ugkwp
{
// Optional pending diagnostic storage. It becomes accepted only when the host
// requested-interval transaction commits. All sums use binary64, independently
// of conservative storage precision. Channels: mass, physical total energy,
// then each species. Positive boundary transport points into the domain.
struct GasBudgetAuditState
{
    bool enabled=false;
    int speciesCount=0;
    double *transport=nullptr,*initialTransport=nullptr,*chemistrySpecies=nullptr;
};
template<class State,class=void>struct GasBudgetAuditCapability:std::false_type{};
template<class State>struct GasBudgetAuditCapability<State,std::void_t<decltype(std::declval<State>().gasBudgetAudit)>>:std::true_type{};
template<class State>UGKWP_GAS_HD inline bool gasBudgetEnabled(const State&s)
{if constexpr(GasBudgetAuditCapability<State>::value)return s.gasBudgetAudit.enabled;return false;}
template<class State>UGKWP_GAS_HD inline bool gasBudgetExternalFace(const State&s,int face)
{return s.faceNeighbour[face]<0 && (!s.facePeriodicPair || s.facePeriodicPair[face]<0);}
template<class State,class Time,class Value>UGKWP_GAS_HD inline void gasBudgetFlux
(const State&s,int cell,int channel,Time dt,Value inwardRate)
{
    if constexpr(GasBudgetAuditCapability<State>::value)
        if(s.gasBudgetAudit.enabled)
            s.gasBudgetAudit.transport[channel*s.nCells+cell]+=double(dt)*double(inwardRate);
}
template<class State>UGKWP_GAS_HD inline void saveGasBudgetAudit(const State&s,int cell)
{
    if constexpr(GasBudgetAuditCapability<State>::value)
        if(s.gasBudgetAudit.enabled)for(int k=0;k<s.gasBudgetAudit.speciesCount+2;++k)
            s.gasBudgetAudit.initialTransport[k*s.nCells+cell]=s.gasBudgetAudit.transport[k*s.nCells+cell];
}
template<class State,class Weight>UGKWP_GAS_HD inline void blendGasBudgetAudit
(const State&s,int cell,Weight initialWeight,Weight stageWeight)
{
    if constexpr(GasBudgetAuditCapability<State>::value)
        if(s.gasBudgetAudit.enabled)for(int k=0;k<s.gasBudgetAudit.speciesCount+2;++k)
        {const int i=k*s.nCells+cell;s.gasBudgetAudit.transport[i]=double(initialWeight)*s.gasBudgetAudit.initialTransport[i]+double(stageWeight)*s.gasBudgetAudit.transport[i];}
}
template<class State,class Real,int Ns>UGKWP_GAS_HD inline void gasBudgetChemistry
(const State&s,int cell,const Real(&publishedDensity)[Ns],Real volume)
{
    if constexpr(GasBudgetAuditCapability<State>::value)
        if(s.gasBudgetAudit.enabled)for(int k=0;k<Ns;++k)
            s.gasBudgetAudit.chemistrySpecies[k*s.nCells+cell]+=
                (double(publishedDensity[k])-double(s.gasSpecies.rho[k*s.nCells+cell]))*double(volume);
}
}
