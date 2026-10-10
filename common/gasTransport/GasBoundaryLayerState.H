#pragma once
#include <type_traits>
#include <utility>
#include "GasModel.H"

namespace ugkwp
{
// Compact, application-owned stage outputs. Fluxes use the ordinary common
// gas convention: outward from the owner, integrated over the physical face.
// Species flux is species-major [species*count+slot], in kg/s. These are total
// physical-wall species fluxes, never matching-plane fluxes or reaction rates.
template<class Real> struct GasBoundaryLayerExchange
{
    Real mass=0, momentumX=0, momentumY=0, momentumZ=0, energy=0;
    Real k=0;
    // Thermodynamic/BVP pressure used in the published force/work. The face
    // operator replaces only this mechanical contribution with local pressure.
    Real matchingPressure=0;
    bool ready=false;
    bool massReady=false; // prescribed physical Js are available before thermal solve
};
template<class Real> struct GasBoundaryLayerSstClosure
{
    Real integratedKSource=0; // complete net k source, over actual owner volume
    Real ownerOmega=0;       // finite owner-distance algebraic constraint
    Real volume=0;           // geometry volume used for the source quadrature
};
template<class Real> struct GasBoundaryLayerState
{
    bool enabled=false;
    int count=0;
    bool preparedFirstStage=false; // host trial-phase cache, cleared on consume/rollback
    unsigned long long preparedGeneration=0;
    unsigned long long generation=0;
    double preparedTime=0,preparedInterval=0;
    const int* faceSlot=nullptr;  // nFaces; -1 means not a selected wall
    const int* ownerSlot=nullptr; // nCells; -1 means unconstrained; unique owner
    int* status=nullptr;         // count; zero success, nonzero rejects trial
    GasBoundaryLayerExchange<Real>* exchange=nullptr;
    GasBoundaryLayerSstClosure<Real>* sst=nullptr;
    Real* speciesFlux=nullptr;
    // No second chemical inventory exists here. The auxiliary reaction integral
    // stays with model diagnostics and is never sent to the bulk source update.
    // omega source/transport audit channels describe the replaced coarse-grid
    // discrete row. They are not integrals of the singular smooth-wall profile.
};
template<class State,class=void> struct GasBoundaryLayerCapability:std::false_type{};
template<class State> struct GasBoundaryLayerCapability<State,
    std::void_t<decltype(std::declval<State>().gasBoundaryLayer)>>:std::true_type{};
template<class State> UGKWP_GAS_HD inline bool gasBoundaryLayerEnabled(const State& s)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)return s.gasBoundaryLayer.enabled;
    return false;
}
template<class State> UGKWP_GAS_HD inline int gasBoundaryLayerFaceSlot(const State& s,int face)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)
        if(gasBoundaryLayerEnabled(s) && face>=0 && face<s.nFaces && s.gasBoundaryLayer.faceSlot)
            return s.gasBoundaryLayer.faceSlot[face];
    return -1;
}
template<class State> UGKWP_GAS_HD inline int gasBoundaryLayerOwnerSlot(const State& s,int cell)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)
        if(gasBoundaryLayerEnabled(s) && cell>=0 && cell<s.nCells && s.gasBoundaryLayer.ownerSlot)
            return s.gasBoundaryLayer.ownerSlot[cell];
    return -1;
}
template<class State> UGKWP_GAS_HD inline bool gasBoundaryLayerSlotReady(const State& s,int slot)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)
    {
        const auto&w=s.gasBoundaryLayer;
        return w.enabled && slot>=0 && slot<w.count && w.status && w.exchange && w.sst
            && w.status[slot]==0 && w.exchange[slot].ready;
    }
    return false;
}
}
