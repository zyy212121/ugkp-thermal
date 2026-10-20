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
    Real wallShearStress=0;  // magnitude of numerical-wall traction, Pa
    Real ownerOmega=0;       // finite destruction-equivalent closure (point constraint in other adapters)
    Real volume=0;           // stage owner volume, m^3
    Real ownerDistance=0;    // physical wall to owner centre, m
    // Volume-projected SST closure: source evaluated from the same profile,
    // destruction linearised in the sole macro rho*k inventory.
    Real nonDestructiveKSource=0;
    Real normal[3]{};
    const Real* kFluxHeight=nullptr;
    const Real* kTransportFlux=nullptr;
    const Real* omegaDiffusiveFlux=nullptr;
    int kFluxCount=0;
    Real normalMassFlux=0; // prescribed relative mass through the wall, positive into gas
};
// k uses total normal transport (m*k - Dk'); omega retains diffusion.
// Both use the same converged BVP control faces. The macro k operator removes
// only the prescribed normal-mass part of its upwind advection.
template<class R> UGKWP_GAS_HD inline R boundaryLayerSstFlux(
 const GasBoundaryLayerSstClosure<R>& closure,R y,bool omega=false){
    const R* flux=omega?closure.omegaDiffusiveFlux:closure.kTransportFlux;
    int l=0;while(l+1<closure.kFluxCount&&closure.kFluxHeight[l+1]<y)++l;
    if(l+1==closure.kFluxCount)return flux[l];
    const R t=(y-closure.kFluxHeight[l])/(closure.kFluxHeight[l+1]-closure.kFluxHeight[l]);
    return flux[l]+t*(flux[l+1]-flux[l]);
}
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
    int* status=nullptr;
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
template<class State,class=void> struct GasSuppliedWallInputCapability:std::false_type{};
template<class State> struct GasSuppliedWallInputCapability<State,
    std::void_t<decltype(std::declval<State>().gasBoundaryLayerModel.useSuppliedWallState),
        decltype(std::declval<State>().gasBoundaryLayerModel.input)>>:std::true_type{};
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
            && w.exchange[slot].ready;
    }
    return false;
}
}
