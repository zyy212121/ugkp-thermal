#pragma once
#include <type_traits>
#include <utility>
#include "GasModel.H"
#include "../gasWall/WallModelTypes.H"

namespace ugkwp
{




template<class Real> struct GasBoundaryLayerExchange
{
    Real mass=0, momentumX=0, momentumY=0, momentumZ=0, energy=0;
    Real k=0;


    Real matchingPressure=0;
    bool ready=false;
};
template<class Real> struct GasBoundaryLayerSstClosure
{
    Real wallShearStress=0;
    Real ownerOmega=0;
    Real volume=0;
    Real ownerDistance=0;


    Real nonDestructiveKSource=0;
    Real normal[3]{};
    const Real* kFluxHeight=nullptr;
    const Real* kTransportFlux=nullptr;
    const Real* omegaDiffusiveFlux=nullptr;
    int kFluxCount=0;
    Real kDestructionRate=0;
    Real normalMassFlux=0;
};



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



    bool useMacroSstInventory=false;
    int count=0;
    bool preparedFirstStage=false;
    unsigned long long preparedGeneration=0;
    unsigned long long generation=0;
    double preparedTime=0,preparedInterval=0;
    const int* faceSlot=nullptr;
    const int* ownerSlot=nullptr;
    int* status=nullptr;
    GasBoundaryLayerExchange<Real>* exchange=nullptr;
    GasBoundaryLayerSstClosure<Real>* sst=nullptr;
    Real* speciesFlux=nullptr;




};
template<class State,class=void> struct GasBoundaryLayerCapability:std::false_type{};
template<class State> struct GasBoundaryLayerCapability<State,
    std::void_t<decltype(std::declval<State>().gasBoundaryLayer)>>:std::true_type{};
template<class State> UGKWP_GAS_HD inline bool gasBoundaryLayerEnabled(const State& s)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)return s.gasBoundaryLayer.enabled;
    return false;
}
template<class State> UGKWP_GAS_HD inline bool gasBoundaryLayerUsesMacroSst(const State& s)
{
    if constexpr(GasBoundaryLayerCapability<State>::value)
        return s.gasBoundaryLayer.enabled && s.gasBoundaryLayer.useMacroSstInventory;
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
template<class State,class=void> struct GasWallKModelCapability:std::false_type{};
template<class State> struct GasWallKModelCapability<State,
    std::void_t<decltype(std::declval<State>().gasBoundaryLayerModel.config.model)>>:std::true_type{};






template<class State,class R> UGKWP_GAS_HD inline R gasBoundaryLayerWallKFlux(
 const State& state,int slot,R outwardMass,R ownerK){
    if constexpr(GasWallKModelCapability<State>::value){
        if(state.gasBoundaryLayerModel.config.model==gaswall::BoundaryLayerModel::EquilibriumOde)
            return outwardMass>R(0)?outwardMass*ownerK:R(0);
    }
    return state.gasBoundaryLayer.exchange[slot].k;
}



template<class Wall,class Model> inline void configureEquilibriumOdeMacroSst(Wall& wall,Model& model){
    if(model.config.model!=gaswall::BoundaryLayerModel::EquilibriumOde)return;
    wall.useMacroSstInventory=true;
    model.volumeSstClosure=false;
    model.config.enableSst=false;
}

}
