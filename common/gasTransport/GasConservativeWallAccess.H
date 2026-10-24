#pragma once
#include "GasStateView.H"
namespace ugkwp {
template <class S, class = void>
struct GasConservativeWallCapability : std::false_type {};
template <class S>
struct GasConservativeWallCapability<
    S,
    std::void_t<decltype(std::declval<S>().gasBoundaryLayerModel.conservative)>>
    : std::true_type {};
template <class S>
UGKWP_GAS_HD inline bool conservativeWallCellReady(const S &s, int c) {
  if constexpr (GasConservativeWallCapability<S>::value) {
    const auto &d = s.gasBoundaryLayerModel.conservative;
    if (d.enabled && d.cellSlot && d.cellSlot[c] >= 0)
      return d.active[d.cellSlot[c]] &&
             gasBoundaryLayerSlotReady(s, d.cellSlot[c]);
  }
  return false;
}
} // namespace ugkwp
