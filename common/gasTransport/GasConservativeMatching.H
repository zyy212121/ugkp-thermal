#pragma once
#include "../gasWall/WallModelTypes.H"

namespace ugkwp {



template<class R>
UGKWP_GAS_HD inline R physicalFaceTemperature(R centre, R adjacent, R centreWeight)
{
    return centreWeight*centre+(R(1)-centreWeight)*adjacent;
}







template<class State, class T, class R, int N>
UGKWP_GAS_HD inline void sampleConservativeWallThermocomposition(
    const State& state, int face, T physicalTemperature,
    gaswall::WallMatchingState<R,N>& matching)
{
    const int owner=state.faceOwner[face], neighbour=state.faceNeighbour[face];
    const R rawWeight=R(state.faceWeight[face]);
    const R weight=rawWeight<R(0)?R(0):(rawWeight>R(1)?R(1):rawWeight);
    matching.temperature=R(physicalTemperature);
    for(int species=0;species<N;++species)
        matching.massFraction[species]=
            weight*state.gasSpecies.rho[species*state.nCells+owner]/state.rho[owner]
            +(R(1)-weight)*state.gasSpecies.rho[species*state.nCells+neighbour]/state.rho[neighbour];
}
}
