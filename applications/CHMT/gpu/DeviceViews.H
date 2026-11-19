#ifndef CHMT_GPU_DEVICEVIEWS_H
#define CHMT_GPU_DEVICEVIEWS_H
#include "core/Configuration.H"
namespace chmt {

struct NormalThermalView {
    int nFaces=0,nLayers=0;
    const int* offsets=nullptr;
    const Real* coordinate=nullptr;const Real* thickness=nullptr;
    const Real* area=nullptr;
    const Real* pressure=nullptr;const Real* pressureRate=nullptr;






    const Real* mass=nullptr;const Real* oldVolume=nullptr;const Real* newVolume=nullptr;
    Real* enthalpy=nullptr;Real* rhs=nullptr;Real* temperature=nullptr;
    Real* liquidFraction=nullptr;
    Real* bottomHeatRate=nullptr;Real* topHeatRate=nullptr;
    Real* acceptedWallHeat=nullptr;


    const Real* bottomTemperature=nullptr;const Real* topTemperature=nullptr;
    const Real* bottomHeatFlux=nullptr;const Real* topHeatFlux=nullptr;



};
}
#endif
