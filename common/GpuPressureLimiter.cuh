#pragma once
// Identical pressure positivity and face-limiting equations for every application.
__device__ PressureReal pressureKickInternalEnergy
(
    const PressureReal rhoP,
    const PressureReal px,
    const PressureReal py,
    const PressureReal pz,
    const PressureReal energy
)
{
    return energy - PressureReal(0.5)*sqr3(px, py, pz)/clampMin(rhoP, OfVSmall);
}

__global__ void scaleCollisionalPressureFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }
    const int own = s.faceOwner[f];
    if(!finiteDevice(s.solidPressurePhiMomX[f])||!finiteDevice(s.solidPressurePhiMomY[f])
       ||!finiteDevice(s.solidPressurePhiMomZ[f])||!finiteDevice(s.solidPressurePhiEnergy[f]))
        recordPressureFailure(s,pressureBadFace,own>=0?own:0);
    const int nei = s.faceNeighbour[f];
    PressureReal scale = own >= 0 && own < s.nCells ? s.pressureKickScale[own] : PressureReal(0.0);
    if (nei >= 0 && nei < s.nCells)
    {
        scale = fmin(scale, s.pressureKickScale[nei]);
    }
    scale = clampRange(finiteOr(scale, PressureReal(0.0)), PressureReal(0.0), PressureReal(1.0));
    s.solidPressurePhiMomX[f] *= scale;
    s.solidPressurePhiMomY[f] *= scale;
    s.solidPressurePhiMomZ[f] *= scale;
    s.solidPressurePhiEnergy[f] *= scale;
}
