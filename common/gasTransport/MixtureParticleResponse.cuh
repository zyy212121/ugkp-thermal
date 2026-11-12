#pragma once
#include "../GpuPrecisionTypes.H"
#include "../gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "../gasNumerics/GpuDragModels.cuh"

// Source-epoch values only: rebuild after gas or parcel state publication.
// The two precision trees share this code; transport algebra remains GpuReal,
// while time, exponential response fractions and exchanged energy stay double.
namespace mixtureParticles {
struct DragCellProperties {
    GpuReal rho,ux,uy,uz,viscosity;
};
struct HeatCellProperties {
    DragCellProperties gas;
    GpuReal conductivity,prandtlOneThird;
};
template<class State>
__device__ DragCellProperties dragCellProperties(const State& s,int c) {
    return {s.rho[c],s.Ux[c],s.Uy[c],s.Uz[c],
        ugkwp::gasCellMolecularViscosityAt(s,c,s.Tgas[c])};
}
template<class State>
__device__ HeatCellProperties heatCellProperties(const State& s,int c) {
    return {dragCellProperties(s,c),molecularGasConductivity(s,c,s.Tgas[c]),
        ugkwp::gasCellPrandtlOneThird(s,c)};
}
template<class DragModel,class State>
__device__ GpuReal beta(const State& s,int i,GpuReal epsG,const DragCellProperties& gas) {
    const GpuReal dx=gas.ux-s.pux[i],dy=gas.uy-s.puy[i],dz=gas.uz-s.puz[i];
    const ugkwpGpuDrag::DragInput input={gas.rho,gas.viscosity,epsG,s.rhoSolid,
        s.pd[i],sqrt(dx*dx+dy*dy+dz*dz),GPU_TINY(1e-300),
        GpuReal(s.dragParameter0),GpuReal(s.dragParameter1),
        GpuReal(s.dragParameter2),GpuReal(s.dragParameter3)};
    return fmax(GpuReal(0),DragModel::inverseResponseTime(input));
}
// Retain the scalar property-evaluation entry for reference callers.
template<class DragModel,class State>
__device__ GpuReal beta(const State& s,int c,int i,GpuReal epsG) {
    return beta<DragModel>(s,i,epsG,dragCellProperties(s,c));
}
template<class State>
__device__ double heatFraction(const State& s,int i,double dt,const HeatCellProperties& properties) {
    const auto& gas=properties.gas;
    const GpuReal dx=gas.ux-s.pux[i],dy=gas.uy-s.puy[i],dz=gas.uz-s.puz[i];
    const GpuReal re=gas.rho*s.pd[i]*sqrt(dx*dx+dy*dy+dz*dz)/fmax(gas.viscosity,GpuReal(1e-30));
    const GpuReal nu=ugkwp::ranzMarshallNuFromPrOneThird(fmax(GpuReal(0),re),properties.prandtlOneThird);
    const GpuReal rate=GpuReal(6)*nu*properties.conductivity
        /(s.particleThermalCapacity*s.pd[i]*s.pd[i]+GPU_TINY(1e-300));
    return -expm1(-fmax(0.,rate)*dt);
}
template<class State>
__device__ double heatFraction(const State& s,int c,int i,double dt) {
    return heatFraction(s,i,dt,heatCellProperties(s,c));
}
// Same operand precision and clamp semantics as particleGasHeatRelaxationRate.
// The caller retains the caloric path's existing double Reynolds evaluation,
// then converts at the same rate-function boundary as before.
__device__ inline GpuReal caloricHeatRate(const HeatCellProperties& properties,
    GpuReal reynolds,GpuReal diameter,GpuReal volumetricHeatCapacity) {
    const GpuReal nu=ugkwp::ranzMarshallNuFromPrOneThird(
        reynolds<GpuReal(0)?GpuReal(0):reynolds,properties.prandtlOneThird);
    return GpuReal(6)*nu*properties.conductivity
        /(volumetricHeatCapacity*diameter*diameter+GPU_TINY(1e-300));
}
}
