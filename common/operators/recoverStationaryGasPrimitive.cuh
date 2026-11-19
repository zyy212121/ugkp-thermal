#pragma once

template<class Energy,class GasState>
__device__ inline void recoverStationaryGasPrimitive(GasState& s,int cell,Energy density,
    Energy momentumX,Energy momentumY,Energy momentumZ,Energy totalEnergy)
{
    gasInventoryFloorWarning(s,cell,density,s.rhoMin,-1,2625,30,ugkp::diagnostics::Quantity::Density);
    const Energy boundedDensity=clampMin(density,Energy(s.rhoMin));
    const Energy ux=momentumX/boundedDensity;
    const Energy uy=momentumY/boundedDensity;
    const Energy uz=momentumZ/boundedDensity;
    if(!gasProducerScalarValid(s,cell,ux,30012)||!gasProducerScalarValid(s,cell,uy,30104)
        ||!gasProducerScalarValid(s,cell,uz,30105))return;
    const Energy kinetic=ugkwp::gasLegacyKineticEnergy(boundedDensity,ux,uy,uz);
    const Energy internal=totalEnergy-kinetic;
    if(!gasProducerScalarValid(s,cell,kinetic,30109)||!gasProducerScalarValid(s,cell,internal,30110))return;

    const Energy minimum=fmax(s.rhoMin,ugkwp::gasLegacyThermalMinimumInternalEnergy(
        boundedDensity,Energy(s.Rgas),Energy(s.TgasMin),Energy(s.gammaGas),Energy(OfSmall)));
    gasInventoryFloorWarning(s,cell,internal,minimum,-1,2626,30,ugkp::diagnostics::Quantity::InternalEnergy);
    const Energy boundedInternal=clampMin(internal,minimum);
    const Energy pressure=(s.gammaGas-Energy(1))*boundedInternal;
    const Energy temperature=pressure/(boundedDensity*s.Rgas);
    if(!gasProducerScalarValid(s,cell,pressure,30013)||!gasProducerScalarValid(s,cell,temperature,30111,true))return;

    s.rho[cell]=boundedDensity;
    if(density<s.rhoMin){
        s.rhoUx[cell]=boundedDensity*ux;
        s.rhoUy[cell]=boundedDensity*uy;
        s.rhoUz[cell]=boundedDensity*uz;
    }
    if(internal<minimum)s.rhoE[cell]=kinetic+boundedInternal;
    s.Ux[cell]=ux;
    s.Uy[cell]=uy;
    s.Uz[cell]=uz;
    s.p[cell]=pressure;
    s.Tgas[cell]=temperature;
}
