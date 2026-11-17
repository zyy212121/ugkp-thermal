#ifndef CHMT_DRY_PORE_TRANSPORT_H
#define CHMT_DRY_PORE_TRANSPORT_H
#include "Types.H"

namespace chmt { namespace dry {


template<std::size_t Ns>
struct PoreFaceState {
    std::array<PhysicalReal,Ns> massFraction{};
    PhysicalReal density=0;ThermalReal temperature=0;
};

template<std::size_t Ns>
struct PoreFlux {
    std::array<PhysicalReal,Ns> advection{};
    std::array<PhysicalReal,Ns> diffusion{};
    std::array<PhysicalReal,Ns> species{};
    std::array<ThermalReal,Ns> speciesEnergy{};
    ThermalReal conductionEnergy=0;
    ThermalReal energy=0;
    PhysicalReal darcyVelocity=0;


    PhysicalReal diffusionLossLeft=0, diffusionLossRight=0;
};




CHMT_DRY_HD inline PhysicalReal darcyVelocity(PhysicalReal permeability, PhysicalReal viscosity,
    PhysicalReal pressureGradient, PhysicalReal density, PhysicalReal gravityNormal=0)
{
    return -permeability/viscosity*(pressureGradient-density*gravityNormal);
}







template<std::size_t Ns>
CHMT_DRY_HD std::array<PhysicalReal,Ns> correctedDiffusion(PhysicalReal density,
    const std::array<PhysicalReal,Ns>& massFraction,
    const std::array<PhysicalReal,Ns>& gradient,
    const std::array<PhysicalReal,Ns>& effectiveDiffusivity)
{
    std::array<PhysicalReal,Ns> flux{};
    PhysicalReal sum=0;
    for(std::size_t s=0;s<Ns;++s) {
        flux[s]=-density*effectiveDiffusivity[s]*gradient[s];sum+=flux[s];
    }
    for(std::size_t s=0;s<Ns;++s)flux[s]-=massFraction[s]*sum;
    return flux;
}

template<std::size_t Ns, class GasThermo>
CHMT_DRY_HD PoreFlux<Ns> poreFaceFlux(const PoreFaceState<Ns>& left,
    const PoreFaceState<Ns>& right, PhysicalReal distance,
    PhysicalReal superficialVelocity, const std::array<PhysicalReal,Ns>& effectiveDiffusivity,
    PhysicalReal effectiveConductivity, const GasThermo& gas)
{
    PoreFlux<Ns> out;
    out.darcyVelocity=superficialVelocity;
    std::array<PhysicalReal,Ns> gradient{};
    const PhysicalReal faceDensity=0.5f*(left.density+right.density);
    PhysicalReal rawSum=0, maximumDiffusivity=0;
    for(std::size_t s=0;s<Ns;++s) {
        if(effectiveDiffusivity[s]>maximumDiffusivity)maximumDiffusivity=effectiveDiffusivity[s];
        gradient[s]=(right.massFraction[s]-left.massFraction[s])/distance;
        rawSum-=faceDensity*effectiveDiffusivity[s]*gradient[s];
    }







    const PhysicalReal diagonal=faceDensity*maximumDiffusivity/distance;
    out.diffusionLossLeft=diagonal+(rawSum<0?-rawSum:0);
    out.diffusionLossRight=diagonal+(rawSum>0?rawSum:0);
    const auto& correctionDonor=rawSum<=0 ? left : right;
    out.diffusion=correctedDiffusion(faceDensity,
        correctionDonor.massFraction,gradient,effectiveDiffusivity);
    const auto& donor=superficialVelocity>=0 ? left : right;
    const ThermalReal faceTemperature=ThermalReal(.5)*(left.temperature+right.temperature);
    out.conductionEnergy=-ThermalReal(effectiveConductivity)*(right.temperature-left.temperature)/ThermalReal(distance);
    out.energy=out.conductionEnergy;
    for(std::size_t s=0;s<Ns;++s) {
        out.advection[s]=superficialVelocity*donor.density*donor.massFraction[s];
        out.species[s]=out.advection[s]+out.diffusion[s];
        out.speciesEnergy[s]=ThermalReal(out.advection[s])*ThermalReal(gas.enthalpy(s,donor.temperature))
            +ThermalReal(out.diffusion[s])*ThermalReal(gas.enthalpy(s,faceTemperature));
        out.energy+=out.speciesEnergy[s];
    }
    return out;
}







template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD void exchangePoreFlux(MaterialState<Nc,Ns>& left, MaterialState<Nc,Ns>& right,
    const PoreFlux<Ns>& flux, PhysicalReal area, PhysicalReal dt)
{
    const PhysicalReal weight=area*dt;
    for(std::size_t s=0;s<Ns;++s) {
        const PhysicalReal mass=weight*flux.species[s];
        left.poreMass[s]-=mass;right.poreMass[s]+=mass;
    }
    const ThermalReal energy=ThermalReal(area)*ThermalReal(dt)*flux.energy;
    left.energy-=energy;right.energy+=energy;
}
} }
#endif
