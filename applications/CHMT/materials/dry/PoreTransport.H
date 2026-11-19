#ifndef CHMT_DRY_PORE_TRANSPORT_H
#define CHMT_DRY_PORE_TRANSPORT_H
#include "Types.H"

namespace chmt { namespace dry {


template<std::size_t Ns>
struct PoreFaceState {
    std::array<double,Ns> massFraction{};
    double density=0, temperature=0;
};

template<std::size_t Ns>
struct PoreFlux {
    std::array<double,Ns> advection{};
    std::array<double,Ns> diffusion{};
    std::array<double,Ns> species{};
    std::array<double,Ns> speciesEnergy{};
    double conductionEnergy=0;
    double energy=0;
    double darcyVelocity=0;


    double diffusionLossLeft=0, diffusionLossRight=0;
};




CHMT_DRY_HD inline double darcyVelocity(double permeability, double viscosity,
    double pressureGradient, double density, double gravityNormal=0)
{
    return -permeability/viscosity*(pressureGradient-density*gravityNormal);
}







template<std::size_t Ns>
CHMT_DRY_HD std::array<double,Ns> correctedDiffusion(double density,
    const std::array<double,Ns>& massFraction,
    const std::array<double,Ns>& gradient,
    const std::array<double,Ns>& effectiveDiffusivity)
{
    std::array<double,Ns> flux{};
    double sum=0;
    for(std::size_t s=0;s<Ns;++s) {
        flux[s]=-density*effectiveDiffusivity[s]*gradient[s];sum+=flux[s];
    }
    for(std::size_t s=0;s<Ns;++s)flux[s]-=massFraction[s]*sum;
    return flux;
}

template<std::size_t Ns, class GasThermo>
CHMT_DRY_HD PoreFlux<Ns> poreFaceFlux(const PoreFaceState<Ns>& left,
    const PoreFaceState<Ns>& right, double distance,
    double superficialVelocity, const std::array<double,Ns>& effectiveDiffusivity,
    double effectiveConductivity, const GasThermo& gas)
{
    PoreFlux<Ns> out;
    out.darcyVelocity=superficialVelocity;
    std::array<double,Ns> gradient{};
    const double faceDensity=0.5*(left.density+right.density);
    double rawSum=0, maximumDiffusivity=0;
    for(std::size_t s=0;s<Ns;++s) {
        if(effectiveDiffusivity[s]>maximumDiffusivity)maximumDiffusivity=effectiveDiffusivity[s];
        gradient[s]=(right.massFraction[s]-left.massFraction[s])/distance;
        rawSum-=faceDensity*effectiveDiffusivity[s]*gradient[s];
    }







    const double diagonal=faceDensity*maximumDiffusivity/distance;
    out.diffusionLossLeft=diagonal+(rawSum<0?-rawSum:0);
    out.diffusionLossRight=diagonal+(rawSum>0?rawSum:0);
    const auto& correctionDonor=rawSum<=0 ? left : right;
    out.diffusion=correctedDiffusion(faceDensity,
        correctionDonor.massFraction,gradient,effectiveDiffusivity);
    const auto& donor=superficialVelocity>=0 ? left : right;
    const double faceTemperature=0.5*(left.temperature+right.temperature);
    out.conductionEnergy=-effectiveConductivity*(right.temperature-left.temperature)/distance;
    out.energy=out.conductionEnergy;
    for(std::size_t s=0;s<Ns;++s) {
        out.advection[s]=superficialVelocity*donor.density*donor.massFraction[s];
        out.species[s]=out.advection[s]+out.diffusion[s];
        out.speciesEnergy[s]=out.advection[s]*gas.enthalpy(s,donor.temperature)
            +out.diffusion[s]*gas.enthalpy(s,faceTemperature);
        out.energy+=out.speciesEnergy[s];
    }
    return out;
}







template<std::size_t Nc, std::size_t Ns>
CHMT_DRY_HD void exchangePoreFlux(MaterialState<Nc,Ns>& left, MaterialState<Nc,Ns>& right,
    const PoreFlux<Ns>& flux, double area, double dt)
{
    const double weight=area*dt;
    for(std::size_t s=0;s<Ns;++s) {
        const double mass=weight*flux.species[s];
        left.poreMass[s]-=mass;right.poreMass[s]+=mass;
    }
    left.energy-=weight*flux.energy;right.energy+=weight*flux.energy;
}
} }
#endif
