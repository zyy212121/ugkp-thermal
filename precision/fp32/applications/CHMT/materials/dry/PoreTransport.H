#ifndef CHMT_DRY_PORE_TRANSPORT_H
#define CHMT_DRY_PORE_TRANSPORT_H
#include "Types.H"

namespace chmt { namespace dry {
// Intrinsic gas density and mass fractions at either side of a material face.
// Fluxes below are per TOTAL face area, not per pore cross-sectional area.
template<std::size_t Ns>
struct PoreFaceState {
    std::array<PhysicalReal,Ns> massFraction{};
    PhysicalReal density=0;ThermalReal temperature=0;
};

template<std::size_t Ns>
struct PoreFlux {
    std::array<PhysicalReal,Ns> advection{};
    std::array<PhysicalReal,Ns> diffusion{};
    std::array<PhysicalReal,Ns> species{};       // kg/(m2 s), positive left -> right
    std::array<ThermalReal,Ns> speciesEnergy{}; // W/m2, h-weighted transport per species
    ThermalReal conductionEnergy=0;
    ThermalReal energy=0;                      // W/m2, including species enthalpy
    PhysicalReal darcyVelocity=0;               // m/s superficial, NOT pore speed
    // Largest diagonal loss coefficient of the frozen diffusion operator,
    // kg/(m2 s) per donor mass fraction. This remains nonzero at uniform Y.
    PhysicalReal diffusionLossLeft=0, diffusionLossRight=0;
};

// Darcy volume flux on a face, g projected along the same left->right normal.
// Permeability and viscosity are configured material closures, not inferred
// universally from porosity. For zero g this is pressure-driven Darcy flow.
CHMT_DRY_HD inline PhysicalReal darcyVelocity(PhysicalReal permeability, PhysicalReal viscosity,
    PhysicalReal pressureGradient, PhysicalReal density, PhysicalReal gravityNormal=0)
{
    return -permeability/viscosity*(pressureGradient-density*gravityNormal);
}

// Minimal mixture-averaged Fick transport with mass-average correction:
// J_i*=-rho D_i,eff grad(Y_i); J_i=J_i* - Y_i sum(J_j*).
// D_eff already includes porosity/tortuosity. Do NOT multiply by porosity
// again; do NOT add a DustyGas pressure term here (Darcy is already separate).
// No Soret, barodiffusion, Knudsen, or secondary gas kinetics is implied.
// Cf. PATO volume-averaged species transport and Cantera mass-averaged fluxes.
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
    // Discretize the common mass-correction velocity with its UPWIND
    // composition, not a centered Y. C=-sum(J*) is the correction mass flux.
    // This preserves sum(J)=0 and cannot export an absent species from its
    // zero-inventory side. Centered correction can do so for unequal D and
    // three or more species, making any positive explicit step impossible.
    // This is one fixed conservative scheme (no clipping/fallback); its
    // unequal-D correction is generally first-order in space.
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

// Same computed flux applied with opposite signs; neither side recomputes
// donor states. Conserves each species and thermal energy to roundoff.
// For gravity-enabled material energy include rho*u_D dot g as a volume
// power source, or consistently store/transport potential energy instead.
// Boundary recession sweep is a separate INTERNAL-energy channel, not this
// pressure-work-bearing Darcy/relative-diffusion enthalpy channel.
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
