#ifndef CHMT_DRY_PORE_TRANSPORT_H
#define CHMT_DRY_PORE_TRANSPORT_H
#include "Types.H"

namespace chmt { namespace dry {
// Intrinsic gas density and mass fractions at either side of a material face.
// Fluxes below are per TOTAL face area, not per pore cross-sectional area.
template<std::size_t Ns>
struct PoreFaceState {
    std::array<double,Ns> massFraction{};
    double density=0, temperature=0;
};

template<std::size_t Ns>
struct PoreFlux {
    std::array<double,Ns> advection{};
    std::array<double,Ns> diffusion{};
    std::array<double,Ns> species{};       // kg/(m2 s), positive left -> right
    std::array<double,Ns> speciesEnergy{}; // W/m2, h-weighted transport per species
    double conductionEnergy=0;
    double energy=0;                      // W/m2, including species enthalpy
    double darcyVelocity=0;               // m/s superficial, NOT pore speed
};

// Darcy volume flux on a face, g projected along the same left->right normal.
// Permeability and viscosity are configured material closures, not inferred
// universally from porosity. For zero g this is pressure-driven Darcy flow.
inline double darcyVelocity(double permeability, double viscosity,
    double pressureGradient, double density, double gravityNormal=0)
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
std::array<double,Ns> correctedDiffusion(double density,
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
PoreFlux<Ns> poreFaceFlux(const PoreFaceState<Ns>& left,
    const PoreFaceState<Ns>& right, double distance,
    double superficialVelocity, const std::array<double,Ns>& effectiveDiffusivity,
    double effectiveConductivity, const GasThermo& gas)
{
    PoreFlux<Ns> out;
    out.darcyVelocity=superficialVelocity;
    std::array<double,Ns> faceY{},gradient{};
    for(std::size_t s=0;s<Ns;++s) {
        faceY[s]=0.5*(left.massFraction[s]+right.massFraction[s]);
        gradient[s]=(right.massFraction[s]-left.massFraction[s])/distance;
    }
    out.diffusion=correctedDiffusion(0.5*(left.density+right.density),
        faceY,gradient,effectiveDiffusivity);
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

// Same computed flux applied with opposite signs; neither side recomputes
// donor states. Conserves each species and thermal energy to roundoff.
// For gravity-enabled material energy include rho*u_D dot g as a volume
// power source, or consistently store/transport potential energy instead.
// Boundary recession sweep is a separate INTERNAL-energy channel, not this
// pressure-work-bearing Darcy/relative-diffusion enthalpy channel.
template<std::size_t Nc, std::size_t Ns>
void exchangePoreFlux(MaterialState<Nc,Ns>& left, MaterialState<Nc,Ns>& right,
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
