#pragma once
#include "../WallModel.H"
namespace ugkwp
{
namespace gaswall
{
namespace physical
{
template <class R, int N> struct Face
{
    R mass = 0, momentum[3]{}, energy = 0, species[N]{}, k = 0, omega = 0, normalStress = 0,
      normalVelocity = 0;
};
template <class R, int N> struct Primitive
{
    R rho = 0, T = 0, U[3]{}, Y[N]{}, k = 0, omega = 0;
    R gradU[3][3]{}, gradT[3]{}, gradY[N][3]{}, gradK[3]{}, gradOmega[3]{};
};
template <class R, int N> struct Source
{
    R species[N]{}, k = 0, omega = 0, nut = 0, f1 = 0, f2 = 0, pressure = 0,
                    divergence = 0;
};
// Continuum physical constitutive evaluation on an arbitrary non-equilibrium
// reconstruction. No constant-pressure BVP assumptions and no flux-derived source.
template <class R, int N, class T>
UGKWP_GAS_HD inline bool
evaluatePrimitive(const ugkwp::gaswall::WallInput<R, N, T> &in,
                  const ugkwp::gaswall::WallModelConfig<R> &cfg, const Primitive<R, N> &p,
                  R wallDistance, const R *normal, R meshNormal, Face<R, N> &f, Source<R, N> &src)
{
    using namespace ugkwp;
    namespace d = ugkwp::gaswall::detail;
    f = {};
    src = {};
    R cp = 0, gasR = 0, h[N], normalU = 0, speed2 = 0, div = 0, s2 = 0, cross = 0;
    for (int j = 0; j < N; ++j)
    {
        R c;
        speciesCpH(j, p.T, in.model.thermo, c, h[j]);
        cp += p.Y[j] * c;
        gasR += p.Y[j] * universalGasConstant<R>() / in.model.thermo.species[j].molarMass;
    }
    for (int a = 0; a < 3; ++a)
    {
        normalU += p.U[a] * normal[a];
        speed2 += p.U[a] * p.U[a];
        div += p.gradU[a][a];
        s2 += R(2) * p.gradU[a][a] * p.gradU[a][a];
        cross += p.gradK[a] * p.gradOmega[a];
        for (int b = a + 1; b < 3; ++b)
            s2 += (p.gradU[a][b] + p.gradU[b][a]) * (p.gradU[a][b] + p.gradU[b][a]);
    }
    R nut = 0;
    const auto &co = in.model.sstCoefficients;
    if (cfg.enableSst && wallDistance > 0)
    {
        R cd = sstCrossDiffusion(p.omega, cross, co);
        src.f1 = sstF1(p.k, p.omega, in.model.viscosity / p.rho, wallDistance, cd, co);
        src.f2 = sstF2(p.k, p.omega, in.model.viscosity / p.rho, wallDistance, co);
        nut = sstNut(p.k, p.omega, s2, src.f2, co);
        R g = s2 - R(2) / R(3) * div * div;
        src.k = sstKSource(p.rho, p.k, p.omega, div, nut, g, co);
        src.omega = sstOmegaSource(p.rho, p.k, p.omega, div, g, s2, src.f1, src.f2, cd, co);
    }
    src.nut = nut;
    src.divergence = div;
    src.pressure = p.rho * gasR * p.T;
    R mu = in.model.viscosity + p.rho * nut, traction[3]{}, gradTn = 0, gradKn = 0, gradOn = 0,
      gradYn[N];
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
            traction[a] += mu *
                           (p.gradU[a][b] + p.gradU[b][a] - (a == b ? R(2) / R(3) * div : R(0))) *
                           normal[b];
        gradTn += p.gradT[a] * normal[a];
        gradKn += p.gradK[a] * normal[a];
        gradOn += p.gradOmega[a] * normal[a];
    }
    for (int j = 0; j < N; ++j)
    {
        gradYn[j] = 0;
        for (int a = 0; a < 3; ++a)
            gradYn[j] += p.gradY[j][a] * normal[a];
    }
    R diffusion[N], diffH = 0;
    bool available = false;
    if (!correctedSpeciesDiffusion(p.Y, gradYn, p.rho, in.model.diffusivity, nut,
                                   cfg.turbulentSchmidt, p.T, in.model.thermo, diffusion, diffH,
                                   true, &available) ||
        !available)
        return false;
    R conductivity =
        (in.model.molecularPrandtl > 0 ? in.model.viscosity * cp / in.model.molecularPrandtl
                                       : in.model.conductivity) +
        p.rho * nut * cp / cfg.turbulentPrandtl;
    f.mass = p.rho * (normalU - meshNormal);
    f.normalVelocity = normalU;
    R work = 0;
    for (int a = 0; a < 3; ++a)
    {
        f.momentum[a] = f.mass * p.U[a] + src.pressure * normal[a] - traction[a];
        work += traction[a] * p.U[a];
        f.normalStress += traction[a] * normal[a];
    }
    f.energy = f.mass * (mixtureEnergy(p.Y, p.T, in.model.thermo) + speed2 / R(2)) +
               src.pressure * normalU - work - conductivity * gradTn + diffH;
    for (int j = 0; j < N; ++j)
        f.species[j] = f.mass * p.Y[j] + diffusion[j];
    f.k = f.mass * p.k - (in.model.viscosity + p.rho * sstAlphaK(src.f1, co) * nut) * gradKn;
    f.omega =
        f.mass * p.omega - (in.model.viscosity + p.rho * sstAlphaOmega(src.f1, co) * nut) * gradOn;
    d::LayerPoint<R, N> rp;
    rp.rho = p.rho;
    rp.T = p.T;
    for (int j = 0; j < N; ++j)
        rp.Y[j] = p.Y[j];
    if (!d::pointReaction(in, rp))
        return false;
    for (int j = 0; j < N; ++j)
        src.species[j] = rp.source[j];
    return true;
}
} // namespace physical

} // namespace gaswall
} // namespace ugkwp
