#include "../legacy/common/gasNumerics/LegacySstWallFunction.H"
#include "CompressibleWallFunction.H"
#include <algorithm>
#include <cmath>
#include <iostream>

bool verifyCompressibleWallFunction()
{
    using R = GpuReal;
    using Wall = CompressibleWallFunction;
    constexpr R tolerance = R(32)*GPU_REAL_EPSILON;
    Wall::Input input{0, R(2000), R(4), R(9.3e-5), R(.4), R(1722), R(.85),
        R(.41), R(9.8), R(360), R(2800), 0, true};
    input.thermalYPlus = ugkpwall::jayatillekeThermalYPlus
        (input.viscosity*input.cp/(input.conductivity*input.turbulentPrandtl),
            input.kappa, input.E);

    auto result = Wall::evaluate(input, 64, tolerance);
    if (!result.thermal.valid || result.thermal.conductivity != input.conductivity
        || result.momentum.uTau != 0 || result.recoveryRise != 0
        || result.thermal.heatFlux != input.conductivity*input.inverseDistance
            *(input.cellTemperature-input.wallTemperature))
        return false;

    input.slipSpeed = R(20);
    input.inverseDistance = R(1e14);
    result = Wall::evaluate(input, 64, tolerance);
    const R shear = input.referenceDensity*result.momentum.uTau*result.momentum.uTau;
    const R molecularRecovery = shear*input.slipSpeed
        /(R(2)*input.conductivity*input.inverseDistance);
    const R molecularShear = input.viscosity*input.slipSpeed*input.inverseDistance;
    if (!result.thermal.valid || result.thermal.conductivity != input.conductivity
        || std::abs(shear/molecularShear-R(1)) > R(4)*tolerance
        || std::abs(result.recoveryRise/molecularRecovery-R(1)) > R(1e-6))
        return false;

    input.slipSpeed = R(1600);
    input.inverseDistance = R(1000);
    input.wallTemperature = input.cellTemperature;
    result = Wall::evaluate(input, 64, tolerance);
    if (!result.thermal.valid || result.recoveryRise <= 0 || result.thermal.heatFlux <= 0
        || std::abs(result.residual) > R(4)*tolerance)
        return false;

    input.prescribedWallTemperature = false;
    const auto adiabatic = Wall::evaluate(input, 64, tolerance);
    if (!adiabatic.thermal.valid || adiabatic.thermal.heatFlux != 0
        || adiabatic.wallTemperature <= input.cellTemperature)
        return false;

    input.prescribedWallTemperature = true;
    input.wallTemperature = adiabatic.wallTemperature;
    input.referenceDensity *= input.cellTemperature/input.wallTemperature;
    const auto prescribed = Wall::evaluate(input, 64, tolerance);
    return prescribed.thermal.valid
        && std::abs(prescribed.momentum.uTau/adiabatic.momentum.uTau-R(1)) <= R(8)*tolerance
        && std::abs(prescribed.thermal.conductivity/adiabatic.thermal.conductivity-R(1))
            <= R(8)*tolerance
        && std::abs(prescribed.thermal.heatFlux)/(prescribed.thermal.conductivity
            *input.inverseDistance*adiabatic.recoveryRise) <= R(8)*tolerance;
}

bool verifyMolecularConductivityFloor()
{
    using R = GpuReal;
    using Wall = CompressibleWallFunction;
    constexpr R tolerance = R(32)*GPU_REAL_EPSILON;
    constexpr R mu = R(2e-5), cp = R(1000), h = R(3.88516e-5);
    constexpr R pr = R(.7), prt = R(1.5), kappa = R(.41), E = R(9.8);
    const R conductivity = mu*cp/pr;
    Wall::Input input{R(100), R(1)/h, R(1), mu, conductivity, cp, prt,
        kappa, E, R(300), R(3000), ugkpwall::jayatillekeThermalYPlus(pr/prt,kappa,E), true};
    const auto counterexample = Wall::evaluate(input, 64, tolerance);
    if (!counterexample.thermal.valid || counterexample.thermal.conductivity < conductivity
        || counterexample.thermal.heatFlux < conductivity*input.inverseDistance
            *(input.cellTemperature-input.wallTemperature))
        return false;

    for (R molecularPrandtl : {R(.4), R(.7), R(1), R(2)})
    {
        input.conductivity = mu*cp/molecularPrandtl;
        for (R turbulentPrandtl : {R(.5), R(.85), R(1.5)})
        {
            input.turbulentPrandtl = turbulentPrandtl;
            input.thermalYPlus = ugkpwall::jayatillekeThermalYPlus
                (molecularPrandtl/turbulentPrandtl, kappa, E);
            for (R reynolds : {R(1), R(150), R(200), R(1000), R(20000)})
            {
                input.inverseDistance = input.referenceDensity*input.slipSpeed/(mu*reynolds);
                for (R temperature : {R(150), R(300), R(600), R(3000)})
                {
                    input.cellTemperature = temperature;
                    const auto state = Wall::evaluate(input, 64, tolerance);
                    if (!state.thermal.valid || state.thermal.conductivity < input.conductivity
                        || std::abs(state.residual) > R(8)*tolerance)
                        return false;
                }
                auto adiabaticInput = input;
                adiabaticInput.slipSpeed = R(1800);
                adiabaticInput.inverseDistance *= R(18);
                adiabaticInput.prescribedWallTemperature = false;
                const auto adiabatic = Wall::evaluate(adiabaticInput, 64, tolerance);
                if (!adiabatic.thermal.valid || adiabatic.thermal.conductivity < input.conductivity
                    || adiabatic.thermal.heatFlux != R(0))
                    return false;
                adiabaticInput.prescribedWallTemperature = true;
                adiabaticInput.wallTemperature = adiabatic.wallTemperature;
                adiabaticInput.referenceDensity *= adiabaticInput.cellTemperature/adiabatic.wallTemperature;
                const auto prescribed = Wall::evaluate(adiabaticInput, 64, tolerance);
                if (!prescribed.thermal.valid
                    || std::abs(prescribed.momentum.uTau/adiabatic.momentum.uTau-R(1)) > R(32)*tolerance
                    || std::abs(prescribed.thermal.conductivity/adiabatic.thermal.conductivity-R(1)) > R(32)*tolerance
                    || std::abs(prescribed.thermal.heatFlux) >
                        R(32)*tolerance*prescribed.thermal.conductivity
                        *adiabaticInput.inverseDistance*adiabaticInput.cellTemperature)
                    return false;
            }
        }
    }
    return true;
}

bool verifyJayatillekeLimit()
{
    using R = GpuReal;
    constexpr R mu=R(2e-5), cp=R(1000), rho=R(1), U=R(.001), T=R(300);
    constexpr R kappa=R(.41), E=R(9.8), tolerance=R(32)*GPU_REAL_EPSILON;
    for (R pr : {R(.4), R(.7), R(1.5)})
        for (R prt : {R(.6), R(.85), R(1.5)})
        {
            const R molecular = mu*cp/pr;
            const R P = ugkpwall::jayatillekeSmoothP(pr/prt);
            const R yt = ugkpwall::jayatillekeThermalYPlus(pr/prt,kappa,E);
            for (R re : {R(1), R(60), R(140), R(300), R(500), R(1000), R(4000), R(30000)})
            {
                const R h = re*mu/(rho*U);
                const auto momentum = ugkpwall::spaldingWallStateFromNormalGradient
                    (U,h,mu/rho,U/h,kappa,E);
                const R reference = ugkpwall::sstJayatillekeThermalTransport
                    (molecular,pr,prt,kappa,E,P,yt,momentum.yPlus,R(0)).conductivity;
                const auto state = CompressibleWallFunction::evaluate(
                    {U,R(1)/h,rho,mu,molecular,cp,prt,kappa,E,T,T,yt,true},64,tolerance);
                const R limitTolerance = R(8)*tolerance+R(2)*U*U/(cp*T);
                if (!state.thermal.valid || state.thermal.conductivity < molecular
                    || std::abs(state.thermal.conductivity/reference-R(1)) > limitTolerance
                    || std::abs(state.momentum.uTau/momentum.uTau-R(1)) > limitTolerance)
                    return false;
            }
        }
    return true;
}

bool verifyThermalTransition()
{
    using R = GpuReal;
    using Wall = CompressibleWallFunction;
    const R pr = R(.7);
    const R viscosity = R(2e-5);
    const R cp = R(1000);
    Wall::Input input{R(100), R(41868.78125), R(1), viscosity, viscosity*cp/pr, cp,
        R(.9), R(.41), R(9.8), R(300), R(300), R(0), true};
    input.thermalYPlus = ugkpwall::jayatillekeThermalYPlus(pr/input.turbulentPrandtl, input.kappa, input.E);
    for (R scale : {R(.999), R(1), R(1.001)})
    {
        const R centre = R(41868.78125)*scale;
        for (R inverseDistance : {std::nextafter(centre, R(0)), centre,
            std::nextafter(centre, R(1e6))})
        {
            input.inverseDistance = inverseDistance;
            const R tolerance = R(32)*GPU_REAL_EPSILON;
            const auto state = Wall::evaluate(input, 64, tolerance);
            if (!state.thermal.valid || std::abs(state.residual) > R(2)*tolerance
                || state.thermal.conductivity < input.conductivity)
                return false;
        }
    }
    return true;
}

int main()
{
    if (!verifyCompressibleWallFunction()) return 3;
    if (!verifyMolecularConductivityFloor()) return 4;
    if (!verifyJayatillekeLimit()) return 5;
    if (!verifyThermalTransition()) return 6;
    constexpr GpuReal mu=GpuReal(9.29152148664344e-5), cp=GpuReal(1722);
    constexpr GpuReal prt=GpuReal(.85), kappa=GpuReal(.41), E=GpuReal(9.8);
    int count=0;
    double worst=0;
    for (GpuReal Pr : {GpuReal(.1), GpuReal(.4), GpuReal(.71), GpuReal(2)})
    {
        const auto P=ugkpwall::jayatillekeSmoothP(Pr/prt);
        const auto transition=ugkpwall::jayatillekeThermalYPlus(Pr/prt,kappa,E);
        const GpuReal molecular=mu*cp/Pr;
        for (GpuReal yPlus : {GpuReal(0), transition*GpuReal(.5), transition,
                             transition*GpuReal(1.1), GpuReal(8150), GpuReal(15815)})
        {
            double multiplier=1;
            if (yPlus > transition)
            {
                const double TPlus=double(prt)*(std::log(double(E)*double(yPlus))/double(kappa)+double(P));
                multiplier=std::max(1.,double(Pr)*double(yPlus)/TPlus);
            }
            const double expectedConductivity=double(molecular)*multiplier;
            for (GpuReal gradient : {GpuReal(-2700), GpuReal(-1e-8), GpuReal(0), GpuReal(1e-8), GpuReal(2700)})
            {
                const auto actual=ugkpwall::sstJayatillekeThermalTransport
                    (molecular,Pr,prt,kappa,E,P,transition,yPlus,gradient);
                if (!actual.valid || (gradient == 0 && actual.heatFlux != 0)) return 1;
                const double expectedHeat=-expectedConductivity*double(gradient);
                const double error=std::abs(actual.heatFlux-expectedHeat)/std::max(1.,std::abs(expectedHeat));
                worst=std::max(worst,error);
                if (error > 5e-6 || std::abs(actual.conductivity/expectedConductivity-1) > 3e-6) return 2;
                ++count;
            }
        }
    }
    std::cout<<"PASS "<<count<<" thermal cases, worst relative error "<<worst<<"\n";
}
