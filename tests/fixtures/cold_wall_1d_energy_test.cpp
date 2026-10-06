#include "GpuColdWallSolidification.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace Foam::gpuThermal;
using Nodes = std::array<GpuReal, coldWallAxialNodeCount>;

namespace
{
constexpr GpuReal volume = GPU_R(9.047786842338605e-13);
constexpr GpuReal mass = GPU_R(3200.0)*volume;
constexpr GpuReal maximumArea = GPU_R(2.2e-8);
constexpr GpuReal contactArea = GPU_R(1.7e-8);
constexpr GpuTime duration = 8.0e-6;
constexpr GpuReal peak = GPU_R(0.42);
constexpr GpuReal wallTemperature = GPU_R(486.0);
constexpr GpuReal efficiency = GPU_R(0.004);

ColdWallSolidificationParameters parameters(const int iterations = 12)
{
    return {GPU_R(2327), GPU_R(20), GPU_R(1.16e6), GPU_R(3990),
            GPU_R(1273), GPU_R(5.9), GPU_R(0.25), GPU_R(0), 0, iterations};
}

void require(const bool ok, const char* message, const double value = 0)
{
    if (!ok)
    {
        std::fprintf(stderr, "%s: %.17g\n", message, value);
        std::exit(1);
    }
}

struct Profile
{
    Nodes enthalpy{}, ring{};
    GpuTime age = duration;
    GpuReal frozen = 0;
};

Profile uniform(const GpuReal temperature, const ColdWallSolidificationParameters& p)
{
    Profile result;
    initialiseColdWallProfile(result.enthalpy.data(), result.ring.data(), temperature, p);
    return result;
}

ColdWallSolidificationStep advance(
    Profile& state, const ColdWallSolidificationParameters& p, const GpuTime dt,
    const GpuReal gasTemperature = GPU_R(1500), const GpuReal gasConductance = 0,
    const bool gasOnly = false, const GpuTime gasExchangeDuration = -1.0)
{
    return advanceColdWallProfile(
        state.enthalpy.data(), state.ring.data(), state.age, state.frozen, p,
        volume, mass, maximumArea, contactArea, gasOnly ? 1.0e6 : duration,
        peak, dt, wallTemperature, GPU_R(13000), efficiency,
        gasTemperature, gasConductance, gasExchangeDuration);
}

long double totalEnergy(const Profile& state)
{
    long double sum = 0;
    for (const auto h : state.enthalpy) sum += h;
    return sum*static_cast<long double>(mass)/8;
}

// Independent dense Gaussian elimination, not the production tridiagonal solver.
std::array<long double, 8> referenceBackwardEuler(
    const Profile& old, const ColdWallSolidificationParameters& p, const double dt)
{
    long double matrix[8][9]{};
    const long double dz = static_cast<long double>(volume)/contactArea/8;
    const long double k = p.solidThermalConductivityWmK;
    const long double face = k*contactArea/dz;
    const long double wall = efficiency*contactArea/(p.interfaceResistanceM2KW + dz/(2*k));
    const long double capacity = static_cast<long double>(mass)/8*p.solidSpecificHeatJkgK/dt;
    for (int n = 0; n < 8; ++n)
    {
        matrix[n][n] = capacity + (n > 0 ? face : 0) + (n < 7 ? face : 0);
        matrix[n][8] = capacity*old.enthalpy[n]/p.solidSpecificHeatJkgK;
        if (n > 0) matrix[n][n - 1] = -face;
        if (n < 7) matrix[n][n + 1] = -face;
    }
    matrix[0][0] += wall;
    matrix[0][8] += wall*wallTemperature;
    for (int pivot = 0; pivot < 8; ++pivot)
    {
        const long double scale = matrix[pivot][pivot];
        for (int col = pivot; col <= 8; ++col) matrix[pivot][col] /= scale;
        for (int row = 0; row < 8; ++row)
        {
            if (row == pivot) continue;
            const long double multiplier = matrix[row][pivot];
            for (int col = pivot; col <= 8; ++col)
                matrix[row][col] -= multiplier*matrix[pivot][col];
        }
    }
    std::array<long double, 8> result{};
    for (int n = 0; n < 8; ++n) result[n] = matrix[n][8];
    return result;
}

void constantCp()
{
    double maximumError = 0;
    for (const double dt : {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2})
    for (const int iterations : {1, 4, 12})
    {
        const auto p = parameters(iterations);
        auto state = uniform(GPU_R(2000), p);
        for (int n = 0; n < 8; ++n)
            state.enthalpy[n] = coldWallSpecificEnthalpyJkg(GPU_R(1800 + 30*n), p);
        const auto expected = referenceBackwardEuler(state, p, dt);
        const auto before = totalEnergy(state);
        const auto step = advance(state, p, dt);
        require(step.valid, "constant-cp step must be valid");
        for (int n = 0; n < 8; ++n)
        {
            const double error = std::fabs(
                coldWallTemperatureFromSpecificEnthalpyK(state.enthalpy[n], p) - expected[n]);
            maximumError = std::max(maximumError, error);
            require(error < (UGKWP_GPU_REAL_BITS == 32 ? 0.15 : 1.0e-7),
                    "constant-cp result differs from backward Euler", error);
        }
        const double balance = totalEnergy(state) - before + step.wallEnergyJ;
        require(std::fabs(balance) < (UGKWP_GPU_REAL_BITS == 32 ? 2e-9 : 2e-17),
                "wall-only energy is not conserved", balance);
    }
    std::printf("constant_cp max_temperature_error=%.12g K\n", maximumError);
}

void nonlinearResidual()
{
    double maximumResidual = 0;
    for (const bool phase : {false, true})
    {
        auto p = parameters();
        auto state = uniform(GPU_R(3200), p);
        for (int n = 0; n < 8; ++n)
        {
            const GpuReal temperature = phase ? GPU_R(2315 + 4*n) : GPU_R(3000 + 50*n);
            state.enthalpy[n] = coldWallSpecificEnthalpyJkg(temperature, p);
        }
        const auto old = state;
        const double dt = phase ? 1e-7 : 1e-5;
        const auto step = advance(state, p, dt);
        require(step.valid, "nonlinear step must be valid");
        double temperature[8], conductivity[8], power[8]{};
        for (int n = 0; n < 8; ++n)
        {
            temperature[n] = coldWallTemperatureFromSpecificEnthalpyK(state.enthalpy[n], p);
            conductivity[n] = coldWallThermalConductivity(state.enthalpy[n], p);
        }
        const double dz = static_cast<double>(volume)/contactArea/8;
        for (int n = 0; n < 7; ++n)
        {
            const double face = 2*conductivity[n]*conductivity[n+1]
                /(conductivity[n]+conductivity[n+1])*contactArea/dz;
            const double flux = face*(temperature[n]-temperature[n+1]);
            power[n] -= flux;
            power[n+1] += flux;
        }
        const double wall = efficiency*contactArea/(p.interfaceResistanceM2KW+dz/(2*conductivity[0]));
        power[0] -= wall*(temperature[0]-wallTemperature);
        for (int n = 0; n < 8; ++n)
        {
            const double residual = state.enthalpy[n]-old.enthalpy[n]-dt*power[n]/(mass/8);
            maximumResidual = std::max(maximumResidual, std::fabs(residual));
            require(std::fabs(residual) < (UGKWP_GPU_REAL_BITS == 32 ? 2.0 : 1.0e-5),
                    "nonlinear enthalpy equation residual", residual);
        }
    }
    std::printf("nonlinear max_specific_enthalpy_residual=%.12g J/kg\n", maximumResidual);
}

double referenceGasIncrement(
    const Profile& state, const ColdWallSolidificationParameters& p,
    const double gasTemperature, const double conductance, const double dt)
{
    long double meanH = 0;
    for (const auto h : state.enthalpy) meanH += h/8.0L;
    const double bulkTemperature = coldWallTemperatureFromSpecificEnthalpyK(GpuReal(meanH), p);
    const double differenceH = coldWallSpecificEnthalpyJkg(GpuReal(gasTemperature), p)-double(meanH);
    if (conductance == 0 || dt == 0 || differenceH == 0 || gasTemperature == bulkTemperature) return 0;
    const double secantCp = differenceH/(gasTemperature-bulkTemperature);
    return differenceH*(-std::expm1(-conductance*dt/(mass*secantCp)));
}

void uniformGas()
{
    double maximumError = 0;
    for (const auto temperatures : {std::array<double, 2>{1600, 1200},
                                   std::array<double, 2>{1600, 2100},
                                   std::array<double, 2>{2322, 2332}})
    for (const double stiffness : {1e-6, 0.1, 1.0, 10.0, 100.0})
    for (const int iterations : {1, 4, 12})
    {
        const auto p = parameters(iterations);
        const GpuReal initialTemperature = GpuReal(temperatures[0]);
        const GpuReal gasTemperature = GpuReal(temperatures[1]);
        const double cp = coldWallApparentSpecificHeat(initialTemperature, p);
        // Vary gas stiffness without also making the internal diffusion solve
        // arbitrarily ill-conditioned in FP32.
        const double dt = 1e-5;
        const GpuReal conductance = GpuReal(stiffness*mass*cp/dt);
        auto state = uniform(initialTemperature, p);
        state.age = 0;
        const double expected = gasTemperature+(initialTemperature-gasTemperature)*std::exp(-stiffness);
        const auto step = advance(state, p, dt, gasTemperature, conductance, true);
        require(step.valid, "uniform gas-only step must be valid");
        require(step.wallEnergyJ == 0, "gas-only fixture must have no wall flux");
        for (const auto h : state.enthalpy)
        {
            const double temperature = coldWallTemperatureFromSpecificEnthalpyK(h, p);
            const double error = std::fabs(temperature-expected);
            maximumError = std::max(maximumError, error);
            require(error < (UGKWP_GPU_REAL_BITS == 32 ? 0.15 : 1e-7),
                    "uniform gas source differs from whole-particle exponential", error);
            require(temperature >= std::min(double(initialTemperature),double(gasTemperature))-0.15
                    && temperature <= std::max(double(initialTemperature),double(gasTemperature))+0.15,
                    "uniform gas source overshot equilibrium", temperature);
        }
    }
    std::printf("uniform_gas max_temperature_error=%.12g K\n", maximumError);
}

void phaseGas()
{
    const auto p = parameters();
    for (const auto temperatures : {std::array<double, 2>{3000, 2000},
                                   std::array<double, 2>{2000, 3000}})
    {
        auto state = uniform(GpuReal(temperatures[0]), p);
        state.age = 0;
        const auto old = state;
        const GpuReal gasTemperature = GpuReal(temperatures[1]);
        const GpuReal conductance = GPU_R(0.3);
        const double dt = 1e-5;
        const double deltaH = referenceGasIncrement(old, p, gasTemperature, conductance, dt);
        const auto step = advance(state, p, dt, gasTemperature, conductance, true);
        require(step.valid, "phase-crossing gas step must be valid");
        for (int n = 0; n < 8; ++n)
        {
            const double error = state.enthalpy[n]-old.enthalpy[n]-deltaH;
            require(std::fabs(error) < (UGKWP_GPU_REAL_BITS == 32 ? 3 : 1e-6),
                    "phase-crossing gas source omitted latent enthalpy", error);
        }
    }
    // Nonuniform enthalpies: a single bulk source, not eight local gas sources.
    auto state = uniform(GPU_R(3000), p);
    for (int n = 0; n < 8; ++n)
        state.enthalpy[n] = coldWallSpecificEnthalpyJkg(GPU_R(2100 + 150*n), p);
    const auto old = state;
    const double dt = 1e-7;
    const double deltaH = referenceGasIncrement(old, p, 1500, 3e-4, dt);
    const auto step = advance(state, p, dt, GPU_R(1500), GPU_R(3e-4));
    require(step.valid, "nonuniform coupled step must be valid");
    const double balance = totalEnergy(state)-totalEnergy(old)+step.wallEnergyJ-mass*deltaH;
    require(std::fabs(balance) < (UGKWP_GPU_REAL_BITS == 32 ? 2e-9 : 2e-17),
            "whole-particle gas plus wall energy is not conserved", balance);
}

void zeroGas()
{
    const auto p = parameters();
    for (const double dt : {0.0, 1e-6})
    {
        auto state = uniform(GPU_R(2000), p);
        state.age = 0;
        const auto old = state;
        const auto step = advance(state, p, dt, GPU_R(1200), GPU_R(0), true);
        require(step.valid, "zero-source step must be valid");
        for (int n = 0; n < 8; ++n)
            require(std::fabs(state.enthalpy[n]-old.enthalpy[n]) < 1,
                    "disabled gas changed a uniform isolated profile");
    }
}

void localCoolingRollback()
{
    const auto p = parameters();
    auto state = uniform(GPU_R(5000), p);
    state.enthalpy[0] = coldWallSpecificEnthalpyJkg(GPU_R(1), p);
    state.age = 0;
    const auto old = state;
    const auto step = advance(state, p, 1e-8, GPU_R(300), GPU_R(1e9), true);
    require(!step.valid, "locally inadmissible uniform cooling must be rejected");
    require(state.enthalpy == old.enthalpy && state.ring == old.ring
            && state.age == old.age && state.frozen == old.frozen,
            "rejected gas cooling mutated profile");
}

void sourceLimits()
{
    const auto p = parameters();
    for (const GpuReal temperature : {GPU_R(300), GPU_R(2317), GPU_R(2327),
                                      GPU_R(2337), GPU_R(4000), GPU_R(5000)})
    {
        const GpuReal h = coldWallSpecificEnthalpyJkg(temperature, p);
        for (const GpuReal gas : {temperature, std::nextafter(temperature, GPU_R(0)),
                                  std::nextafter(temperature, GPU_REAL_MAX)})
        {
            GpuReal increment = GPU_R(-1);
            require(coldWallGasSpecificEnthalpyIncrement(h, mass, gas, GPU_R(3e-4),
                        1e-5, p, increment), "near-equilibrium source rejected");
            const double target = coldWallSpecificEnthalpyJkg(gas, p)-h;
            require(increment*target >= 0 && std::fabs(increment) <= std::fabs(target),
                    "near-equilibrium source crossed enthalpy target", increment);
        }
    }
    GpuReal increment = 1;
    require(!coldWallGasSpecificEnthalpyIncrement(GPU_R(2e6), mass,
                std::numeric_limits<GpuReal>::infinity(), GPU_R(1), 1e-5, p, increment),
            "infinite gas temperature accepted");
    require(!coldWallGasSpecificEnthalpyIncrement(GPU_R(2e6), mass, GPU_R(1200),
                std::numeric_limits<GpuReal>::quiet_NaN(), 1e-5, p, increment),
            "NaN gas conductance accepted");
}

void phaseRefinement()
{
    const auto p = parameters();
    const auto integrate = [&](const int count)
    {
        GpuReal h = coldWallSpecificEnthalpyJkg(GPU_R(3000), p);
        for (int step = 0; step < count; ++step)
        {
            GpuReal increment = 0;
            require(coldWallGasSpecificEnthalpyIncrement(h, GPU_R(3.2e-9), GPU_R(2000),
                        GPU_R(3e-4), 0.05/count, p, increment), "refinement source rejected");
            h += increment;
        }
        return double(h);
    };
    // This is a convergence check for the explicitly documented first-order
    // nonlinear integration, not a claim of exact exponential phase kinetics.
    const double reference = integrate(4096);
    const double coarse = std::fabs(integrate(64)-reference);
    const double medium = std::fabs(integrate(128)-reference);
    const double fine = std::fabs(integrate(256)-reference);
    require(medium < 0.65*coarse && fine < 0.65*medium,
            "phase integration failed timestep refinement", fine);
    std::printf("phase_refinement enthalpy_errors=%.12g,%.12g,%.12g J/kg\n", coarse, medium, fine);
}

void separateGasDuration()
{
    const auto p = parameters();
    auto state = uniform(GPU_R(2000), p);
    const auto old = state;
    const GpuTime wallDt = 1e-6, gasDt = 4e-6;
    const GpuReal gas = GPU_R(1500), conductance = GPU_R(3e-4);
    const double deltaH = referenceGasIncrement(old, p, gas, conductance, gasDt);
    auto storageTarget = old;
    for (auto& h : storageTarget.enthalpy) h += GpuReal(deltaH);
    const auto expected = referenceBackwardEuler(storageTarget, p, wallDt);
    const auto step = advance(state, p, wallDt, gas, conductance, false, gasDt);
    require(step.valid, "separate gas/contact duration must be valid");
    for (int n = 0; n < 8; ++n)
    {
        const double error = coldWallTemperatureFromSpecificEnthalpyK(state.enthalpy[n], p)-expected[n];
        require(std::fabs(error) < (UGKWP_GPU_REAL_BITS == 32 ? 0.005 : 1e-7),
                "gas-then-contact implicit storage target differs from dense BE", error);
    }
    require(state.age == old.age+wallDt, "profile age must use only active contact duration");
    const long double dz = static_cast<long double>(volume)/contactArea/8;
    const long double wall = efficiency*contactArea/(p.interfaceResistanceM2KW
        + dz/(2*p.solidThermalConductivityWmK));
    const double expectedWallEnergy = double(wall*(expected[0]-wallTemperature)*wallDt);
    require(std::fabs(step.wallEnergyJ-expectedWallEnergy)
                < (UGKWP_GPU_REAL_BITS == 32 ? 1e-11 : 1e-18),
            "wall energy must use active duration and the same implicit temperature");
    const double balance = totalEnergy(state)-totalEnergy(old)+step.wallEnergyJ-mass*deltaH;
    require(std::fabs(balance) < (UGKWP_GPU_REAL_BITS == 32 ? 2e-9 : 2e-17),
            "separate gas/contact duration energy closure", balance);
    auto gasOff = old, legacyGasOff = old;
    const auto off = advance(gasOff, p, wallDt, gas, GPU_R(0), false, gasDt);
    const auto legacy = advance(legacyGasOff, p, wallDt, gas, GPU_R(0));
    require(off.valid && legacy.valid && gasOff.enthalpy == legacyGasOff.enthalpy
            && off.wallEnergyJ == legacy.wallEnergyJ,
            "gas-off independent duration changed the wall-only solution");
    for (const GpuTime badDuration : {-2.0, std::numeric_limits<GpuTime>::quiet_NaN(),
                                     std::numeric_limits<GpuTime>::infinity()})
    {
        auto rejected = old;
        const auto bad = advance(rejected, p, wallDt, gas, conductance, false, badDuration);
        require(!bad.valid && rejected.enthalpy == old.enthalpy && rejected.ring == old.ring
                && rejected.age == old.age && rejected.frozen == old.frozen,
                "invalid explicit gas duration must reject without mutation");
    }
}

void invalidRollback()
{
    const auto p = parameters();
    for (const int invalid : {0, 1, 2, 3, 4, 5, 6})
    {
        auto state = uniform(GPU_R(2000), p);
        if (invalid == 3) state.enthalpy[0] = GPU_R(-1);
        if (invalid == 4) state.ring[0] = GPU_R(-1);
        if (invalid == 5) state.age = -1;
        if (invalid == 6) state.frozen = GPU_R(-1);
        const auto old = state;
        auto bad = p;
        if (invalid == 0) bad.nonlinearIterations = 0;
        const GpuTime dt = invalid == 1 ? -1e-6 : 1e-6;
        const GpuReal conductance = invalid == 2 ? GPU_R(-1) : GPU_R(0);
        const auto step = advance(state, bad, dt, GPU_R(1500), conductance);
        require(!step.valid, "invalid input was accepted");
        require(state.enthalpy == old.enthalpy && state.ring == old.ring
                && state.age == old.age && state.frozen == old.frozen,
                "rejected step mutated profile");
    }
}
}

int main(int argc, char** argv)
{
    require(argc == 2, "expected test mode");
    if (std::strcmp(argv[1], "constant_cp") == 0) constantCp();
    else if (std::strcmp(argv[1], "nonlinear") == 0) nonlinearResidual();
    else if (std::strcmp(argv[1], "rollback") == 0) invalidRollback();
    else if (std::strcmp(argv[1], "gas_uniform") == 0) uniformGas();
    else if (std::strcmp(argv[1], "gas_phase") == 0) phaseGas();
    else if (std::strcmp(argv[1], "gas_zero") == 0) zeroGas();
    else if (std::strcmp(argv[1], "gas_rollback") == 0) localCoolingRollback();
    else if (std::strcmp(argv[1], "source_limits") == 0) sourceLimits();
    else if (std::strcmp(argv[1], "phase_refinement") == 0) phaseRefinement();
    else if (std::strcmp(argv[1], "separate_gas_duration") == 0) separateGasDuration();
    else require(false, "unknown test mode");
}
