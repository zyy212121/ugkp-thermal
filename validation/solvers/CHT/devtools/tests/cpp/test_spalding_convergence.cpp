#include "OpenFoamWallFunctions.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>

int main()
{
    long double worst = 0;
    int count = 0;
    for (int exponent = -6; exponent <= 12; ++exponent)
    {
        for (double factor : {1.0, 2.5, 7.5})
        {
            const GpuReal velocity = 100;
            const GpuReal distance = 0.001;
            const GpuReal viscosity = velocity*distance/(factor*std::pow(10.0, exponent));
            const auto state = ugkpwall::spaldingWallState(velocity, distance, viscosity);
            const long double uPlus = (long double)velocity/state.uTau;
            const long double z = (long double)GpuReal(0.41)*uPlus;
            const long double rhs = uPlus + (std::expm1(z)-z-z*z/2-z*z*z/6)/GpuReal(9.8);
            const long double yPlus = (long double)distance*state.uTau/viscosity;
            const long double residual = std::abs(rhs-yPlus)/std::max(rhs,yPlus);
            worst = std::max(worst,residual);
            if (!std::isfinite(state.uTau) || !std::isfinite(state.nut)
                || state.uTau <= 0 || state.nut < 0 || !(residual <= 0.0101L))
            {
                std::cerr << "FAIL Re=" << factor*std::pow(10.0,exponent)
                          << " residual=" << residual << '\n';
                return 1;
            }
            ++count;
        }
    }
    const auto rest = ugkpwall::spaldingWallState(0,0.001,1e-5);
    if (rest.uTau != 0 || rest.yPlus != 0 || rest.nut != 0) return 2;
    std::cout << "PASS samples=" << count << " worst_equation_residual=" << worst << '\n';
}
