#include "ThermalCouplingSchedule.H"

#include <cstdlib>
#include <iostream>

int main()
{
    const double roundedElapsed = 1.0999999999999961 - 1.0;
    if (!ugkpcht::thermalEventIsDue(roundedElapsed, 0.1))
    {
        std::cerr << "rounded 0.1 s event was missed\n";
        return EXIT_FAILURE;
    }
    if (!ugkpcht::thermalEventIsDue(0.1, 0.1))
    {
        std::cerr << "exact event was missed\n";
        return EXIT_FAILURE;
    }
    if (ugkpcht::thermalEventIsDue(0.0999999999, 0.1))
    {
        std::cerr << "event tolerance is physically too broad\n";
        return EXIT_FAILURE;
    }
    if (ugkpcht::thermalEventIsDue(0.09, 0.1))
    {
        std::cerr << "early event was accepted\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASS\n";
    return EXIT_SUCCESS;
}
