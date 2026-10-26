#pragma once
#include "gasTransport/GasBuildConfig.H"

// Scalar-only public report. No active fields or parcel arrays cross to host.
struct GpuSteadySamplingReport
{
    static constexpr int fields=14+ugkwp::compiledGasSpecies;
    static constexpr int gasFluxFields=5+ugkwp::compiledGasSpecies;
    static constexpr int budgetFields=gasFluxFields+12;
    double meanChange[fields]{};
    double fluctuation[fields]{};
    double characteristicInventory[fields]{};
    double inventoryChange[fields]{};
    double inventoryBegin[fields]{};
    double inventoryEnd[fields]{};
    double gasBoundaryExchange[gasFluxFields]{};
    double particleInjection[6]{};
    // Includes wall momentum/energy exchange as well as removed parcel outflow.
    double particleTrackingExchange[6]{};
};
