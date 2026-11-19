#pragma once
#include "gasTransport/GasBuildConfig.H"


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

    double particleTrackingExchange[6]{};
};
