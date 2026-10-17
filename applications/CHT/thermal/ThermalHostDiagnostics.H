#ifndef UGKP_THERMAL_HOST_DIAGNOSTICS_H
#define UGKP_THERMAL_HOST_DIAGNOSTICS_H
#include "error.H"
#include <cstdlib>
#include <cstring>
namespace Foam { namespace gpuThermal {
// Matches the common diagnostic output-only switch; never controls solving.
inline bool hostDiagnosticWarningsEnabled()
{
    const char* value = std::getenv("UGKP_DIAGNOSTIC_WARNINGS");
    return !value || (std::strcmp(value, "0") && std::strcmp(value, "off")
                  && std::strcmp(value, "false"));
}
}}
#endif
