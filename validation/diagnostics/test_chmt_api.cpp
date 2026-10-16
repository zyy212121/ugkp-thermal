#include "gpu/Backend.H"
#include "DiagnosticsHost.H"
#include <type_traits>
static_assert(std::is_same<decltype(&chmt::setDiagnosticsSink),void(*)(chmt::Backend&,const ugkp::diagnostics::HostSink&)>::value,"CHMT binds the host sink without linking OpenFOAM into CUDA");
