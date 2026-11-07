#pragma once
#include "GasDriveChange.H"
#include "../CompensatedTime.H"
namespace ugkwp {
struct GasDriveControls {
    ugkp::CompensatedTime begin{},requested{},total{};
    double tolerance=0,tractionScale=1;
};
enum class GasDriveSelectionError : int {None,InvalidControl,NoSample,LayoutChanged,NonfiniteChange,NonAdvancing};
struct GasDriveSelection {
    ugkp::CompensatedTime end{};
    unsigned long long held=0,first=0;
    GasDriveSelectionError error=GasDriveSelectionError::None;
};
}
