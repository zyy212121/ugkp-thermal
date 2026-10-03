#pragma once
// The cache retains its established allocation/layout: four full-directory
// moments followed by nine particle-state parameters; split state starts at zero.
enum class FlatPressureSegment { full, base, injection };
struct FlatPressureLayout
{
    static constexpr int fullStateOffset = 4;
    static constexpr int splitStateOffset = 0;
    static constexpr int stateSize = 9;
    static constexpr int parameterStride = fullStateOffset + stateSize;
    static constexpr int flagStride = 2;
    enum Moment { momentumX = 0, momentumY, momentumZ, energy };
    enum State { ux0 = 0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale };
    enum Flag { active = 0, resolved };
};
static_assert(FlatPressureLayout::parameterStride == 13, "flat pressure cache layout changed");
