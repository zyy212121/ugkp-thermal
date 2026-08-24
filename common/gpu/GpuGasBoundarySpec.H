#ifndef GpuGasBoundarySpec_H
#define GpuGasBoundarySpec_H

#include "word.H"

namespace Foam
{
namespace ugkpboundary
{

                                                                             
                                                                          
enum class PatchGeometry : int
{
    standard = 0,
    symmetry = 1,
    wall = 2,
    processor = 3,
    empty = 4,
    cyclic = 5
};

enum class ScalarOperator : int
{
    unsupported = -1,
    zeroGradient = 0,
    fixedValue = 1,
    waveTransmissive = 2,
    calculated = 3
};

enum class VelocityOperator : int
{
    unsupported = -1,
    zeroGradient = 0,
    fixedValue = 1,
    inletOutlet = 2
};

inline ScalarOperator scalarOperator
(
    const word& patchFieldType,
    const bool fixesValue,
    const bool waveTransmissive
)
{
    if (waveTransmissive)
    {
        return ScalarOperator::waveTransmissive;
    }
    if (fixesValue)
    {
        return ScalarOperator::fixedValue;
    }
    if (patchFieldType == "zeroGradient")
    {
        return ScalarOperator::zeroGradient;
    }
    if (patchFieldType == "calculated")
    {
        return ScalarOperator::calculated;
    }
    return ScalarOperator::unsupported;
}

inline VelocityOperator velocityOperator
(
    const word& patchFieldType,
    const bool fixesValue,
    const bool inletOutlet
)
{
    if (inletOutlet)
    {
        return VelocityOperator::inletOutlet;
    }
    if (fixesValue)
    {
        return VelocityOperator::fixedValue;
    }
    if (patchFieldType == "zeroGradient")
    {
        return VelocityOperator::zeroGradient;
    }
    return VelocityOperator::unsupported;
}

}                          
}                  

#endif
