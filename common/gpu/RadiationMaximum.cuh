#pragma once
#include "RadiationPrecision.H"
#include "OrderedMaximum.cuh"
namespace ugkp { namespace gpu {
using RadiationMaximum=OrderedMaximumSegment<RadiationReal>;
}}
