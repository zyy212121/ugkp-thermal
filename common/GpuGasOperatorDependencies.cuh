#pragma once
// Include at global scope before an application's operator namespace. These
// headers own ugkwp types and the chemistry kernel; first inclusion inside a
// namespace would create a second ugkwp and break the shared-state templates.
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/GasGeometryValidation.H"
#include "gasTransport/SpeciesDiffusion.H"
#include "operators/advanceGasChemistryKernel.cuh"
