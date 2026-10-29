#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasBuildConfig.H"
namespace ugkp { namespace thermal {
// Borrowed pointers remain owned by the gas application. The gas owner must
// outlive the particle owner; rebind after replacing any allocation. Binding
// transfers no field values and the phase API uses CUDA events for ordering.
struct ParticleGasBinding {
    int nCells=0,nFaces=0,nInternalFaces=0;
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) Type* Name=nullptr;
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
    double gammaGas=1.4,Rgas=287,gasCp=1004.5,gasMu=0,gasPrClamped=.7,gasPrOneThird=0;
    double rhoMin=1e-12,TgasMin=1,gasThermalConductivity=-1;
    int sstConfigured=0;
    ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies> species;
    ugkwp::MolecularViscosityView<double> viscosity;
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
};
template<class Gas> ParticleGasBinding particleGasBinding(const Gas& gas) {
    ParticleGasBinding out;
    out.nCells=gas.nCells;out.nFaces=gas.nFaces;out.nInternalFaces=gas.nInternalFaces;
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) out.Name=gas.Name;
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
    out.gammaGas=gas.gammaGas;out.Rgas=gas.Rgas;out.gasCp=gas.gasCp;out.gasMu=gas.gasMu;
    out.gasPrClamped=gas.gasPrClamped;out.gasPrOneThird=gas.gasPrOneThird;
    out.rhoMin=gas.rhoMin;out.TgasMin=gas.TgasMin;out.gasThermalConductivity=gas.gasThermalConductivity;
    out.sstConfigured=gas.sstConfigured;out.species=gas.gasSpecies;
    out.viscosity=gas.gasViscosity;out.diagnostics=gas.diagnostics;
    return out;
}
// Configuration/state publication boundary. Existing private physical-field
// allocations are released on first binding; borrowed memory is never freed.
int bindParticleGas(void* particleOwner,const ParticleGasBinding& gas);
// Submit the shared physical particle phase between gasStream events. The
// caller observes its own diagnostics after this stream reaches completion.
int advanceBoundParticlePhase(void* particleOwner,double dt,double endTime,cudaStream_t gasStream);
}}
