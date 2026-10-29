#pragma once
#include <cstdint>
#include <cmath>
#include "ParticlePhaseConfiguration.H"
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
    out.gasPrClamped=gas.gasPrClamped;out.gasPrOneThird=std::pow(gas.gasPrClamped,1./3.);
    out.rhoMin=gas.rhoMin;out.TgasMin=gas.TgasMin;out.gasThermalConductivity=gas.gasThermalConductivity;
    out.sstConfigured=gas.sstConfigured;out.species=gas.gasSpecies;
    out.viscosity=gas.gasViscosity;out.diagnostics=gas.diagnostics;
    return out;
}
// Configuration/state publication boundary. Existing private physical-field
// allocations are released on first binding; borrowed memory is never freed.
int bindParticleGas(void* particleOwner,const ParticleGasBinding& gas);
// Allocate/configure the same owner used by CHT and bind the externally owned
// gas mesh/state. The caller selects contact faces in its mesh adapter. It must
// upload complete particle history and refresh tracking geometry before advance.
int createBoundParticlePhase(const ParticlePhaseConfiguration& phase,
    const ParticleContactConfiguration& contact,const ParticleGasBinding& gas,
    int nCellPlanes,const unsigned char* contactFaces,const double* gravity,void** owner);
const double* particleGasVolumeFraction(void* particleOwner);
// Refresh contact occupancy before gas conductive fluxes, on the caller stream.
int prepareParticleGasInterval(void* particleOwner,cudaStream_t gasStream);

// Submit the shared physical particle phase between gasStream events. The
// caller observes its own diagnostics after this stream reaches completion.
int advanceBoundParticlePhase(void* particleOwner,double dt,double endTime,cudaStream_t gasStream);
}}
