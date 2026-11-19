#pragma once
#include <cstdint>
#include <cmath>
#include "ParticlePhaseConfiguration.H"
#include <cuda_runtime_api.h>
#include "ParticleResidentInterval.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasBuildConfig.H"
namespace ugkp { namespace thermal {



struct ParticleGasBinding {
    int nCells=0,nFaces=0,nInternalFaces=0;
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) Type* Name=nullptr;
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
    double gammaGas=1.4,Rgas=287,gasCp=1004.5,gasMu=0,gasPrClamped=.7,gasPrOneThird=0;
    double rhoMin=1e-12,TgasMin=1,gasThermalConductivity=-1;
    ugkp::gasNumerics::ScalarSolverControls<double> gasScalarSolvers;
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
    out.gasScalarSolvers=gas.gasScalarSolvers;
    out.sstConfigured=gas.sstConfigured;out.species=gas.gasSpecies;
    out.viscosity=gas.gasViscosity;out.diagnostics=gas.diagnostics;
    return out;
}


int bindParticleGas(void* particleOwner,const ParticleGasBinding& gas);



int createBoundParticlePhase(const ParticlePhaseConfiguration& phase,
    const ParticleContactConfiguration& contact,const ParticleGasBinding& gas,
    int nCellPlanes,const unsigned char* contactFaces,const double* gravity,void** owner);
const double* particleGasVolumeFraction(void* particleOwner);

int prepareParticleGasInterval(void* particleOwner,cudaStream_t gasStream);



int orderParticlePhaseAfterGas(void* particleOwner,cudaStream_t gasStream);
int orderGasAfterParticlePhase(void* particleOwner,cudaStream_t gasStream);



int advanceBoundParticlePhase(void* particleOwner,double dt,double endTime,cudaStream_t gasStream);


struct BoundParticleMirror {const void* activeCells=nullptr;int baseReady=0;};
cudaGraphNode_t appendBoundParticlePhase(void*,cudaGraph_t,cudaGraphNode_t,
    const ugkwp::ParticleResidentInterval*,cudaStream_t);
int enqueueBoundParticleMirror(void*,BoundParticleMirror*,cudaStream_t);
int acceptBoundParticleMirror(void*,const BoundParticleMirror&);

}}
