#pragma once
#include "gpu/Buffer.H"
#include "gpu/ImplicitMaterialWindow.H"
#include "gpu/GasWindowControl.H"
#include "materials/ImplicitMaterialBalance.H"
#include "materials/MaterialWindowExchange.H"
#include "../../../common/gpu/SparseConjugateGradient.H"

namespace chmt {
struct ImplicitWindowControls {
    Real interval=0;
    PhysicalReal dt=0,changeGoal=0;
    EnergyReal absolute=0,relative=0,linearFraction=0;
    int iterations=0;
    ugkp::gpu::ConjugateGradientControls linear;
};
struct ImplicitWindowMetrics {
    PhysicalReal mass=0,change=0;
    EnergyReal energy=0,boundary=0,initialResidual=0,scale=0;
};
struct ImplicitWindowRemovalSum {MaterialWindowSum<PhysicalReal> mass[Nc];};

struct ImplicitWindowStorage {
    Buffer<ImplicitWindowControls> controls;
    Buffer<ImplicitWindowReport> report;
    Buffer<MaterialWindowCell> exchange;
    Buffer<ImplicitWindowRemovalSum> removed;
    Buffer<ImplicitWindowRemoval> removals;
    Buffer<unsigned> receivers;
    Buffer<int> surfaceOffsets,surfaceFaces,surfaceCells;
    std::vector<int> surfaceOwners;
    bool surfaceReady=false;
    Buffer<implicitMaterial::Face> face;
    Buffer<int> incidentOffsets,incident,orderedFaces;
    std::vector<int> colorOffsets;
    std::vector<Real> sweepValues;
    std::uint64_t colorTopology=0;
    int colorCells=0,colorFaces=0;
    Buffer<Real> swept;
    Buffer<PhysicalReal> nativeVolume;
    Buffer<implicitMaterial::Amount> fixedMass,mass;
    struct Extents {PhysicalReal values[Nr]{};};
    Buffer<Extents> extents;
    Buffer<EnergyReal> fixedEnergy,target,conducted;
    Buffer<PhysicalReal> temperature;
    Buffer<MaterialCaloric> caloric;
    Buffer<implicitMaterial::CellThermal> thermal;
    Buffer<MaterialFaceTransport> splitFlux;
    Buffer<GpuReal> capacity,source,conductivity;
    Buffer<ImplicitWindowMetrics> cellMetrics,faceMetrics,parts;
    Buffer<ugkp::gpu::ConjugateGradientControls> linear;
    ugkp::gpu::ImplicitDiffusion* diffusion=nullptr;
    ugkp::gpu::SparseConjugateGradient* sparse=nullptr;
    ugkp::gpu::ExecutionGraphSignature signature,continuationSignature;
    cudaGraphExec_t continuation=nullptr;
    cudaEvent_t ready=nullptr;
    bool transport=false,darcy=false,diffusionActive=false,hasSweepStorage=false;
    unsigned long long revision=0;
    ~ImplicitWindowStorage(){
        if(continuation)cudaGraphExecDestroy(continuation);
        if(ready)cudaEventDestroy(ready);
    }
};
}
