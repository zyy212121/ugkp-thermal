#pragma once
#include "core/Types.H"
#include "../../../common/GpuPrecisionTypes.H"
#include "../../../common/gpu/ExecutionGraph.H"
#include <string>
#include <vector>

namespace ugkp::diagnostics {struct DeviceContext;}
namespace ugkp::gpu {class ImplicitDiffusion;class SparseConjugateGradient;}
namespace chmt {
struct MaterialBackend;
struct HostState;
struct HostMesh;
struct CouplingInterval;
struct CpuMaterialControls;
struct CoupledSurfaceDownload;
struct ResidentMaterialDriveHistory;
struct GasWindowProgress;
struct IntervalRadiationDeviceView;
struct ImplicitWindowRemoval {PhysicalReal mass[Nc]{};};
struct ImplicitWindowReport {
    std::uint64_t packets=0,frames=0;
    int iterations=0,linearSolves=0,linearIterations=0;
    bool initialized=false,linearPrepared=false,requiresLinearSolve=false,complete=false,converged=false,capacity=false;
    PhysicalReal massResidual=0,temperatureResidual=0;
    EnergyReal energyResidual=0,boundaryEnergy=0;
};
struct ImplicitWindowThermalFields {
    GpuReal* capacity=nullptr;
    GpuReal* source=nullptr;
    GpuReal* conductivity=nullptr;
    void* stream=nullptr;
};

bool beginImplicitMaterialExchange(MaterialBackend&,const HostState&,const CouplingInterval&,
    const CpuMaterialControls&,std::string&);
ugkp::gpu::ExecutionGraphNode appendImplicitMaterialExchange(MaterialBackend&,const ResidentMaterialDriveHistory&,
    const GasWindowProgress*,bool,ugkp::gpu::ExecutionGraph,ugkp::gpu::ExecutionGraphNode,void*);
void enqueueImplicitMaterialFrame(MaterialBackend&,const CoupledSurfaceDownload*,std::size_t faces,
    std::uint64_t acceptedFrame,bool particles,void* stream);
bool collectImplicitWindowRemoval(MaterialBackend&,std::vector<ImplicitWindowRemoval>&,std::string&);
bool implicitWindowMeshConfigured(const MaterialBackend&,const HostMesh&,const std::vector<Real>&);
bool configureImplicitWindowMesh(MaterialBackend&,const HostMesh&,const std::vector<Real>&,
    const std::vector<Real>& nativeVolumes,std::string&);
ImplicitWindowThermalFields implicitWindowThermalFields(MaterialBackend&);
bool configureImplicitWindowConduction(MaterialBackend&,ugkp::gpu::ImplicitDiffusion&,
    ugkp::gpu::SparseConjugateGradient&,std::string&);
unsigned long long implicitWindowRevision(const MaterialBackend&);
const ugkp::diagnostics::DeviceContext* implicitWindowDiagnostics(const MaterialBackend&);
void orderImplicitWindow(MaterialBackend&,void* destination);
void acceptImplicitWindowExecution(MaterialBackend&,void* producer);
ugkp::gpu::ExecutionGraphNode appendImplicitMaterialSolve(MaterialBackend&,const EnergyReal* radiation,
    ugkp::gpu::ExecutionGraph,ugkp::gpu::ExecutionGraphNode,void*);
bool runImplicitMaterialSolve(MaterialBackend&,const IntervalRadiationDeviceView&,std::string&);
bool collectImplicitMaterialSolve(MaterialBackend&,std::vector<SolidQ>&,ImplicitWindowReport&,std::string&);
bool commitImplicitMaterialInventory(MaterialBackend&,const std::vector<SolidQ>&,std::string&);
}
