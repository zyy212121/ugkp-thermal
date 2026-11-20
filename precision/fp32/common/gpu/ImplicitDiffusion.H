#ifndef UGKP_GPU_IMPLICIT_DIFFUSION_H
#define UGKP_GPU_IMPLICIT_DIFFUSION_H
#include "SparseConjugateGradient.H"
#include "ThermalFieldOperators.H"
#include "ThermalInterfaceDeviceView.H"
#include <memory>
#include "../gasBoundary/Definition.H"
#include <vector>
namespace ugkp { namespace gpu {
struct DiffusionInput {
    int cells=0,internalFaces=0;
    std::vector<int> owner,neighbour;
    std::vector<double> volume,gamma,area,delta,gradientInternal,gradientBoundary,correction;
    bool corrected=false;


    const GpuReal* deviceConductivity=nullptr;
    const GpuReal* deviceBoundaryConductivity=nullptr;
    void* conductivityStream=nullptr;
    std::vector<double> conductivityWeights,conductivityBoundary,conductivityNeighbour;
    std::vector<int> conductivityCoupled,conductivityAdjacentCells;
    std::vector<double> conductivityJump;
    bool harmonicConductivity=false;

    const GpuReal* deviceTemperature=nullptr;
    std::vector<double> gradientWeights,valueInternal,valueBoundary,normalInternal,normalBoundary,jump,boundaryValueDelta;
    std::vector<int> gradientNeighbour;
    std::vector<ThermalVector> faceArea,correctionDirection,neighbourTransform;

};


struct DiffusionTransientInput {
    const std::vector<double>* capacity=nullptr;
    const std::vector<double>* rhs=nullptr;
    double dt=0;bool sourceFromCapacity=false;
    const GpuReal* deviceCapacity=nullptr;
    const GpuReal* deviceRhs=nullptr;
    void* readyStream=nullptr;
};
struct DiffusionMatrix {
    std::vector<double> diagonal,source,upper,lower,boundaryDiagonal,boundarySource,fluxCorrection;
};
class ImplicitDiffusion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ImplicitDiffusion();~ImplicitDiffusion();
    ExecutionGraphSignature residentExecutionSignature() const;


    void assembleResident(const DiffusionInput&,const DiffusionTransientInput* transient=nullptr);



    void enqueueResidentAssembly(void* executionStream,const double* deviceInterval=nullptr);
    void enqueueResidentSnapshot(void* executionStream);

    void prepareResidentWindowSnapshot();
    void enqueueResidentWindowSnapshot(void* executionStream);
    void enqueueResidentWindowRestore(void* executionStream);
    void enqueueResidentBoundaryTemperature(void* executionStream);
    void enqueueResidentFlux(void* executionStream,bool boundaryOnly);
    void enqueueResidentBoundaryFlux(void* executionStream);
    void enqueueResidentInterfaceEnergy(void* executionStream);
    void configureBoundaryJumpSchedule(const boundary::Definition&,const std::vector<int>& signs,double initialTime);
    void enqueueBoundaryJumpSchedule(void* executionStream,const double* interval);
    void retain(const DiffusionInput&,const DiffusionMatrix&);
    SparseDeviceLduValues deviceValues() const;
    GpuReal* deviceSolution();
    const GpuReal* deviceTemperature() const;
    GpuReal* deviceResidual();
    const GpuReal* devicePreviousSolution() const;
    void snapshotSolution();
    void refreshBoundaryTemperature();
    void prescribeBoundaryEnergy(const std::vector<double>& energy,const std::vector<int>& selected,
        double interval,const GpuReal* deviceBoundaryConductivity,void* propertyStream);
    void setInterfaceEnergy(const std::vector<int>& boundaryFaces,const ThermalInterfaceEnergyDeviceView&,
        const ThermalInterfaceEnergyDeviceView& appended={});
    void initializeBoundaryTemperature(const std::vector<double>&);
    const GpuReal* deviceBoundaryTemperature() const;
    bool residentIteration() const;
    void uploadSolution(const double*,int);
    void copySolution(const GpuReal*,int,void* producerStream);
    void downloadSolution(double*,int);
    void downloadResidual(double*,int);
    void downloadBoundaryState(std::vector<double>& temperature,std::vector<double>& normalGradient);
    const GpuReal* deviceBoundaryFlux() const;
    const GpuReal* deviceVolume() const;
    const GpuReal* deviceBoundaryArea() const;
    const GpuReal* deviceBoundaryGradient() const;
    const GpuReal* deviceBoundaryEnergy() const;
    void downloadBoundaryCoefficients(std::vector<double>&,std::vector<double>&);
    void flux(const std::vector<double>& boundaryNeighbour,const std::vector<int>& coupled,
        const std::vector<int>& adjacentCells,const std::vector<double>& jump,
        std::vector<double>& output,bool boundaryOnly=false,bool download=true);
    void assemble(const DiffusionInput&,DiffusionMatrix&,const DiffusionTransientInput* transient=nullptr);
    void transient(const std::vector<double>& volume,const std::vector<double>& capacity,
        const std::vector<double>& rhs,double dt,bool sourceFromCapacity,DiffusionMatrix&);
};
}}
#endif
