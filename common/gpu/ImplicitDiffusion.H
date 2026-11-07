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
    // Optional resident cell conductivity for the selected common linear
    // interpolation. Boundary values still follow their native patch policy.
    const double* deviceConductivity=nullptr;
    const double* deviceBoundaryConductivity=nullptr;
    void* conductivityStream=nullptr;
    std::vector<double> conductivityWeights,conductivityBoundary,conductivityNeighbour;
    std::vector<int> conductivityCoupled,conductivityAdjacentCells;
    std::vector<double> conductivityJump;
    bool harmonicConductivity=false;
    // Selected Gauss-linear/corrected operators consume the retained iterate.
    const double* deviceTemperature=nullptr;
    std::vector<double> gradientWeights,valueInternal,valueBoundary,normalInternal,normalBoundary,jump,boundaryValueDelta;
    std::vector<int> gradientNeighbour;
    std::vector<ThermalVector> faceArea,correctionDirection,neighbourTransform;

};
// Borrowed host inputs; assemble uploads these once and keeps its matrix on
// device through transient completion. The caller still owns native FV output.
struct DiffusionTransientInput {
    const std::vector<double>* capacity=nullptr;
    const std::vector<double>* rhs=nullptr;
    double dt=0;bool sourceFromCapacity=false;
    const double* deviceCapacity=nullptr;
    const double* deviceRhs=nullptr;
    void* readyStream=nullptr;
};
struct DiffusionMatrix {
    std::vector<double> diagonal,source,upper,lower,boundaryDiagonal,boundarySource,fluxCorrection;
};
class ImplicitDiffusion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ImplicitDiffusion();~ImplicitDiffusion();
    // This owner retains matrix factors and the iterate across its shared
    // assembly, sparse solve, residual and flux operators.
    void assembleResident(const DiffusionInput&,const DiffusionTransientInput* transient=nullptr);
    // Replay the configured discretization with updated in-place device fields.
    // The enclosing device graph orders producers and owns executionStream;
    // topology, policies and borrowed addresses remain fixed until reconfigured.
    void enqueueResidentAssembly(void* executionStream,const double* deviceInterval=nullptr);
    void enqueueResidentSnapshot(void* executionStream);
    // Contact fixed-point candidates restart from the same accepted window.
    void prepareResidentWindowSnapshot();
    void enqueueResidentWindowSnapshot(void* executionStream);
    void enqueueResidentWindowRestore(void* executionStream);
    void enqueueResidentBoundaryTemperature(void* executionStream);
    void enqueueResidentBoundaryFlux(void* executionStream);
    void enqueueResidentInterfaceEnergy(void* executionStream);
    void configureBoundaryJumpSchedule(const boundary::Definition&,const std::vector<int>& signs,double initialTime);
    void enqueueBoundaryJumpSchedule(void* executionStream,const double* interval);
    void retain(const DiffusionInput&,const DiffusionMatrix&);
    SparseDeviceLduValues deviceValues() const;
    double* deviceSolution();
    const double* deviceTemperature() const;
    double* deviceResidual();
    const double* devicePreviousSolution() const;
    void snapshotSolution();
    void refreshBoundaryTemperature();
    void prescribeBoundaryEnergy(const std::vector<double>& energy,const std::vector<int>& selected,
        double interval,const double* deviceBoundaryConductivity,void* propertyStream);
    void setInterfaceEnergy(const std::vector<int>& boundaryFaces,const ThermalInterfaceEnergyDeviceView&,
        const ThermalInterfaceEnergyDeviceView& appended={});
    void initializeBoundaryTemperature(const std::vector<double>&);
    const double* deviceBoundaryTemperature() const;
    bool residentIteration() const;
    void uploadSolution(const double*,int);
    void copySolution(const double*,int,void* producerStream);
    void downloadSolution(double*,int);
    void downloadResidual(double*,int);
    void downloadBoundaryState(std::vector<double>& temperature,std::vector<double>& normalGradient);
    const double* deviceBoundaryFlux() const;
    const double* deviceVolume() const;
    const double* deviceBoundaryArea() const;
    const double* deviceBoundaryGradient() const;
    const double* deviceBoundaryEnergy() const;
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
