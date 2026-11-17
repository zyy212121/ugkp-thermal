#ifndef UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#define UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#include <memory>
#include <future>
#include <limits>
#include "../GpuPrecisionTypes.H"
#include "ExecutionGraph.H"
namespace ugkp { namespace gpu {


struct SparseMatrixView {
    int rows=0, entries=0;
    const int* offsets=nullptr;
    const int* columns=nullptr;
    const double* values=nullptr;
    const double* rowSum=nullptr;
};


struct SparseLduView {
    int rows=0, faces=0, boundaryFaces=0;
    const int* owner=nullptr;
    const int* neighbour=nullptr;
    const int* boundaryOwner=nullptr;
    const int* boundaryNeighbour=nullptr;
    const double* diagonal=nullptr;
    const double* upper=nullptr;
    const double* lower=nullptr;
    const double* source=nullptr;
    const double* boundaryDiagonal=nullptr;
    const double* boundarySource=nullptr;


    const double* boundaryJump=nullptr;
};


struct SparsePhysicalDiffusionValues {
    int internalFaces=0;
    const int* cellOffsets=nullptr;
    const int* cellFaces=nullptr;
    const float* gamma=nullptr;
    const float* area=nullptr;
    const float* delta=nullptr;
    const float* volume=nullptr;
    const float* capacity=nullptr;
    const float* interval=nullptr;
    const float* gradientInternal=nullptr;
    const float* gradientBoundary=nullptr;
    bool transient=false,active=false;
};
struct SparseDeviceLduValues {
    const GpuReal* diagonal=nullptr;
    const GpuReal* upper=nullptr;
    const GpuReal* lower=nullptr;
    const GpuReal* source=nullptr;
    const GpuReal* boundaryDiagonal=nullptr;
    const GpuReal* boundarySource=nullptr;
    void* readyStream=nullptr;
    const GpuReal* boundaryJump=nullptr;
    SparsePhysicalDiffusionValues diffusion{};
};
struct ConjugateGradientControls {
    int maximumIterations=300, minimumIterations=0;
    GpuReal tolerance=1e-6f, relativeTolerance=0;
    GpuReal normalizationSmall=1e-20f, singularSmall=std::numeric_limits<GpuReal>::min();
};
struct ConjugateGradientReport {
    int preconditionerFallbackBlocks=0;
    int iterations=0;
    GpuReal initialResidual=0, finalResidual=0;
    bool converged=false, singular=false;
    bool stagnated=false, budgetExhausted=false;
    int residualReplacements=0;
    GpuReal targetResidual=0;
};
struct ConjugateGradientDeviceReport {
    const ConjugateGradientReport* report=nullptr;
    void* readyStream=nullptr;
    std::shared_future<void> completion;
};



struct SparseCommunication {
    void* context=nullptr;
    void (*globalSum)(void*,GpuReal*,int)=nullptr;
    void (*addInterface)(void*,const GpuReal*,GpuReal*,int)=nullptr;


    const int* interfaceCells=nullptr;
    int interfaceCellCount=-1;
};
class SparseConjugateGradient {
    struct Implementation;
    std::unique_ptr<Implementation> data_;
    ConjugateGradientReport solvePrepared(int,double*,const ConjugateGradientControls&,const SparseCommunication&,const GpuReal* initialAffine=nullptr,GpuReal* deviceSolution=nullptr);
    void prepareLdu(const SparseLduView&,const SparseDeviceLduValues*,bool assemble=true);
public:
    SparseConjugateGradient();
    ExecutionGraphSignature residentExecutionSignature() const;
    ~SparseConjugateGradient();
    SparseConjugateGradient(const SparseConjugateGradient&)=delete;
    SparseConjugateGradient& operator=(const SparseConjugateGradient&)=delete;
    ConjugateGradientReport solveLdu(const SparseLduView&,double* solution,
        const ConjugateGradientControls&,const SparseCommunication& communication={});
    ConjugateGradientReport solveResidentLdu(const SparseLduView&,const SparseDeviceLduValues&,
        GpuReal* deviceSolution,const ConjugateGradientControls&,const SparseCommunication& communication={});



    void configureResidentLocalLdu(const SparseLduView&,const SparseDeviceLduValues&,
        const ConjugateGradientControls&);
    ConjugateGradientDeviceReport enqueueResidentLocalLdu(GpuReal* deviceSolution);


    ExecutionGraphNode appendResidentLocalLdu(ExecutionGraph,ExecutionGraphNode dependency,GpuReal* deviceSolution,
        const ConjugateGradientControls* deviceControls=nullptr);
    ExecutionGraphNode appendResidentLocalPreparation(ExecutionGraph,ExecutionGraphNode dependency);
    ExecutionGraphNode appendResidentLocalPreconditioner(ExecutionGraph,ExecutionGraphNode dependency,const GpuReal* rhs,GpuReal* result);
    ExecutionGraphNode appendResidentLocalResidual(ExecutionGraph,ExecutionGraphNode dependency,
        GpuReal* deviceSolution,GpuReal* deviceResidual);
    const int* capacityPendingDevice() const;
    bool servicePendingCapacity();
    const ConjugateGradientReport* deviceReport() const;
    ConjugateGradientReport collectResidentLocalReport();
    void residualResidentLdu(const SparseLduView&,const SparseDeviceLduValues&,
        const GpuReal* deviceSolution,GpuReal* deviceResidual,const SparseCommunication& communication={});
    ConjugateGradientReport solve(const SparseMatrixView&,const double* source,
        double* solution,const ConjugateGradientControls&,
        const SparseCommunication& communication={});
};
}}
#endif
