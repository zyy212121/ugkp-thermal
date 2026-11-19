#ifndef UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#define UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#include <memory>
#include <future>
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
    const double* gamma=nullptr;
    const double* area=nullptr;
    const double* delta=nullptr;
    const double* volume=nullptr;
    const double* capacity=nullptr;
    const double* interval=nullptr;
    const double* gradientInternal=nullptr;
    const double* gradientBoundary=nullptr;
    bool transient=false,active=false;
};
struct SparseDeviceLduValues {
    const double* diagonal=nullptr;
    const double* upper=nullptr;
    const double* lower=nullptr;
    const double* source=nullptr;
    const double* boundaryDiagonal=nullptr;
    const double* boundarySource=nullptr;
    void* readyStream=nullptr;
    const double* boundaryJump=nullptr;
    SparsePhysicalDiffusionValues diffusion{};
};
struct ConjugateGradientControls {
    int maximumIterations=300, minimumIterations=0;
    double tolerance=1e-6, relativeTolerance=0;
    double normalizationSmall=1e-20, singularSmall=1e-300;
};
struct ConjugateGradientReport {
    int preconditionerFallbackBlocks=0;
    int iterations=0;
    double initialResidual=0, finalResidual=0;
    bool converged=false, singular=false;
    bool stagnated=false, budgetExhausted=false;
    int residualReplacements=0;
    double targetResidual=0;
};
struct ConjugateGradientDeviceReport {
    const ConjugateGradientReport* report=nullptr;
    void* readyStream=nullptr;
    std::shared_future<void> completion;
};



struct SparseCommunication {
    void* context=nullptr;
    void (*globalSum)(void*,double*,int)=nullptr;
    void (*addInterface)(void*,const double*,double*,int)=nullptr;


    const int* interfaceCells=nullptr;
    int interfaceCellCount=-1;
};
class SparseConjugateGradient {
    struct Implementation;
    std::unique_ptr<Implementation> data_;
    ConjugateGradientReport solvePrepared(int,double*,const ConjugateGradientControls&,const SparseCommunication&,const double* initialAffine=nullptr,double* deviceSolution=nullptr);
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
        double* deviceSolution,const ConjugateGradientControls&,const SparseCommunication& communication={});



    void configureResidentLocalLdu(const SparseLduView&,const SparseDeviceLduValues&,
        const ConjugateGradientControls&);
    ConjugateGradientDeviceReport enqueueResidentLocalLdu(double* deviceSolution);


    ExecutionGraphNode appendResidentLocalLdu(ExecutionGraph,ExecutionGraphNode dependency,double* deviceSolution,
        const ConjugateGradientControls* deviceControls=nullptr);
    ExecutionGraphNode appendResidentLocalPreparation(ExecutionGraph,ExecutionGraphNode dependency);
    ExecutionGraphNode appendResidentLocalPreconditioner(ExecutionGraph,ExecutionGraphNode dependency,const double* rhs,double* result);
    ExecutionGraphNode appendResidentLocalResidual(ExecutionGraph,ExecutionGraphNode dependency,
        double* deviceSolution,double* deviceResidual);
    const int* capacityPendingDevice() const;
    bool servicePendingCapacity();
    const ConjugateGradientReport* deviceReport() const;
    ConjugateGradientReport collectResidentLocalReport();
    void residualResidentLdu(const SparseLduView&,const SparseDeviceLduValues&,
        const double* deviceSolution,double* deviceResidual,const SparseCommunication& communication={});
    ConjugateGradientReport solve(const SparseMatrixView&,const double* source,
        double* solution,const ConjugateGradientControls&,
        const SparseCommunication& communication={});
};
}}
#endif
