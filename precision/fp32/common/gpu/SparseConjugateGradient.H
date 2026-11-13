#ifndef UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#define UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#include <memory>
#include <limits>
#include "../GpuPrecisionTypes.H"
#include "ExecutionGraph.H"
namespace ugkp { namespace gpu {
// Scalar finite-volume matrix in ordered CSR form. Each row starts with its
// diagonal; subsequent entries retain the original face accumulation order.
struct SparseMatrixView {
    int rows=0, entries=0;
    const int* offsets=nullptr;
    const int* columns=nullptr;
    const double* values=nullptr;
    const double* rowSum=nullptr;
};
// Native LDU factors plus boundary coefficients, before boundary completion.
// Boundary neighbour: >=0 local cyclic cell, -1 physical source, -2 remote.
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
    // Optional signed coupled jump per boundary face. Its weighted affine
    // contribution belongs to the initial A*solution, not A*KrylovDirection.
    const double* boundaryJump=nullptr;
};
// Coefficients are borrowed device storage. Topology and boundary semantics
// remain in SparseLduView; readyStream orders their producer without a host wait.
struct SparseDeviceLduValues {
    const GpuReal* diagonal=nullptr;
    const GpuReal* upper=nullptr;
    const GpuReal* lower=nullptr;
    const GpuReal* source=nullptr;
    const GpuReal* boundaryDiagonal=nullptr;
    const GpuReal* boundarySource=nullptr;
    void* readyStream=nullptr;
    const GpuReal* boundaryJump=nullptr;
};
struct ConjugateGradientControls {
    int maximumIterations=1000, minimumIterations=0;
    GpuReal tolerance=1e-6f, relativeTolerance=0;
    GpuReal normalizationSmall=1e-20f, singularSmall=std::numeric_limits<GpuReal>::min();
};
struct ConjugateGradientReport {
    int iterations=0;
    GpuReal initialResidual=0, finalResidual=0;
    bool converged=false, singular=false;
};
struct ConjugateGradientDeviceReport {
    const ConjugateGradientReport* report=nullptr;
    void* readyStream=nullptr;
};
// Communication hooks only: all local matrix/vector arithmetic remains CUDA.
// addInterface supplies the missing coupled-interface contribution to A*x.
// Local cyclic interfaces may instead be represented directly in the CSR.
struct SparseCommunication {
    void* context=nullptr;
    void (*globalSum)(void*,GpuReal*,int)=nullptr;
    void (*addInterface)(void*,const GpuReal*,GpuReal*,int)=nullptr;
    // Exact union of local cells read/written by addInterface. Negative count
    // retains the unrestricted callback contract; native FV supplies this set.
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
    // Bind local topology/controls once. In-place device coefficient updates
    // are consumed on each enqueue; output publication is an explicit action.
    // Native distributed-interface callbacks remain in solveResidentLdu.
    void configureResidentLocalLdu(const SparseLduView&,const SparseDeviceLduValues&,
        const ConjugateGradientControls&);
    ConjugateGradientDeviceReport enqueueResidentLocalLdu(GpuReal* deviceSolution);
    // Compose the same local solve into an enclosing device-controlled loop.
    // Caller orders input/output ownership through graph dependencies.
    ExecutionGraphNode appendResidentLocalLdu(ExecutionGraph,ExecutionGraphNode dependency,GpuReal* deviceSolution,
        const ConjugateGradientControls* deviceControls=nullptr);
    ExecutionGraphNode appendResidentLocalResidual(ExecutionGraph,ExecutionGraphNode dependency,
        GpuReal* deviceSolution,GpuReal* deviceResidual);
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
