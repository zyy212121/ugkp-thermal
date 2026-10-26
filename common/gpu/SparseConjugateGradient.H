#ifndef UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#define UGKP_GPU_SPARSE_CONJUGATE_GRADIENT_H
#include <memory>
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
};
struct ConjugateGradientControls {
    int maximumIterations=1000, minimumIterations=0;
    double tolerance=1e-6, relativeTolerance=0;
    double normalizationSmall=1e-20, singularSmall=1e-300;
};
struct ConjugateGradientReport {
    int iterations=0;
    double initialResidual=0, finalResidual=0;
    bool converged=false, singular=false;
};
// Communication hooks only: all local matrix/vector arithmetic remains CUDA.
// addInterface supplies the missing coupled-interface contribution to A*x.
// Local cyclic interfaces may instead be represented directly in the CSR.
struct SparseCommunication {
    void* context=nullptr;
    void (*globalSum)(void*,double*,int)=nullptr;
    void (*addInterface)(void*,const double*,double*,int)=nullptr;
};
class SparseConjugateGradient {
    struct Implementation;
    std::unique_ptr<Implementation> data_;
    ConjugateGradientReport solvePrepared(int,double*,const ConjugateGradientControls&,const SparseCommunication&);
public:
    SparseConjugateGradient();
    ~SparseConjugateGradient();
    SparseConjugateGradient(const SparseConjugateGradient&)=delete;
    SparseConjugateGradient& operator=(const SparseConjugateGradient&)=delete;
    ConjugateGradientReport solveLdu(const SparseLduView&,double* solution,
        const ConjugateGradientControls&,const SparseCommunication& communication={});
    ConjugateGradientReport solve(const SparseMatrixView&,const double* source,
        double* solution,const ConjugateGradientControls&,
        const SparseCommunication& communication={});
};
}}
#endif
