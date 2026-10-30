#ifndef UGKP_GPU_FOAM_SPARSE_SOLVE_H
#define UGKP_GPU_FOAM_SPARSE_SOLVE_H
#include "SparseConjugateGradient.H"
#include "fvMatrices.H"
#include "cyclicFvPatchFields.H"
#include "jumpCyclicFvPatchFields.H"
#include "cyclicFvPatch.H"
#include "Residuals.H"
#include "PstreamReduceOps.H"
#include <vector>
#include <algorithm>
namespace ugkp { namespace gpu {
// Bridge the native FV boundary contract to the shared CUDA sparse solver.
// This bridge is also the independent assembly reference while cell/face
// assembly is migrated. It never dispatches to a CPU linear solver.
inline Foam::solverPerformance solveFoamScalarMatrix(
    Foam::fvScalarMatrix& equation,const Foam::dictionary& dictionary,
    SparseConjugateGradient& solver) {
    using namespace Foam;
    auto& temperature=const_cast<volScalarField&>(equation.psi());
    const int count=temperature.size();
    lduInterfaceFieldPtrsList interfaces(temperature.boundaryField().scalarInterfaces());
    std::vector<int> boundaryOwner,boundaryNeighbour,interfaceCells;
    std::vector<double> boundaryDiagonal,boundarySource,boundaryJump;
    bool haveJump=false;
    bool haveInterface=false;
    forAll(equation.internalCoeffs(),patch){
        const auto& cells=equation.lduAddr().patchAddr(patch);
        const bool coupled=temperature.boundaryField()[patch].coupled();
        const bool localCyclic=isA<cyclicFvPatchScalarField>(temperature.boundaryField()[patch]);
        const labelUList* adjacent=nullptr;
        if(localCyclic){
            adjacent=&refCast<const cyclicFvPatch>(temperature.mesh().boundary()[patch]).neighbFvPatch().faceCells();
            interfaces.set(patch,nullptr);
        }
        // jumpCyclic applies its affine term only to the original solution
        // field, never to Krylov vectors. Transfer that term explicitly before
        // removing this native interface; copying psi into a callback loses
        // the field-identity test used by OpenFOAM's jump implementation.
        tmp<scalarField> jump;scalar jumpSign=1;
        const auto& field=temperature.boundaryField()[patch];
        if(localCyclic && isA<jumpCyclicFvPatchScalarField>(field)){
            jump=refCast<const jumpCyclicFvPatchScalarField>(field).jump();
            if(!refCast<const cyclicFvPatch>(temperature.mesh().boundary()[patch]).owner())jumpSign=-1;
            if(!haveJump){boundaryJump.resize(boundaryOwner.size(),0);haveJump=true;}
        }
        if(interfaces.set(patch)){haveInterface=true;interfaceCells.insert(interfaceCells.end(),cells.begin(),cells.end());}
        forAll(cells,f){
            if(haveJump)boundaryJump.push_back(jump.valid()?jumpSign*jump()[f]:0);
            boundaryOwner.push_back(cells[f]);boundaryNeighbour.push_back(localCyclic?(*adjacent)[f]:coupled?-2:-1);
            boundaryDiagonal.push_back(equation.internalCoeffs()[patch][f]);boundarySource.push_back(equation.boundaryCoeffs()[patch][f]);
        }
    }
    std::sort(interfaceCells.begin(),interfaceCells.end());
    interfaceCells.erase(std::unique(interfaceCells.begin(),interfaceCells.end()),interfaceCells.end());
    struct Context {fvScalarMatrix& equation;const lduInterfaceFieldPtrsList& interfaces;int communicator;
        const std::vector<int>& cells;scalarField input,output;};
    Context context{equation,interfaces,equation.mesh().comm(),interfaceCells,scalarField(count,0),scalarField(count,0)};
    SparseCommunication communication;communication.context=&context;
    communication.interfaceCells=interfaceCells.data();communication.interfaceCellCount=int(interfaceCells.size());
    if(Pstream::parRun())communication.globalSum=[](void* data,double* values,int n){
        const auto& context=*static_cast<Context*>(data);
        for(int i=0;i<n;++i)reduce(values[i],sumOp<scalar>(),Pstream::msgType(),context.communicator);
    };
    if(haveInterface)communication.addInterface=[](void* data,const double* x,double* result,int n){
        auto& context=*static_cast<Context*>(data);auto& input=context.input;auto& output=context.output;
        for(int c:context.cells){input[c]=x[c];output[c]=result[c];}
        context.equation.initMatrixInterfaces(context.equation.boundaryCoeffs(),context.interfaces,input,output,0);
        context.equation.updateMatrixInterfaces(context.equation.boundaryCoeffs(),context.interfaces,input,output,0);
        for(int c:context.cells)result[c]=output[c];
        (void)n;
    };
    ConjugateGradientControls controls;
    controls.maximumIterations=dictionary.lookupOrDefault<label>("maxIter",1000);
    controls.minimumIterations=dictionary.lookupOrDefault<label>("minIter",0);
    controls.tolerance=dictionary.lookupOrDefault<scalar>("tolerance",1e-6);
    controls.relativeTolerance=dictionary.lookupOrDefault<scalar>("relTol",0);
    controls.normalizationSmall=solverPerformance::small_;
    controls.singularSmall=solverPerformance::vsmall_;
    const lduMatrix& ldu=equation;
    const auto& owner=equation.lduAddr().lowerAddr();const auto& neighbour=equation.lduAddr().upperAddr();
    SparseLduView matrix{count,int(owner.size()),int(boundaryOwner.size()),owner.begin(),neighbour.begin(),
        boundaryOwner.data(),boundaryNeighbour.data(),equation.diag().begin(),
        owner.size()?ldu.upper().begin():nullptr,owner.size()?ldu.lower().begin():nullptr,equation.source().begin(),
        boundaryDiagonal.data(),boundarySource.data(),haveJump?boundaryJump.data():nullptr};
    const auto report=solver.solveLdu(matrix,temperature.primitiveFieldRef().begin(),controls,communication);
    solverPerformance performance("CudaDiagonalPCG",temperature.name(),report.initialResidual,
        report.finalResidual,report.iterations,report.converged,report.singular);
    temperature.correctBoundaryConditions();Residuals<scalar>::append(temperature.mesh(),performance);
    return performance;
}
}}
#endif
