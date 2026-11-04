#include "gpu/ImplicitDiffusion.H"
#include "gpu/SparseConjugateGradient.H"
#include "tests/TestSupport.H"
#include "materials/DryMaterialAdapter.H"
#include <iostream>
int main(){
 using namespace ugkp::gpu;
 DiffusionInput input;input.cells=1;input.owner={0};input.volume={1};
 input.gamma={2000};input.area={1};input.delta={1};
 input.gradientInternal={-1};input.gradientBoundary={300};
 std::vector<double> capacity={1000},rhs={-100000};
 DiffusionTransientInput transient{&capacity,&rhs,1,false};DiffusionMatrix matrix;
 ImplicitDiffusion assembly;assembly.assemble(input,matrix,&transient);
 const int owner=0,neighbour=-1;SparseLduView view;view.rows=1;view.boundaryFaces=1;
 view.boundaryOwner=&owner;view.boundaryNeighbour=&neighbour;
 view.diagonal=matrix.diagonal.data();view.source=matrix.source.data();
 view.boundaryDiagonal=matrix.boundaryDiagonal.data();view.boundarySource=matrix.boundarySource.data();
 SparseConjugateGradient solver;ConjugateGradientControls controls;controls.tolerance=1e-12;
 double temperature=300;auto report=solver.solveLdu(view,&temperature,controls);
 chmt_test::check(report.converged,"implicit CUDA conduction convergence");
 chmt_test::near(temperature,500./3,"implicit RHS temperature");
 chmt_test::near(1000*temperature+100000,2000*(300-temperature),"energy budget");
 auto physics=chmt_test::physics();physics.condensed[0].cp0=1000;physics.condensed[0].cp1=0;physics.condensed[0].e0=0;
 chmt::SolidQ state;state.condensed[0]=1;state.energy=1000*temperature;state.porosity=.5;
 chmt::MaterialPrimitive primitive;chmt_test::check(chmt::recoverMaterial(state,.002,physics,primitive),"final physical temperature");
 chmt_test::near(primitive.temperature,temperature,"final EOS");
 std::cout<<"PASS actual CUDA diffusion assembly and sparse solve: T="<<temperature<<" energy closure\n";
}
