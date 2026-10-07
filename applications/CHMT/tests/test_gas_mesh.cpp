// Real geometry plus host gas-flux arithmetic; no CUDA kernel execution.
#define main existing_sweep_fixture_main
#include "../devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "tests/TestSupport.H"
#include "gas/AleFlux.H"
int main(){
 auto p=chmt_test::physics();auto w=chmt_test::gas(p);auto old=twoHexahedra(0);auto points=old.points;
 for(auto& x:points)x={1.1*x.x+.05*x.y+.1,.9*x.y+.07*x.z-.1,1.2*x.z+.02*x.x};
 HostMesh next;std::vector<Real>sweep;std::string error;const double dt=.03;
 require(makeStageGeometry(old,points,dt,next,sweep,error),error);require(maximumGclResidual(old,next,sweep)<1e-12,"independent moving-grid GCL");
 const GasQ density=conservativeGas(w,1,p);std::vector<GasQ>flux(old.owner.size());
 for(size_t f=0;f<flux.size();++f){require(aleRusanov(w,w,next.areaVectors[f],sweep[f]/dt,p,flux[f]),"ALE equal-state flux");GasQ reverse;require(aleRusanov(w,w,-next.areaVectors[f],-sweep[f]/dt,p,reverse),"reverse face flux");chmt_test::near(flux[f].mass,-reverse.mass,"face mass antisymmetry");chmt_test::near(flux[f].energy,-reverse.energy,"face energy antisymmetry");}
 for(size_t cell=0;cell<old.volumes.size();++cell){GasQ q=density*old.volumes[cell];for(int e=old.cellFaceOffsets[cell];e<old.cellFaceOffsets[cell+1];++e)q=q-flux[old.cellFaces[e]]*(dt*old.cellFaceSigns[e]);
  chmt_test::near(q.mass/next.volumes[cell],density.mass,"ALE uniform density");chmt_test::near(q.energy/next.volumes[cell],density.energy,"ALE uniform total energy");chmt_test::near(mag(q.momentum/next.volumes[cell]-density.momentum),0,"ALE uniform momentum",1e-10,1e-9);
  GasPrimitive recovered;require(recoverGas(q,next.volumes[cell],p,recovered),"ALE state EOS");chmt_test::near(recovered.temperature,w.temperature,"ALE uniform temperature");}
 GasPrimitive bad=w;bad.Y[0]=-.1;GasQ out;out.mass=123;require(!aleRusanov(w,bad,{1,0,0},0,p,out),"invalid gas face refuses");require(out.mass==123,"face rejection preserves output");
 Symmetric3 matrix;Vec3 rhs;const Vec3 exact={2,-3,4};for(Vec3 d:std::vector<Vec3>{{1,.2,.3},{-.4,1,.5},{.2,-.7,1}})addLeastSquares(matrix,rhs,d,dot(d,exact));chmt_test::near(mag(solveLeastSquares(matrix,rhs)-exact),0,"oblique full-rank gradient",1e-12,1e-12);
 std::cout<<"gas/mesh host regressions: actual polyhedral ALE free stream, face symmetry, gradient passed\n";
}
