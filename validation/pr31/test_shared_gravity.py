from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_shared_gravity_keeps_physical_density(tmp_path):
 source=tmp_path/'gravity.cpp'
 source.write_text(r"""
#include <cassert>
#include <cmath>
#include "Diagnostics.H"
#define __device__
#define __global__
#define __forceinline__ inline
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
struct Index {int x;}; Index blockIdx{0},blockDim{1},threadIdx{0};
struct DeviceState {
 int nCells=1;double rho[1],rhoUx[1]{},rhoUy[1]{},rhoUz[1]{},rhoE[1]{7};
 double rhoMin=.001,gravityX=3,gravityY=0,gravityZ=0;
 ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
};
double clampMin(double a,double b){return a<b?b:a;}
double sqr3(double a,double b,double c){return a*a+b*b+c*c;}
#include "operators/applyGasGravityKernel.cuh"
int main(){
 for(double rho:{0.,1e-12,1e-3,1.}) {
  DeviceState a;a.rho[0]=rho;auto b=a;
  applyGasGravityKernel(&a,2.);applyGasOnlyGravityKernel(&b,2.);
  assert(a.rho[0]==rho && b.rho[0]==rho);
  assert(a.rhoUx[0]==6*rho && b.rhoUx[0]==a.rhoUx[0]);
  assert(std::abs(a.rhoE[0]-(7+18*rho))<1e-14);
  assert(a.rhoE[0]==b.rhoE[0]);
  if(rho>0)assert(std::abs(a.rhoE[0]-a.rhoUx[0]*a.rhoUx[0]/(2*rho)-7)<1e-14);
 }
}
""")
 exe=tmp_path/'gravity'
 subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)
