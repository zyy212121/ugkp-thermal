"""T12 real cell-drag caller lineage for T06 finite Energy and T10 cache work."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT
SOURCE=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <limits>
#define __host__
#define __device__
#define __forceinline__ inline
#include "gasTransport/GasEnergy.H"
using Real=STORAGE;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) Real(x)
struct Dim{int x=0;};Dim blockIdx,threadIdx,blockDim{1};
template<class T>T clampMin(T a,T b){return std::max(a,b);}
template<class T>T clampRange(T a,T b,T c){return std::min(std::max(a,b),c);}
Real finiteOr(Real a,Real b){return std::isfinite(a)?a:b;}
template<class T>bool finiteDevice(T x){return std::isfinite(x);}
constexpr Real OfSmall=Real(1e-12);
Real sqr3(Real x,Real y,Real z){return x*x+y*y+z*z;}
struct DeviceState{
 int nCells=1;Real rhoMin=Real(1e-12),TgasMin=Real(.1),epsSMin=0,rhoSolid=4,particleDiameterFallback=.001,gammaGas=Real(1.4),Rgas=1;
 Real rho[1]={1},Ux[1]={1},Uy[1]={},Uz[1]={},Tgas[1]={300},rhoUx[1]={1},rhoUy[1]={},rhoUz[1]={};double rhoE[1]={1.4e300};
 Real couplingRhoOld[1]={-99},couplingUxOld[1]={-99},couplingUyOld[1]={-99},couplingUzOld[1]={-99},couplingTgasOld[1]={-99},thetaDragAlpha[1]={-99};
 Real momRhoP[1]={1},momRhoUPx[1]={1},momRhoUPy[1]={},momRhoUPz[1]={},momRhoEP[1]={Real(.5)},momRhoPD[1]={.001};};
Real solidEpsFromMomentDevice(const DeviceState&s,int){return s.momRhoP[0]/s.rhoSolid;}
template<int M>Real cellDragInverseTime(DeviceState&,Real,Real,Real,Real,int){return 1;}
template<int DragModel>void drag(DeviceState*sp,double dt,int dragModel){
#include "GpuCellDragUpdate.inl"
}
void close(long double x,long double expected){assert(std::abs(x-expected)<=std::abs(expected)*8e-15L+1e-12L);}
int main(){
 for(double dt:{0.,.25}){DeviceState s;s.rhoE[0]=1e300;const double energy=s.rhoE[0];Real before[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0]};drag<0>(&s,dt,0);
 assert(s.rhoE[0]==energy&&s.rho[0]==before[0]&&s.rhoUx[0]==before[1]&&s.rhoUy[0]==before[2]&&s.rhoUz[0]==before[3]);
 assert(s.couplingRhoOld[0]==1&&s.couplingUxOld[0]==1&&s.couplingUyOld[0]==0&&s.couplingUzOld[0]==0&&s.couplingTgasOld[0]==300);
 assert(s.thetaDragAlpha[0]==(dt==0?Real(1):Real(exp(Real(-2)*dt))));}
 // Equal velocities are insufficient for the strict-zero shortcut: real internal repair runs.
 {DeviceState s;s.rhoE[0]=0;drag<0>(&s,.25,0);assert(s.rhoE[0]>.5&&s.rhoUx[0]==1&&s.thetaDragAlpha[0]<1);}
 {DeviceState s;s.rhoUx[0]=s.Ux[0]=0;drag<0>(&s,.25,0);assert(s.rhoUx[0]>0&&s.rhoUx[0]<1);const long double igMin=.75L*s.Rgas*s.TgasMin/(s.gammaGas-1);
 const long double diss=.5L-(.5L*.75L*s.rhoUx[0]*s.rhoUx[0]+.5L*std::pow((1.-.75*s.rhoUx[0]),2));
 close(s.rhoE[0],(.5L*.75L*s.rhoUx[0]*s.rhoUx[0]+igMin+diss)/.75L);}
 {
  DeviceState s;const Real u=1e151;s.Ux[0]=s.rhoUx[0]=u;s.momRhoUPx[0]=-u;s.momRhoEP[0]=Real(.5)*u*u;s.rhoE[0]=u*u;
  const Real mg=.75,ms=1,alpha=exp(-.25*(1+ms/(mg+OfSmall))),w=2*u,us=(mg*u-u-mg*w*alpha)/(mg+ms),ug=us+w*alpha;
  const long double kOld=.5L*mg*u*u+.5L*ms*u*u,kNew=.5L*mg*ug*ug+.5L*ms*us*us;
  const long double diss=kOld-kNew,expected=(.5L*mg*ug*ug+mg*s.Rgas*s.TgasMin/(s.gammaGas-1))/.75L;
  assert(diss>1e300L&&std::isfinite(expected));drag<0>(&s,.25,0);close(s.rhoE[0],expected);close(s.rhoUx[0],ug);assert(s.rhoUy[0]==0&&s.rhoUz[0]==0);
 }
}
'''
def test_actual_drag_large_energy_zero_exchange_and_repair(tmp_path):
 src=tmp_path/'drag.cpp';src.write_text(SOURCE.replace('STORAGE','double'));exe=tmp_path/'drag'
 q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert q.returncode==0,q.stderr
 q=subprocess.run([str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stdout+q.stderr

def test_actual_zero_time_zero_exchange_and_repair_branch_counts(tmp_path):
    actual=(ROOT/'common/GpuCellDragUpdate.inl').read_text()
    for old,new in [
      ('    s.couplingRhoOld[c] =','    ++counts[0];s.couplingRhoOld[c] ='),
      ('    const Energy trueGasKineticDensity','    ++counts[1];const Energy trueGasKineticDensity'),
      ('    const GPU_OPERATOR_REAL alphaI =','    ++counts[2];const GPU_OPERATOR_REAL alphaI ='),
      ('    const GPU_OPERATOR_REAL kW =','    ++counts[3];const GPU_OPERATOR_REAL kW ='),
      ('    s.rho[c] = rhoG;','    ++counts[4];s.rho[c] = rhoG;'),
      ('        igOld = igMin;','        ++counts[5];igOld = igMin;')]:
        assert old in actual;actual=actual.replace(old,new)
    code='int counts[6]={};\n'+SOURCE.split('int main(){',1)[0].replace('#include "GpuCellDragUpdate.inl"',actual)
    code+=r'''
int main(){for(int mode:{0,1,2}){DeviceState s;for(int&v:counts)v=0;s.rhoE[0]=mode==2?0.:1e300;drag<0>(&s,mode==0?0.:.25,0);
const int expected[3][6]={{1,0,0,0,0,0},{1,1,1,0,0,0},{1,1,1,1,1,1}};
for(int j=0;j<6;++j)assert(counts[j]==expected[mode][j]);
std::printf("mode=%d cache=%d kinetic=%d alphaI=%d alphaW=%d publish=%d repair=%d\n",mode,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5]);}}
'''
    src=tmp_path/'counts.cpp';src.write_text(code.replace('STORAGE','double'));exe=tmp_path/'counts'
    q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stdout+q.stderr
    print(q.stdout)
