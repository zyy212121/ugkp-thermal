from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]/'common'
def test_low_re_omega_cell_constraint(tmp_path):
    src=(ROOT/'operators/computeGasPrimitiveGradientsKernel.cuh').read_text()
    functions=src[src.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue'):src.index('template<class GasState>\n__global__ void initialiseSstConservativeStateKernel')]
    recovery=src[src.index('template<class GasState>\n__global__ void recoverSstPrimitivesKernel'):]
    flux_src=(ROOT/'operators/computeSstFaceFluxKernel.cuh').read_text()
    update=flux_src[flux_src.index('template<class GasState>\n__global__ void applySstFluxAndSourceKernel'):flux_src.index('template<class GasState>\n__global__ void computeGasCourantFieldKernel')]
    pre=r'''
#include <cmath>
#include <algorithm>
#include <iostream>
#include "GpuSstAlgebra.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "gasTransport/GasGeometryValidation.H"
#define __device__
#define __global__
#define GPU_OPERATOR_TIME GpuReal
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
using R=GpuReal;
struct Index{int x=0;};Index blockIdx,threadIdx;struct Block{int x=1;}blockDim;
const R OfVSmall=R(1e-30),OfSmall=R(1e-15);
R clampMin(R a,R b){return std::max(a,b);}R finiteOr(R a,R b){return std::isfinite(a)?a:b;}
bool finiteDevice(R value){return std::isfinite(value);}
struct DeviceState{
 int faceOwner[2]={0,0};
 R sstPhiRhoK[2]={.1,.2},sstPhiRhoOmega[2]={3,7},V[1]={1},sstSourceNumber[1]={0},sstF1[1]={1},sstF2[1]={1};
 R gradKX[1]={0},gradKY[1]={0},gradKZ[1]={0},gradOmegaX[1]={0},gradOmegaY[1]={0},gradOmegaZ[1]={0};
 int nCells=1,nFaces=2,nInternalFaces=0,sstConfigured=1,sstWallTreatment=0;
 int riemannBoundaryKind[2]={2,2},riemannBoundaryUFix[2]={1,1},cellPlaneStart[1]={0},cellPlaneCount[1]={2},cellFaceId[2]={0,1};
 int sstBoundaryOmegaMode[2]={0,0},sstBoundaryKMode[2]={0,0};
 R rho[1]={1},rhoMin=1e-12,gasMu=1.8e-5,wallRho[2]={4,8};
 R Ux[1]={2},Uy[1]={0},Uz[1]={0},riemannBoundaryUx[2]={0,0},riemannBoundaryUy[2]={0,0},riemannBoundaryUz[2]={0,0};
 R Sfx[2]={1,1},Sfy[2]={0,0},Sfz[2]={0,0};
 R k[1]={.1},omega[1]={7},rhoOmega[1]={7},rhoK[1]={.1},sstWallDistance[1]={.001};
 R sstWallCmu=.09,sstWallKappa=.41,sstWallE=9.8,sstOmegaMin=1e-8,sstKMin=1e-10;
 R sstBoundaryOmega[2]={42,42},sstBoundaryK[2]={.2,.2};
 ugkwp::SstCoefficients sstCoefficients=ugkwp::defaultSstCoefficients();
};
struct Prim{R rho;};Prim riemannFacePrimitiveForGradient(const DeviceState&s,int,int f){return {s.wallRho[f]};}
bool isPeriodicFace(const DeviceState&,int){return false;}
'''
    stubs=r'''
void sstVelocityInvariants(const DeviceState&,int,R&d,R&s,R&g){d=s=g=0;}
R sstKProductionForCell(const DeviceState&,int,R){return 0;}
'''
    body=pre+functions+recovery+stubs+update+r'''
int failures=0;void check(const char*name,R got,R want){if(std::abs(got-want)>(sizeof(R)==4?3e-5:1e-11)*std::max(R(1),std::abs(want))){std::cerr<<name<<" got="<<got<<" expected="<<want<<"\n";++failures;}}
int main(){
 DeviceState s;applySstWallFunctionStateKernel(&s);check("two-face corner target",s.omega[0],270);check("conservative",s.rhoOmega[0],270);
 check("omega copied to wall",sstBoundaryValue(s,0,0,true),s.omega[0]);check("wall omega normal gradient",1000*(sstBoundaryValue(s,0,0,true)-s.omega[0]),0);check("wall k zero",sstBoundaryValue(s,0,0,false),0);
 s.rho[0]=2;s.rhoOmega[0]=18;recoverSstPrimitivesKernel(&s);check("projection after stage/RK recovery",s.omega[0],270);check("updated conservative density",s.rhoOmega[0],540);
 s.cellPlaneCount[0]=1;applySstWallFunctionStateKernel(&s);check("single face wall nu",s.omega[0],360);
 s.riemannBoundaryKind[0]=0;s.rhoOmega[0]=18;recoverSstPrimitivesKernel(&s);check("interior remains evolved",s.omega[0],9);
 s=DeviceState();s.sstWallTreatment=1;R expected=(sstDynamicOmegaWallValue(s,0,0)+sstDynamicOmegaWallValue(s,1,0))/2;applySstWallFunctionStateKernel(&s);check("highRe unchanged",s.omega[0],expected);s.rhoOmega[0]=18;recoverSstPrimitivesKernel(&s);check("highRe recovery constrained",s.omega[0],expected);
 s=DeviceState();s.sstConfigured=0;applySstWallFunctionStateKernel(&s);check("disabled unchanged",s.omega[0],7);
 s=DeviceState();applySstWallFunctionStateKernel(&s);R oldOmega=s.rhoOmega[0],oldK=s.rhoK[0];applySstFluxAndSourceKernel(&s,R(1e-4));check("constrained equation rejects omega flux/source",s.rhoOmega[0],oldOmega);if(s.rhoK[0]==oldK){std::cerr<<"k equation was frozen\n";++failures;}
 s.riemannBoundaryKind[0]=s.riemannBoundaryKind[1]=0;applySstFluxAndSourceKernel(&s,R(1e-4));if(s.rhoOmega[0]==oldOmega){std::cerr<<"interior omega equation was frozen\n";++failures;}
 return failures?1:0;
}
'''
    source=tmp_path/'test.cpp';source.write_text(body)
    for bits in (64,32):
        exe=tmp_path/f'probe{bits}'
        subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT),'-I'+str(ROOT/'gasNumerics'),str(source),'-o',str(exe)],check=True)
        p=subprocess.run([str(exe)],capture_output=True,text=True)
        assert p.returncode==0, f'FP{bits}: {p.stderr}'

