"""The Courant mass-only route must match the complete production Riemann operator."""
from pathlib import Path
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('bits',[32,64])
def test_mass_only_matches_full_operator(tmp_path,bits):
    src=(ROOT/'common/operators/computeRiemannGasFaceFluxDevice.cuh').read_text()
    assert 'bool MassOnly' in src, 'Courant needs a mass-only production specialization'
    flux=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    courant=flux[flux.index('template<class GasState>\n__global__ void computeGasCourantFieldKernel'):flux.index('template<class GasState>\n__global__ void computeGasDiffusionNumberKernel')]
    primitive=(ROOT/'common/operators/computeGasPrimitiveGradientsKernel.cuh').read_text().split('template<class GasState>\n__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue')[0]
    sensor=(ROOT/'common/operators/computeGasHllcAdcSensorKernel.cuh').read_text()
    full_flux=(ROOT/'common/operators/computeGasInternalFaceFluxKernel.cuh').read_text().split('template<class GasState>\n__global__ void enforcePeriodicGasFluxAntisymmetryKernel')[0]
    fields_source=src+courant+primitive+sensor+full_flux
    arrays=sorted(set(re.findall(r's\.(\w+)\[',fields_source))|{'faceNeighbour'})
    scalars=sorted(set(re.findall(r's\.(\w+)',fields_source))-set(arrays)-{'gasSpecies','gasGeometry','gasSstAudit'})
    int_arrays={'faceOwner','faceNeighbour','riemannBoundaryKind','riemannBoundaryTFix','riemannBoundaryUFix','cellPlaneStart','cellPlaneCount','cellFaceId'}
    int_scalars={'nFaces','nCells','gasReconstruction','gasFluxScheme','nInternalFaces','sstConfigured'}
    fields='\n'.join(('int' if x in int_arrays else 'R')+' '+x+'[2]={};' for x in arrays)
    fields+='\n'+'\n'.join(('int' if x in int_scalars else 'R')+' '+x+'=0;' for x in scalars)
    pre=r'''
#include <cmath>
using std::isfinite;
#include <algorithm>
#include <iostream>
#include "RiemannGasFlux.cuh"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#define __host__
#define __device__
#define __global__
#define GPU_OPERATOR_TIME GpuReal
#include "GpuCellNeighbour.cuh"
struct Idx{int x=0;}; Idx blockIdx,threadIdx;struct Block{int x=1;}blockDim;
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
using R=GpuReal;const R OfSmall=R(1e-15),OfGreat=R(1e30);
R clampMin(R x,R y){return std::max(x,y);}R clampRange(R x,R lo,R hi){return std::max(lo,std::min(x,hi));}R finiteOr(R x,R y){return std::isfinite(x)?x:y;}
bool finiteDevice(R x){return std::isfinite(x);}
struct DeviceState {FIELDS};
bool isPeriodicFace(const DeviceState&,int){return false;}
struct GasPrimDevice{R rho,ux,uy,uz,p,T;};
GasPrimDevice makeGasPrimDevice(R rho,R ux,R uy,R uz,R p,R gas,R rmin,R tmin){rho=clampMin(rho,rmin);p=clampMin(p,rho*gas*tmin);return{rho,ux,uy,uz,p,p/(rho*gas)};}
GasPrimDevice gasCellPrimitive(const DeviceState&s,int c){return makeGasPrimDevice(s.rho[c],s.Ux[c],s.Uy[c],s.Uz[c],s.p[c],s.Rgas,s.rhoMin,s.TgasMin);}
GasPrimDevice reconstructGasCellToFace(const DeviceState&s,int c,int){return makeGasPrimDevice(s.rho[c],s.Ux[c],s.Uy[c],s.Uz[c],s.p[c],s.Rgas,s.rhoMin,s.TgasMin);}
GasPrimDevice riemannFacePrimitiveForGradient(const DeviceState&s,int c,int){return reconstructGasCellToFace(s,c,0);}
GasPrimDevice riemannExteriorStateForFace(const DeviceState&s,int,const GasPrimDevice&){return reconstructGasCellToFace(s,1,0);}
GasPrimDevice riemannBoundaryState(const DeviceState&s,int,const GasPrimDevice&){return reconstructGasCellToFace(s,1,0);}
int coupledFaceNeighbour(const DeviceState&s,int f){return s.faceNeighbour[f];}
void periodicMappedCellCentre(const DeviceState&s,int,int c,R&x,R&y,R&z){x=s.Cx[c];y=s.Cy[c];z=s.Cz[c];}
bool useRiemannBoundaryVelocity(const DeviceState&,int,const GasPrimDevice&){return false;}
R molecularGasConductivity(const DeviceState&){return R(.07);}
void gasFaceSubgridTransportProperties(const DeviceState&,int,int,int,int,R,R&mu,R&kt,R&q,int&a){mu=R(.03);kt=R(.05);q=0;a=0;}
'''.replace('FIELDS',fields)
    body=pre+(src+courant+sensor+primitive+full_flux).replace('asm("trap;");','std::abort();')+r'''
int main(){int fails=0,count=0;for(int scheme=1;scheme<=9;++scheme){ugkpriemann::Scheme kind;if(!ugkpriemann::schemeFromCreateCode(scheme,kind))continue;
for(int reconstruction:{0,1,2})for(int boundary:{-1,0,1,2,3,4})for(int state=0;state<8;++state){
 DeviceState s;s.nFaces=1;s.nCells=2;s.faceOwner[0]=0;s.faceNeighbour[0]=boundary<0?1:-1;s.riemannBoundaryKind[0]=std::max(boundary,0);s.gasFluxScheme=scheme;s.gasReconstruction=reconstruction;
 s.rho[0]=R(.5)+R(.3)*state;s.rho[1]=R(2.4)-R(.2)*state;s.p[0]=R(1)+R(.7)*state;s.p[1]=R(5)-R(.3)*state;
 s.Ux[0]=state-4;s.Ux[1]=2-state;s.Uy[0]=R(.3);s.Uy[1]=R(-.7);s.Uz[0]=R(.2);s.Uz[1]=R(.9);
 s.Rgas=1;s.rhoMin=R(1e-8);s.TgasMin=R(1e-6);s.gammaGas=R(1.4);s.gasCp=R(3.5);s.gasMu=R(.1);
 s.Tgas[0]=s.p[0]/s.rho[0];s.Tgas[1]=s.p[1]/s.rho[1];s.magSf[0]=R(1.7);s.Sfx[0]=s.magSf[0]*R(.6);s.Sfy[0]=s.magSf[0]*R(.8);s.Cx[1]=1;s.Cy[1]=1;s.faceWeight[0]=R(.3);s.deltaCoeffs[0]=R(.8);s.gasHllcAdcSensor[0]=R(.2);s.gasHllcAdcSensor[1]=R(.7);
 s.gradRhoX[0]=R(.1);s.gradRhoY[1]=R(-.1);s.gradUxY[0]=R(.2);s.gradUyX[1]=R(.5);s.gradTX[0]=R(.3);
 R full,x,y,z,e,mass,mx,my,mz,me;bool a=computeRiemannGasFaceFluxDevice<true>(s,0,full,x,y,z,e);bool b=computeRiemannGasFaceFluxDevice<false,true>(s,0,mass,mx,my,mz,me);
 if(a!=b||!std::isfinite(full)||full!=mass){std::cerr<<"scheme="<<scheme<<" reconstruction="<<reconstruction<<" boundary="<<boundary<<" full="<<full<<" mass="<<mass<<"\n";++fails;}++count;
 // Poisoned scratch must be overwritten by fresh actual mass and acoustic data.
 s.sstConfigured=1;s.nInternalFaces=boundary<0?1:0;s.gasPhiRho[0]=12345;s.gasPhiRhoE[0]=-12345;
 computeGasCourantFieldKernel(&s,R(.01));
 if(s.gasPhiRho[0]!=full || !(s.gasPhiRhoE[0]>=0)){std::cerr<<"Courant fresh mass/acoustics mismatch\n";++fails;}
 s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=1;s.cellFaceId[0]=0;s.V[0]=2;
 R wantCo=R(.5)*R(.01)*s.gasPhiRhoE[0]/s.V[0];computeGasConvectiveCourantByCellKernel(&s,R(.01));
 if(std::abs(s.gasFluxPositivityScale[0]-wantCo)>R(1e-6) || s.gasPhiRho[0]!=full){std::cerr<<"Courant scratch lifetime mismatch\n";++fails;}
 // Laminar Courant has no mass-flux consumer: it must leave that array alone.
 // Its acoustic scratch and the next complete flux still match the SST route.
 {
  DeviceState laminar=s;laminar.sstConfigured=0;
  laminar.gasPhiRho[0]=R(12345);laminar.gasPhiRhoE[0]=R(-12345);
  computeGasCourantFieldKernel(&laminar,R(.01));
  if(laminar.gasPhiRho[0]!=R(12345) || laminar.gasPhiRhoE[0]!=s.gasPhiRhoE[0]){std::cerr<<"laminar Courant touched unused mass or changed acoustics\n";++fails;}
  computeGasConvectiveCourantByCellKernel(&laminar,R(.01));
  if(laminar.gasFluxPositivityScale[0]!=s.gasFluxPositivityScale[0]){std::cerr<<"laminar Courant changed\n";++fails;}
  computeGasInternalFaceFluxKernel<false>(&laminar,R(.01));
  if(laminar.gasPhiRho[0]!=full){std::cerr<<"next full flux retained stale mass\n";++fails;}
  DeviceState invalid=s;invalid.faceOwner[0]=-1;invalid.gasPhiRho[0]=R(12345);
  computeGasCourantFieldKernel(&invalid,R(.01));
  if(invalid.gasPhiRho[0]!=R(0) || invalid.gasPhiRhoE[0]!=OfGreat){std::cerr<<"SST early return retained stale mass\n";++fails;}
 }
 if(scheme==7){
  computeGasHllcAdcSensorKernel(&s);R expected=s.gasHllcAdcSensor[0];
  s.gasHllcAdcSensor[0]=R(-1);computeGasPrimitiveGradientsKernel(&s,true);
  if(s.gasHllcAdcSensor[0]!=expected){std::cerr<<"fused sensor differs from production standalone\n";++fails;}
  s.gasHllcAdcSensor[0]=R(.123);computeGasPrimitiveGradientsKernel(&s);
  if(s.gasHllcAdcSensor[0]!=R(.123)){std::cerr<<"normal advance sensor changed\n";++fails;}
 }

}}
std::cout<<"cases="<<count<<" failures="<<fails<<"\n";return fails?1:0;}
'''
    path=tmp_path/'probe.cpp';path.write_text(body.replace('#pragma once',''));exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(path),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
