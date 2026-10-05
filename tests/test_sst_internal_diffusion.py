"""Numerically regress rho*D interpolation and its stability bound.

Compiles production flux/stability blocks against controlled two-cell states.
Correlated density/F1/nut and non-midpoint weights catch the old order of
operations; literal expectations also cover omega-dominated stability,
orientation, periodic faces, molecular limits and unchanged boundaries.
"""
from pathlib import Path
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1] / 'common'

def test_sst_internal_diffusion_and_stability(tmp_path):
    W = tmp_path
    src = (ROOT / 'operators/computeSstFaceFluxKernel.cuh').read_text()
    helpers=src[:src.index('__global__ void computeSstFaceFluxKernel')].replace('#pragma once','')
    flux=src[src.index('    const GPU_OPERATOR_REAL rhoFace'):src.index('    GPU_OPERATOR_REAL snGradK')]
    tail=src[src.index('    const GPU_OPERATOR_REAL area = s.magSf'):src.index('\n}\n',src.index('    const GPU_OPERATOR_REAL area = s.magSf'))]
    st=src[src.index('        const int other =',src.index('__global__ void computeSstStabilityNumberKernel')):src.index('    const GPU_OPERATOR_REAL diffusionNumber')].rsplit('    }',1)[0]
    preamble=r'''
    #include <cmath>
    #include <iostream>
    #include <iomanip>
    #include <algorithm>
    #include "GpuSstAlgebra.cuh"
    #include "OpenFoamWallFunctions.cuh"
    #define GPU_OPERATOR_REAL GpuReal
    #define GPU_OPERATOR_R GPU_R
    #define __device__
    using R=GpuReal;
    R clampMin(R a,R b){return std::max(a,b);} R clampRange(R a,R b,R c){return std::max(b,std::min(a,c));} R finiteOr(R a,R b){return std::isfinite(a)?a:b;}
    struct DeviceState {
     int faceOwner[1]={0},faceNeighbour[1]={1},riemannBoundaryKind[1]={0},riemannBoundaryUFix[1]={1};
     int nInternalFaces=1,sstWallTreatment=0; bool periodic=false;
     R rho[2]={2,8},sstF1[2]={1,0},nut[2]={3,1},faceWeight[1]={.25},gasMu=.2,rhoMin=1e-6;
     R riemannBoundaryUx[1]={0},riemannBoundaryUy[1]={0},riemannBoundaryUz[1]={0};
     R Ux[2]={15,20},Uy[2]={0,0},Uz[2]={0,0},sstWallDistance[2]={.01,.02};
     R sstWallKappa=.41,sstWallE=9.8,magSf[1]={1},deltaCoeffs[1]={1};
     R sstPhiRhoK[1],sstPhiRhoOmega[1],boundaryRho=4;
     ugkwp::SstCoefficients sstCoefficients=ugkwp::defaultSstCoefficients();
    };
    bool isPeriodicFace(const DeviceState&s,int){return s.periodic;}
    int coupledFaceNeighbour(const DeviceState&s,int f){return (f<s.nInternalFaces||s.periodic)?s.faceNeighbour[f]:-1;}
    struct Prim{R rho;}; Prim riemannFacePrimitiveForGradient(const DeviceState&s,int,int){return {s.boundaryRho};}
    '''
    body=preamble+helpers+'\nvoid flux(DeviceState&s){int f=0,own=0,nei=coupledFaceNeighbour(s,f),boundaryKind=nei>=0?0:s.riemannBoundaryKind[f];R ownerWeight=clampRange(s.faceWeight[f],R(0),R(1));\n'+flux+'R massFlux=0,kUpwind=0,omegaUpwind=0,snGradK=-1,snGradOmega=-1;\n'+tail+'\n}\nR stability(DeviceState&s,int c){int f=0;R diffusionRate=0;\n'+st+'return diffusionRate;}\n'
    body+=r'''
    int failures=0; void check(const char*n,double got,double want){double tol=sizeof(R)==4?3e-6:2e-13;if(std::abs(got-want)>tol*std::max(1.,std::abs(want))){std::cerr<<n<<" got "<<got<<" want "<<want<<std::endl;++failures;}}
    int main(int argc,char**argv){
     if(argc>1 && std::string(argv[1])=="boundary") {std::cout<<std::setprecision(17);for(int kind:{0,2})for(int treatment:{0,1})for(double rho:{2.,8.}) {DeviceState s;s.nInternalFaces=0;s.riemannBoundaryKind[0]=kind;s.sstWallTreatment=treatment;s.rho[0]=rho;flux(s);std::cout<<s.sstPhiRhoK[0]<<" "<<s.sstPhiRhoOmega[0]<<" "<<stability(s,0)<<std::endl;}return 0;}
    
     if(argc>1){DeviceState s;double ro,rn,fo,fn,no,nn,w;std::cout<<std::setprecision(17);while(std::cin>>ro>>rn>>fo>>fn>>no>>nn>>w){s.rho[0]=ro;s.rho[1]=rn;s.sstF1[0]=fo;s.sstF1[1]=fn;s.nut[0]=no;s.nut[1]=nn;s.faceWeight[0]=w;s.gasMu=1.8e-5;flux(s);std::cout<<s.sstPhiRhoK[0]<<" "<<s.sstPhiRhoOmega[0]<<" "<<stability(s,0)<<" "<<stability(s,1)<<std::endl;}return 0;}
     DeviceState s;flux(s);check("correlated endpoints k",s.sstPhiRhoK[0],7.475);check("correlated endpoints omega",s.sstPhiRhoOmega[0],6.086);
     check("owner diffusion rate",stability(s,0),3.7375);check("neighbour diffusion rate",stability(s,1),.934375);
     s.nInternalFaces=0;s.periodic=true;flux(s);check("periodic k",s.sstPhiRhoK[0],7.475);check("periodic omega",s.sstPhiRhoOmega[0],6.086);check("periodic rate",stability(s,0),3.7375);
     s=DeviceState();std::swap(s.rho[0],s.rho[1]);std::swap(s.nut[0],s.nut[1]);std::swap(s.sstF1[0],s.sstF1[1]);s.faceWeight[0]=.75;flux(s);check("orientation k",s.sstPhiRhoK[0],7.475);check("orientation omega",s.sstPhiRhoOmega[0],6.086);
     s=DeviceState();s.faceWeight[0]=0;flux(s);check("weight zero k",s.sstPhiRhoK[0],8.2);check("weight zero omega",s.sstPhiRhoOmega[0],7.048);
     s.faceWeight[0]=1;flux(s);check("weight one k",s.sstPhiRhoK[0],5.3);check("weight one omega",s.sstPhiRhoOmega[0],3.2);
     s=DeviceState();s.nut[0]=s.nut[1]=0;flux(s);check("molecular k",s.sstPhiRhoK[0],.2);check("molecular omega",s.sstPhiRhoOmega[0],.2);check("molecular owner rate",stability(s,0),.1);
     s=DeviceState();s.nInternalFaces=0;s.riemannBoundaryKind[0]=2;flux(s);check("lowRe wall k",s.sstPhiRhoK[0],.2);check("lowRe wall omega",s.sstPhiRhoOmega[0],.2);check("lowRe wall rate",stability(s,0),.1);
     s.riemannBoundaryKind[0]=0;flux(s);check("open boundary k",s.sstPhiRhoK[0],5.3);check("open boundary omega",s.sstPhiRhoOmega[0],3.2);check("open boundary rate",stability(s,0),2.65);
     s=DeviceState();s.sstCoefficients.alphaOmega1=2;s.sstCoefficients.alphaOmega2=2;flux(s);check("omega dominates flux",s.sstPhiRhoOmega[0],15.2);check("omega dominates stability",stability(s,0),7.6);
     s=DeviceState();s.rho[1]=s.rho[0];s.nut[1]=s.nut[0];s.sstF1[1]=s.sstF1[0];flux(s);check("uniform k",s.sstPhiRhoK[0],5.3);check("uniform omega",s.sstPhiRhoOmega[0],3.2);
     s=DeviceState();s.rhoMin=4;s.nut[0]=s.nut[1]=0;flux(s);check("cell density floor",s.sstPhiRhoK[0],.175);check("cell density floor rate",stability(s,0),.04375);
     std::cout<<"failures="<<failures<<std::endl;return failures?1:0;
    }
    '''
    (W/'probe.cpp').write_text(body)
    results=[]
    for bits in (64,32):
     exe=W/f'probe{bits}'
     subprocess.run(['g++','-std=c++14','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}', '-I'+str(ROOT),'-I'+str(ROOT/'gasNumerics'),str(W/'probe.cpp'),'-o',str(exe)],check=True)
     p=subprocess.run([str(exe)],capture_output=True,text=True)
     results.append({'bits':bits,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    assert all(r['exit'] == 0 for r in results), json.dumps(results, indent=2)
