"""Host verification of the shared ALE frame/metric policy, using shared fluxes.

These are geometry and Riemann-header tests, not CUDA/native-solver evidence.
All expected ALE fluxes and GCL results are independently analytic.
"""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / 'common/gasTransport/GasGeometryPolicy.H'

SOURCE = r'''
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "gasNumerics/RiemannGasFlux.cuh"
#include "gasTransport/GasGeometryPolicy.H"
using Real=GpuReal;
using namespace ugkwp;
using namespace ugkpriemann;
void check(bool good,const char*message){if(!good){std::fprintf(stderr,"%s\n",message);std::exit(1);}}
void near(Real actual,Real expected,const char*message){
 const Real tolerance=Real(300)*std::numeric_limits<Real>::epsilon()*std::fmax(Real(1),std::abs(expected));
 if(!(std::abs(actual-expected)<=tolerance)){std::fprintf(stderr,"%s: %.17g != %.17g\n",message,double(actual),double(expected));std::exit(1);}
}
GasFaceFrame<Real> frame(Real sweep=Real(.21),Real dt=Real(.01),Real sign=1,Real tx=0,Real ty=0,Real tz=0){
 GasFaceFrame<Real> f;check(makeGasFaceFrame(sign*2,sign*-3,sign*6,sign*sweep,dt,f,tx,ty,tz),"frame creation");return f;
}
Primitive primitive(){return {Real(1.25),Real(19),Real(-7),Real(4),Real(81000)};}
Conservative state(Primitive p,Real internal=-2300000){
 return {{p.rho,p.rho*p.ux,p.rho*p.uy,p.rho*p.uz,internal+Real(.5)*p.rho*(p.ux*p.ux+p.uy*p.uy+p.uz*p.uz)}};
}
FluxResult physicalFlux(const Primitive&l,const Conservative&ul,const Primitive&r,const Conservative&ur,const GasFaceFrame<Real>&f){
 Primitive lm,rm;Conservative uml,umr;
 check(shiftGasStateToFaceFrame(l,ul,f,lm,uml),"left shift");check(shiftGasStateToFaceFrame(r,ur,f,rm,umr),"right shift");
 FluxResult moving=rusanovTadmorFluxUnitNormal(lm,uml,Real(300)+l.rho,rm,umr,Real(300)+r.rho,f.nx,f.ny,f.nz,false), result;
 check(restoreGasFluxFromFaceFrame(moving,f,result),"flux restore");
 for(int k=0;k<5;++k) {result.flux[k]*=f.area;}
 return result;
}
void equal_state(){
 auto p=primitive();auto q=state(p);auto f=frame();auto flux=physicalFlux(p,q,p,q,f);
 Real un=p.ux*f.nx+p.uy*f.ny+p.uz*f.nz,wn=f.meshVolumeRate/f.area;
 Real expected[5]={p.rho*(un-wn),q.q[1]*(un-wn)+p.p*f.nx,q.q[2]*(un-wn)+p.p*f.ny,q.q[3]*(un-wn)+p.p*f.nz,q.q[4]*(un-wn)+p.p*un};
 for(int k=0;k<5;++k)near(flux.flux[k],expected[k]*f.area,"physical Euler-minus-mesh flux");
 check(flux.flux[4]<0,"formation-inclusive negative total energy must remain legal");
}
void zero_motion(){
 Primitive l=primitive(),r{Real(.7),Real(-11),Real(3),Real(2),Real(53000)};auto f=frame(0);
 Primitive lm,rm;Conservative ul=conservative(l,Real(1.4)),ur=conservative(r,Real(1.4)),uml,umr;
 check(shiftGasStateToFaceFrame(l,ul,f,lm,uml),"stationary left shift");check(shiftGasStateToFaceFrame(r,ur,f,rm,umr),"stationary right shift");
 check(std::memcmp(&l,&lm,sizeof l)==0&&std::memcmp(&ul,&uml,sizeof ul)==0,"zero motion state identity");
 for(int code=1;code<=8;++code){Scheme scheme;check(schemeFromCreateCode(code,scheme),"scheme code");
 auto original=fluxUnitArea(l,r,f.nx,f.ny,f.nz,Real(1.4),scheme);auto moving=fluxUnitArea(lm,rm,f.nx,f.ny,f.nz,Real(1.4),scheme);FluxResult restored;
 check(original.valid&&restoreGasFluxFromFaceFrame(moving,f,restored),"stationary scheme validity");
 check(std::memcmp(original.flux,restored.flux,sizeof original.flux)==0,"zero-motion changes selected scheme flux");
 check(restored.evaluatedScheme==original.evaluatedScheme&&restored.usedFallback==original.usedFallback&&restored.maxSignalSpeed==original.maxSignalSpeed,"flux metadata lost");}
}
void constant_gamma_schemes(){
 Primitive p=primitive(),shifted;auto q=conservative(p,Real(1.4));Conservative shiftedQ;auto f=frame();
 check(shiftGasStateToFaceFrame(p,q,f,shifted,shiftedQ),"caloric state shift");DensityGradient gradient{0,0,0};
 const Real un=p.ux*f.nx+p.uy*f.ny+p.uz*f.nz,relative=un-f.meshVolumeRate/f.area;
 Real expected[5]={p.rho*relative,q.q[1]*relative+p.p*f.nx,q.q[2]*relative+p.p*f.ny,q.q[3]*relative+p.p*f.nz,q.q[4]*relative+p.p*un};
 for(int code=1;code<=9;++code){Scheme scheme;check(schemeFromCreateCode(code,scheme),"all scheme codes");
 auto moving=code==9?slau22FluxUnitNormal(shifted,shifted,gradient,gradient,f.nx,f.ny,f.nz,Real(1.4),Real(1e-12),Real(1e-12)):fluxUnitArea(shifted,shifted,f.nx,f.ny,f.nz,Real(1.4),scheme);
 FluxResult result;check(restoreGasFluxFromFaceFrame(moving,f,result),"selected scheme ALE transform");check(result.evaluatedScheme==scheme,"ALE changed flux scheme");for(int k=0;k<5;++k)near(result.flux[k],expected[k],"selected scheme equal-state ALE flux");}
 auto stationary=frame(0);auto original=slau22FluxUnitNormal(p,p,gradient,gradient,stationary.nx,stationary.ny,stationary.nz,Real(1.4),Real(1e-12),Real(1e-12));FluxResult result;
 check(restoreGasFluxFromFaceFrame(original,stationary,result),"stationary SLAU2.2 transform");check(std::memcmp(original.flux,result.flux,sizeof original.flux)==0,"SLAU2.2 zero-motion identity");
}
void tangential_frame(){
 auto p=primitive();auto q=state(p);auto f=frame();auto tangent=frame(Real(.21),Real(.01),1,3,2,0);
 auto a=physicalFlux(p,q,p,q,f),b=physicalFlux(p,q,p,q,tangent);for(int k=0;k<5;++k)near(a.flux[k],b.flux[k],"tangential frame changed flux");
 Primitive r{Real(.9),Real(-9),Real(2),Real(5),Real(70000)};auto ur=state(r,Real(1500000));
 a=physicalFlux(p,q,r,ur,f);b=physicalFlux(p,q,r,ur,tangent);for(int k=0;k<5;++k)near(a.flux[k],b.flux[k],"tangential frame changed jump dissipation");
 Primitive shifted;Conservative shiftedQ;check(shiftGasStateToFaceFrame(p,q,tangent,shifted,shiftedQ),"formation state shift");
 Real internal=shiftedQ.q[4]-Real(.5)*shifted.rho*(shifted.ux*shifted.ux+shifted.uy*shifted.uy+shifted.uz*shifted.uz);
 near(internal,Real(-2300000),"frame changed internal formation energy");
}
void orientation(){
 auto l=primitive();Primitive r{Real(.9),Real(-9),Real(2),Real(5),Real(70000)};auto ul=state(l),ur=state(r,Real(1500000));
 auto f=frame(),reverse=frame(Real(.21),Real(.01),-1);auto a=physicalFlux(l,ul,r,ur,f),b=physicalFlux(r,ur,l,ul,reverse);
 near(f.wx,reverse.wx,"orientation changed mesh velocity");near(f.wy,reverse.wy,"orientation changed mesh velocity");near(f.wz,reverse.wz,"orientation changed mesh velocity");
 for(int k=0;k<5;++k)near(a.flux[k],-b.flux[k],"paired face antisymmetry");
}
void uniform_gcl(){
 const Real oldV=2,newV=Real(2.03),dt=Real(.01);auto p=primitive();auto q=state(p);Real density[7]={q.q[0],q.q[1],q.q[2],q.q[3],q.q[4],Real(.25),Real(1)};
 Real sums[7]={},sweeps=0;
 for(int face=0;face<6;++face){Real normal[3]={};normal[face/2]=face%2?1:-1;Real sweep=face%2?Real(.015):Real(-.005);GasFaceFrame<Real> f;
 check(makeGasFaceFrame(normal[0],normal[1],normal[2],sweep,dt,f),"Cartesian swept face");auto flux=physicalFlux(p,q,p,q,f);
 for(int k=0;k<5;++k){sums[k]+=flux.flux[k];}
 sums[5]+=Real(.2)*flux.flux[0];sums[6]+=Real(.8)*flux.flux[0];sweeps+=sweep;}
 Real output[7];check(advanceGasEulerGeometry<Real,2>(density,GasConservativeStorage::Density,oldV,newV,sweeps,dt,sums,output),"uniform Euler update");
 for(int k=0;k<7;++k)near(output[k],density[k],"moving uniform state violates GCL");
 Real integrated[7];check(convertGasConservativeStorage<Real,2>(density,oldV,GasConservativeStorage::Density,GasConservativeStorage::Integral,integrated),"uniform density to integral");
 check(advanceGasEulerGeometry<Real,2>(integrated,GasConservativeStorage::Integral,oldV,newV,sweeps,dt,sums,output),"integral Euler update");
 for(int k=0;k<7;++k)near(output[k],density[k]*newV,"integral update uses wrong new volume");
}
void moving_wall(){
 auto f=frame();Primitive wall{Real(1.2),f.wx,f.wy,f.wz,Real(95000)};auto q=state(wall);auto flux=physicalFlux(wall,q,wall,q,f);
 near(flux.flux[0],Real(0),"moving impermeable wall transported mass");
 near(flux.flux[1],wall.p*f.nx*f.area,"moving wall pressure force");near(flux.flux[2],wall.p*f.ny*f.area,"moving wall pressure force");near(flux.flux[3],wall.p*f.nz*f.area,"moving wall pressure force");
 near(flux.flux[4],wall.p*f.meshVolumeRate,"moving wall pressure work missing or doubled");
}
void storage_conversion(){
 Real density[7]={2,6,-4,3,-2000000,Real(.2),Real(1.8)},integral[7],back[7];
 for(Real volume:{Real(.03),Real(7)}){check(convertGasConservativeStorage<Real,2>(density,volume,GasConservativeStorage::Density,GasConservativeStorage::Integral,integral),"density to inventory");
 for(int k=0;k<7;++k)near(integral[k],density[k]*volume,"nonuniform volume conversion");
 check(convertGasConservativeStorage<Real,2>(integral,volume,GasConservativeStorage::Integral,GasConservativeStorage::Density,back),"inventory to density");for(int k=0;k<7;++k)near(back[k],density[k],"conversion round trip");}
 Real flux[7]={Real(.1),2,-3,1,400,Real(.01),Real(.09)},a[7],b[7];
 check(advanceGasEulerGeometry<Real,2>(density,GasConservativeStorage::Density,2,3,1,Real(.1),flux,a),"density nonuniform update");
 check(convertGasConservativeStorage<Real,2>(density,2,GasConservativeStorage::Density,GasConservativeStorage::Integral,integral),"inventory input");
 check(advanceGasEulerGeometry<Real,2>(integral,GasConservativeStorage::Integral,2,3,1,Real(.1),flux,b),"integral nonuniform update");
 for(int k=0;k<7;++k){near(a[k],(2*density[k]-Real(.1)*flux[k])/3,"Euler volume semantics");near(3*a[k],b[k],"storage-dependent update");}
}
void reject_transaction(){
 auto f=frame();auto keep=f;
 check(!makeGasFaceFrame(Real(0),Real(0),Real(0),Real(1),Real(.1),f),"zero area accepted");check(std::memcmp(&f,&keep,sizeof f)==0,"failed frame changed output");
 check(!makeGasFaceFrame(Real(1),Real(0),Real(0),Real(1),Real(0),f),"zero interval accepted");
 check(!makeGasFaceFrame(Real(1),Real(0),Real(0),Real(1),Real(.1),f,Real(1),Real(0),Real(0)),"nontangential supplemental frame accepted");
 check(!makeGasFaceFrame(std::numeric_limits<Real>::infinity(),Real(0),Real(0),Real(1),Real(.1),f),"nonfinite area accepted");
 auto p=primitive();auto q=state(p);Primitive po{99,98,97,96,95},poKeep=po;Conservative qo{{1,2,3,4,5}},qoKeep=qo;
 q.q[0]=-1;check(!shiftGasStateToFaceFrame(p,q,keep,po,qo),"invalid state accepted");check(std::memcmp(&po,&poKeep,sizeof po)==0&&std::memcmp(&qo,&qoKeep,sizeof qo)==0,"failed frame state published");
 q=state(p);q.q[1]+=2;check(!shiftGasStateToFaceFrame(p,q,keep,po,qo),"inconsistent primitive/conservative accepted");
 FluxResult input=invalidResult(Scheme::HLLC),out=invalidResult(Scheme::Roe),saved=out;out.flux[0]=saved.flux[0]=57;
 check(!restoreGasFluxFromFaceFrame(input,keep,out),"invalid flux accepted");check(std::memcmp(&out,&saved,sizeof out)==0,"failed frame flux published");
 Real base[7]={1,2,3,4,-50000,Real(.3),Real(.7)},flux[7]={},result[7]={91,92,93,94,95,96,97},prior[7];std::memcpy(prior,result,sizeof prior);
 check(!advanceGasEulerGeometry<Real,2>(base,GasConservativeStorage::Density,1,0,-1,Real(.1),flux,result),"nonpositive volume accepted");
 check(!advanceGasEulerGeometry<Real,2>(base,GasConservativeStorage::Density,1,2,Real(.8),Real(.1),flux,result),"GCL mismatch accepted");
 flux[5]=10;check(!advanceGasEulerGeometry<Real,2>(base,GasConservativeStorage::Density,1,1,0,Real(.1),flux,result),"negative species accepted");
 flux[5]=0;flux[4]=std::numeric_limits<Real>::quiet_NaN();check(!advanceGasEulerGeometry<Real,2>(base,GasConservativeStorage::Density,1,1,0,Real(.1),flux,result),"nonfinite residual accepted");
 check(std::memcmp(prior,result,sizeof prior)==0,"failed Euler update published");
 check(!convertGasConservativeStorage<Real,2>(base,Real(-1),GasConservativeStorage::Density,GasConservativeStorage::Integral,result),"negative conversion volume accepted");check(std::memcmp(prior,result,sizeof prior)==0,"failed conversion published");
}
void alias_and_extreme(){
 GasGeometryState<Real> view;check(!view.enabled&&!view.oldVolume&&!view.newVolume&&!view.faceSweptVolume&&view.interval==0,"default geometry changed legacy mode");
 Real q[5]={1,2,3,4,-50000};check(convertGasConservativeStorage<Real,0>(q,Real(.2),GasConservativeStorage::Density,GasConservativeStorage::Integral,q),"in-place zero-species conversion");near(q[4],Real(-10000),"negative formation energy rejected");
 check(convertGasConservativeStorage<Real,0>(q,Real(.2),GasConservativeStorage::Integral,GasConservativeStorage::Density,q),"in-place zero-species recovery");near(q[4],Real(-50000),"in-place state damaged");
 Real flux[5]={};check(advanceGasEulerGeometry<Real,0>(q,GasConservativeStorage::Density,1,1,0,Real(.1),flux,q),"in-place Euler update");
 Real output[5]={91,92,93,94,95},saved[5];std::memcpy(saved,output,sizeof saved);
 q[4]=std::numeric_limits<Real>::max();check(!convertGasConservativeStorage<Real,0>(q,Real(3),GasConservativeStorage::Density,GasConservativeStorage::Integral,output),"conversion overflow accepted");check(std::memcmp(saved,output,sizeof saved)==0,"overflow changed output");
 check(!convertGasConservativeStorage<Real,0>(q,Real(1),static_cast<GasConservativeStorage>(9),GasConservativeStorage::Integral,output),"invalid storage enum accepted");
 check(validateGasEulerGeometry(Real(1e-15),Real(1.1e-15),Real(1e-16)),"tiny cell exact GCL rejected");check(!validateGasEulerGeometry(Real(1e-15),Real(1.1e-15),Real(0)),"absolute tolerance masked tiny cell GCL error");
 auto f=frame(),keep=f;check(!makeGasFaceFrame(std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max(),Real(0),Real(0),Real(1),f),"area norm overflow accepted");check(std::memcmp(&f,&keep,sizeof f)==0,"overflow frame published");
 auto p=primitive();auto u=state(p);check(shiftGasStateToFaceFrame(p,u,keep,p,u),"in-place state frame transform");
 auto moving=rusanovTadmorFluxUnitNormal(p,u,Real(300),p,u,Real(300),keep.nx,keep.ny,keep.nz,false);check(restoreGasFluxFromFaceFrame(moving,keep,moving),"in-place flux restore");
}
void tolerance_threshold(){
 GasGeometryState<Real> geometry;check(geometry.absoluteGeometryTolerance<0&&geometry.relativeGeometryTolerance<0,"moving view silently defaults configured tolerances");
 check(!validateGasEulerGeometry(Real(1000),Real(1000),Real(.1),Real(.001)),"configured absolute volume tolerance treated as relative");
 geometry.absoluteGeometryTolerance=Real(.002);geometry.relativeGeometryTolerance=Real(0);
 check(validateGasEulerGeometry(Real(1),Real(1.25),Real(.251),geometry.absoluteGeometryTolerance,geometry.relativeGeometryTolerance),"configured loose absolute tolerance ignored");
 check(!validateGasEulerGeometry(Real(1),Real(1.25),Real(.251),Real(.0001),Real(0)),"configured strict absolute tolerance ignored");
 check(validateGasEulerGeometry(Real(1),Real(2),Real(1.015),Real(0),Real(.01)),"relative tolerance failed to use max old/new volume");
 check(!validateGasEulerGeometry(Real(1),Real(2),Real(1.015),Real(0),Real(.001)),"strict relative tolerance ignored");
 check(!validateGasEulerGeometry(Real(1),Real(1),Real(2.1),Real(0),Real(2)),"sweep incorrectly inflated relative tolerance scale");
 check(validateGasEulerGeometry(Real(1e-15),Real(1.1e-15),Real(0),Real(2e-16),Real(0)),"absolute tolerance omitted for tiny cell");
 check(!validateGasEulerGeometry(Real(1e-15),Real(1.1e-15),Real(0),Real(1e-17),Real(0)),"tiny cell absolute threshold ignored");
 const Real delta=Real(4)*std::numeric_limits<Real>::epsilon();
 check(!validateGasEulerGeometry(Real(1),Real(1.25),Real(.25)+delta,Real(0),Real(0)),"strict configured tolerance replaced by epsilon floor");
 check(!validateGasEulerGeometry(Real(1),Real(1),Real(0),Real(-1),Real(0)),"negative absolute tolerance accepted");
 check(!validateGasEulerGeometry(Real(1),Real(1),Real(0),Real(0),std::numeric_limits<Real>::infinity()),"nonfinite relative tolerance accepted");
 Real base[5]={1,2,3,4,-50000},flux[5]={},output[5]={91,92,93,94,95},prior[5];std::memcpy(prior,output,sizeof prior);
 check(!advanceGasEulerGeometry<Real,0>(base,GasConservativeStorage::Density,1,Real(1.25),Real(.25)+delta,Real(.1),flux,output,Real(0),Real(0)),"Euler update discarded strict configured tolerance");
 check(std::memcmp(prior,output,sizeof prior)==0,"configured tolerance failure published state");
 check(!advanceGasEulerGeometry<Real,0>(base,GasConservativeStorage::Density,1,1,0,Real(.1),flux,output,Real(-1),Real(-1)),"unconfigured moving tolerances accepted");
 check(std::memcmp(prior,output,sizeof prior)==0,"unconfigured tolerance failure published state");
 check(advanceGasEulerGeometry<Real,0>(base,GasConservativeStorage::Density,1,Real(1.25),Real(.251),Real(.1),flux,output,geometry.absoluteGeometryTolerance,geometry.relativeGeometryTolerance),"Euler update discarded loose configured tolerance");
 for(int k=0;k<5;++k)near(output[k],base[k]/Real(1.25),"configured tolerance changed conservative update");
 check(advanceGasEulerGeometry<Real,0>(base,GasConservativeStorage::Density,1,Real(1.25),Real(.251),Real(.1),flux,output,Real(0),Real(.002)),"Euler update discarded configured relative tolerance");
 for(int k=0;k<5;++k)near(output[k],base[k]/Real(1.25),"relative tolerance changed conservative update");
}
void deformation_metrics(){
 // Uniform diagonal dilation: V=abc. Sum exact time-integrated swept faces
 // equals abc at t1 minus abc at t0, while endpoint area*speed*dt does not.
 const Real oldV=Real(2)*3*4,newV=Real(2.2)*Real(3.3)*Real(4.4);const Real expansion=Real(.1);
 const Real sweepPerAxis=oldV*(expansion+expansion*expansion+expansion*expansion*expansion/3);
 check(validateGasEulerGeometry(oldV,newV,3*sweepPerAxis),"exact deformation sweep rejected");
 check(!validateGasEulerGeometry(oldV,newV,3*oldV*expansion*(1+expansion)*(1+expansion)),"endpoint metric mismatch hidden");
 check(validateGasGeometryTimeIntegrator(false,1)&&validateGasGeometryTimeIntegrator(false,2)&&validateGasGeometryTimeIntegrator(false,3),"fixed RK rejected");
 check(validateGasGeometryTimeIntegrator(true,1),"ALE Euler rejected");check(!validateGasGeometryTimeIntegrator(true,2)&&!validateGasGeometryTimeIntegrator(true,3),"unverified moving RK allowed");
 check(!validateGasGeometryTimeIntegrator(false,0)&&!validateGasGeometryTimeIntegrator(true,4),"invalid RK selector allowed");
}
int main(int argc,char**argv){check(argc==2,"test name required");
 if(!std::strcmp(argv[1],"equal_state"))equal_state();else if(!std::strcmp(argv[1],"zero_motion"))zero_motion();else if(!std::strcmp(argv[1],"constant_gamma_schemes"))constant_gamma_schemes();else if(!std::strcmp(argv[1],"tangential_frame"))tangential_frame();else if(!std::strcmp(argv[1],"orientation"))orientation();else if(!std::strcmp(argv[1],"uniform_gcl"))uniform_gcl();else if(!std::strcmp(argv[1],"moving_wall"))moving_wall();else if(!std::strcmp(argv[1],"alias_and_extreme"))alias_and_extreme();else if(!std::strcmp(argv[1],"storage_conversion"))storage_conversion();else if(!std::strcmp(argv[1],"reject_transaction"))reject_transaction();else if(!std::strcmp(argv[1],"tolerance_threshold"))tolerance_threshold();else if(!std::strcmp(argv[1],"deformation_metrics"))deformation_metrics();else check(false,"unknown test");}
'''

def test_shared_ale_geometry_policy_available():
    assert POLICY.exists(), 'The shared ALE geometry/frame policy is not implemented'


@pytest.fixture(scope='module', params=[32, 64])
def ale_probe(tmp_path_factory, request):
    assert POLICY.exists(), 'The shared ALE geometry/frame policy is not implemented'
    directory = tmp_path_factory.mktemp(f'ale_geometry_{request.param}')
    source = directory / 'geometry.cpp'
    source.write_text('#include <initializer_list>\n' + SOURCE)
    executable = directory / 'geometry'
    build = subprocess.run([
        'g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
        f'-DUGKWP_GPU_REAL_BITS={request.param}', '-I'+str(ROOT/'common'),
        '-I'+str(ROOT/'common/gasNumerics'), str(source), '-o', str(executable)
    ], text=True, capture_output=True)
    assert build.returncode == 0, build.stdout + build.stderr
    return executable

@pytest.mark.parametrize('scenario', [
    'equal_state', 'zero_motion', 'tangential_frame', 'orientation',
    'uniform_gcl', 'storage_conversion', 'reject_transaction', 'deformation_metrics',
    'moving_wall', 'alias_and_extreme', 'constant_gamma_schemes', 'tolerance_threshold'
])
def test_shared_ale_geometry_invariants(ale_probe, scenario):
    result = subprocess.run([str(ale_probe), scenario], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
