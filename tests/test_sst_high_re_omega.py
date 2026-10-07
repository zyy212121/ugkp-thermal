"""CPU execution of production wall, RHS, recovery and RK kernels (FP32/64).

Removing either high-Re RHS suppression or recovery projection must fail this
regression. No test-only implementation of the constraint is substituted.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_high_re_omega_stage_constraint(tmp_path):
    primitive = (ROOT / 'common/operators/computeGasPrimitiveGradientsKernel.cuh').read_text()
    flux = (ROOT / 'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    functions = primitive[primitive.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue'):
                          primitive.index('template<class GasState>\n__global__ void initialiseSstConservativeStateKernel')]
    recovery = primitive[primitive.index('template<class GasState>\n__global__ void recoverSstPrimitivesKernel'):]
    update = flux[flux.index('template<class GasState>\n__global__ void applySstFluxAndSourceKernel'):
                  flux.index('template<class GasState>\n__global__ void computeGasCourantFieldKernel')]
    rk = flux[flux.index('template<class GasState>\n__global__ void saveGasConservativeStateKernel'):
              flux.index('template<class GasState>\n__device__ void recoverGasPrimitiveCell')]
    pre = (ROOT / 'tests/fixtures/sst_wall_constraint_host.hpp').read_text()
    controlled_invariants = '''
void sstVelocityInvariants(const DeviceState&,int,R&d,R&s,R&g){d=s=g=0;}
R sstKProductionForCell(const DeviceState&,int,R){return 0;}
'''
    body = '#include "gasTransport/GasGeometryValidation.H"\n' + pre + '\nbool finiteDevice(R value){return std::isfinite(value); }\n' + functions + recovery + controlled_invariants + update + rk + r'''
int failures=0;
void check(const char*name,R got,R want){
 if(!std::isfinite(got)||std::abs(got-want)>(sizeof(R)==4?6e-5:2e-11)*std::max(R(1),std::abs(want))){
  std::cerr<<name<<" got="<<got<<" expected="<<want<<"\n";++failures;
 }
}
void changed(const char*name,R got,R old){if(got==old){std::cerr<<name<<" frozen\n";++failures;}}
R logTarget(R k,R y){return std::sqrt(k)/(std::sqrt(std::sqrt(R(.09)))*R(.41)*y);}
void checkLogState(const char*name,const DeviceState&s){
 check(name,s.omega[0],logTarget(s.rhoK[0]/s.rho[0],s.sstWallDistance[0]));
 check("rhoOmega matches current density",s.rhoOmega[0],s.rho[0]*s.omega[0]);
}
int main(){
 // The formula and equal-face corner averaging remain the existing production law.
 DeviceState s;s.sstWallTreatment=1;applySstWallFunctionStateKernel(&s);
 checkLogState("high-Re pre-stage",s);
 check("high-Re wall k remains zero-gradient",sstBoundaryValue(s,0,0,false),s.k[0]);
 check("wall omega zero normal difference",sstBoundaryValue(s,0,0,true)-s.omega[0],0);
 s.rho[0]=2;s.rhoK[0]=R(.4);s.rhoOmega[0]=18;recoverSstPrimitivesKernel(&s);
 checkLogState("recovery uses current density and k",s);
 // Mixed viscous/log branches at a corner: nu differs by face, not cell rho.
 s=DeviceState();s.sstWallTreatment=1;s.wallRho[0]=1;s.wallRho[1]=8;
 applySstWallFunctionStateKernel(&s);
 check("mixed-branch corner arithmetic average",s.omega[0],(R(1440)+logTarget(R(.1),R(.001)))/2);
 s.cellPlaneCount[0]=1;applySstWallFunctionStateKernel(&s);check("single viscous face",s.omega[0],1440);
 s.cellPlaneCount[0]=2;std::swap(s.cellFaceId[0],s.cellFaceId[1]);applySstWallFunctionStateKernel(&s);
 check("corner face order independent",s.omega[0],(R(1440)+logTarget(R(.1),R(.001)))/2);
 // A prescribed inlet and an internal face are never included in wall averaging.
 s.riemannBoundaryKind[0]=0;applySstWallFunctionStateKernel(&s);check("ignore nonwall face",s.omega[0],logTarget(R(.1),R(.001)));
 s.riemannBoundaryKind[0]=2;s.nInternalFaces=1;applySstWallFunctionStateKernel(&s);check("ignore internal face",s.omega[0],logTarget(R(.1),R(.001)));
 // Source/flux suppression and signed correction accounting at different dt.
 for(R dt:{R(1e-7),R(1e-5),R(1e-4)}){
  s=DeviceState();s.sstWallTreatment=1;applySstWallFunctionStateKernel(&s);
  R oldOmega=s.rhoOmega[0],oldK=s.rhoK[0];
  DeviceState free=s;free.riemannBoundaryKind[0]=free.riemannBoundaryKind[1]=0;
  applySstFluxAndSourceKernel(&free,dt);applySstFluxAndSourceKernel(&s,dt);
  changed("unconstrained omega RHS",free.rhoOmega[0],oldOmega);
  check("wall omega equation rejects RHS",s.rhoOmega[0],oldOmega);
  changed("k equation continues",s.rhoK[0],oldK);
  check("wall k update equals free k update",s.rhoK[0],free.rhoK[0]);
  check("constrained omega source excluded",s.sstSourceNumber[0],dt*s.sstCoefficients.betaStar*s.omega[0]);
  R suppressed=s.rhoOmega[0]-free.rhoOmega[0];
  s.rho[0]=R(1.25);R beforeProjection=s.rhoOmega[0];recoverSstPrimitivesKernel(&s);
  checkLogState("Euler endpoint current-state projection",s);
  R projection=s.rhoOmega[0]-beforeProjection;
  check("suppression plus projection ledger",suppressed+projection,s.rhoOmega[0]-free.rhoOmega[0]);
  R after=s.rhoOmega[0];recoverSstPrimitivesKernel(&s);check("repeat recovery idempotent",s.rhoOmega[0]-after,0);
 }
 // Real production RK save/blend: the endpoint constraint must survive all
 // SSPRK2/3 blend weights, even when the saved state is not constrained.
 for(R weight:{R(.5),R(.25),R(2)/R(3)}){
  s=DeviceState();s.sstWallTreatment=1;saveGasConservativeStateKernel(&s);
  s.rho[0]=2;s.rhoK[0]=R(.8);s.rhoOmega[0]=123;s.rhoUx[0]=4;
  blendGasConservativeStateKernel(&s,1-weight,weight);
  check("RK density still blends",s.rho[0],1+weight);
  check("RK k still blends",s.rhoK[0],R(.1)*(1-weight)+R(.8)*weight);
  check("RK momentum still blends",s.rhoUx[0],4*weight);
  R rawBlend=R(7)*(1-weight)+R(123)*weight;check("RK conservative blend precedes projection",s.rhoOmega[0],rawBlend);
  recoverSstPrimitivesKernel(&s);checkLogState("RK endpoint projection",s);
  R finalCorrection=s.rhoOmega[0]-rawBlend;
  check("RK projection ledger",rawBlend+finalCorrection,s.rhoOmega[0]);
 }
 // Low-Re behavior from current main is retained, including wall-nu averaging.
 s=DeviceState();applySstWallFunctionStateKernel(&s);check("low-Re target unchanged",s.omega[0],270);
 applySstFluxAndSourceKernel(&s,R(1e-5));check("low-Re RHS stays constrained",s.rhoOmega[0],270);
 s.rho[0]=2;recoverSstPrimitivesKernel(&s);check("low-Re recovery unchanged",s.omega[0],270);check("low-Re conservative unchanged",s.rhoOmega[0],540);
 // Disabled SST, both wall treatments in nonwall cells, and floors.
 s=DeviceState();s.sstWallTreatment=1;s.sstConfigured=0;
 applySstWallFunctionStateKernel(&s);applySstFluxAndSourceKernel(&s,R(.1));recoverSstPrimitivesKernel(&s);
 check("disabled omega untouched",s.rhoOmega[0],7);check("disabled k untouched",s.rhoK[0],R(.1));
 for(int treatment:{0,1}){
  s=DeviceState();s.sstWallTreatment=treatment;s.riemannBoundaryKind[0]=s.riemannBoundaryKind[1]=0;
  saveGasConservativeStateKernel(&s);applySstFluxAndSourceKernel(&s,R(1e-4));R advanced=s.rhoOmega[0];changed("nonwall Euler omega",advanced,7);
  s.rho[0]=2;blendGasConservativeStateKernel(&s,R(.5),R(.5));R blended=(R(7)+advanced)/2;
  recoverSstPrimitivesKernel(&s);check("nonwall RK omega remains evolved",s.rhoOmega[0],blended);check("nonwall primitive follows density",s.omega[0],blended/R(1.5));
 }
 s=DeviceState();s.sstWallTreatment=1;s.sstOmegaMin=2000;recoverSstPrimitivesKernel(&s);check("wall target respects floor",s.omega[0],2000);
 std::cout<<"failures="<<failures<<"\n";return failures?1:0;
}
'''
    source = tmp_path / 'probe.cpp'
    source.write_text(body)
    failures = []
    for bits in (64, 32):
        exe = tmp_path / f'probe{bits}'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                        f'-DUGKWP_GPU_REAL_BITS={bits}', '-I'+str(ROOT/'common'),
                        '-I'+str(ROOT/'common/gasNumerics'), str(source), '-o', str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        if result.returncode:
            failures.append(f'FP{bits}: {result.stdout}{result.stderr}')
    assert not failures, '\n'.join(failures)


def test_wall_constraint_stage_hooks_are_shared():
    """Supplement numeric kernels with production host ordering/ownership gates."""
    advance = (ROOT / 'common/GpuGasAdvance.cuh').read_text()
    euler = advance.split('int advanceGasEulerSubstage', 1)[1].split('int blendGasRungeKuttaStage', 1)[0]
    after_rhs = euler.split('applySstFluxAndSourceKernel<<<', 1)[1]
    assert after_rhs.index('applyGasFluxDivergenceByCellKernel<<<') < after_rhs.index('recoverGasPrimitivesKernel<<<')
    assert after_rhs.index('recoverGasPrimitivesKernel<<<') < after_rhs.index('recoverSstPrimitivesKernel<<<')
    assert euler.index('applySstWallFunctionStateKernel<<<') < euler.index('computeSstGradientsKernel<<<')
    blend = advance.split('int blendGasRungeKuttaStage', 1)[1].split('int advanceGasFluxStage', 1)[0]
    assert blend.index('blendGasConservativeStateKernel<<<') < blend.index('recoverGasPrimitivesKernel<<<')
    assert blend.index('recoverGasPrimitivesKernel<<<') < blend.index('recoverSstPrimitivesKernel<<<')
    for application in ('gasUGKP/private_backend', 'FSH/private_backend', 'CHT/gpu'):
        backend = (ROOT / 'applications' / application / 'GpuResidentStrict.cu').read_text()
        for shared in ('GpuGasAdvance.cuh', 'operators/computeGasPrimitiveGradientsKernel.cuh',
                       'operators/computeSstFaceFluxKernel.cuh'):
            assert (f'common/{shared}"' if shared == 'GpuGasAdvance.cuh' else f'#include "{shared}"') in backend
