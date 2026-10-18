"""Execute real producer headers with and without the optional detailed context."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
import pytest

@pytest.mark.parametrize('context', [False, True])
@pytest.mark.parametrize('fault', ['species_nan', 'density_zero', 'volume_zero', 'rk_species_nan', 'rk_energy_nan'])
def test_physical_fault_preserves_optional_coarse_status(tmp_path, context, fault):
    action = {
        'species_nan': 's.gasSpecies.rho[0]=NAN;recoverGasPrimitiveCell(s,0,true);',
        'density_zero': 's.rho[0]=0;recoverGasPrimitiveCell(s,0,true);',
        'volume_zero': 's.V[0]=0;applyGasFluxDivergenceByCellKernel(&s,.001);',
        'rk_species_nan': 's.gasSpecies.rho[0]=NAN;blendGasConservativeStateKernel(&s,.5,.5);',
        'rk_energy_nan': 's.rhoE[0]=NAN;blendGasConservativeStateKernel(&s,.5,.5);',
    }[fault]
    compile_probe(tmp_path, fixture() + r'''
int main(){State s;initialise(s);threadIdx.x=0;blockIdx.x=0;
 ugkp::diagnostics::DeviceContext detailed{};
 s.diagnostics=CONTEXT?&detailed:nullptr;
 // Stage inventories are legitimate copies; only ACTION introduces a fault.
 for(int c=0;c<s.nCells;++c){s.rhoNext[c]=s.rho[c];s.rhoUxNext[c]=s.rhoUx[c];s.rhoUyNext[c]=s.rhoUy[c];s.rhoUzNext[c]=s.rhoUz[c];s.rhoENext[c]=s.rhoE[c];}
 for(int k=0;k<2;++k)for(int c=0;c<s.nCells;++c)s.gasSpecies.initial[k*s.nCells+c]=s.gasSpecies.rho[k*s.nCells+c];
 const auto oldTemperature=s.Tgas[0],oldPressure=s.p[0];
 ACTION
 if(s.gasSpecies.cellStatus[0]==0){fprintf(stderr,"physical fault lost coarse status\n");return 1;}
 if(s.Tgas[0]!=oldTemperature||s.p[0]!=oldPressure){fprintf(stderr,"failed producer changed primitives\n");return 2;}
 if(CONTEXT && (!ugkp::diagnostics::hasFatal(&detailed)||detailed.firstFatal.sourceSiteId!=2620)){
 fprintf(stderr,"coarse propagation lost detailed first fatal\n");return 3;}
}
'''.replace('CONTEXT', 'true' if context else 'false').replace('ACTION', action))

@pytest.mark.parametrize('context', [False, True])
def test_preclip_audit_sum_overflow_does_not_refuse_finite_clipped_state(tmp_path, context):
    compile_probe(tmp_path, fixture() + r'''
int main(){State s;initialise(s);threadIdx.x=0;blockIdx.x=0;
 ugkp::diagnostics::DeviceContext detailed{};s.diagnostics=CONTEXT?&detailed:nullptr;
 s.rho[0]=1;s.rhoUx[0]=s.rhoUy[0]=s.rhoUz[0]=0;
 s.gasSpecies.rho[0]=s.gasSpecies.rho[s.nCells]=(std::numeric_limits<Real>::max)()/Real(1.5);
 const Real clipped[2]={1,1};const Real target=300;
 s.rhoE[0]=ugkwp::mixtureEnergy(clipped,target,s.gasSpecies.thermo);s.Tgas[0]=target;s.p[0]=-777;
 recoverGasPrimitiveCell(s,0);
 if(s.gasSpecies.cellStatus[0]!=0||ugkp::diagnostics::hasFatal(s.diagnostics)){
 fprintf(stderr,"audit-only preclip sum overflow refused physical evaluation\n");return 1;}
 if(!(s.p[0]>0)||!std::isfinite(s.p[0])||!std::isfinite(s.Tgas[0])||std::abs(s.Tgas[0]-target)>Real(.01)){
 fprintf(stderr,"finite clipped state was not recovered\n");return 2;}
}
'''.replace('CONTEXT', 'true' if context else 'false'))

@pytest.mark.parametrize('context', [False, True])
def test_warning_helper_preserves_success_status(tmp_path, context):
    compile_probe(tmp_path, fixture() + r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext detailed{};
 s.diagnostics=CONTEXT?&detailed:nullptr;
 gasCellPhysicalDiagnostic(s,0,30,ugkp::diagnostics::Code::ConservationResidual,Real(.1));
 if(s.gasSpecies.cellStatus[0]!=0||ugkp::diagnostics::hasFatal(s.diagnostics)){
 fprintf(stderr,"warning helper created a physical failure\n");return 1;}
}
'''.replace('CONTEXT', 'true' if context else 'false'))

@pytest.mark.parametrize('action, aborts', [
    ('gasCellPhysicalDiagnostic(s,0,30,ugkp::diagnostics::Code::NonFinite,Real(NAN));', True),
    ('gasCellPhysicalDiagnostic(s,0,30,ugkp::diagnostics::Code::ConservationResidual,Real(.1));', False),
    ('s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.gasGeometry.enabled=true;s.rho[0]=0;recoverGasPrimitiveCell(s,0);', True),
])
def test_detected_fault_without_either_channel_uses_existing_fatal_fallback(tmp_path, action, aborts):
    import signal
    import subprocess
    from test_mixture_state import HERE, ROOT
    source = tmp_path / 'no_channel.cpp'
    executable = tmp_path / 'no_channel'
    pre = (HERE / 'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    source.write_text(pre + fixture() + '''
int main(){State s;initialise(s);s.diagnostics=nullptr;s.gasSpecies.cellStatus=nullptr;
ACTION
return 0;}
'''.replace('ACTION', action))
    build = subprocess.run(['g++', '-std=c++17', '-O0', '-I'+str(HERE),
                            '-I'+str(ROOT/'common'), '-I'+str(ROOT/'common/gasNumerics'),
                            str(source), '-o', str(executable)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == (-signal.SIGABRT if aborts else 0), result.stdout + result.stderr
