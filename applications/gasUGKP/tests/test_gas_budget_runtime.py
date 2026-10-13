"""Actual resident transaction policy, host-memory CUDA shim; no GPU claim."""
import json
import re
import subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]


def build(tmp_path, precision=False, wall_audit=False):
    original=(Path(__file__).with_name('test_shared_gas_trial_policy.py')).read_text()
    code=re.search("code=r'''(.*?)'''",original,re.S).group(1).split('int main(){')[0]
    scalar_fields='rho rhoUx rhoUy rhoUz rhoE Ux Uy Uz p Tgas gasFluxPositivityScale gasDiffusionNumber sstSourceNumber rhoK rhoOmega k omega nut gasBoundaryRho gasBoundaryUx gasBoundaryUy gasBoundaryUz gasBoundaryP gasBoundaryT V'.split()
    code=code.replace('__DECLARATIONS__','\n'.join('double* '+name+'=nullptr;' for name in scalar_fields))
    code=code.replace('#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"',
        '#include "common/gasTransport/GasBudgetAudit.H"\nstruct GasBudgetHostLedger;\n#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"')
    code=code.replace('SharedGasTrialStorage gasTrial;', 'SharedGasTrialStorage gasTrial;ugkwp::GasBudgetAuditState gasBudgetAudit;GasBudgetHostLedger* gasBudgetHost=nullptr;bool gasBudgetCoverageReported=false;bool hostGravityActive=false;int particleCapacity=0,nBoundarySources=0;')
    code=code.replace('#include <cassert>', '#include <cassert>\n#include <cstdlib>\n#include <new>\nstd::size_t hostAllocations=0;int hostAllocationFailures=0;\nvoid* operator new(std::size_t n){++hostAllocations;if(hostAllocationFailures>0){--hostAllocationFailures;throw std::bad_alloc();}if(void*p=std::malloc(n))return p;throw std::bad_alloc();}\nvoid operator delete(void*p)noexcept{std::free(p);}\nvoid operator delete(void*p,std::size_t)noexcept{std::free(p);}')
    code=code.replace('int allocations=0,deviceCopyFailures=0;', 'int allocations=0,deviceCopyFailures=0;int hostCopies=0,hostCopyFailures=0;')
    code=code.replace('int copyToHost(T*d,const T*s,std::size_t n,const char*){', 'int copyToHost(T*d,const T*s,std::size_t n,const char*){++hostCopies;if(hostCopyFailures>0){--hostCopyFailures;return 1;}')
    native=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    def entry_prefix(name,stop):
        body=native.split('extern "C" int '+name+'\n',1)[1].split(')\n{',1)[1]
        return body.split(stop,1)[0]
    code+='\nDeviceState* asState(void*p){return static_cast<DeviceState*>(p);}\nint validateState(DeviceState*,const char*){return 0;}\n'
    for name,entry,stop in [('probePure','ugkwpGpuResidentStrictAdvanceGasOnly','    if (s->gasSpecies.mode =='),
                            ('probeCombined','ugkwpGpuResidentStrictAdvance','    if(s->gasSpecies.mode'),
                            ('probeMesh','ugkwpGpuResidentStrictUploadMesh','    int rc = 0;')]:
        code+='int '+name+'(void*handle){'+entry_prefix(entry,stop)+'return 0;}\n'
    code+=r'''
#include "common/gasTransport/GasModelIO.H"
int main(int argc,char**argv){assert(argc==3);DeviceState state;state.deviceState=&state;
__ASSIGNMENTS__
double partial[4]={.4,.4,.6,.6},sound[2]{},cp[2]{},r[2]{};int cells[2]{},faces[2]{};
state.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;state.gasSpecies.rho=partial;
state.gasSpecies.soundSpeed=sound;state.gasSpecies.heatCapacity=cp;state.gasSpecies.gasConstant=r;
state.gasSpecies.cellStatus=cells;state.gasSpecies.faceStatus=faces;
auto model=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);elements(X);
speciesThermo {A {model linearCp;molarMass .028;minTemperature 100;maxTemperature 3000;coefficients(1000 0 0);atoms(1);}
B {model linearCp;molarMass .028;minTemperature 100;maxTemperature 3000;coefficients(1000 0 0);atoms(1);}}diffusion{model none;})");
if(std::string(argv[1])=="off"){
 unsetenv("UGKP_GAS_BUDGET_OUTPUT");assert(configureGasBudget(&state,model)==0);
 assert(prepareGasBudgetRequest(&state,.1,0)==0);assert(beginGasBudgetInterval(&state)==0);assert(commitGasBudgetInterval(&state)==0);
 assert(allocations==0&&hostCopies==0&&!state.gasBudgetHost&&!state.gasBudgetAudit.enabled);
 const auto before=hostAllocations;assert(SharedGasTrialPolicy::beginInterval(&state)==0);assert(SharedGasTrialPolicy::begin(&state)==0);
 // Original policy allocates one cell/face vector per snapshot level. No audit-count vector.
 assert(hostAllocations-before==4);assert(hostCopies==0);releaseSharedGasTrialStorage(state.gasTrial);assert(allocations==0);return 0;}
setenv("UGKP_GAS_BUDGET_OUTPUT",argv[2],1);
if(std::string(argv[1])=="scope_legacy"){state.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;assert(probePure(&state)==0);assert(probePure(&state)==0);return 0;}
if(std::string(argv[1])=="scope_combined"){assert(probeCombined(&state)==0);return 0;}
if(std::string(argv[1])=="owner_failure"){hostAllocationFailures=1;assert(configureGasBudget(&state,model)!=0);assert(!state.gasBudgetHost&&!state.gasBudgetAudit.enabled&&allocations==0);return 0;}
assert(configureGasBudget(&state,model)==0);
if(std::string(argv[1])=="scope_mesh"){assert(probeMesh(&state)==0);assert(prepareGasBudgetRequest(&state,.1,0)==0);assert(probeMesh(&state)==0);releaseGasBudget(&state);return 0;}
if(std::string(argv[1])=="reuse"){releaseGasBudget(&state);return 0;}

if(std::string(argv[1])=="diagnostic_nonfinite"){
 assert(prepareGasBudgetRequest(&state,.1,0)==0);assert(beginGasBudgetInterval(&state)==0);
 state.gasBudgetAudit.transport[0]=INFINITY;state.gasBudgetAudit.chemistrySpecies[0]=NAN;
 assert(commitGasBudgetInterval(&state)==0&&state.rho[0]==1);
 releaseGasBudget(&state);return 0;}
if(std::string(argv[1])=="lifecycle"){
 assert(beginGasBudgetInterval(&state)==0);assert(commitGasBudgetInterval(&state)==0);
 assert(prepareGasBudgetRequest(&state,.1,-2)==0);assert(beginGasBudgetInterval(&state)==0);assert(commitGasBudgetInterval(&state)==0);
 assert(prepareGasBudgetRequest(&state,1.,1e20)==0);assert(beginGasBudgetInterval(&state)==0);assert(commitGasBudgetInterval(&state)==0);
 releaseGasBudget(&state);return 0;}
const double origin=std::string(argv[1])=="restart"?.1:0;
assert(prepareGasBudgetRequest(&state,.1,origin)==0);assert(SharedGasTrialPolicy::beginInterval(&state)==0);
assert(SharedGasTrialPolicy::begin(&state)==0);
state.rho[0]+=.1;partial[0]+=.04;partial[2]+=.06;state.rhoE[0]+=2;
state.gasBudgetAudit.transport[0]=.1;state.gasBudgetAudit.transport[2]=2;
state.gasBudgetAudit.transport[4]=.04;state.gasBudgetAudit.transport[6]=.06;
assert(SharedGasTrialPolicy::commit(&state)==0);
if(std::string(argv[1])=="read_failure"){
 hostCopyFailures=1;assert(SharedGasTrialPolicy::commitInterval(&state)!=0);SharedGasTrialPolicy::rollback(&state);SharedGasTrialPolicy::rollbackInterval(&state);
 assert(state.rho[0]==1&&!state.gasBudgetHost->requestActive&&state.gasBudgetHost->sequence==0);
 releaseSharedGasTrialStorage(state.gasTrial);releaseGasBudget(&state);assert(allocations==0);return 0;
}
if(std::string(argv[1])=="energy_residual")state.rhoE[0]+=1;
assert(SharedGasTrialPolicy::commitInterval(&state)==0);
assert(prepareGasBudgetRequest(&state,.1,origin+.1)==0);assert(SharedGasTrialPolicy::beginInterval(&state)==0);assert(SharedGasTrialPolicy::begin(&state)==0);
state.rho[0]=999;state.gasBudgetAudit.transport[0]=999;SharedGasTrialPolicy::rollback(&state);SharedGasTrialPolicy::rollbackInterval(&state);
assert(state.rho[0]==1.1);releaseSharedGasTrialStorage(state.gasTrial);releaseGasBudget(&state);assert(allocations==0);
}
'''.replace('__ASSIGNMENTS__','\n'.join(f'double {name}[2]={{1,1}};state.{name}={name};' for name in scalar_fields))
    if wall_audit:
        wall_setup=original.split(' double audit[13][2]{};',1)[1].split(' assert(SharedGasTrialPolicy::beginInterval',1)[0]
        code=code.replace('const double origin=', 'double audit[13][2]{};'+wall_setup+'\nconst double origin=')
        code=code.replace('state.rho[0]+=.1;', 'state.gasSstAudit.sourceK[0]=5;wallOutput[0].conductiveHeatFlux=6;state.rho[0]+=.1;')
        code=code.replace('state.rho[0]=99;', 'state.gasSstAudit.sourceK[0]=99;wallOutput[0].conductiveHeatFlux=99;state.rho[0]=99;')
        code=code.replace('state.rho[0]=999;', 'assert(state.gasSstAudit.sourceK[0]==5&&wallOutput[0].conductiveHeatFlux==6);state.rho[0]=999;')
        code=code.replace('assert(state.rho[0]==1&&!state.gasBudgetHost', 'assert(state.gasSstAudit.sourceK[0]==0&&wallOutput[0].conductiveHeatFlux==0);assert(state.rho[0]==1&&!state.gasBudgetHost')
        code=code.replace('state.rho[0]=999;', 'state.gasSstAudit.sourceK[0]=999;wallOutput[0].conductiveHeatFlux=999;state.rho[0]=999;')
        code=code.replace('assert(state.rho[0]==1.1);', 'assert(state.gasSstAudit.sourceK[0]==5&&wallOutput[0].conductiveHeatFlux==6);assert(state.rho[0]==1.1);')
    if precision:
        code=code.replace('setenv("UGKP_GAS_BUDGET_OUTPUT",argv[2],1);', 'state.rho[0]=1e16;state.rho[1]=1;setenv("UGKP_GAS_BUDGET_OUTPUT",argv[2],1);')
        code=code.replace('state.rho[0]+=.1;', 'state.rho[1]+=1;').replace('state.gasBudgetAudit.transport[0]=.1;', 'state.gasBudgetAudit.transport[1]=1;')
        code=code.replace('assert(state.rho[0]==1.1&&state.gasBudgetAudit.transport[0]==.1);', 'assert(state.rho[0]==1e16&&state.gasBudgetAudit.transport[1]==1);')
        code=code.replace('assert(state.rho[0]==1.1);', 'assert(state.rho[0]==1e16);')
    source=tmp_path/'runtime.cpp';source.write_text(code);binary=tmp_path/'runtime'
    result=subprocess.run(['g++','-std=c++17','-O0','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return binary


def test_disabled_runtime_adds_no_allocation_copy_or_output(tmp_path):
    binary=build(tmp_path);path=tmp_path/'off.jsonl'
    run=subprocess.run([str(binary),'off',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert not path.exists()


@pytest.mark.parametrize('mode,origin',[('normal',0.),('restart',.1)])
def test_accepted_interval_only_publishes_once_and_restart_is_new_segment(tmp_path,mode,origin):
    binary=build(tmp_path);path=tmp_path/'ledger.jsonl'
    run=subprocess.run([str(binary),mode,str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2 and rows[0]['record']=='NEW_SEGMENT'
    assert rows[0]['accepted_initial_time']==origin and rows[0]['continuity']=='UNVERIFIED_NEW_SEGMENT'
    assert rows[1]['record']=='ACCEPTED_INTERVAL' and rows[1]['sequence']==1
    assert rows[1]['accepted_time']==pytest.approx(origin+.1)
    assert rows[1]['cumulative_boundary_transport'][0]==pytest.approx(.1)
    assert rows[1]['cumulative_boundary_transport'][1]==pytest.approx(2.)
    assert max(map(abs,rows[1]['constraint_or_roundoff_residual']))<1e-12
    assert rows[1]['audit_extra_kernel_launches']==0
    assert rows[1]['audit_host_read_bytes']>0


def test_existing_ledger_is_not_silently_overwritten(tmp_path):
    binary=build(tmp_path);path=tmp_path/'ledger.jsonl';path.write_text('preserve\n')
    run=subprocess.run([str(binary),'reuse',str(path)],capture_output=True,text=True)
    assert run.returncode!=0
    assert path.read_text()=='preserve\n'


def test_large_inventory_does_not_round_small_delta_before_residual(tmp_path):
    from decimal import Decimal
    binary=build(tmp_path,precision=True);path=tmp_path/'precision.jsonl'
    run=subprocess.run([str(binary),'normal',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line,parse_float=Decimal) for line in path.read_text().splitlines()]
    assert rows[0]['initial_inventory'][0]==Decimal('10000000000000001')
    assert rows[1]['inventory'][0]==Decimal('10000000000000002')
    assert rows[1]['constraint_or_roundoff_residual'][0]==0


@pytest.mark.parametrize('mode',['scope_legacy','scope_combined','scope_mesh'])
def test_real_api_entry_guards_reject_unsupported_budget_paths(tmp_path,mode):
    binary=build(tmp_path);path=tmp_path/'scope.jsonl'
    run=subprocess.run([str(binary),mode,str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    if mode=="scope_legacy":assert run.stderr.count("legacy ledger unavailable")==1


def test_failed_commit_read_rolls_back_without_accepted_output(tmp_path):
    binary=build(tmp_path);path=tmp_path/'failed.jsonl'
    run=subprocess.run([str(binary),'read_failure',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [row['record'] for row in rows]==['NEW_SEGMENT']


def test_actual_energy_change_outside_flux_remains_visible_residual(tmp_path):
    binary=build(tmp_path);path=tmp_path/'residual.jsonl'
    run=subprocess.run([str(binary),'energy_residual',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]['cumulative_boundary_transport'][1]==2
    assert rows[-1]['constraint_or_roundoff_residual'][1]==1


def test_owner_allocation_failure_does_not_escape_or_leave_dangling_storage(tmp_path):
    binary=build(tmp_path);path=tmp_path/'allocation.jsonl'
    run=subprocess.run([str(binary),'owner_failure',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert not path.exists()


@pytest.mark.parametrize('mode',['normal','read_failure'])
def test_wall_sst_and_bulk_budget_share_trial_and_interval_rollback(tmp_path,mode):
    binary=build(tmp_path,wall_audit=True);path=tmp_path/'wall-budget.jsonl'
    run=subprocess.run([str(binary),mode,str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==(2 if mode=='normal' else 1)
    if mode=='normal':assert rows[-1]['cumulative_boundary_transport'][0]==pytest.approx(.1)


@pytest.mark.parametrize('mode',['diagnostic_nonfinite','lifecycle'])
def test_optional_diagnostic_overflow_and_unprepared_lifecycle_are_honest(tmp_path,mode):
    binary=build(tmp_path);path=tmp_path/'diagnostics.jsonl'
    run=subprocess.run([str(binary),mode,str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    if mode=='diagnostic_nonfinite':
        assert rows[-1]['cumulative_boundary_transport'][0]=='Infinity'
        assert rows[-1]['cumulative_chemistry_species'][0]=='NaN'
        assert rows[-1]['inventory'][0]==2
    else:
        assert rows[1]['record']=='OBSERVED_STATE' and rows[1]['completed_duration']=='NaN'
        assert rows[-1]['accepted_time']==1e20 and rows[-1]['completed_duration']==1
        assert rows[-3]['accepted_time']==pytest.approx(-1.9)


def test_relative_path_keeps_no_clobber_semantics(tmp_path):
    binary=build(tmp_path)
    run=subprocess.run([str(binary),'normal','relative.jsonl'],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert (tmp_path/'relative.jsonl').exists()

@pytest.mark.parametrize('truncate_failure',[False,True])
def test_partial_real_output_write_reports_error_and_restores_trial(tmp_path,truncate_failure):
    binary=build(tmp_path)
    source=tmp_path/'runtime.cpp';code=source.read_text()
    overrides=r'''
#include <sys/syscall.h>
int writeFailure=0,truncateFailure=0;
extern "C" ssize_t write(int fd,const void*data,size_t bytes){
 if(writeFailure==1){writeFailure=2;return syscall(SYS_write,fd,data,bytes>3?3:bytes);}
 if(writeFailure==2){writeFailure=0;return -1;}return syscall(SYS_write,fd,data,bytes);}
extern "C" int ftruncate(int fd,off_t length){if(truncateFailure)return -1;return syscall(SYS_ftruncate,fd,length);}
'''
    code=code.replace('int main(int argc,char**argv)',overrides+'\nint main(int argc,char**argv)')
    needle='const double origin=std::string(argv[1])=="restart"?.1:0;'
    main=r'''
assert(prepareGasBudgetRequest(&state,.1,0)==0);assert(beginGasBudgetInterval(&state)==0);assert(SharedGasTrialPolicy::begin(&state)==0);
state.rho[0]=1.25;state.gasBudgetAudit.transport[0]=.25;
const auto offset=lseek(fileno(state.gasBudgetHost->output),0,SEEK_CUR);writeFailure=1;truncateFailure=TRUNCATE;
assert(SharedGasTrialPolicy::commitInterval(&state)!=0&&state.gasBudgetHost->sequence==0);
if(TRUNCATE)assert(lastError.find("output rollback failed")!=std::string::npos);
else assert(lastError.find("record write failed")!=std::string::npos&&lseek(fileno(state.gasBudgetHost->output),0,SEEK_CUR)==offset);
SharedGasTrialPolicy::rollback(&state);SharedGasTrialPolicy::rollbackInterval(&state);
assert(state.rho[0]==1&&!state.gasBudgetHost->requestActive);
releaseSharedGasTrialStorage(state.gasTrial);releaseGasBudget(&state);assert(allocations==0);return 0;
'''.replace('TRUNCATE','1'if truncate_failure else'0')
    code=code.replace(needle,main+needle);source.write_text(code)
    b=subprocess.run(['g++','-std=c++17','-O0','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True);assert b.returncode==0,b.stderr
    path=tmp_path/'write-error.jsonl';r=subprocess.run([str(binary),'normal',str(path)],capture_output=True,text=True);assert r.returncode==0,r.stderr
    content=path.read_text();rows=content.splitlines()
    assert json.loads(rows[0])['record']=='NEW_SEGMENT'
    assert len(rows)==(2 if truncate_failure else 1)
    if truncate_failure:assert rows[1]=='{"s'
