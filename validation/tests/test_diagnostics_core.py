"""Compile and execute the portable diagnostics and real OF10 host adapters."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

ROOT = SOURCE_ROOT
SOURCES = ROOT / 'validation' / 'diagnostics'
ENV = Path(os.environ.get('UGKP_TEST_ENV', '/workspace/shared/toolchains/env.sh'))
CUDA_ARCH = os.environ.get('UGKWP_CUDA_ARCH', 'sm_80')

def toolchain_setup():
    return f'source {ENV} >/dev/null 2>&1' if ENV.exists() else ':'


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize('name', ['core', 'view'])
def test_portable_diagnostics(name, tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    binary = tmp_path / name
    run([compiler, '-std=c++17', '-Wall', '-Wextra', '-pedantic', '-pthread',
         '-I'+str(ROOT/'common'), str(SOURCES/f'test_{name}.cpp'), '-o', str(binary)])
    run([str(binary)])


def test_real_openfoam_dispatch(tmp_path):
    if not ENV.exists() and not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('OpenFOAM environment unavailable')
    binary = tmp_path / 'foam'
    command = f'''{toolchain_setup()}
    g++ -std=c++17 -pthread -DWM_DP -DWM_LABEL_SIZE=32 -I{ROOT}/common \
      -I$WM_PROJECT_DIR/src/OpenFOAM/lnInclude -I$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude \
      {SOURCES}/test_foam.cpp -L$FOAM_LIBBIN -L$FOAM_LIBBIN/$FOAM_MPI -lOpenFOAM -lPstream -o {binary} && {binary}
    '''
    result=run(['bash','-c',command])
    assert 'FOAM Warning' in result.stdout
    muted=run(['bash','-c',f'{toolchain_setup()}; UGKP_DIAGNOSTIC_WARNINGS=0 {binary}'])
    assert 'FOAM Warning' not in muted.stdout+muted.stderr
    result=subprocess.run(['bash','-c',f'{toolchain_setup()}; {binary} exit'],capture_output=True,text=True)
    assert result.returncode==1
    assert 'FOAM FATAL ERROR' in result.stdout+result.stderr
    assert 'observed=-4' in result.stdout+result.stderr
    result=subprocess.run(['bash','-c',f'{toolchain_setup()}; mpirun --allow-run-as-root -np 2 {binary} mpi'],capture_output=True,text=True,timeout=20)
    assert result.returncode==1,result.stdout+result.stderr
    assert 'MPI_ABORT' in result.stdout+result.stderr
    assert '[1] --> FOAM Warning' in result.stdout+result.stderr
    assert 'FOAM parallel run exiting' in result.stdout+result.stderr


def test_cuda_diagnostics_compile_and_device_if_available(tmp_path):
    if not ENV.exists() and not shutil.which('nvcc'):
        pytest.skip('CUDA compiler unavailable')
    binary=tmp_path/'device'
    run(['bash','-c',f'{toolchain_setup()}; nvcc -std=c++17 -arch={CUDA_ARCH} -I{ROOT}/common {SOURCES}/test_device.cu -o {binary}'])
    result=subprocess.run(['bash','-c',f'{toolchain_setup()}; {binary}'],capture_output=True,text=True)
    if result.returncode==77:
        pytest.skip('CUDA graph test compiled; GPU unavailable for execution')
    assert result.returncode==0,result.stdout+result.stderr


def test_chmt_direct_api_compiles(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    run([compiler,'-std=c++17','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),
         '-c',str(SOURCES/'test_chmt_api.cpp'),'-o',str(tmp_path/'chmt-api.o')])


def test_chmt_failed_microstep_has_no_snapshot_replay():
    source=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text()
    assert 'restoreGasSnapshot' not in source
    assert 'CoupledTrialPolicy::rollback' not in source
    assert 'rollbackGasWindow' not in source


def test_returning_fatal_sink_and_missing_context_cannot_continue(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    binary=tmp_path/'backstop'
    run([compiler,'-std=c++17','-I'+str(ROOT/'common'),str(SOURCES/'test_backstop.cpp'),'-o',str(binary)])
    for argument in ('','missing-context'):
        result=subprocess.run(['bash','-c',f'ulimit -c 0; {binary} {argument}'],capture_output=True,text=True)
        assert result.returncode not in (0,99)


def test_catalogue_has_unique_real_source_locations():
    import re
    catalogue=(ROOT/'common/DiagnosticsCatalogue.H').read_text()
    entries=re.findall(r'\{(\d+),"([^"]+)","([^"]+)",(\d+),"([^"]+)"\}',catalogue)
    assert entries
    seen=set()
    actual_calls=production_diagnostic_callsites()
    for site,path,function,line,description in entries:
        assert site not in seen,site
        seen.add(site)
        source=(ROOT/path).read_text()
        lines=source.splitlines()
        assert 0<int(line)<=len(lines),(site,path,line)
        assert re.search(r'\b'+site+r'\b',lines[int(line)-1]),(site,path,line)
        name=function.split('::')[-1]
        assert name not in ('if','constexpr','for','while','switch'),(site,function)
        assert re.search(r'\b'+re.escape(name)+r'\s*(?:<[^>]*>)?\s*\(',source),(site,function)
        assert description
        if int(site) in actual_calls:
            assert actual_calls[int(site)]==(path,int(line)), (site,actual_calls[int(site)],(path,int(line)))


def test_cuda_transfer_catalogue_points_to_record_producer():
    import re
    catalogue=(ROOT/'common/DiagnosticsCatalogue.H').read_text()
    match=re.search(r'\{1,"([^"]+)","([^"]+)",(\d+),"',catalogue)
    assert match is not None
    assert (match[1],int(match[3]))==production_diagnostic_callsites()[1]
    assert match[2]=='recordCudaFailure'


def test_chmt_reduction_does_not_read_fields_after_fatal(tmp_path):
    source=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text()
    functors=source[source.index('struct CoupledCourantMaximum'):source.index('static bool currentGasCourant')]
    probe=tmp_path/'reduction.cpp'
    probe.write_text('''#define __host__
#define __device__
#include "Diagnostics.H"
#include "gasTransport/GasCapabilities.H"
#include <cmath>
#include <limits>
#include <cassert>
'''+functors+'''
int main(){
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,1,1);
    double field=4; int status=0;
    CoupledScalarRead scalar{&field,&context};
    CoupledCourantRead combined{&field,&status,&context};
    assert(scalar(0)==4 && combined(0).value==4);
    ugkp::diagnostics::recordFatal(&context,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::NonFinite));
    scalar.field=reinterpret_cast<const double*>(1);
    combined.field=reinterpret_cast<const double*>(1);
    combined.status=reinterpret_cast<const int*>(1);
    assert(scalar(0)==-std::numeric_limits<double>::infinity());
    assert(combined(0).value==-std::numeric_limits<double>::infinity());
}
''')
    binary=tmp_path/'reduction'
    run(['c++','-std=c++17','-I'+str(ROOT/'common'),str(probe),'-o',str(binary)])
    run([str(binary)])


def test_chmt_missing_selected_mixture_operator_has_no_config_fallback():
    source=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text()
    assert 'CHMT mixture requested flux=' not in source
    assert 'selected mixture gas operator is unavailable' in source


def production_diagnostic_callsites():
    import re
    positions={'makeRecord':1,'wallFatal':2,'wallWarning':2,'reportPhysicsWarning':2,
               'clipPhysicsMassFraction':2,'gasRecordFaceFailure':3,
               'gasRecordEvaluatedFaceFailure':3,'gasRecordTransportDiagnostic':4,'failure':1,
               'gasEddyFinite':3,'gasProducerScalarValid':3,'gasStencilIndexFailure':3,'interfaceFailure':1,'materialInversionFailure':1,'reactionControlFailure':2}
    seen={}
    resolved_sources=set()
    for folder in ('common','applications'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.suffix not in ('.H','.cuh','.cu','.C') or path.name.startswith('Diagnostics'):
                continue
            # A compatibility symlink is the same physical call site, not a second ID.
            resolved=path.resolve()
            if resolved in resolved_sources:
                continue
            resolved_sources.add(resolved)
            source=path.read_text()
            masked=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                          lambda m: ''.join('\n' if c=='\n' else ' ' for c in m[0]),source)
            for match in re.finditer(r'\b('+'|'.join(positions)+r')\s*\(',masked):
                start=masked.index('(',match.start());depth=0;args=[];last=start+1
                for end in range(start+1,len(masked)):
                    ch=masked[end]
                    if ch==')' and depth==0:
                        args.append((last,end));break
                    if ch in '([{':depth+=1
                    elif ch in ')]}':depth-=1
                    elif ch==',' and depth==0:args.append((last,end));last=end+1
                index=positions[match[1]]
                if match[1]=='failure':
                    if path.name=='GasBoundaryLayerEvaluation.H':index=0
                    elif path.name!='gasFaceSubgridTransportProperties.cuh':continue
                if len(args)<=index:continue
                a,b=args[index];expression=masked[a:b]
                if re.search(r'\b(?:int|unsigned|uint32_t|sourceSiteId)\b|=',expression):continue
                for value in re.finditer(r'\b\d+\b',expression):
                    site=int(value[0])
                    if site!=1 and not 1000<=site<=65535:continue
                    location=(str(path.relative_to(ROOT)),source[:a+value.start()].count('\n')+1)
                    assert site not in seen,(site,seen.get(site),location)
                    seen[site]=location
    return seen


def test_production_diagnostic_callsite_ids_are_unique():
    import re
    seen=production_diagnostic_callsites()
    catalogueIds={int(site) for site in re.findall(r'\{(\d+),"', (ROOT/'common/DiagnosticsCatalogue.H').read_text())}
    assert set(seen)<=catalogueIds, sorted(set(seen)-catalogueIds)


def test_cuda_fatal_read_is_load_not_atomic_rmw(tmp_path):
    if not ENV.exists() and not shutil.which('nvcc'):
        pytest.skip('CUDA compiler unavailable')
    ptx=tmp_path/'diagnostic-load.ptx'
    run(['bash','-c',f'{toolchain_setup()}; nvcc -std=c++17 -arch={CUDA_ARCH} -I{ROOT}/common -ptx {SOURCES}/test_atomic_load.cu -o {ptx}'])
    code=ptx.read_text()
    assert 'atom.global.add' not in code
    assert 'ld.relaxed.gpu.b32' in code or 'ld.relaxed.gpu.global.u32' in code


def test_native_gas_boundary_layer_input_compiles(tmp_path):
    """Compile the real OF frontend adapter, not a mock of its diagnostic API."""
    if not ENV.exists() and not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('configured OpenFOAM toolchain unavailable')
    source=tmp_path/'boundary_layer_input.cpp'
    source.write_text('#include "BoundaryLayerInput.H"\n')
    setup=f'{toolchain_setup()}' if ENV.exists() else ':'
    command=f'''{setup}
    g++ -std=c++17 -fsyntax-only -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \\
      -DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=2 \\
      -I$WM_PROJECT_DIR/src/finiteVolume/lnInclude -I$WM_PROJECT_DIR/src/meshTools/lnInclude \\
      -I$WM_PROJECT_DIR/src/OpenFOAM/lnInclude -I$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude \\
      -I{ROOT}/common -I{ROOT}/applications/gasUGKP/gpu {source}
    '''
    run(['bash','-c',command])
