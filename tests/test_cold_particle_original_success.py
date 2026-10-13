"""Actual cold group/wrapper and finalizer comparisons to immutable original."""
from pathlib import Path
import subprocess
import hashlib
import re
import math
import pytest
from test_particle_original_success import ROOT, ORIGINAL, CURRENT_ROOT, FIXTURES, original_root


def function(text,name):
    start=text.index(name);start=text.rfind('\n',0,start)+1
    if text[start:].startswith(name):start=text.rfind('\n',0,start-1)+1
    brace=text.index('{',start);depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


def current_storage_fixture(text, fsh):
    """Historical source uses frozen declarations; current source uses M02 owners."""
    text=text.replace('    float coldH[24]{}, coldRing[24]{}, coldFrozen[3]{}, pCold2DFrozenArea[3]{};',
        '    ProfileStorage coldH[24]{}; double coldRing[24]{}, coldFrozen[3]{}, pCold2DFrozenArea[3]{};')
    text=text.replace('struct DeviceState {', ('using ProfileStorage=double;\n' if fsh else 'using ProfileStorage=float;\n')+'struct DeviceState {')
    text=text.replace('float* compactPColdNodeSpecificEnthalpy','ProfileStorage* compactPColdNodeSpecificEnthalpy').replace('float* compactPCold2DNodeSpecificEnthalpy','ProfileStorage* compactPCold2DNodeSpecificEnthalpy')
    text=text.replace('float cold2DH','ProfileStorage cold2DH').replace('float* pCold2DNodeSpecificEnthalpy','ProfileStorage* pCold2DNodeSpecificEnthalpy').replace('float* pColdNodeSpecificEnthalpy','ProfileStorage* pColdNodeSpecificEnthalpy')
    for field in ['pContactMaximumArea','pContactDuration','pColdRingSolidMass','pColdFrozenArea']:
        text=text.replace('float '+field,'double '+field).replace('float* '+field,'double* '+field)
    text=text.replace('double inventory(const float* h','template<class Storage> double inventory(const Storage* h')
    if fsh:
        start=text.index('#define publishPairedParticleGasHeat actualPublishPairedParticleGasHeat')
        stop=text.index('#include "GpuColdWall1DDevice.cuh"',start)
        text=text[:start]+'std::atomic<int> pairedCallbackCalls{0};\n'+text[stop:]
    return text


def compare_historical_physics(reference, current):
    # The historical file represents float publications. This comparison bounds
    # that reference's quantization; it is not a current solver tolerance.
    token=r'[-+]?(?:0x[0-9a-f.]+p[-+]?\d+|inf|nan)'
    before=re.findall(token,reference);after=re.findall(token,current)
    assert len(before)==len(after)
    assert re.sub(token,'VALUE',reference)==re.sub(token,'VALUE',current)
    for a,b in zip(before,after):
        x,y=float.fromhex(a),float.fromhex(b)
        if math.isnan(x):assert math.isnan(y)
        elif math.isinf(x):assert x==y
        else:assert math.isfinite(y) and abs(x-y)<=64*2**-23*max(abs(x),abs(y),1e-30)


def build(folder,root,fsh):
    folder.mkdir(parents=True,exist_ok=True)
    text=(root/'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh').read_text()
    body=function(text,'__device__ GPU_OPERATOR_REAL molecularGasConductivity')
    (folder/'route_conductivity.cuh').write_text('#include "gasTransport/GasStateView.H"\n'+body)
    text=(root/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    body=function(text,'__device__ void recoverGasPrimitiveCell')
    (folder/'actual_gas_recovery.cuh').write_text('#include "gasTransport/GasCapabilities.H"\n#include "gasTransport/MixtureThermo.H"\n#include "gasTransport/GasGeometryValidation.H"\n'+body)
    two=(root/'common/wall/GpuColdWall2DDevice.cuh').read_text()
    two=two.replace('        ColdWallThermalReal power[','        if(radialNode==0)observedStepTimes.push_back(stepTime);\n        ColdWallThermalReal power[')
    marker='''        for(int traceNode=0;traceNode<8;++traceNode)observedTrace[radialNode].push_back(enthalpy[traceNode]);
        observedTrace[radialNode].push_back(localWallEnergyJ);
#ifdef GPU_PAIRED_PARTICLE_HEAT
        observedTrace[radialNode].push_back(localGasEnergyJ);
#endif
'''
    anchor='        if (substep < 256)\n        {' if '        if (substep < 256)\n        {' in two else ('        const GpuTime correctedStep' if '        const GpuTime correctedStep' in two else '        elapsedTime += stepTime;')
    two=two.replace(anchor,marker+anchor)
    (folder/'actual_2d_group.cuh').write_text(two)
    prefix=(FIXTURES/'cold_prefix.cpp').read_text()
    cases=(FIXTURES/'cold_cases.cpp').read_text()
    if root.resolve()==CURRENT_ROOT.resolve():
        prefix=current_storage_fixture(prefix,fsh)
        cases=cases.replace('float(', 'ProfileStorage(').replace('for(float damage:', 'for(double damage:')
        if fsh:
            cases=cases.replace('int interval(double multiple){','int interval(double multiple,bool closed=false){')
            cases=cases.replace('areaFactor=GPU_R(1e-5),gasG=GPU_R(1e-5);','areaFactor=closed?0.:GPU_R(1e-5),gasG=closed?0.:GPU_R(1e-5);')
            cases=cases.replace(' const double dt=stable*multiple;', ' long double initialSum=0;for(int j=0;j<64;++j)initialSum+=s.cold2DH[j];\n const double dt=stable*multiple;')
            witness=r'''
 if(closed){long double finalSum=0;for(int j=0;j<64;++j)finalSum+=s.cold2DH[j];
  require(errors==0,"closed FSH solve succeeds");require(wall==0,"closed FSH wall energy is zero");
  require(std::abs(finalSum-initialSum)<=64*std::numeric_limits<double>::epsilon()*std::abs(initialSum),"independent closed-node enthalpy conservation");
  static_assert(sizeof(s.cold2DH[0])==8&&sizeof(s.coldRing[0])==8&&sizeof(s.pContactDuration[0])==8,"current FSH physical owners are double");
  std::puts("CLOSED_DOUBLE_CONSERVATION");
 }
'''
            cases=cases.replace(' printf("multiple=',witness+' printf("multiple=')
            cases=cases.replace('if(mode=="interval")return interval', 'if(mode=="closed")return interval(7.125,true);if(mode=="interval")return interval')

    src=folder/'probe.cpp';src.write_text(prefix+cases)
    exe=folder/'probe';cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-pthread',f'-DROUTE_FSH={fsh}','-I'+str(folder),'-I'+str(root/'common'),'-I'+str(root/'common/wall'),'-I'+str(ROOT/'tests/gas_transport'),str(src),'-o',str(exe)]
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=90);(folder/'build.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
    return exe


@pytest.fixture(scope='module',params=[0,1])
def executables(request,tmp_path_factory,original_root):
    fsh=request.param;folder=tmp_path_factory.mktemp(f'cold_original_{64}_{fsh}')
    return fsh,build(folder/'original',original_root,fsh),build(folder/'corrected',CURRENT_ROOT,fsh)


def run(exe,*args):
    result=subprocess.run([str(exe),*args],capture_output=True,text=True,timeout=60)
    (exe.parent/('_'.join(args)+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


@pytest.mark.parametrize('multiple',['7.125','3.0000000000015','256'])
def test_cold2d_original_success_full_fields(executables,multiple):
    fsh,before,after=executables
    a=run(before,'interval',multiple);b=run(after,'interval',multiple)
    assert 'errors=0' in a
    expected_steps={'7.125':8,'3.0000000000015':3,'256':256}[multiple]
    assert f'steps={expected_steps} ' in a
    compare_historical_physics(a,b)


def test_masked_resistance_old_success(executables):
    fsh,before,after=executables
    a=run(before,'resistance');b=run(after,'resistance');assert 'errors=0' in a;compare_historical_physics(a,b)


def test_fsh_effusivity_complete_wrapper(executables):
    fsh,before,after=executables
    if not fsh:pytest.skip('this route is the FSH configured-property fallback')
    a=run(before,'effusivity');b=run(after,'effusivity');assert 'errors=0' in a;compare_historical_physics(a,b)


def test_adhesion_complete_finalizer(executables):
    fsh,before,after=executables
    a=run(before,'adhesion');b=run(after,'adhesion');assert 'errors=0' in a;compare_historical_physics(a,b)


def test_cold2d_terminal_extension_preserves_original_prefix(executables):
    fsh,before,after=executables
    a=run(before,'interval','320.25');b=run(after,'interval','320.25')
    assert 'errors=8' in a and 'errors=0' in b
    assert 'steps=256 ' in a
    original_times=a.split('TIMES',1)[1].split('NODES',1)[0].split()
    corrected_times=b.split('TIMES',1)[1].split('NODES',1)[0].split()
    assert len(original_times)==256
    assert corrected_times[:256]==original_times
    for lane in range(8):
        aa=a.split(f'TRACE{lane}',1)[1].splitlines()[0].split()
        bb=b.split(f'TRACE{lane}',1)[1].splitlines()[0].split()
        assert aa
        assert hashlib.sha256(" ".join(bb[:len(aa)]).encode()).hexdigest()==hashlib.sha256(" ".join(aa).encode()).hexdigest()


@pytest.mark.parametrize('case',['contact','geometry'])
def test_census_bounded_actual_consumers(executables,case):
    fsh,before,after=executables
    a=run(before,case);b=run(after,case)
    compare_historical_physics(a,b)


@pytest.mark.parametrize('case',['multiplicity1d','multiplicity2d'])
def test_original_multiplicity_wrapper_outcome(executables,case):
    fsh,before,after=executables
    if not fsh:pytest.skip('nonpaired FSH wrapper outcome, not a finite-ledger claim')
    a=run(before,case);b=run(after,case)
    assert 'errors=0' in a
    compare_historical_physics(a,b)


def test_current_fsh_double_profile_closed_energy(executables):
    fsh, _, current = executables
    if not fsh:
        pytest.skip('FSH double-profile owner witness; CHT carry has its own tests')
    assert 'CLOSED_DOUBLE_CONSERVATION' in run(current, 'closed')
