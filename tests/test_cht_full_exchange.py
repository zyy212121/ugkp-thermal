"""Full production CHT thermal coupler with a recording resident boundary.

This executes actual CPU numerical/state/IO owners, not CUDA kernels or the
native resident adapter. Recording payloads are not claimed codec coverage.
"""
from pathlib import Path
import importlib.util
import shutil
import re
import subprocess
import os
import pytest
ROOT=Path(__file__).resolve().parents[1];THERMAL=ROOT/'applications/CHT/thermal'
spec=importlib.util.spec_from_file_location('t14_case_writer',ROOT/'tests/test_cht_surface_temperature.py');cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
def observe_restart(source):
    """Count actual .C open/helper executions; never alter production sources."""
    source=source.replace('Foam::IFstream stream(path);','Foam::IFstream stream(path); ++t14DictionaryOpens;')
    for symbol in ['canonicalConfigurationSha1','canonicalMeshTopologySha1','canonicalWallTemperatureSha1','fileSha1','validateFiniteRestartFields']:
        counter='t14FieldScans' if symbol=='validateFiniteRestartFields' else 't14HashCalls'
        source=re.sub(r'('+symbol+r'\s*\([^{}]*\)\s*\{)',lambda m:m[1]+' ++'+counter+';',source,count=1)
    return 'int t14DictionaryOpens=0, t14HashCalls=0, t14FieldScans=0;\n'+source

@pytest.fixture(scope='module')
def exchange_probe(tmp_path_factory):
    if not shutil.which('wmake'):pytest.skip('Foundation OpenFOAM required')
    build=tmp_path_factory.mktemp('exchange_native');(build/'Make').mkdir()
    for n in ['exchange.C','RecordingResident.H']:shutil.copy(ROOT/'tests/fixtures/cht_thermal'/n,build/n)
    names=['GpuSolidThermalCoupler','GpuSolidThermalProperties','GpuSolidPatchMapper','GpuThermalExchangeState','GpuParticleRadiationCoupler','MieDoRadiationSolver','MieTable']
    for n in names:
        source=(THERMAL/(n+'.C')).read_text()
        if n=='GpuSolidThermalCoupler':source=source.replace('#include "../gpu/GpuResidentStrict.H"','#include "RecordingResident.H"')
        if n=='GpuThermalExchangeState':source=observe_restart(source)
        (build/(n+'.C')).write_text(source)
    (build/'Make/files').write_text('exchange.C\n'+'\n'.join(n+'.C' for n in names)+f'\nEXE = {build}/exchange\n')
    (build/'Make/options').write_text(f'EXE_INC = -DUGKWP_USE_CUDA -I{THERMAL} -I{ROOT}/applications/CHT/gpu -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    p=subprocess.run(['wmake'],cwd=build,capture_output=True,text=True);(build/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    return build/'exchange'

def make_case(path,mode):
    cases.make_case(path,1000,transverse_cells=3 if mode.startswith("pending_") else 1)
    t=path/'0/T';t.write_text(t.read_text().replace('type fixedGradient; gradient uniform 0;','type fixedValue; value uniform 300;'))
    for region,start,T in [('solid',-1,300),('aux',-2,100)]:
        seed=path/(region+'_mesh');cases.make_case(seed,0,transverse_cells=3 if mode.startswith("pending_") else 1)
        mesh=seed/'system/blockMeshDict';text=mesh.read_text()

        text=text.replace('vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1))',f'vertices (({start} 0 0) ({start+1} 0 0) ({start+1} 1 0) ({start} 1 0) ({start} 0 1) ({start+1} 0 1) ({start+1} 1 1) ({start} 1 1))')
        mesh.write_text(text);p=subprocess.run(['blockMesh','-case',str(seed)],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
        for base in ['constant','system','0']:(path/base/region).mkdir(parents=True,exist_ok=True)
        shutil.copytree(seed/'constant/polyMesh',path/'constant'/region/'polyMesh')
        for f in ['fvSchemes','fvSolution']:shutil.copy(seed/'system'/f,path/'system'/region/f)
        value=(seed/'0/T').read_text().replace('uniform 300',f'uniform {T}').replace('right { type zeroGradient;','right { type fixedGradient; gradient uniform 0;')
        (path/'0'/region/'T').write_text(value)
    props='rho {type constant; value 1000;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}'
    three=mode in ('outer_one','outer_zero','outer_reversed','outer_zero_k');outer=0 if mode=='outer_zero' else 1
    if three:
        selection=('formatVersion 2; fluidRegion region0; primarySolidRegion solid; solidRegions {solid {properties {%s}} aux {properties {%s}}} fluidSolidInterfaces ({fluidPatch left; solidPatch right; solidRegion solid; mapping conservativeNonConformal;}); solidSolidInterfaces ({firstRegion aux;secondRegion solid; firstPatch right;secondPatch left; mapping oneToOneConformal;}); solidInterfaceMaxIterations %d; solidInterfaceTemperatureRelativeTolerance 0; solidInterfaceEnergyRelativeTolerance 0;' % (props,props,outer))
    else:
        selection='formatVersion 1; solidRegion solid; mapping oneToOneConformal; coupledInterfaces ({fluidPatch left;solidPatch right;}); properties {%s}' % props

    if mode=='outer_zero_k':selection=selection.replace('kappa {type constant; value 6;}','kappa {type constant; value 0;}')
    if mode=='outer_reversed':selection=selection.replace('firstRegion aux;secondRegion solid; firstPatch right;secondPatch left;', 'firstRegion solid;secondRegion aux; firstPatch left;secondPatch right;').replace('formatVersion 2','formatVersion 99')
    radiation='radiationEnabled false;'
    if mode=='radiation':
        spec=importlib.util.spec_from_file_location('t14_mie_fixture',ROOT/'applications/CHT/devtools/tests/test_radiation_continuation.py');mie=importlib.util.module_from_spec(spec);spec.loader.exec_module(mie);mie.table(path/'constant/mieTable')
        radiation='radiationEnabled true; radiationCouplingFrequency 1; radiation { mieTable "constant/mieTable"; do { IBoundaryCondition diffuseGrey; nTheta 2; nPhi 4; maxSourceIterations 200; sourceRelativeTolerance 1e-12; phaseBalanceTolerance 1e-12; maxPhaseBalanceIterations 200; radiationActiveEpsilonMin 0; conservationRelativeTolerance 1e-10; maxPhaseScalingCorrection 100; }}'
    (path/'constant/exchangeConfig').write_text(cases.header('dictionary','exchangeConfig')+f'testMode {mode}; resolvedParticleContract {{rhoSolid 1000;particleHeatFactorJkgK 1;TpMin 1;TpMax 4000;particleCapacity 10;}} solidThermalCoupling {{{selection} couplingInterval 0.1; {radiation} nonlinearMaxIterations 200; nonlinearRelativeTolerance 1e-10;energyRelativeTolerance 1e-8;energyAbsoluteTolerance 1e-8;}}')
@pytest.mark.parametrize('mode',['radiation','outer_one','outer_zero','outer_zero_k','outer_reversed','mismatch','consume_error','pending_reordered','pending_missing','legacy_absent','legacy_one_absent','legacy_two_absent','pending_legacy_present','missing_current','missing_unknown'])
def test_full_actual_exchange(exchange_probe,tmp_path,mode):
    make_case(tmp_path,mode)
    env=os.environ.copy();env.pop('FOAM_SIGFPE',None)
    p=subprocess.run([str(exchange_probe),'-case',str(tmp_path)],capture_output=True,text=True,env=env)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'PASS '+mode in p.stdout
