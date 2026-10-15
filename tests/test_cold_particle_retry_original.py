"""Original conditional PCR retry through the actual cold-wall wrapper."""
from pathlib import Path
import subprocess
import hashlib
import json
import pytest
import test_cold_particle_original_success as cold
from test_particle_original_success import ROOT, CURRENT_ROOT, original_root

AUDIT=r'''
#include <vector>
std::vector<double> auditEntry[2][8],auditIterations[2][8];
int auditAttempts[2][8]{},auditPublishes[2][8]{},auditFailures[2][8]{},auditSignFailures[2][8]{},auditUnchanged[2][8]{},auditWallPosts=0;
std::vector<double> auditSnapshot(const DeviceState& s,int lane){return {double(s.coldH[lane]),double(s.coldRing[lane]),double(s.coldAge[0]),double(s.coldFrozen[0]),s.pThermalEnthalpyResidual[0],s.heatBudget[0],double(s.pT[0]),s.rhoE[0],double(s.deposited[0]),double(s.reflected[0]),double(s.pStatus[0]),double(s.pStuck[0])};}
'''
def build(folder,root):
 folder.mkdir(parents=True,exist_ok=True)
 for source,name,header in [('updateLegacyGasBoundaryMirrorKernel.cuh','molecularGasConductivity','#include "gasTransport/GasStateView.H"\n'),('computeSstFaceFluxKernel.cuh','recoverGasPrimitiveCell','#include "gasTransport/GasCapabilities.H"\n#include "gasTransport/MixtureThermo.H"\n#include "gasTransport/GasGeometryValidation.H"\n')]:
  text=(root/'common/operators'/source).read_text();fn=cold.function(text,('__device__ GPU_OPERATOR_REAL ' if name=='molecularGasConductivity' else '__device__ void ')+name);(folder/('route_conductivity.cuh' if name=='molecularGasConductivity' else 'actual_gas_recovery.cuh')).write_text(header+fn)
 (folder/'actual_2d_group.cuh').write_text((root/'common/wall/GpuColdWall2DDevice.cuh').read_text())
 source=(root/'common/wall/GpuColdWall1DDevice.cuh').read_text().replace('            atomicAddParticleWallEnergyByFace\n','            ++auditWallPosts;\n            atomicAddParticleWallEnergyByFace\n')
 (folder/'GpuColdWall1DDevice.cuh').write_text(source)
 advance=(root/'common/wall/GpuColdWall1DAdvance.inl').read_text();advance='''\n const int auditAttempt=0;
 ++auditAttempts[auditAttempt][lane];auditEntry[auditAttempt][lane]=auditSnapshot(s,lane);
'''+advance
 anchor='        acceptedWallPower = wallPower;';advance=advance.replace(anchor,anchor+'\n        auditIterations[auditAttempt][lane].push_back(candidateEnthalpy);auditIterations[auditAttempt][lane].push_back(guessedTemperature);')
 anchor='            if (linearSolveFailed != nullptr) *linearSolveFailed = true;';advance=advance.replace(anchor,anchor+'''\n            ++auditFailures[auditAttempt][lane];
            auditSignFailures[auditAttempt][lane]=std::isfinite(candidateEnthalpy)&&std::isfinite(guessedTemperature)&&(candidateEnthalpy<0||!(guessedTemperature>0));
            auditUnchanged[auditAttempt][lane]=(auditEntry[auditAttempt][lane]==auditSnapshot(s,lane));
''');(folder/'GpuColdWall1DAdvance.inl').write_text(advance)
 pub=(root/'common/wall/GpuColdWall1DPublish.inl').read_text();pub=pub.replace('    s.pColdNodeSpecificEnthalpy[nodeBase + lane] =','    ++auditPublishes[auditAttempt][lane];\n    s.pColdNodeSpecificEnthalpy[nodeBase + lane] =');(folder/'GpuColdWall1DPublish.inl').write_text(pub)
 prefix=(cold.FIXTURES/'cold_prefix.cpp').read_text().replace('#include "GpuColdWall1DDevice.cuh"',AUDIT+'\n#include "GpuColdWall1DDevice.cuh"')

 if root.resolve()==CURRENT_ROOT.resolve():prefix=cold.current_storage_fixture(prefix,False)
 src=folder/'probe.cpp';src.write_text(prefix+(cold.FIXTURES/'cold_retry_cases.cpp').read_text());exe=folder/'probe';cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-pthread','-DROUTE_FSH=0','-I'+str(folder),'-I'+str(root/'common'),'-I'+str(root/'common/wall'),'-I'+str(ROOT/'tests/gas_transport'),str(src),'-o',str(exe)];r=subprocess.run(cmd,capture_output=True,text=True);(folder/'build.log').write_text(r.stdout+r.stderr);(folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');assert r.returncode==0,r.stderr;return exe


@pytest.fixture(scope='module')
def retry_executables(tmp_path_factory,original_root):
    folder=tmp_path_factory.mktemp(f'cold_retry_{64}')
    return 64,build(folder/'original',original_root),build(folder/'corrected',CURRENT_ROOT)


def output(exe,case):
    run=subprocess.run([str(exe),str(case)],capture_output=True,text=True,timeout=30)
    (exe.parent/f'case_{case}.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    return run.stdout


@pytest.mark.parametrize('case',[0])
def test_original_pcr_wrapper_and_publication_exact(retry_executables,case):
    _,before,after=retry_executables
    a=output(before,case);b=output(after,case)
    header=a.splitlines()[0]
    for expected in ['errors=0','first=1','unchanged=1','gascalls=1','wallposts=1']:
        assert expected in header
    for expected in ['retry=0','publish0=1','publish1=0']:
        assert expected in header
    cold.compare_historical_physics(a,b)
