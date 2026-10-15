"""C03A (deeeef2) original d6110c0 face-walk lineage: existing hop bound,
rounded remainder and contact stop. This does not demand the retired compensated
unbounded candidate trajectory.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT
@pytest.mark.parametrize('fsh',[0, 1])
@pytest.mark.parametrize('scenario',['progress','periodic_finish','contact_stop'])
def test_facewalk_full_time_lineage(tmp_path,fsh,scenario):
    fixture=((VALIDATION_ROOT / 'tests/fixtures/particle_tracking/wall_incident_velocity.cpp')).read_text().split('void run(const char* scenario',1)[0]
    body=(ROOT/'common/GpuParticleTransport.cuh').read_text()
    low='remainingDtLow' if 'remainingDtLow' in body else 'GpuTime(0)'
    if scenario=='progress':
        anchor='    for (int hop = 0; hop < s.maxFaceWalkHops && remainingDt > GPU_OPERATOR_R(0.0); ++hop)\n    {'
        assert anchor in body
        body='int observedHops=0;\n'+body.replace(anchor,anchor+'\n        ++observedHops;')
        main=r'''
int main(){DeviceState s;s.cellFaceKind[0]=s.cellFaceKind[1]=1;s.cellFaceRestitution[0]=s.cellFaceRestitution[1]=GPU_R(1);s.px[0]=GPU_R(-.25);s.pux[0]=s.puxOld[0]=GPU_R(1);
trackOneParticleLocalFaceWalk(s,0,1e17);require(observedHops==s.maxFaceWalkHops,"existing hop bound must retain its exact traversal count");require(std::isfinite(s.px[0])&&std::abs(s.pux[0])==1,"elastic bounded traversal preserves finite state and speed");}
'''
    elif scenario=='periodic_finish':
        main=r'''
int main(){DeviceState s;const GpuTime dt=1e17;s.nCells=2;s.cellFaceKind[0]=6;s.cellFaceNeighbor[0]=1;s.facePeriodicDx[0]=GpuReal(dt);s.planeD[2]=GPU_R(32);s.planeD[3]=GpuReal(2*dt);s.faceCx[2]=GPU_R(32);s.faceCx[3]=GpuReal(-2*dt);s.px[0]=GPU_R(-.25);s.pux[0]=s.puxOld[0]=GPU_R(1);
trackOneParticleLocalFaceWalk(s,0,dt);require(s.pStatus[0]==1&&s.pCellId[0]==1,"actual periodic traversal must finish in next cell");const double firstFraction=.25/dt;const double remaining=dt*(1.-firstFraction);const double expected=double(GPU_PERIODIC_COORDINATE(GPU_R(0),GpuReal(dt),GPU_R(1e-9),GPU_R(1)))+remaining;require(std::abs(double(s.px[0])-expected)<2e-7,"original rounded residual and periodic translation determine final position");require(s.pux[0]==1&&s.pStuck[0]==0,"ordinary velocity and physical state survive");}
'''
    else:
        body=body.replace('    inside = pointInsideCell(s, c, x, y, z);',f'    if(s.pStuck[i]!=0 && (remainingDt!=0 || {low}!=0))throw 44;\n    inside = pointInsideCell(s, c, x, y, z);')
        main=r'''
int main(){DeviceState s;s.cellFaceKind[0]=1;s.cellFaceKind[1]=5;s.cellFaceRestitution[0]=GPU_R(1);s.px[0]=GPU_R(-.25);s.pux[0]=s.puxOld[0]=GPU_R(1);trackOneParticleLocalFaceWalk(s,0,1e17);require(s.pStatus[0]==1&&s.pStuck[0]!=0&&s.pux[0]==0,"actual contact must settle full remaining interval");}
'''
    source=tmp_path/'walk.cpp';exe=tmp_path/'walk';source.write_text(fixture.replace('#include "GpuParticleTransport.cuh"',body)+main)
    command=['g++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all',f'-DINCIDENT_FSH={fsh}','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/wall'),str(source),'-o',str(exe)]
    compiled=subprocess.run(command,capture_output=True,text=True);assert compiled.returncode==0,compiled.stderr
    ran=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);assert ran.returncode==0,ran.stdout+ran.stderr
