"""Wall inputs must pass the real build gate and preserve paired ownership."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
APP=(
    'gpu/BoundaryLayerInput.H',
    'private_backend/BoundaryLayerAdapter.cuh',
    'private_backend/BoundaryLayerStorage.cuh',
    'tests/test_boundary_layer_lifecycle.py',
    'tests/test_boundary_layer_protocol.py',
    'tests/test_boundary_layer_storage.py',
    'tests/test_native_boundary_layer_frontend.py',
)
COMMON=(
    'gasTransport/GasBoundaryLayerEvaluation.H',
    'gasTransport/GasBoundaryLayerHost.H',
    'gasTransport/GasBoundaryLayerModelState.H',
    'gasTransport/GasBoundaryLayerState.H',
    'gasTransport/GasBoundaryLayerWorkspace.H',
    'gasWall/ConstantTransportWall.H',
    'gasWall/ReactingWallLayer.H',
    'gasWall/WallGeometryBuilder.H',
    'gasWall/WallLayerMath.H',
    'gasWall/WallModel.H',
    'gasWall/WallModelTypes.H',
    'gasWall/WallPrecision.H',
    'gasWall/WallThermoMath.H',
    'operators/evaluateGasBoundaryLayerKernel.cuh',
)
WALL_FILES=tuple('applications/gasUGKP/'+p for p in APP)+tuple('common/'+p for p in COMMON)
REPOS=('gpu-riemann-gkp-main','ugkp-thermal')


def test_wall_inputs_pass_the_actual_standalone_build_gate():
    env=dict(os.environ);env.pop('UGKP_MANAGED_MIRROR_ROOT',None)
    result=subprocess.run([sys.executable,str(ROOT/'tools/managed_mirrors.py')],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def make_wall_pair(tmp_path):
    data=json.loads((ROOT/'tools/managed_mirrors.json').read_text())
    entries={entry['path']:entry for entry in data['mirrors']}
    for path in WALL_FILES:
        upstream='ugkp-thermal' if path.startswith('common/') else 'gpu-riemann-gkp-main'
        assert path in entries,'unregistered wall input: '+path
        assert entries[path]['upstream']==upstream
        assert entries[path]['downstream']==next(repo for repo in REPOS if repo!=upstream)
    # A real mirror checker over a bounded isolated pair, using the production
    # classification and actual wall input bytes; no external repository writes.
    data['mirrors']=[entries[path] for path in WALL_FILES]
    data['local_only']={repo:[] for repo in REPOS}
    for repo in REPOS:
        root=tmp_path/repo;(root/'tools').mkdir(parents=True)
        (root/'tools/managed_mirrors.json').write_text(json.dumps(data))
        shutil.copy2(ROOT/'tools/managed_mirrors.py',root/'tools/managed_mirrors.py')
        for path in WALL_FILES:
            dst=root/path;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,dst)
    return tmp_path


def check(pair,*args):
    return subprocess.run([sys.executable,str(pair/'ugkp-thermal/tools/managed_mirrors.py'),'--pair-root',str(pair),*args],capture_output=True,text=True)


def test_wall_pair_checks_and_repairs_only_in_declared_directions(tmp_path):
    wall_pair=make_wall_pair(tmp_path)
    checked=check(wall_pair);assert checked.returncode==0,checked.stdout+checked.stderr
    controls=(('common/gasWall/WallModel.H','ugkp-thermal','gpu-riemann-gkp-main'),
              ('applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh','gpu-riemann-gkp-main','ugkp-thermal'))
    expected={}
    for path,upstream,downstream in controls:
        src=wall_pair/upstream/path;dst=wall_pair/downstream/path
        expected[path]=src.read_bytes();dst.write_bytes(b'downstream drift\n')
    rejected=check(wall_pair);assert rejected.returncode==1
    assert all('mirror drift: '+path in rejected.stdout for path,_,_ in controls)
    repaired=check(wall_pair,'--sync');assert repaired.returncode==0,repaired.stdout+repaired.stderr
    for path,upstream,downstream in controls:
        assert (wall_pair/upstream/path).read_bytes()==expected[path]
        assert (wall_pair/downstream/path).read_bytes()==expected[path]
