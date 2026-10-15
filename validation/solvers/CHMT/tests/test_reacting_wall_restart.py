from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
APP=(SOURCE_ROOT / 'applications/CHMT')
def test_wall_solver_metadata_does_not_refuse_restart(tmp_path):
    # Exercise the real checkpoint round-trip fixture with additional changes
    # to the new model identity, rather than inspecting serializer source text.
    source=((VALIDATION_ROOT / 'solvers/CHMT/tests/test_restart.cpp')).read_text()
    source=source.replace('change<9','change<13').replace('if(change==8)wrong.physics.singleGasSpecies=0;','if(change==8)wrong.physics.singleGasSpecies=0;\n  if(change==9)wrong.physics.wallModel.nodes+=1;\n  if(change==10)wrong.physics.wallModel.relativeTolerance*=2;\n  if(change==11)wrong.physics.wallWorkspaceSlots=3;\n  if(change==12)wrong.physics.wallModel.stretch+=1;')
    fixture=tmp_path/'restart.cpp';fixture.write_text(source)
    executable=tmp_path/'restart'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str((VALIDATION_ROOT / 'solvers/CHMT/tests')),'-I'+str(APP.parents[1]/'common'),str(fixture),str(APP/'restart/Checkpoint.C'),str(APP/'mesh/Geometry.C'),'-Wl,--wrap=fsync','-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)
