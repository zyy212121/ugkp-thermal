"""Real OF10 serial/MPI reductions under default dangerous-operation FPE traps."""
from pathlib import Path
import os,shlex,subprocess,tempfile,unittest
from validation.tools.paths import SOURCE_ROOT
ROOT=SOURCE_ROOT
class HostAuditReduction(unittest.TestCase):
 def test_serial_and_two_rank(self):
  foam=Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
  if not (foam/'etc/bashrc').exists():self.skipTest('OpenFOAM 10 unavailable')
  with tempfile.TemporaryDirectory(prefix='ugkp-host-audit-') as tmp:
   out=Path(tmp);q=shlex.quote
   script='source '+q(str(foam/'etc/bashrc'))+'\nset -eo pipefail\n'
   script+='export FOAM_USER_APPBIN='+q(str(out/'bin'))+' FOAM_USER_LIBBIN='+q(str(out/'lib'))+' XDG_CACHE_HOME='+q(str(out/'cache'))+' CUDA_CACHE_PATH='+q(str(out/'cuda-cache'))+' TMPDIR='+q(str(out/'tmp'))+'\n'
   script+='mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$TMPDIR"\n'
   script+='export PATH="$FOAM_USER_APPBIN:$FOAM_APPBIN:/usr/bin:/bin" LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN/dummy"\n'
   script+='g++ -std=c++17 -O2 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I'+q(str(ROOT/'applications/CHMT'))+' -I'+q(str(ROOT/'common'))+' -I$WM_PROJECT_DIR/src/OpenFOAM/lnInclude -I$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude '+q(str(Path(__file__).parent/'fixtures/host_audit/probe.C'))+' -L$FOAM_LIBBIN -lOpenFOAM -L$FOAM_LIBBIN/$FOAM_MPI -lPstream -o '+q(str(out/'probe'))+'\n'
   script+=q(str(out/'probe'))+'\n'
   script+='mpirun --allow-run-as-root --oversubscribe -np 2 '+q(str(out/'probe'))+' -parallel\n'
   (out/'run.sh').write_text(script);run=subprocess.run(['bash',str(out/'run.sh')],capture_output=True,text=True)
   evidence=os.environ.get('UGKP_HOST_AUDIT_OUTPUT')
   if evidence:
    p=Path(evidence);p.mkdir(exist_ok=True,parents=True);(p/'run.log').write_text(run.stdout+run.stderr);(p/'run.sh').write_text(script)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr);self.assertIn('PASS host audit ranks=1',run.stdout);self.assertIn('PASS host audit ranks=2',run.stdout)
