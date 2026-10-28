from pathlib import Path
import subprocess, sys, unittest
R=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles/repo')
sys.path.insert(0,str(R))
scope={'__name__':'original_contract'}
text=subprocess.check_output(['git','show','503d484f:validation/solvers/gasUGKP/tests/test_common_gas_numerics_source_contract.py'],cwd=R,text=True)
exec(compile(text,'original_contract.py','exec'),scope)
with (R.parent/'review-before-source-contract.log').open('w') as f:
 result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(scope['CommonGasNumericsSourceContract']))
assert result.testsRun==4 and len(result.failures)==1 and not result.errors
print('Original contract reproduced: 3 pass, 1 failure')
