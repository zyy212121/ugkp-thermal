from r3_runner import *
import difflib
old=D/'inputs/chemistry-fixture-S301.cu';text=old.read_text();new=text.replace('check(successful->gasSpecies.chemistryStatus[0],','check(bool(successful->gasSpecies.chemistryStatus[0]),');assert new!=text
source=D/'inputs/chemistry-fixture-S302.cu';source.write_text(new);patch=D/'patches/S302-explicit-status-bool.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/chemistry-fixture-S301.cu',tofile='b/chemistry-fixture-S302.cu')))
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS301首次nvcc支持编译红保留：ChemistryStatus explicit bool转换不能隐式传check(bool)。S302只加显式bool转换，不改任何assertion、状态/阈值、输入或生产算法；source/patch独立保存，compile900s。\n')
command=json.loads((ROOT/'work/fixtures/chemistry-trial-formal/command.json').read_text());binary=ROOT/'work/fixtures/chemistry-trial-S302';command[-3]=str(source);command[-1]=str(binary)
(D/'inputs/chemistry-fixture-S302-command.json').write_text(json.dumps(command,indent=2)+'\n')
t=run('BUILD-CHEMISTRY-S302',cmd(*command),[10,12,14,26],900,'Explicitstatusboolsupportcompileonly; S301red preserved; no physical/assertionchange.',inputs=[source,patch],kind='BUILD',variant='SUPPORT_S302');assert t['exit_code']==0
