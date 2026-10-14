import difflib,json
from r2_native import *
src=D/'inputs/resident_check_fp64_mean.py';old=src.read_text()
new=old.replace("carry=p['pThermalEnthalpyResidual'];", "assert nodes.dtype==np.float64 if name.startswith('FSH') else nodes.dtype==np.float32\n   carry=np.zeros(len(p['pT']),dtype=np.float64) if name.startswith('FSH') else p['pThermalEnthalpyResidual'];")
p=D/'inputs/resident_check_fsh_double.py';p.write_text(new)
(D/'patches/S204-FSH-double-profile-checker.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/checker.py',tofile='b/checker-fsh-double.py')))
(D/'patches/S204-FSH-double-profile-checker.json').write_text(json.dumps({'classification':'TEST_SUPPORT','production_modified':False,'reason':'FSH current schema8 has double profiles and no CHT float-cache carry channel. Original checker CHT-only key raised KeyError. Require actualFSH float64 dtype and mean inverse1e-9 unchanged; CHT still requires realfloat32/carry. No tolerance/assert removed.','old_sha256':sha(src),'new_sha256':sha(p)},indent=2)+'\n')
name='FSH-coldWall1D';case=ROOT/'work/cases/resident'/name
t=run('FIELDS-S204-'+name,cmd('python3',p,'finite',name),[18,23,26,33],expected='Actualunchanged existing FSHrun output doubleprofile, no carrychannel; original1e-9Kinverse/finiteallfields.',inputs=[p],kind='ANALYSIS')
assertion(t,t['exit_code']==0,(D/t['log']).read_text());amend('NATIVE-'+name,verdict='PASS' if t['exit_code']==0 else 'FAIL',checker_result_id=t['id'],test_support='S204')
restart(case)
def fsh(case):
 p=case/'system/controlDict';p.write_text(re.sub(r'application\s+CHT;',r'application FSH;',p.read_text()))
 p=case/'constant/particleProperties';s=p.read_text()
 for key,value in [('wallDensity','1800'),('wallSpecificHeat','710'),('wallThermalConductivity','100')]:s=re.sub(r'\b'+key+r'\s+[^;]+;',key+' '+value+';',s)
 p.write_text(s)
case=resident('FSH-coldWall2D','CHT-coldWall2D-INPUT2',modified=fsh);restart(case)
