from pathlib import Path
import subprocess,json,hashlib
r=Path.cwd();o=r.parent/'chmt_efficiency_refresh_20261011/implementation/chmt_input';base='360bf94dd6c5344bb771d5f1f8b02a79bd041d92';env=r.parent/'chmt_implementation_20261011/native-environment/environment.sh';records=[]
for label,rev in [('old',base),('current',None)]:
 out=o/label;out.mkdir(exist_ok=True);include=out/'include';(include/'configuration').mkdir(parents=True,exist_ok=True)
 if rev:
  for name in ['InputValues.H','ModelIO.H','SharedGasModel.H']:
   src=subprocess.check_output(['git','show',rev+':applications/CHMT/configuration/'+name],text=True);src=src.replace('"../../../common/gasTransport/GasModelIO.H"','"'+str(r/'common/gasTransport/GasModelIO.H')+'"');(include/'configuration'/name).write_text(src)
 command=f'''set -eo pipefail
source {env}
export TMPDIR={o/'tmp'}
g++ -std=c++17 -O0 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I{include} -I{r/'applications/CHMT'} -I{r/'common'} -I{o.parent/'build_launch/frontend-chmt/generated'} -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" {r/'applications/CHMT/tests/test_input_admission_of.C'} -L"$FOAM_LIBBIN" -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o {out/'input_of'}
'''
 (out/'build.sh').write_text(command);result=subprocess.run(['bash',str(out/'build.sh')],capture_output=True,text=True);(out/'compile.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 rec={'label':label,'compile_exit':result.returncode,'build_script_sha256':hashlib.sha256(command.encode()).hexdigest(),'runs':{}}
 for mode in ['normal','missing','token','short']+(['admitted'] if not rev else []):
  run=f'source {env}; "{out/"input_of"}" {mode}';result=subprocess.run(['bash','-c',run],capture_output=True,text=True);(out/f'{mode}.log').write_text(result.stdout+result.stderr);rec['runs'][mode]={'command':run,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
  assert result.returncode==({'normal':0,'missing':1,'token':7,'short':1,'admitted':0}[mode]),(label,mode,result.stdout,result.stderr)
 records.append(rec);(o/'of-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
assert records[0]['runs']['normal']['stdout']==records[1]['runs']['normal']['stdout']
for mode in ['missing','token','short']:assert records[0]['runs'][mode]['exit']==records[1]['runs'][mode]['exit']
print('OF10 old/current2 normal serialized model outputs exact; actual missing/type/short failures retained; admitted typed nonfinite/count/modes passed')
