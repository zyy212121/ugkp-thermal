from pathlib import Path
import subprocess,json,hashlib,re
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b4/v2')
src=root/'receipts/zero-rank-unused-caloric.cpp';src.write_bytes((root.parent/'review/zero-rank-unused-caloric.cpp').read_bytes());exe=root/'receipts/zero-rank-v2';cmd=['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-DFINAL','-I'+str(repo),str(src),'-o',str(exe)]
build=subprocess.run(cmd,capture_output=True,text=True);(root/'logs/paired-build.log').write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
result=subprocess.run([str(exe)],capture_output=True,text=True);(root/'logs/paired-v2.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
base=root.parent/'review/zero-rank-base.log';expected=base.read_text();normalize=lambda x:re.sub(r' available=[01]','',x)
assert normalize(result.stdout)==normalize(expected)
assert all('available=1' in x for x in result.stdout.splitlines())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'paired-receipt.json').write_text(json.dumps({'build_command':cmd,'build_exit_code':build.returncode,'run_command':[str(exe)],'run_exit_code':result.returncode,'source_sha256':sha(src),'executable_sha256':sha(exe),'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'reused_original_stdout':str(base),'reused_original_stdout_sha256':sha(base),'comparison':'Original status, thermo status and all untouched Jacobian sentinels match exactly in finite, missing-backing and unused-overflow modes. The original API has no output-availability argument, so only its printed available=0 field is excluded. The revised API has actual empty-output available=1 in all three cases.'},indent=2)+'\n')
print(result.stdout)
