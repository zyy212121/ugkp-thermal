from r2_runner import *
import urllib.request
result_path=ROOT/'work/round2-publication-result.json';q=json.loads(result_path.read_text());commit=q['commit']
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin','refs/heads/'+q['branch']],text=True,env=env).split()[0]
assert remote==commit,(remote,commit)
request=urllib.request.Request('https://api.github.com/repos/zyy212121/ugkp-thermal/pulls/14',headers={'User-Agent':'PR14-independent-FAST-review'})
with urllib.request.urlopen(request,timeout=30) as response:pr=json.load(response)
assert pr['head']['sha']==commit and pr['head']['ref']==q['branch'] and pr['state']=='open' and not pr['merged'],(pr['head']['sha'],pr['state'],pr['merged'])
assert pr['head']['repo']['full_name']=='zyy212121/ugkp-thermal'
manifest=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show',commit+':review/pr14/manifest.json']))
assert manifest['latest_reviewer_delivery']==str(D.relative_to(ROOT/'review/pr14'))
with urllib.request.urlopen(urllib.request.Request('https://raw.githubusercontent.com/zyy212121/ugkp-thermal/'+commit+'/'+str((D/'REPORT.md').relative_to(ROOT)),headers={'User-Agent':'PR14-independent-FAST-review'}),timeout=30) as response:raw=response.read()
assert hashlib.sha256(raw).hexdigest()==sha(D/'REPORT.md')
src=json.loads((D/'metadata/source-hashes.json').read_text());source_matches=0
for t in src:
 assert git('rev-parse',commit+':'+t['path'])==t['blob'],t['path'];source_matches+=1
for line in (D/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert sha(D/name)==digest
q.update(verified_utc=utc(),remote_head=remote,github_PR_head=pr['head']['sha'],PR_open=True,PR_merged=False,raw_report_sha256=hashlib.sha256(raw).hexdigest(),production_blobs_verified=source_matches,remote_tree_equals_record_tree=True,document_validation=json.loads((D/'logs/validate-records.json').read_text()),verification_scope='Remote git head, GitHub PR metadata and raw fixed-commit report verified; all1152 production/source blob IDs unchanged in published record commit. No PR comments/description/merge actions.')
result_path.write_text(json.dumps(q,indent=2)+'\n')
shutil.copy2(result_path,P/'round2-publication-result.json');shutil.copy2(D/'REPORT.md',P/'PR14-round2-REPORT.md')
print(json.dumps(q,indent=2),flush=True)
