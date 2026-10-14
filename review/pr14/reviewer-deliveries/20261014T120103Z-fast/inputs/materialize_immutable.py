from pathlib import Path
import sys,json,subprocess,hashlib
D=Path(__file__).resolve().parents[1];ROOT=D.parents[3];target=Path(sys.argv[1]).resolve();target.mkdir(parents=True,exist_ok=True)
assert not target.is_relative_to(ROOT/'review'), 'Use an independent testdirectory, not historicalreview paths'
for q in json.loads((D/'metadata/immutable-references.json').read_text())['references']:
 p=target/q['alias'];assert p.resolve().is_relative_to(target)
 data=subprocess.check_output(['git','-C',str(ROOT),'show',q['fixed_commit']+':'+q['repo_path']]);assert hashlib.sha256(data).hexdigest()==q['sha256']
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==q['sha256']
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print('Verifiedmaterialized references to',target)
