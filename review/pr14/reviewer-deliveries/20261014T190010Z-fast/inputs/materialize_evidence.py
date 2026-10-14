from pathlib import Path
import sys,json,subprocess,hashlib,gzip,shutil
D=Path(__file__).resolve().parents[1];repo=D.parents[3];target=Path(sys.argv[1]).resolve();area=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test').resolve()
assert target.is_relative_to(area) and not target.is_relative_to(repo/'review'), 'Use an independent authorizedtestoutput directory, never historicalreview paths'
target.mkdir(parents=True,exist_ok=True)
for p in D.rglob('*'):
 if p.is_file():
  q=target/p.relative_to(D);assert q.resolve().is_relative_to(target);q.parent.mkdir(parents=True,exist_ok=True)
  if not q.exists():shutil.copy2(p,q)
for r in json.loads((D/'metadata/immutable-references.json').read_text())['references']:
 b=subprocess.check_output(['git','-C',str(repo),'show',r['fixed_commit']+':'+r['repo_path']]);assert hashlib.sha256(b).hexdigest()==r['sha256'];p=target/r['alias'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
for r in json.loads((D/'metadata/lossless-storage.json').read_text())['entries']:
 p=target/r['alias'];p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(D/r['stored_path'],'rb') as a,p.open('wb') as b:shutil.copyfileobj(a,b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['original_sha256']
print('Exacthistorical anddecompressed rawbyte aliases verified:',target)
