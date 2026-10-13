import pathlib,shutil,difflib
from run_fast import D
source=pathlib.Path(__file__).with_name('profile_native.py');old=source.read_text()
shutil.copy2(source,D/'inputs/profile_native_v0.py')
(D/'logs/NSYS-analysis-v0-failure.log').write_text('NSYS-NATIVE-baseline exit0; NSYS-EXPORT-baseline exit0. Original analysis then raised:\nTraceback (most recent call last):\n  File "profile_native.py", line20, in <module>\n    with p.open("w") as h:w=csv.writer(h);w.writerow(cols);w.writerows(rows)\n_csv.Error: need to escape, but no escapechar set\nNo production or numerical failure. Existing nsys-rep/sqlite reused for next analysis.\n')
new=old.replace('shutil.copytree(src,case)','\n if not case.exists():shutil.copytree(src,case)')
new=new.replace("for relative in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(mesh/relative,case/relative)","for relative in ['constant/polyMesh','constant/solid/polyMesh']:\n  if not (case/relative).exists():shutil.copytree(mesh/relative,case/relative)")
start=new.index(" t=run('NSYS-NATIVE-'");end=new.index(' conn=sqlite3.connect(db)',start)
block=new[start:end]
replacement=" if variant=='baseline':\n  t={'id':'NSYS-NATIVE-baseline'};rep=prefix.with_suffix('.nsys-rep');db=prefix.with_suffix('.sqlite')\n else:\n"+''.join(' '+line+'\n' for line in block.splitlines())
new=new[:start]+replacement+new[end:]
new=new.replace("csv.writer(h)","csv.writer(h,quoting=csv.QUOTE_ALL,escapechar=chr(92))")
target=D/'inputs/profile_native_current.py';target.write_text(new)
(D/'patches/P012-nsys-csv-serialization.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/reviewer/profile_native_v0.py',tofile='b/reviewer/profile_native_current.py')))
for p in (D/'artifacts/performance/nsys').glob('baseline-*.csv'):
 if p.name=='baseline-StringIds.csv':shutil.copy2(p,D/'logs/NSYS-StringIds-v0-partial.csv')
exec(compile(new,str(target),'exec'))
