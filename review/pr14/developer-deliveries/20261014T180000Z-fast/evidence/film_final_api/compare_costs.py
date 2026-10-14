from pathlib import Path
import importlib.util,subprocess,json
OUT=Path(__file__).resolve().parent
ROOT=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');APP=ROOT/'applications/CHMT'
spec=importlib.util.spec_from_file_location('counts',ROOT/'tests/test_film_incidence.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
results=[]
for label in ('before','after'):
 work=OUT/('cost_'+label);work.mkdir(exist_ok=True)
 includes=[]
 if label=='before':
  (work/'film').mkdir(exist_ok=True)
  header=(OUT/'CpuFilmDriver.H').read_text()
  types=subprocess.check_output(['git','show','HEAD:applications/CHMT/core/Types.H'],cwd=ROOT,text=True)
  start=types.index('struct FilmRateBudget {');end=types.index('};',start)+2
  header=header.replace('#include "core/HostState.H"','#include "core/HostState.H"\nnamespace chmt { '+types[start:end]+' }')
  (work/'film/CpuFilmDriver.H').write_text(header);includes=['-I'+str(work)]
 source=(OUT/'CpuFilmDriver.C').read_text() if label=='before' else (APP/'film/CpuFilmDriver.C').read_text()
 owner=work/'owner.C';owner.write_text(module.instrument(source))
 fixture=((OUT/'test_cpu_film.cpp') if label=='before' else (APP/'devtools/multirate/test_cpu_film.cpp')).read_text().replace('std::cout<<"CPU film tests passed\\n";}','std::cout<<"CPU film tests passed\\n";return 0;}')
 (work/'fixture.cpp').write_text(fixture)
 adapter=module.legacy_transport_adapter(header) if label=='before' else ''
 probe=adapter+(ROOT/'tests/fixtures/film_incidence/probe.cpp').read_text().replace('FILM_FIXTURE',str(work/'fixture.cpp')).replace('FILM_OWNER',str(owner));(work/'probe.cpp').write_text(probe)
 cmd=['g++','-std=c++14','-O2','-ffp-contract=off',*includes,'-I'+str(APP),'-I'+str(ROOT/'common'),str(work/'probe.cpp'),'-o',str(work/'probe')]
 b=subprocess.run(cmd,capture_output=True,text=True);(work/'build.log').write_text(b.stdout+b.stderr);assert b.returncode==0,b.stderr
 r=subprocess.run([str(work/'probe')],capture_output=True,text=True);(work/'stdout').write_text(r.stdout);(work/'stderr').write_text(r.stderr);assert r.returncode==0,r.stderr;results.append(r)
assert results[0].stdout==results[1].stdout,'physical stdout differs'
rows=[]
for a,b in zip(results[0].stderr.splitlines(),results[1].stderr.splitlines()):
 if a.startswith('COUNT ') or a.startswith('ADVCOUNT '):
  old=list(map(int,a.split()[1:]));new=list(map(int,b.split()[1:]));index=4 if a.startswith('COUNT ') else 1;rows.append({'kind':a.split()[0],'before':old,'after':new,'allocations_saved':old[index]-new[index],'bytes_saved':old[index+1]-new[index+1]})
 else:assert a==b,(a,b)
assert len(rows)==8
edges={}
for row in rows:
 if row['kind']=='COUNT':
  n,f=row['before'][:2];edges[n]=f
  assert row['allocations_saved']==2 and row['bytes_saved']==64*n+32*f
 else:
  n=row['before'][0]
  assert row['allocations_saved']==4 and row['bytes_saved']==2*(64*n+32*edges[n])
(OUT/'cost_physical_comparison.json').write_text(json.dumps({'scope':'Observed before-report-removal source snapshot versus current; not unavailable ancient historical incidence baseline','physical_stdout_exact':True,'advance_rows':rows},indent=2))
print(json.dumps(rows));print('PASS: actual physical stdout identical; measured COUNT/ADVCOUNT reductions; Evaluation output and report vectors removed')
