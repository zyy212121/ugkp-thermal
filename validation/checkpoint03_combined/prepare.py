#!/usr/bin/env python3
"""Compile-only against complete materialized trees. Never launch a compiled fixture."""
from pathlib import Path
import argparse,ast,hashlib,json,os,shlex,shutil,subprocess,time
CORE=('pressure','face','coupling','bath','probe')
EXTRA=('ordered','radiation','sparse','coldwall','cht_codec','chmt_codec','chmt_closure')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def fixture_inputs(here,controls,family):
 paths=[]
 if family=='face':paths += [controls/'generate_face.py',controls/'face_fixture.cpp']
 elif family in ('pressure','coupling','bath'):paths += [controls/(family+'_fixture.'+('cu' if family=='pressure' else 'cpp'))]
 elif family in ('ordered','sparse','coldwall'):paths += [here/(family+'_fixture.cu')]
 elif family=='cht_codec':paths += [here/'generate_cht_codec.py']
 elif family=='radiation':paths += [here/'radiation_fixture.cpp']
 elif family in ('chmt_codec','chmt_closure'):
  paths += [here/(family+'.cpp'),here/'chmt_support.H',here/'chmt_build_recipe.json']
  if family=='chmt_codec':paths += [here/'chmt_seed.H',here/'chmt_legacy_writer.cpp']
 return paths
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--variant',choices=('baseline','candidate'),required=True)
 ap.add_argument('--families',nargs='+',choices=CORE+EXTRA);ap.add_argument('--bits',type=int,nargs='+',choices=(32,64),default=(64,32))
 ap.add_argument('--source32',type=Path);ap.add_argument('--source64',type=Path)
 ap.add_argument('--manifest32',type=Path);ap.add_argument('--manifest64',type=Path)
 ap.add_argument('--arch',default='sm_89');ap.add_argument('--revision',default='01');a=ap.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((out/'compiled.json').read_text()) if (out/'compiled.json').exists() else {'status':'PREPARING_NOT_RUN','programs':{},'variants':{},'commands':[]}
 root=(a.baseline if a.variant=='baseline' else a.candidate).resolve();here=Path(__file__).resolve().parent
 controls=here/'legacy' if a.variant=='baseline' else here
 cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))
 families=a.families or (CORE if a.variant=='baseline' else CORE+EXTRA)
 def command(argv,key):
  argv=list(map(str,argv));log=out/(key+'.build.log')
  with log.open('x') as f:
   record={'argv':argv,'cwd':str(root),'log':log.name,'started':time.time(),'build_coordinator_sha256':digest(Path(__file__).resolve())};meta['commands'].append(record);save(out/'compiled.json',meta)
   result=subprocess.run(argv,cwd=root,stdout=f,stderr=subprocess.STDOUT)
  record.update(exit=result.returncode,finished=time.time())
  if '-o' in argv:
   artifact=Path(argv[argv.index('-o')+1])
   if artifact.is_file():record['output']={'path':str(artifact),'sha256':digest(artifact),'bytes':artifact.stat().st_size}
  if '-MF' in argv:
   dep=Path(argv[argv.index('-MF')+1])
   if dep.is_file():
    names=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
    record['dependency_file']={'path':str(dep),'sha256':digest(dep)}
    record['non_system_dependencies']={str(Path(n).resolve()):digest(Path(n)) for n in sorted(set(names)) if Path(n).is_file()}
  save(out/'compiled.json',meta)
  if result.returncode:raise RuntimeError('Compile failed; evidence '+str(log))
 for bits in a.bits:
  source=(getattr(a,f'source{bits}') or root/'.build/precision'/f'fp{bits}'/'source').resolve()
  manifest_path=(getattr(a,f'manifest{bits}') or source.parent/'source-manifest.json').resolve()
  manifest=json.loads(manifest_path.read_text())
  hashes={name:digest(source/name) for name in manifest['files'] if name not in ('validation/checkpoint03/prepare.py','validation/checkpoint03_combined/prepare.py')}
  meta['variants'].setdefault(a.variant,{})[str(bits)]={'tree':str(source),'manifest_path':str(manifest_path),'manifest_sha256':digest(manifest_path),'manifest':manifest,'source_sha256':hashes,
   'lineage_review':'Parent reviews complete production binary/source build evidence manually; no new automatic build-admission gate.'}
  includes=['-I'+str(source/x) for x in ('common','common/gasNumerics','common/gpu','common/wall','validation/tests/gas_transport')]+['-I'+str(source),'-I'+str(cuda/'include')]
  host=['g++','-std=c++17','-O2','-ffp-contract=off']+includes
  cu=[str(cuda/'bin/nvcc'),'--Ofast-compile=min','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','-arch='+a.arch]+includes
  for family in families:
   key=f'{a.variant}-fp{bits}-{family}';buildkey=key+'-v'+a.revision
   work=out/a.variant/f'fp{bits}'/(family+'-v'+a.revision);work.mkdir(parents=True,exist_ok=False)
   binary=work/family;compiler=host;units=[];args=[];gpu=False
   if family=='face':
    for name in ('generate_face.py','face_fixture.cpp'):shutil.copy2(controls/name,work/name)
    command(['python3',work/'generate_face.py','--root',source,'--output',work/'generated.cpp']+(['--candidate32-trend'] if a.variant=='candidate' and bits==32 else []),buildkey+'-generate');units=[work/'generated.cpp']
   elif family=='probe':
    tree=ast.parse((source/'validation/pr41/native_injection.py').read_text())
    body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROBE' for t in n.targets))
    (work/'probe.cpp').write_text(body);units=[work/'probe.cpp'];compiler=host+['-I'+str(source/'applications/CHMT')]
   elif family in ('pressure','coupling','bath'):
    name=family+'_fixture.'+('cu' if family=='pressure' else 'cpp');shutil.copy2(controls/name,work/name);units=[work/name]
    if family=='pressure':compiler=cu;gpu=True
    # GCC11 ree pass ICE workaround, identical for CP2/candidate and32/64.
    if family=='coupling':compiler=host+['-fno-ree']+(['-DCP3_FP64'] if bits==64 else [])+(['-DCP3_CELL_PREPARES_HEAT'] if a.variant=='candidate' else [])
   elif family in ('ordered','radiation'):
    radiation=source/'validation/checkpoint03/radiation'
    name='ordered_fixture.cu' if family=='ordered' else 'radiation_fixture.cpp';shutil.copy2(here/name,work/name);units=[work/name];gpu=True
    if family=='ordered':compiler=cu
    else:
     objects=[]
     for unit in ('DiscreteOrdinateRadiation','MieOpticalProperties'):
      obj=work/(unit+'.o');command(cu+['-c',source/'common/gpu'/(unit+'.cu'),'-o',obj],buildkey+'-'+unit);objects.append(obj)
     units+=objects;compiler=host;args=['-L'+str(cuda/'lib64'),'-Wl,-rpath,'+str(cuda/'lib64'),'-lcudart']
   elif family in ('sparse','coldwall'):
    name=family+'_fixture.cu';shutil.copy2(here/name,work/name);units=[work/name];compiler=cu;gpu=True
    if family=='sparse':units.append(source/'common/gpu/SparseConjugateGradient.cu')
   elif family=='cht_codec':
    shutil.copy2(here/'generate_cht_codec.py',work/'generate_cht_codec.py')
    command(['python3',work/'generate_cht_codec.py','--root',source,'--output',work/'cht_codec.cpp'],buildkey+'-generate')
    units=[work/'cht_codec.cpp'];compiler=host+(['-DCP3_FP32'] if bits==32 else [])
   else:
    all_recipes=json.loads((here/'chmt_build_recipe.json').read_text());recipe=all_recipes[family]
    # Recipe files list the worker's isolated historical/current translation units.
    # The worker may provide a compile-only helper; it must never run a fixture.
    repl=lambda x:str(x).replace('{source}',str(source)).replace('{controls}',str(here)).replace('{fixture}',str(here)).replace('{output}',str(work)).replace('{build}',str(work)).replace('{bits}',str(bits))
    if family=='chmt_codec':
     for historical in all_recipes.get('precompile_objects',[]):
      historical_argv=[repl(x) for x in historical['argv']]
      historical_argv+=['-MMD','-MF',str(work/(historical['name']+'.d'))]
      command(historical_argv,buildkey+'-'+historical['name'])
      history=Path(historical['source_root']);history_manifest=history.parent/'source-manifest.json'
      if history_manifest.exists():
       hm=json.loads(history_manifest.read_text());meta.setdefault('historical_sources',{})[historical['name']]={'tree':str(history),'manifest':hm,'sha256':{n:digest(history/n) for n in hm['files']}}
    if 'compile_only_argv' in recipe:
     command([repl(x) for x in recipe['compile_only_argv']],buildkey+'-helper')
     candidate=Path(repl(recipe['executable']));shutil.copy2(candidate,binary) if candidate!=binary else None
     units=[]
    else:
     units=[repl(x) for x in recipe['sources']+recipe.get('objects',[])];compiler=host+recipe.get('compile_flags',[])+['-I'+repl(x) for x in recipe.get('include_dirs',[])]+['-D'+x for x in recipe.get('defines',[])];args=[repl(x) for x in recipe.get('link_flags',[])]
   uninstrumented=list(compiler)
   if family=='pressure':compiler=compiler+['-DCP3_COUNT_PRESSURE']+(['-DCP3_EXPECT_SHARED'] if a.variant=='candidate' else [])
   if units:
    if Path(str(compiler[0])).name=='g++':
     objects=[]
     for index,unit in enumerate(units):
      unit=Path(unit)
      if unit.suffix=='.o':objects.append(unit);continue
      obj=work/(f'{index:02d}-'+unit.stem+'.o');dep=obj.with_suffix('.d')
      command(compiler+['-MMD','-MF',dep,'-c',unit,'-o',obj],buildkey+'-object'+str(index));objects.append(obj)
     linkmap=work/'link.map'
     command(compiler+objects+args+['-Wl,-Map='+str(linkmap),'-Wl,-t','-o',binary],buildkey+'-link')
     # readelf inspects ELF data; it never starts the compiled fixture (unlike ldd).
     elf=['readelf','-d','-l',str(binary)]
     info=subprocess.run(elf,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
     (work/'elf-dependencies.txt').write_text(info.stdout)
     loaded={}
     for line in linkmap.read_text().splitlines():
      if line.startswith('LOAD '):
       library=Path(line[5:].strip())
       if library.is_file():loaded[str(library.resolve())]=digest(library)
     save(work/'link-dependencies.json',{'readelf_argv':elf,'readelf_exit':info.returncode,'link_map_sha256':digest(linkmap),'loaded_files_sha256':loaded,'executable_was_run':False})
    else:command(compiler+units+args+['-o',binary],buildkey)
   perf=binary
   if family=='pressure':perf=work/'pressure-perf';command(uninstrumented+units+args+['-o',perf],buildkey+'-perf')
   if key in meta['programs']:meta.setdefault('superseded_programs',[]).append({'id':key,'program':meta['programs'][key]})
   meta['programs'][key]={'executable':str(binary.relative_to(out)),'sha256':digest(binary),'performance_executable':str(perf.relative_to(out)),'performance_sha256':digest(perf),
    'gpu':gpu,'fixture_sources':{str(f.relative_to(here)):digest(f) for f in fixture_inputs(here,controls,family)},
    'correctness_args':['correctness'] if family in CORE else [],'performance_args':['perf']}
   save(out/'compiled.json',meta)
 expected={f'{v}-fp{b}-{f}' for v in ('baseline','candidate') for b in (32,64) for f in (CORE if v=='baseline' else CORE+EXTRA)}
 meta['status']='COMPILED_NOT_RUN' if expected<=meta['programs'].keys() else 'PREPARING_NOT_RUN';save(out/'compiled.json',meta)
 print(meta['status']+'; no numerical execution')
if __name__=='__main__':main()
