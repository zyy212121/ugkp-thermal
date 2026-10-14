from r3_runner import *
import re
# Normalize setup from the immutable base after the interrupted support preparer;
# these controls have never been executed. No failed physical input is replaced.
for name,tm in [('onecell-melting',600),('onecell-solidification',900)]:
 case=ROOT/'work/cases/phase'/name;p=case/'constant/chmtProperties';s=(ROOT/'work/cases/extra/wet-film-formal/constant/chmtProperties').read_text().replace('gasConductivity 1; gasViscosity 0;','gasConductivity 2; gasViscosity .01;').replace('conductivity 1;','conductivity 2;').replace('thickness 1e-5;','thickness .001;').replace('pressure 171600;','pressure 229600;').replace('couplingInterval 0.00050000000000000001; gasMaxDt 1.0000000000000001e-05;','couplingInterval 1e-6; gasMaxDt 1e-6;').replace('mean 350;','mean 800;').replace('e0 100000;','e0 200000;').replace('liquidViscosity .01; liquidReferencePressure 171600;','liquidViscosity 1; liquidReferencePressure 0;')
 s+='\nmeltTemperature '+str(tm)+';\nmaterial { phaseCondensed 0; enableMelting true; phaseFilmY (1 0); evaporation Disabled; }\nmeshMotion { policy CoupledRecession; }\n';p.write_text(s);shutil.copy2(p,D/'inputs/phase'/name/'constant/chmtProperties')
proof={'utc':utc(),'phase_setup_support_errors':['points compact ASCII 8((...)) not newline list; corrected by actual format','R2 has no authoritative ASCII alias; directly load immutable R1 Git blob'],'before_any_phase_onecell_or_active16_GPU':True,'no_physical_failure_replaced':True}
(D/'metadata/control-setup-recovery.json').write_text(json.dumps(proof,indent=2)+'\n')
alias='review/pr14/reviewer-deliveries/20261013T182722Z-fast/inputs/native/resident/CHT-coldWall2D-INPUT2/0/gpuResidentStrictParticles.dat';payload=subprocess.check_output(['git','-C',str(ROOT),'show',PATHS['prior_round1_evidence']+':'+alias]).decode();print('ASCII_ORIGINAL',payload)
assert 'SCHEMA3' in payload
assert git('diff',PATHS['old_source'],SOURCE,'--','applications/CHT','applications/FSH','common')==''
for app in ['CHT','FSH','FSHCudaBackend']:
 p=OLDROOT/'work/install/ns2/bin'/app;q=ROOT/'work/install/ns2/bin'/app;assert p.is_file();shutil.copy2(p,q);assert sha(p)==sha(q) and q.resolve().is_relative_to(Path(PATHS['repo']).resolve())
(D/'metadata/resident-binary-reuse.json').write_text(json.dumps({'unchanged_production_dependencies':True,'source_comparison':[PATHS['old_source'],SOURCE],'reused_binary_producer':PATHS['old_source'],'files':{a:sha(ROOT/'work/install/ns2/bin'/a) for a in ['CHT','FSH','FSHCudaBackend']}},indent=2)+'\n')
for app in ['CHT','FSH']:
 oldcase=OLDROOT/('work/cases/resident/'+app+'-cold2D-valid16-S214');case=ROOT/('work/cases/resident/'+app+'-active16-ASCII-S306');assert not case.exists();shutil.copytree(oldcase,case)
 for p in list(case.iterdir()):
  if p.is_dir() and p.name not in ['0','system','constant']:shutil.rmtree(p)
 lines=payload.splitlines();records=[q for q in lines if q.strip() and not q.startswith('#') and len(q.split())==16];assert len(records)==1,records
 row=records[0].split();assert float(row[5])==-5 and float(row[2])==5e-7 and int(row[13])==0
 rows=[]
 for i in range(16):
  r=row.copy();r[0]=format(.02+.04*(i%4),'.17g');r[1]=format(.02+.04*(i//4),'.17g');r[9]=str(i);r[12]=str(i+1);rows.append(' '.join(r))
 header=[q for q in lines if q not in records]
 new='\n'.join(re.sub(r'(SCHEMA3\s+)1\b',r'\g<1>16',q) for q in header)+'\n'+'\n'.join(rows)+'\n';p=case/'0/gpuResidentStrictParticles.dat';p.write_text(new)
 shutil.copytree(case,D/'inputs/resident'/case.name,ignore=shutil.ignore_patterns('polyMesh'))
 (D/'metadata'/('active16-'+app+'-input-provenance.json')).write_text(json.dumps({'original_fixed_commit':PATHS['prior_round1_evidence'],'original_repo_path':alias,'original_sha256':hashlib.sha256(payload.encode()).hexdigest(),'new_sha256':sha(p),'original_raw':payload,'new_raw':new,'only_record_changes':['x','y','cell','orig_id','SCHEMA3 count'],'mass_from_unchanged_particleProperties':(case/'constant/particleProperties').read_text(),'geometry_sha256':{str(q.relative_to(case)):sha(q) for q in (case/'constant/polyMesh').iterdir() if q.is_file()}},indent=2)+'\n')
 print('COLDREADY',app,new[:380],flush=True)
