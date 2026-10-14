from r3_runner import *
import difflib
text=old_read('inputs/checkpoint_actual_reader_S207.cpp').decode();new=text
new=new.replace('std::printf("},\\\"gasMesh\\\":");mesh(s.gasMesh);',r'''std::printf("},\"physics\":{\"liquidRho\":");num(model.physics.liquid.rho);
 std::printf(",\"phaseCondensed\":%d,\"enableMelting\":%d,\"meltTemperature\":",model.physics.material.phaseCondensed,int(model.physics.material.enableMelting));num(model.physics.meltTemperature);
 std::printf(",\"liquidE0\":");num(model.physics.liquid.e0);std::printf(",\"liquidReferencePressure\":");num(model.physics.liquidReferencePressure);
 std::printf(",\"condensedRho\":[");for(int i=0;i<Nc;++i){if(i)std::printf(",");num(model.physics.condensed[i].rho);}std::printf("]}");
 std::printf(",\"solidMesh\":");mesh(s.solidMesh);
 std::printf(",\"surface\":{\"area\":");arr(s.surface.area);std::printf(",\"gasFace\":");arr(s.surface.gasFace);std::printf(",\"solidFace\":");arr(s.surface.solidFace);std::printf(",\"solidCell\":");arr(s.surface.solidCell);std::printf("}");
 std::printf(",\"gasMesh\":");mesh(s.gasMesh);''')
anchor='std::printf(",\\\"gasFront\\\":");num(a.gasFront);'
assert anchor in new
new=new.replace(anchor,anchor+r'''std::printf(",\"temperature\":");num(a.temperature);std::printf(",\"meanVelocity\":");vec(a.meanVelocity);std::printf(",\"normal\":");vec(a.normal);std::printf(",\"baseVelocity\":");vec(a.baseVelocity);std::printf(",\"normalVelocity\":");num(a.normalVelocity);std::printf(",\"solidNormalVelocity\":");num(a.solidNormalVelocity);''')
anchor='std::printf(",\\\"energy\\\":");num(q.energy);std::printf("}");}'
# Add actual stored material porosity beside condensed/pore/energy.
solidstart=new.index('std::printf("],\\\"solid\\\":[")');solidend=new.index('std::printf("],\\\"sst\\\":[")')
segment=new[solidstart:solidend];assert anchor in segment
new=new[:solidstart]+segment.replace(anchor,'std::printf(",\\\"energy\\\":");num(q.energy);std::printf(",\\\"porosity\\\":");num(q.porosity);std::printf("}");}')+new[solidend:]
source=D/'inputs/checkpoint-full-wet-observer.cpp';source.write_text(new);patch=D/'patches/S303-checkpoint-full-wet-observation.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/prior/checkpoint_actual_reader_S207.cpp',tofile='b/checkpoint-full-wet-observer.cpp')))
assert '"physics' not in new or new!=text
app=ROOT/'applications/CHMT';binary=ROOT/'work/fixtures/checkpoint-full-ASan-2';binary.parent.mkdir(parents=True,exist_ok=True)
flags=['-std=c++17','-O1','-g','-DUGKWP_GAS_SPECIES=2','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(app),'-I'+str(ROOT/'common')]
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS303真实codec只读字段补充：原dry checker明确只计gas+solid、只允许GasSolid/PoreGas，不能验wet film库存；原checker红完整保留为不适用scope。原restart还缺checker-only restart_directory配置，其红不等于solver恢复失败。扩大原真实checkpoint decoder打印实际FilmAux均速/normal/物性/材料选择/porosity/solidVolume，对新GPU产物算完整film H-pM/rho+KE、vector P和逐recipient账；字段观察不变reader/schema/数值标准，hostASan/UBSan编译/run分列。\n')
t=run('BUILD-CHECKPOINT-FULL-ASAN',cmd('g++',*flags,source,app/'restart/Checkpoint.C','-o',binary),[5,6,13,19,20,26,28],900,'Unchangednativecodec underactualhostASan/UBSan, actualstoredwet inventoryfields only; notGPU surrogate.',inputs=[source,patch,app/'restart/Checkpoint.C'],kind='BUILD',variant='SUPPORT_S303');assert t['exit_code']==0
for name in ['WET-FILM-FORMAL','WET-FILM-FORMAL-RESTART']:
 out=D/'artifacts'/name
 for cp in sorted(p for p in out.glob('checkpoint-*') if p.is_dir()):
  t=run('DECODE-'+name+'-'+cp.name,cmd(binary,cp),[5,6,13,19,20,26,28],300,'ActualnewGPU checkpoint, unchangeddecoder hostASan/UBSan; fullstoredstateobserved.',inputs=[cp/'state.chmt',source],binary=binary,kind='HOST_SANITIZER',variant='SUPPORT_S303')
  if t['exit_code']==0:
   state=json.loads((D/t['log']).read_text());(out/(cp.name+'-full-decoded.json')).write_text(json.dumps(state,indent=2)+'\n');assertion(t,True,'Actualnativecodec/noASanUBSanobservederrors; dataavailableforindependentphysicalaudit, notacceptancefromdecode.')
print('FULL_CODEC_READY',sha(binary),flush=True)
