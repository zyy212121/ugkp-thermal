import pathlib,json,re,shutil,difflib,hashlib
from run_fast import ROOT,D,run,cmd,amend
directory=D/'inputs/differential-current';directory.mkdir(exist_ok=True)
original=ROOT/'tests/fixtures/operator_unification/baseline.json';frozen=json.loads(original.read_text());changes=[]
for app in ['gasUGKP','FSH','CHT']:
    key=app+':riemannFacePrimitiveForGradient';old=frozen[key];begin=old.index('    if (kind == 1)');end=old.index('    if (kind == 2)',begin);segment=old[begin:end]
    replacement='''        if (s.riemannBoundaryTFix[f] != 0)
        {
            face.T = clampMin(finiteOr(s.riemannBoundaryT[f], centre.T), s.TgasMin);
            face.rho = face.p/clampMin(s.Rgas*face.T, OfSmall);
        }
        return face;'''
    assert 'riemannBoundaryTFix' not in segment
    updated=segment.replace('        return face;',replacement);assert updated!=segment
    frozen[key]=old[:begin]+updated+old[end:];changes.append({'key':key,'diff':''.join(difflib.unified_diff(old.splitlines(True),frozen[key].splitlines(True)))})
(directory/'baseline-current.json').write_text(json.dumps(frozen,indent=2)+'\n')
generator=ROOT/'tests/fixtures/operator_unification/differential.py';old=generator.read_text()
new=old.replace("(root/'tests/fixtures/operator_unification/baseline.json')","(Path(__file__).with_name('baseline-current.json'))")
new=new.replace(".replace('COLD_TEST',cold)",".replace('COLD_TEST','if(!std::getenv(\"PR14_CORE_ONLY\")){'+cold+'}')")
assert new!=old
target=directory/'differential_current.py';target.write_text(new)
(D/'patches/P005-frozen-slip-wall-oracle.json').write_text(json.dumps({'classification':'test oracle correction/explicit scope','production_source_modified':False,'reason':'Frozen f03e215 snapshot predates base d611 slip-wall independent thermal BC fix c73591f. Exact source expression already present in baseline and PR.','expected_regression':'f21=3.25/500=0.0065, old frozen returns1.625.','changes':changes,'original_baseline_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'original_full_assertions_preserved':True,'core_scope':'All original moment/face/drag/pressure/wall atomic memcmp assertions remain; old cold block still in generated source but explicitly NOT_RUN for this core test. Original full fixture failure preserved. Native CHT cold full app evidence is separate.'},indent=2)+'\n')
(D/'patches/P005-differential-generator.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/tests/fixtures/operator_unification/differential.py',tofile='b/tests/fixtures/operator_unification/differential.py')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n## 追加 P005：冻结算子oracle的baseline适用性\n\n原full differential首败源自f03e215旧snapshot，比d611更早；仅补d611已有slip-wall独立thermal branch，不改输入/TFix或memcmp。CORE测试明确不执行旧cold段，其断言仍保留在生成源，原full失败保留；该段的paired gas producer/carry定义过时，不能据此验收cold。新core同flags同512×9输出/原压力拖曳等完整asserts，每应用300s。\n')
for app in ['gasUGKP','FSH','CHT']:
    out=ROOT/'work/cuda-operators'/('CORE-'+app)
    t=run('CUDA-CORE-P005-'+app,cmd('env','PR14_CORE_ONLY=1','python3',target,ROOT,out,app),[7,8,18,26,31],300,'Current baseline slip thermal formula, all original core bitwise assertions unchanged. Cold old assertions explicit NOT_RUN; no production modification.',inputs=[target,directory/'baseline-current.json'],kind='GPU')
    amend(t['id'],verdict='PASS' if t['exit_code']==0 else 'FAIL',test_support_patch='P005',observations=(D/t['log']).read_text())
    dest=D/'artifacts/cuda-operators'/('CORE-'+app);dest.mkdir(parents=True,exist_ok=True)
    for q in out.iterdir():
        if q.is_file() and q.suffix in ['.cu','.log']:shutil.copy2(q,dest/q.name)
