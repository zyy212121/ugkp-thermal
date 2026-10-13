import pathlib,json,difflib,shutil
from run_fast import ROOT,D,sha
p=ROOT/'work/diagnostic-P011/applications/CHMT/gpu/Backend.cu';previous=p.read_text();old=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text()
for suffix in ['patch','json']:
 q=D/'patches'/('P011-Co-reduction-result-no-padding.'+suffix);shutil.copy2(q,D/'patches'/('P011v1-u64-diagnostic-build-failure.'+suffix))
new=old.replace('struct CoupledCourantValue {double value;bool unavailable;};','struct CoupledCourantValue {double value;bool unavailable;unsigned char initializedTail[7]{};};')
assert new!=old;p.write_text(new)
patch=D/'patches/P011-Co-reduction-result-no-padding.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/applications/CHMT/gpu/Backend.cu',tofile='b/applications/CHMT/gpu/Backend.cu')))
desc=json.loads((D/'patches/P011-Co-reduction-result-no-padding.json').read_text());desc.update({'representation':'v2: unchanged double+bool members/offsets, seven formerly padding bytes are explicitly initialized unsigned-char members; sizeof remains16. aggregate initializer value/bool/{} writes all storage.','diagnostic_v1_build_failure':'logs/BUILD-CHMT2-P011.log: CUDA13.1 sm_80_rt.h expected unqualified-id with uint64_t flag slot; kept separately as P011v1. No original source modification.','after_sha256':sha(p),'patch_sha256':sha(patch)})
(D/'patches/P011-Co-reduction-result-no-padding.json').write_text(json.dumps(desc,indent=2)+'\n')
source=pathlib.Path(__file__).with_name('co_padding_fix.py').read_text();chunk=source[source.index('binary=ROOT/'):]
chunk=chunk.replace("'BUILD-CHMT2-P011'","'BUILD-CHMT2-P011-v2'")
target=D/'inputs/co_padding_execute_v2.py';target.write_text('from run_fast import ROOT,D,run,cmd,amend,sha\nimport pathlib,shutil,json\noverlay=ROOT/"work/diagnostic-P011"\npatch=D/"patches/P011-Co-reduction-result-no-padding.patch"\n'+chunk)
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nP011首版u64 metadata槽在CUDA13.1编译报错，独立失败patch/log保留。v2保留bool成员与偏移，将原padding显式化为7个清零uchar metadata成员，仍16byte，既有max/OR与double计算不改；同native100步+全output bytes+initcheck复验，不降验收标准。\n')
exec(compile(target.read_text(),str(target),'exec'))
