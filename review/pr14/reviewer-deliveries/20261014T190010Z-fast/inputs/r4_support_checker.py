from r4_runner import *
import difflib
orig=ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py';s=orig.read_text()
anchor="            values = {key: float(value) for key, value in row.items()}"
assert s.count(anchor)==1
replacement="""            # Test-side metadata parsing only: preserve the explicit NOT_COLLECTED label.
            metadata_key = 'sst_chemistry_integral_audit_collection'
            if metadata_key in row:
                require(row[metadata_key] == 'NOT_COLLECTED_LEGACY_CHECKPOINT_SLOTS_ONLY',
                        'unexpected audit collection metadata')
            values = {key: float(value) for key, value in row.items() if key != metadata_key}"""
new=s.replace(anchor,replacement);p=D/'inputs/check_case_metadata_support.py';p.write_text(new)
patch=D/'patches/S401-checker-uncollected-metadata.patch';patch.write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='a/check_case.py',tofile='b/test-only/check_case_metadata_support.py')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS401支持修正：正式thermal真实GPU/initcheck完成，但原checker所有CSV值强制float转换，在新增明确NOT_COLLECTED元数据列失败。正式FAIL已留 logs/CHECK-ORIGINAL-thermal-full.log；仅在测试sidecar识别该单列并验证标签后排除数值转换，不改任何physical字段/阈值/schema/输入，不删生产数据。支持版实际物理检查与正式原checker红分别记账；不将缺失audit算守恒零。\n')
t=run('CHECK-S401-thermal-full',cmd('python3',p,ROOT/'work/cases/thermal-full'),[6,14,26,29],expected='OnlynewNOT_COLLECTEDmetadatacolumnparsedseparately; alloriginalphysicalassertionsandthresholds unchanged.',inputs=[p,patch,orig,ROOT/'work/cases/thermal-full/verification.json'],kind='ANALYSIS',variant='SUPPORT_S401')
assertion(t,t['exit_code']==0,(D/t['log']).read_text(errors='replace'))
print((D/t['log']).read_text(),flush=True)
