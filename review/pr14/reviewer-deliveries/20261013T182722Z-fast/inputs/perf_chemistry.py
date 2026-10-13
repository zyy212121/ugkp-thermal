import pathlib,json,shutil,statistics
from run_fast import ROOT,D,run,cmd,amend,sha
prefix=(D/'inputs/chemistry_retry.cu').read_text().split('int main(')[0]
main=pathlib.Path(__file__).with_name('perf_chemistry_main.txt').read_text();src=D/'inputs/perf_chemistry.cu';src.write_text(prefix+main)
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加真实化学kernel分段效率：same d611/PR actual advanceGasChemistryKernel、sm89/O2/lineinfo/fmad=false，Ns2 1和32768cell，原成功dt1e-6/maxSteps100，3warmup/7sample。所有模型和state字段真实device backing，完整publication逐字段bit hash、独立一阶反应解析species绝差1e-9（局部1e-6积分），密度closure1e-10，保守state不写精确相等；kernel CUDA event、host launch、H2D/reset、D2H、wait与pipeline分别报告，CUDA event float仅计时。900s编译/300s执行，各版本当轮分开运行，运行时无其他GPU测试；相同输入未新增门禁。\n')
dest=D/'artifacts/performance/chemistry';dest.mkdir(parents=True,exist_ok=True);summary={}
for variant,inc,macro in [('baseline',ROOT/'work/baseline-common',['-DBASELINE']),('original',ROOT,[])]:
 binary=ROOT/'work'/('perf-chem-'+variant)
 t=run('BUILD-PERF-CHEM-'+variant,cmd('nvcc','-std=c++17','-O2','-arch=sm_89','-lineinfo','--fmad=false','-I'+str(inc),*macro,src,'-o',binary),[10,24,27],900,'Matched actual kernel timing fixture; FP64 physical fields.',inputs=[src],kind='BUILD',variant=variant)
 if t['exit_code']:continue
 for n in [1,32768]:
  t=run('PERF-CHEM-'+variant+'-'+str(n),cmd(binary,str(n)),[10,26,27],300,'3warmup/7samples, all consumed/published fields checked; matched unchanged successful local chemistry workload.',inputs=[src],binary=binary,variant=variant,kind='PERFORMANCE')
  if t['exit_code']:continue
  data=json.loads((D/t['log']).read_text());(dest/(variant+'-'+str(n)+'.json')).write_text(json.dumps(data,indent=2)+'\n')
  amend(t['id'],verdict='PASS' if data['all_fields_pass'] else 'FAIL',observations='Full publication finite/status/conservation/unchanged conservative fields and independent analytic species checked for every cell. FNV bits exclude structure padding.',artifact=str((dest/(variant+'-'+str(n)+'.json')).relative_to(D)))
  metrics={k:{'median':statistics.median([s[k] for s in data['samples'][3:]]),'min':min(s[k] for s in data['samples'][3:]),'max':max(s[k] for s in data['samples'][3:]),'stdev':statistics.stdev([s[k] for s in data['samples'][3:]])} for k in data['samples'][0] if k.endswith(('_us','_ms'))}
  summary[variant+'-'+str(n)]={'metrics':metrics,'publication_hash':data['samples'][0]['publication_hash_fnv1a64'],'malloc_count':data['malloc_count'],'device_allocated_delta_bytes':data['device_allocated_delta_bytes']}
for n in [1,32768]:
 a=summary.get('baseline-'+str(n));b=summary.get('original-'+str(n))
 if not a or not b:continue
 summary['comparison-'+str(n)]={'all_fields_bit_hash_equal':a['publication_hash']==b['publication_hash'],'ratios_PR_over_base':{k:b['metrics'][k]['median']/a['metrics'][k]['median'] for k in a['metrics']}}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
