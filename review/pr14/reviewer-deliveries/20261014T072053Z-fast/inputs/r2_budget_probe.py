from r2_runner import *
for name in ['SST-RK','SST-SSPRK2','SST-SSPRK3','moving-GCL','dry-ablation']:
 p=D/'artifacts/extra'/name/'complete-independent-budget-audit.json'
 if not p.exists():continue
 s=json.loads(p.read_text());print(name,[(x['global_pass'],x['gas_inventory_pass'],x['gas_momentum'],[(q['variable'],q['pass']) for q in x['SST']]) for x in s['checks']])
print('core files',*[str(p) for p in (ROOT/'work/cases/extra').rglob('core*')],sep='\n')
