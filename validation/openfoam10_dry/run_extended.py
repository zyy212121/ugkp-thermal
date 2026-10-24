from run_cases import case,ROOT
import json
rows=[]
for mode in ['conduction','diffusion']:
 for n in [20,40,80]: rows.append(case(mode,n,0.00002,0.5))
for mode in ['isothermal','adiabatic']:
 for dt in [.02,.01,.005]:rows.append(case(mode,8,dt,1))
for n in [10,20,40]:rows.append(case('surface',n,.01,1))
for n in [20,40,80]:rows.append(case('coupled',n,1e-6,.02))
for dt in [4e-5,2e-5,1e-5]:rows.append(case('coupled',20,dt,.02))
(ROOT/'results/extended_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
