import sys,re,json,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1])
s=importlib.util.spec_from_file_location('fields',R/'validation/steady_cpu/export_of.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
def latest(case):return max((p for p in case.iterdir() if p.is_dir() and re.fullmatch('[0-9.eE+-]+',p.name)),key=lambda p:float(p.name))
report={}
for name,expected in [('gas/gasUGKP-table',320),('gas/gasUGKPSteady-table',300),('extended/SSPRK3',320),('extended/Euler',320)]:
 d=latest(out/name);t=(d/'T').read_text();section=t.split('    inlet')[1].split('    outlet')[0]
 actual=float(re.search(r'\bvalue\s+uniform\s+([^;]+);',section)[1]);assert abs(actual-expected)<1e-12,(name,actual)
 rho=f.internal(d/'rho',16);a=f.internal(d/'rhoY_Air',16);b=f.internal(d/'rhoY_CO2',16)
 closure=max(abs(x[0]+y[0]-z[0]) for x,y,z in zip(a,b,rho));assert closure<1e-12,closure
 report[name]={'written_inlet_temperature':actual,'max_species_density_closure':closure}
a=[list(map(float,line.split()[3:])) for line in (out/'reference.log').read_text().splitlines() if line.startswith('REFERENCE ')]
b=next(list(map(float,line.split()[1:])) for line in (out/'resident-memcheck.log').read_text().splitlines() if line.startswith('DEVICE_REFERENCE '))
assert len(a)==4
error=max(abs(x-y) for row in a for x,y in zip(row,b));assert error<1e-10,error
report['native_OpenFOAM_reference']={'faces':len(a),'max_absolute_error':error,'state_Uxyz_p_T':b}
(out/'output-checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
