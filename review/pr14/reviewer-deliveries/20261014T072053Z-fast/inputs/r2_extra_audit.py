import json,csv,decimal,math
from r2_runner import *
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x) if isinstance(x,float) else decimal.Decimal(x)
def total(s):
 mass=sum((dec(q['mass']) for q in s['gas']),dec(0))+sum((sum(map(dec,q['condensed']),dec(0))+sum(map(dec,q['pore']),dec(0)) for q in s['solid']),dec(0))
 energy=sum((dec(q['energy']) for q in s['gas']+s['solid']),dec(0));return mass,energy
all_reports={}
for name in ['SST-RK','moving-GCL','moving-chemistry']:
 folder=D/'artifacts/extra'/name;states=[json.loads((folder/('checkpoint-'+str(i)+'-decoded.json')).read_text()) for i in range(3)];initial=states[0];beforeM,beforeE=total(initial);checks=[]
 for i,s in enumerate(states[1:],1):
  mass,energy=total(s);b=s['budget'];bi=initial['budget'];mr=mass-beforeM+dec(b['boundaryMass'])-dec(bi['boundaryMass']);external=-(dec(b['boundaryEnergy'])-dec(bi['boundaryEnergy']))+dec(b['bodyWork'])-dec(bi['bodyWork'])+dec(b['supportWork'])-dec(bi['supportWork'])+dec(b['radiation'])-dec(bi['radiation']);er=energy-beforeE-external-(dec(b['filmKineticDefect'])-dec(bi['filmKineticDefect']))
  em=sum(map(dec,b['exchangeMass']),dec(0));ee=sum(map(dec,b['exchangeEnergy']),dec(0));sst=[]
  for j,tag in [(0,'K'),(1,'Omega')]:
   old=sum((dec(q[j]) for q in initial['sst']),dec(0));now=sum((dec(q[j]) for q in s['sst']),dec(0));a=s['gasSstAudit'];pred=sum((dec(a[k+tag]) for k in ['transport','source','constraint']),dec(0));sst.append({'variable':tag,'initial':str(old),'current':str(now),'transport':a['transport'+tag],'source':a['source'+tag],'constraint':a['constraint'+tag],'residual':str(now-old-pred)})
  mesh=s['gasMesh'];gcl=[]
  for j,stage in enumerate(s['gasStages']):
   residuals=[]
   for c in range(len(stage['oldVolume'])):
    sweep=sum((dec(stage['sweptVolume'][mesh['cellFaces'][slot]])*dec(mesh['cellFaceSigns'][slot]) for slot in range(mesh['cellFaceOffsets'][c],mesh['cellFaceOffsets'][c+1])),dec(0));residuals.append(float(dec(stage['newVolume'][c])-dec(stage['oldVolume'][c])-sweep))
   gcl.append({'stage':j,'interval':stage['interval'],'oldVolume':stage['oldVolume'],'newVolume':stage['newVolume'],'evaluationVolume':stage['evaluationVolume'],'maximum_gcl_abs':max(map(abs,residuals)) if residuals else None,'all_cell_residuals':residuals})
  packets=list(csv.DictReader((folder/('exchange-'+str(i)+'.csv')).open()));stages=sorted({q.get('stage') for q in packets});row={'checkpoint':i,'time':s['time'],'total_mass_residual':str(mr),'total_energy_residual_J':str(er),'mass_tolerance':float(dec(1e-14)+dec(1e-10)*max(abs(beforeM),abs(mass))),'energy_tolerance_J':float(dec(1e-10)+dec(1e-10)*max(abs(beforeE),abs(energy))),'pair_mass_cancellation':str(em),'pair_energy_cancellation_J':str(ee),'SST_budget':sst,'gas_GCL':gcl,'actual_packet_stage_ids':stages,'packet_count':len(packets),'consumed_packets':b['consumedPackets'],'mass_energy_pass':abs(mr)<=dec(1e-14)+dec(1e-10)*max(abs(beforeM),abs(mass)) and abs(er)<=dec(1e-10)+dec(1e-10)*max(abs(beforeE),abs(energy))};checks.append(row)
 report={'name':name,'source_commit':SOURCE,'checks':checks,'scope':'Independent80-digit sums of originalactualGPUcheckpoint stocks and actualcumulativeexternal/pairedledgers. SST integralclosure from actualfusedtransport/source/constraint audits; NOT gross production/dissipation. Finalgasstage GCL only until allmicrostage data collected.','all_mass_energy_pass':all(x['mass_energy_pass'] for x in checks)};p=folder/'independent-budget-audit.json';p.write_text(json.dumps(report,indent=2)+'\n');all_reports[name]=report
 print(name,[(x['total_mass_residual'],x['total_energy_residual_J'],x['actual_packet_stage_ids'],[y['residual'] for y in x['SST_budget']],[y['maximum_gcl_abs'] for y in x['gas_GCL']]) for x in checks])
(D/'artifacts/extra/all-budgets.json').write_text(json.dumps(all_reports,indent=2)+'\n')
