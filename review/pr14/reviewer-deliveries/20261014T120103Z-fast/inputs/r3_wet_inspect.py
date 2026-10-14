from r3_runner import *
import decimal,collections
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x) if isinstance(x,float) else decimal.Decimal(x)
for name in ['WET-FILM-FORMAL','WET-FILM-FORMAL-RESTART','WET-EVAPORATION-FORMAL-900']:
 folder=D/'artifacts'/name;print('CASE',name,'accepted',[p.read_text() for p in sorted(folder.glob('accepted-*.csv'))])
 states=[json.loads(p.read_text()) for p in sorted(folder.glob('checkpoint-*-full-decoded.json'))]
 for s in states:
  print('STATE',s['time'],'keys',s.keys(),'PHYSICS',s.get('physics'),'LEDGER',len(s['ledger']),collections.Counter(q['kind'] for q in s['ledger']))
  print('MOM',{'gas':[str(sum((dec(q['momentum'][j]) for q in s['gas']),dec(0))) for j in range(3)],'film':[(q.get('meanVelocity'),q['mass']) for q in s['film']],'boundary':s['budget']['boundaryMomentum'],'support':s['budget']['supportImpulse'],'exchange':s['budget']['exchangeMomentum'],'phase_mass_set':sorted(set(q['mass'] for q in s['ledger'] if q['kind']==2)),'phase_heat_minmax':[(min((q['energy'] for q in s['ledger'] if q['kind']==2),default=0)),(max((q['energy'] for q in s['ledger'] if q['kind']==2),default=0))]})
 print('DIF_FILES',[(str(p.relative_to(folder)),p.stat().st_size) for p in folder.glob('checkpoint-*/state.chmt')])
