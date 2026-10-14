from r3_runner import *
raw=(D/'logs/WET-FILM-OBSERVER-D302.log').read_text()
for tag in ['LEDGER_','R3PHASE','R3CONFIG']:
 rows=[q for q in raw.splitlines() if q.startswith(tag)];print(tag,len(rows));print('\n'.join(rows[:6]))
print('SCHEMA',json.loads((D/'artifacts/WET-FILM-FORMAL/checkpoint-1-full-decoded.json').read_text())['ledger'][0])
for p in [ROOT/'applications/CHMT/io/IntervalAudit.H',ROOT/'applications/CHMT/model/ModelConfig.H']:
 if p.exists():print('FILE',p,p.read_text()[:5800])
print('THERMALFILES',[(str(q.relative_to(OLDROOT)),q.stat().st_size) for q in (OLDROOT/'work/cases/chmt/thermal').rglob('*') if q.is_file() and 'chmtOutput' not in q.parts][:50])
print('COLDFIELDS',list((OLDROOT/'work/cases/resident/CHT-cold2D-valid16-S214/0').iterdir()))
