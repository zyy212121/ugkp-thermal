from r2_runner import *
import collections,re
print('RESULT_COUNTS',collections.Counter(x['verdict'] for x in json.loads((D/'results.json').read_text())['tests']))
for p in [D/'artifacts/extra/wet-film-S209-D204/complete-independent-budget-audit.json',D/'metadata/environment.json',D/'metadata/source-hashes.json',OLD/'coverage.json',OLD/'findings.json',ROOT/'review/pr14/manifest.json',ROOT/'review/pr14/responses/R014-010.json']:
 a=json.loads(p.read_text()) if p.exists() else {};print('FILE',p.name,'KEYS',list(a)[:15])
 if 'wet-film' in str(p):print([(x['global_pass'],x['gas_inventory_pass'],x['actual_perrecipient_packets']) for x in a['checks']])
 if p.name=='coverage.json':print('GAPS',a.get('remaining_gaps',a.get('not_run',a.get('unexecuted'))));print('TOPICsample',a['topics'][:2])
 if p.name=='findings.json':print('FINDINGS',[(x.get('id'),x.get('title'),x.get('topics')) for x in a['findings']])
 if p.name=='manifest.json':print('Topics',[(q['id'],q.get('title')) for q in a['topics']]);print('ownpaths',len(a['owned_paths']),'evidence',len(a['evidence']))
 if p.name=='R014-010.json':print('RESPONSE_KEYS',list(a));print('lastrecheck',a.get('recheck_results',[])[-1:]);print('lastreview',[(x.get('id'),x.get('verdict')) for x in a.get('reviewer_results',[])[-2:]])
