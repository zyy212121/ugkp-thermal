from r3_runner import *
for name in ['inputs/resident_check_fp64_mean.py','inputs/resident_check_fsh_double.py']:
 text=old_read(name).decode();print('HELPER',name,text[:5400])
for app in ['CHT','FSH']:
 folder=D/'artifacts/resident'/ (app+'-active16-ASCII-S306');print('TIMES',app,[(p.name,[q.name for q in p.iterdir()][:7]) for p in folder.iterdir() if p.is_dir()])
 print((D/'logs'/('NATIVE-'+app+'-active16-ASCII-S306.log')).read_text()[-2200:])
for name in ['thermal-full','thermal-restart']:
 raw=(D/'logs'/('THERMAL-SEED-'+name+'.log')).read_text();rows=[l for l in raw.splitlines() if l.startswith(('R3SEED','R3DT'))];print('TRACE',name,len(rows),'\n'.join(rows[:12]));p=D/'artifacts'/('thermal-seed-output-'+name+'.json');cmp=json.loads(p.read_text());print('UNEQUAL',[q for q in cmp['files'] if not q['byte_equal']])
