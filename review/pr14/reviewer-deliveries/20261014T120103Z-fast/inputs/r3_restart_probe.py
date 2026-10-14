from r3_runner import *
lines=(ROOT/'applications/CHMT/CHMT.C').read_text().splitlines()
for i,line in enumerate(lines):
 if 'restart' in line.lower():print(i+1,'\n'.join(lines[max(0,i-2):i+4]))
print('OLD_RESTART_CONTROL',old_read('inputs/native/chmt/thermal-formal-resume/system/controlDict').decode())
print('OLD_RESTART_PARAMS',old_read('inputs/native/chmt/thermal-formal-resume/constant/chmtProperties').decode())
for n in ['inputs/native/extra/wet-film-S209/geometry-support.json','inputs/native/extra/wet-evaporation-S209/geometry-support.json']:
 try:print(n,old_read(n).decode())
 except Exception as e:print(type(e).__name__,str(e))
