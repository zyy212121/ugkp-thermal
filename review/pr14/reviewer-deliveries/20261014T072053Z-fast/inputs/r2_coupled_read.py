from r2_runner import *
p=OLD/'inputs/native/resident/CHT-coldWall2D-INPUT2'
print('CONSTANT NAMES',[q.name for q in (p/'constant').iterdir()])
for name in ['solidRegionProperties','radiationProperties','ugkwpProperties']:
 q=p/'constant'/name
 if q.is_file():print('\n'+name+'\n'+q.read_text())
for q in (ROOT/'applications/CHT/thermal').glob('*Coupler*'):print('COUPLER',q)
