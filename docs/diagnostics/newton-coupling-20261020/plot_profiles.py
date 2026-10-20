from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).resolve().parent
rows=list(csv.DictReader((d/'profiles-five-stations.csv').open()))
fig,axs=plt.subplots(3,5,figsize=(15,9),sharey=True)
for j,x in enumerate([.35,.38,.58,.75,.85]):
 for i,key in enumerate(['U','k','omega']):
  ax=axs[i,j]
  for variant,label,color in [('prior250','Before','#c05b30'),('combined','After','#1769aa')]:
   a=sorted([v for v in rows if v['variant']==variant and abs(float(v['time'])-.0233)<1e-9 and float(v['station'])==x and float(v['y'])<=.004],key=lambda v:float(v['y']))
   ax.plot([float(v[key])for v in a],[1000*float(v['y'])for v in a],'-o',ms=2,lw=1.1,label=label,color=color)
  ax.plot([float(v[('omegaRef' if key=='omega' else key+'ref')])for v in a],[1000*float(v['y'])for v in a],'k--',lw=1,label='OF10 reference')
  if key=='omega':ax.set_xscale('log')
  ax.set_xlabel({'U':'U (m/s)','k':'k (m²/s²)','omega':'omega (1/s)'}[key]);ax.grid(alpha=.18)
  if j==0:ax.set_ylabel('Wall distance (mm)')
  if i==0:ax.set_title(f'x ≈ {x:.2f} m')
  ax.set_ylim(0,4)
axs[0,0].legend(fontsize=8)
fig.suptitle('Same restart +250 µs: macro cell fields (not a steady-state acceptance)',fontsize=14)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(d/'five-station-profiles.png',dpi=150)
print('figure saved')
