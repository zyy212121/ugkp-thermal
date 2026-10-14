from r2_native import *
resident('CHT-coldWall1D-WET100','CHT-coldWall1D-P010-WET100')
resident('CHT-coldWall2D-INPUT2',tool='memcheck');resident('CHT-coldWall2D-INPUT2',tool='synccheck')
def fsh(case):
 p=case/'system/controlDict';p.write_text(p.read_text().replace('application CHT;','application FSH;'))
 p=case/'constant/gpuProperties';p.write_text(p.read_text().replace('coldWallLaunchPolicy AUTO;','coldWallLaunchPolicy FIXED;'))
 p=case/'constant/particleProperties';s=p.read_text();s+='\nwallDensityKgM3 1800; wallSpecificHeatJkgK 710; wallConductivityWmK 100;\n';p.write_text(s)
for model in ['1D','2D']:
 case=resident('FSH-coldWall'+model,'CHT-coldWall'+model+'-INPUT2',modified=fsh)
 if (case/'5e-06').exists():restart(case)
