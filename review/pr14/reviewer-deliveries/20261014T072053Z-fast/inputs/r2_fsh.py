from r2_native import *
def fsh(case):
 p=case/'system/controlDict';p.write_text(re.sub(r'application\s+CHT;',r'application FSH;',p.read_text()))
 # FSH actual wall dictionary consumes these three existing tokens; CHT ignores them.
 p=case/'constant/particleProperties';s=p.read_text()
 for key,value in [('wallDensity','1800'),('wallSpecificHeat','710'),('wallThermalConductivity','100')]:s=re.sub(r'\b'+key+r'\s+[^;]+;',key+' '+value+';',s)
 p.write_text(s)
for model in ['1D','2D']:
 case=resident('FSH-coldWall'+model,'CHT-coldWall'+model+'-INPUT2',modified=fsh)
 if (case/'5e-06').exists():restart(case)
