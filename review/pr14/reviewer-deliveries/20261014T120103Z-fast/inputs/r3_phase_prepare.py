from r3_runner import *
import difflib,re
app=ROOT/'applications/CHMT';source=app/'materials/CpuMaterialDriver.C';text=source.read_text()
anchor='for(auto phase:surface.phasePackets){phase.consumerMask=requiredConsumers(phase.kind);trial.ledger.push_back(phase);}'
assert text.count(anchor)==1
observer=r'''
                for(const auto& phase:surface.phasePackets){const int f=phase.filmFace,c=phase.solidCell;
                    MaterialPrimitive material;recoverMaterial(state.solid[c],state.solidMesh.volumes[c],p,material);
                    const double density=phase.mass==0?0:p.condensed[p.material.phaseCondensed].rho;
                    std::fprintf(stderr,"R3PHASE,window=%llu,micro=%llu,time=%a,dt=%a,face=%llu,film=%d,solid=%d,index=%d,mass=%a,rho=%a,porosity=%a,target=%a,energy=%a,heat=%a,mx=%a,my=%a,mz=%a,forcingMass=%a,forcingPhaseMass=%a,oldFilmM=%a,newFilmM=%a,oldFilmH=%a,newFilmH=%a,oldMeanX=%a,newMeanX=%a,oldSolidV=%a,newSolidV=%a\n",
                        static_cast<unsigned long long>(window.identity.sequence),static_cast<unsigned long long>(record.microSequence),state.time,dt,
                        static_cast<unsigned long long>(phase.face),f,c,p.material.phaseCondensed,phase.mass,density,material.porosity,target[f],phase.energy,phase.conductive,
                        phase.momentum.x,phase.momentum.y,phase.momentum.z,forcing[f].mass,forcing[f].phaseMass,state.film[f].mass,trial.film[f].mass,
                        state.film[f].enthalpy,trial.film[f].enthalpy,state.filmAux[f].meanVelocity.x,trial.filmAux[f].meanVelocity.x,state.solidMesh.volumes[c],trial.solidMesh.volumes[c]);
                }
'''
new='#include <cstdio>\n'+text
(D/'metadata/phase-observer-setup-recovery.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','utc':utc(),'original_exception':'AssertionError at phase_prepare line18 before any compile/GPU: assumed unqualified include spelling did not match actual qualified include.','recovery':'Prepend cstdio only to isolated observer source; no production source or assertions changed.'},indent=2)+'\n')
new=new.replace(anchor,anchor+observer)
p=D/'inputs/CpuMaterialDriver-phase-observer.C';p.write_text(new);patch=D/'patches/D301-readonly-phase-consumer.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/applications/CHMT/materials/CpuMaterialDriver.C',tofile='b/diagnostic/CpuMaterialDriver-phase-observer.C')))
# Actual D203 observation CUDA object is byte-preserved from the prior round;
# all production CUDA/common source dependencies are unchanged.
obj=OLDROOT/'work/fixtures/Backend-ledger2.o';versions=json.loads((OLD/'versions.json').read_text());assert sha(obj)==versions['objects'][str(obj.relative_to(OLDROOT))]['sha256']
binary=ROOT/'work/fixtures/CHMT-phase-ledger-observer';gen=ROOT/'work/build/CHMT2/generated'
flags=['-std=c++17','-O2','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-DUGKWP_GAS_SPECIES=2','-I'+str(app),'-I'+str(app/'materials'),'-I'+str(ROOT/'common'),'-I'+str(gen),'-I/opt/openfoam10/src/OpenFOAM/lnInclude','-I/opt/openfoam10/src/finiteVolume/lnInclude','-I/opt/openfoam10/src/meshTools/lnInclude','-I/opt/openfoam10/src/OSspecific/POSIX/lnInclude']
files=[app/'CHMT.C',app/'io/MultirateEvolution.C',p,*[app/n for n in ['materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C','restart/Checkpoint.C']],obj]
libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart']
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nD301只读phase/ledger观察：在正式CpuMaterialDriver成功localphase附入ledger位置，仅fprintf实际mass/热/力/forcing/前后film/porosity/density/solidVolume/target。zero mass输出unusedrho0，绝不读取未选材料。GPU用原已验证D203 finalweightedflux/ledger观察object（source算法未改，完整hash/原patch引用），不新增GPU算法或断言。正式wet/合法非零controls逐文件原输出比较必须等价；I/O不计效率。编译900s/run300s。合法新controls独立于原wet：有效phaseCondensed0/phaseFilmY(1 0)/enableMelting+CoupledRecession；保留实际rho/cp/e0/pressure/film，meltTemperature300或400分别拟覆盖positive/negative，明确小区间1e-6一microstep/一个窗口，两旧4b合法控制与正式9d相同输入/flags比。只对新增控制设小终点，原wet/evap0.001保持。\n')
t=run('BUILD-PHASE-LEDGER-OBSERVER',cmd('g++',*flags,*files,*libs,'-o',binary),[3,6,13,14,20,21,26,29],900,'OnlyreadonlyphaseIO+immutableD203ledgerobject; originalphysicaloutputsproofrequired.',inputs=files+[patch],kind='BUILD',variant='DIAGNOSTIC_D301');assert t['exit_code']==0
for name,tm in [('melting-positive',300),('solidification-negative',400)]:
 case=ROOT/'work/cases/phase'/name;assert not case.exists();shutil.copytree(ROOT/'work/cases/extra/wet-film-formal',case,ignore=shutil.ignore_patterns('chmtOutput'))
 props=case/'constant/chmtProperties';s=props.read_text();s=s.replace('couplingInterval 0.00050000000000000001; gasMaxDt 1.0000000000000001e-05;','couplingInterval 1e-6; gasMaxDt 1e-6;');s+='\nmeltTemperature '+str(tm)+';\nmaterial { phaseCondensed 0; enableMelting true; phaseFilmY (1 0); evaporation Disabled; }\nmeshMotion { policy CoupledRecession; }\n';props.write_text(s)
 control=case/'system/controlDict';s=control.read_text();s=re.sub(r'endTime\s+0\.001;',r'endTime 1e-6;',s);s=re.sub(r'deltaT\s+[^;]+;',r'deltaT 1e-6;',s);control.write_text(s)
 inp=D/'inputs/phase'/name;shutil.copytree(case,inp)
 (D/'patches'/('INPUT-'+name+'.patch')).write_text(''.join(difflib.unified_diff((ROOT/'work/cases/extra/wet-film-formal/constant/chmtProperties').read_text().splitlines(True),props.read_text().splitlines(True),fromfile='a/originalwet/chmtProperties',tofile='b/phasecontrol/chmtProperties')))
print('PHASE_CONTROLS_READY',sha(binary),flush=True)
