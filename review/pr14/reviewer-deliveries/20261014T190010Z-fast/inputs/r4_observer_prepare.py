from r4_runner import *
import io,tarfile,difflib
overlay=ROOT/'work/observer-source';assert not overlay.exists();overlay.mkdir()
payload=subprocess.check_output(['git','-C',str(ROOT),'archive',SOURCE,'applications/CHMT','common'])
with tarfile.open(fileobj=io.BytesIO(payload)) as t:
 for member in t.getmembers():assert (overlay/member.name).resolve().is_relative_to(overlay.resolve())
 t.extractall(overlay)
header=D/'inputs/r4_observer.H';shutil.copy2(P/'r4_observer.H',header)
adv=overlay/'common/GpuGasAdvance.cuh';old=adv.read_text();new=old.replace('#include <type_traits>','#include <type_traits>\n#include "'+str(header)+'"',1)
anchor='    const int cellBlock = s->fixedCellBlockThreads;';assert new.count(anchor)==1
new=new.replace(anchor,'    r4obs::dump(s,"EULER_BEFORE",dt,ledgerDt,stageTime);\n'+anchor)
anchor='        applySstFluxAndSourceKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>'
assert new.count(anchor)==1;new=new.replace(anchor,'        r4obs::dump(s,"SST_BEFORE",dt,ledgerDt,stageTime);\n'+anchor)
anchor='    applyGasFluxDivergenceByCellKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState, dt, nativeRefinement);'
assert new.count(anchor)==1;new=new.replace(anchor,'    r4obs::dump(s,"FLUX_BEFORE",dt,ledgerDt,stageTime);\n'+anchor)
anchor='    return 0;\n}\n\ntemplate<class GasHostState>\nint blendGasRungeKuttaStage'
assert new.count(anchor)==1;new=new.replace(anchor,'    r4obs::dump(s,"EULER_AFTER",dt,ledgerDt,stageTime);\n'+anchor)
anchor='    const int block = s->fixedCellBlockThreads;\n    const int cellGrid = (s->nCells + block - 1)/block;\n    blendGasConservativeStateKernel'
assert new.count(anchor)==1;new=new.replace(anchor,'    r4obs::dump(s,"BLEND_BEFORE",0.,0.,0.,initialWeight,stageWeight);\n'+anchor)
anchor='    return 0;\n}\n\ntemplate<class GasHostState>\nint advanceGasFluxStage'
assert new.count(anchor)==1;new=new.replace(anchor,'    r4obs::dump(s,"BLEND_AFTER",0.,0.,0.,initialWeight,stageWeight);\n'+anchor)
adv.write_text(new)
sst=overlay/'common/operators/computeSstFaceFluxKernel.cuh';sstold=sst.read_text();anchor='    sstSourcesForCell(s, c, divU, sourceK, sourceOmega);';assert sstold.count(anchor)==1
emit='''
    printf("R4_SST_SOURCE {\\\"cell\\\":%d,\\\"dt\\\":\\\"%a\\\",\\\"oldVolume\\\":\\\"%a\\\",\\\"newVolume\\\":\\\"%a\\\",\\\"fluxK\\\":\\\"%a\\\",\\\"fluxOmega\\\":\\\"%a\\\",\\\"sourceK\\\":\\\"%a\\\",\\\"sourceOmega\\\":\\\"%a\\\",\\\"rho\\\":\\\"%a\\\",\\\"k\\\":\\\"%a\\\",\\\"omega\\\":\\\"%a\\\",\\\"divU\\\":\\\"%a\\\",\\\"betaStar\\\":\\\"%a\\\",\\\"constrainedOmega\\\":%d}\\n",c,double(dt),double(oldVolume),double(newVolume),double(fluxK),double(fluxOmega),double(sourceK),double(sourceOmega),double(s.rho[c]),double(s.k[c]),double(s.omega[c]),double(divU),double(s.sstCoefficients.betaStar),int(constrainedOmega));
'''
sstnew=sstold.replace(anchor,anchor+emit);sst.write_text(sstnew)
chem=overlay/'common/operators/advanceGasChemistryKernel.cuh';chemold=chem.read_text();anchor='    ugkwp::advanceGasChemistryCellWithVolumes(*state, cell, chemicalInterval, stageVolumes, nativeRefinement);';assert chemold.count(anchor)==1
replacement='''    constexpr int observedNs=ugkwp::GasStateTraits<GasState>::speciesCount;
    if constexpr(observedNs>0){
        if(state->gasSpecies.mode==ugkwp::GasMode::MixtureChemistry && stageVolumes){
            const double beforeEnergy=state->rhoE[cell];double beforeSpecies[observedNs];
            for(int k=0;k<observedNs;++k)beforeSpecies[k]=state->gasSpecies.rho[k*state->nCells+cell];
            ugkwp::advanceGasChemistryCellWithVolumes(*state, cell, chemicalInterval, stageVolumes, nativeRefinement);
            printf("R4_CHEM_HALF cell=%d dt=%a V=%a rhoE_before=%a rhoE_after=%a",cell,double(chemicalInterval),double(stageVolumes[cell]),beforeEnergy,double(state->rhoE[cell]));
            for(int k=0;k<observedNs;++k)printf(" species%d_before=%a species%d_after=%a",k,beforeSpecies[k],k,double(state->gasSpecies.rho[k*state->nCells+cell]));
            printf("\\n");return;
        }
    }
'''+anchor
chemnew=chemold.replace(anchor,replacement);chem.write_text(chemnew)
patch=D/'patches/D401-readonly-actual-Euler-RK-SST-chemistry.patch'
patch.write_text(''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+name,tofile='b/isolated/'+name)) for name,a,b in [('common/GpuGasAdvance.cuh',old,new),('common/operators/computeSstFaceFluxKernel.cuh',sstold,sstnew),('common/operators/advanceGasChemistryKernel.cuh',chemold,chemnew)]))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD401新增原因：生产SST13audit数组/6D2H及sharedbudget删掉后，原GPU端点和host149项不能证明真实Euler/RK/chemhalf全链。只在隔离SOURCE archive副本插入只读实际resident数组/最终flux/old-newV/本地真实SST源/chem半步读值；原算式/控制/调用顺序不改，不恢复productionaudit。正式原所有physical输出bytes必须与observer逐文件相等才用于测量。记录实际ledgerDt与blend权重，测试侧独立逐cell/全域M/P/E/species与K/omega源/face核算；生产labelsreserved0不参与。新增D2H/sync/printf不作效率计时。原SST三RK以及moving/化学依赖改变，逐个原输入/终点复验，不借缩步/缩端点/阈值得到通过。observer声明/补丁及全部SHA单列。\n')
(D/'metadata/D401-source-receipt.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'overlay':str(overlay),'patch_sha256':sha(patch),'header_sha256':sha(header),'modified_files':[{'path':str(p.relative_to(overlay)),'sha256':sha(p)} for p in [adv,sst,chem]],'production_source_modified':False,'diagnostic_io_not_timing':True},indent=2)+'\n')
print('D401_PREPARED',sha(patch),flush=True)
