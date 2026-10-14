from r4_runner import *
import difflib
overlay=ROOT/'work/observer-source';p=overlay/'common/operators/advanceGasChemistryKernel.cuh';s=p.read_text()
s=s.replace('beforeEnergy,double(state->rhoE[cell]));','beforeEnergy,double(state->rhoE[cell]));\n            printf("\\n");')
s=s.replace('printf(" species%d_before=%a species%d_after=%a",k,beforeSpecies[k],k,double(state->gasSpecies.rho[k*state->nCells+cell]));','printf("R4_CHEM_SPEC cell=%d dt=%a V=%a species=%d before=%a after=%a\\n",cell,double(chemicalInterval),double(stageVolumes[cell]),k,beforeSpecies[k],double(state->gasSpecies.rho[k*state->nCells+cell]));')
p.write_text(s)
patch=D/'patches/D401-readonly-actual-Euler-RK-SST-chemistry.patch';modified=['common/GpuGasAdvance.cuh','common/operators/computeSstFaceFluxKernel.cuh','common/operators/advanceGasChemistryKernel.cuh']
patch.write_text(''.join(''.join(difflib.unified_diff((ROOT/n).read_text().splitlines(True),(overlay/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/isolated/'+n)) for n in modified))
receipt=json.loads((D/'metadata/D401-source-receipt.json').read_text());receipt['patch_sha256']=sha(patch);receipt['modified_files']=[{'path':n,'sha256':sha(overlay/n)} for n in modified];receipt['device_chemistry_lines_single_printf_per_record']=True
(D/'metadata/D401-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for ns in [2,10]:
 build=ROOT/'work/build'/('D401-Ns'+str(ns));install=ROOT/'work/install'/('diag'+str(ns))/'bin';install.mkdir(parents=True,exist_ok=True)
 t=run('BUILD-D401-CHMT-Ns'+str(ns),cmd('env',f'CHMT_BUILD={build}',f'FOAM_USER_APPBIN={install}',f'UGKWP_GAS_SPECIES={ns}','CHMT_CUDA_ARCH=sm_89','bash',overlay/'applications/CHMT/Allwmake'),[8,9,10,14,20,21,26,29],expected='OriginalO2native allphysicalsource+test-only readonlyD401 dumps, noalgorithm/control/mathchange; outputs mustexactformal beforeaccepting observations.',inputs=[patch,D/'inputs/r4_observer.H',*[overlay/n for n in modified]],kind='BUILD',variant='DIAGNOSTIC_D401')
 if t['exit_code']!=0:break
 b=install/'CHMT';amend(t['id'],binary=str(b),binary_sha256=sha(b),object_sha256=sha(build/'Backend.o'))
 print('D401_READY',ns,sha(b),flush=True)
