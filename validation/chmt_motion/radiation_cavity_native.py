'''Native CHT adapter/shared radiation pipeline, with a controlled resident fixture.'''
import argparse,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
from mixture_thermal_native import source
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
def run(args,log,cwd=None):
 with (out/log).open('w') as f:subprocess.run(list(map(str,args)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
s=source();s=s[:s.index('int main')]+s[s.index('int main'):].split(' std::printf("{");')[0]+' return s;\n}\n'
s=s.replace('int main(int argc,char**argv){','extern "C" void* makeCavityOwner(){').replace('const int scenario=argc>1?std::atoi(argv[1]):0;','const int scenario=0;').replace('const int csr=argc>2?std::atoi(argv[2]):1, integrator=argc>3?std::atoi(argv[3]):1;','const int csr=1,integrator=1;').replace('s->particleCapacity=1;','s->particleCapacity=3;')
s+=r'''
extern "C" void resetCavityOwner(void* handle,double t0,double t1,double c0,double c1,int csr){
 auto* s=asState(handle);s->TpMin=100;s->TpMax=5000;s->csrCellLocalPathEnabled=csr;s->preBaseDirectoryReady=csr;
 const int count=3,cell[3]={0,0,0},status[3]={1,1,1},offset[2]={0,3};const unsigned char wall[3]={0,0,1};
 const double mass[3]={.3,.7,10},temp[3]={t0,t1,3500},carry[3]={c0,c1,0},diameter[3]={.001,.001,.001};
 CHECK(copyToDevice(s->particleCountDevice,&count,1,"count")==0);CHECK(copyToDevice(s->pCellId,cell,3,"cell")==0);
 CHECK(copyToDevice(s->pStatus,status,3,"status")==0);CHECK(copyToDevice(s->pStuck,wall,3,"wall")==0);
 CHECK(copyToDevice(s->pm,mass,3,"mass")==0);CHECK(copyToDevice(s->pT,temp,3,"T")==0);
 CHECK(copyToDevice(s->pd,diameter,3,"d")==0);CHECK(copyToDevice(s->pThermalEnthalpyResidual,carry,3,"carry")==0);
 CHECK(copyToDevice(s->preBaseCellOffset,offset,2,"offset")==0);CHECK(syncDeviceState(s,"test restore")==0);
}
extern "C" void readCavityOwner(void* handle,double* x){auto*s=asState(handle);for(int i=0;i<2;++i){x[i]=readOne(s->pT+i);x[i+2]=readOne(s->pThermalEnthalpyResidual+i);}}
'''
(out/'owner.cu').write_text(s)
run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89','-I'+str(ROOT/'common/thermal'),'-I'+str(ROOT/'common'),'-c',out/'owner.cu','-o',out/'owner.o'],'build-owner.log')
(out/'probe.C').write_text((ROOT/'validation/chmt_motion/radiation_cavity.cpp').read_text())
(out/'Make').mkdir(exist_ok=True);thermal=ROOT/'common/thermal/radiation'
(out/'Make/files').write_text(f'probe.C\n{thermal}/MieTable.C\n{thermal}/MieDoRadiationSolver.C\n{thermal}/GpuParticleRadiationCoupler.C\nEXE = {out}/probe\n')
includes=[ROOT/'applications/CHT/gpu',thermal,ROOT/'common',ROOT/'common/gasNumerics',ROOT/'common/wall',ROOT/'common/thermal']
lib=ROOT/'applications/CHT/gpu/libugkpcht_cuda_internal.a'
(out/'Make/options').write_text('EXE_INC = -std=c++17 -DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=2 '+' '.join('-I'+str(x) for x in includes)+' -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = '+str(out/'owner.o')+' '+str(lib)+' -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lfiniteVolume -lmeshTools -lOpenFOAM\n')
run(['wmake'],'build-probe.log',out)
case=out/'case';(case/'system').mkdir(parents=True,exist_ok=True);(case/'constant').mkdir(exist_ok=True)
def header(name):return f'FoamFile {{version 2.0; format ascii; class dictionary; object {name};}}\n'
(case/'system/controlDict').write_text(header('controlDict')+'application radiationCavity; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .1; writeControl timeStep; writeInterval 1;')
(case/'system/fvSchemes').write_text(header('fvSchemes')+'ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}')
(case/'system/fvSolution').write_text(header('fvSolution')+'solvers {}')
(case/'system/blockMeshDict').write_text(header('blockMeshDict')+'convertToMeters 1; vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1)); blocks (hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)); edges (); boundary (walls {type wall; faces ((0 4 7 3) (1 2 6 5) (0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2));}); mergePatchPairs ();')
(case/'constant/mie.table').write_text('Tvals 1000 4000\nDvals .001 .002\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'+''.join(f'NODE {d} {t} 0.66666666666666666667 0 0.66666666666666666667\nPHI 1 1\n' for d in range(2) for t in range(2))+'END_MIE_TABLE\n')
run(['blockMesh','-case',case],'mesh.log')
run([out/'probe','-case',case],'run.log')
print('NATIVE_CAVITY_PIPELINE_PASS')
