"""Execute production particle and task-construction bodies on the host.

The theta proxy detects otherwise invisible loads of rejected particles.
CUDA block/thread coordinates are simulated only for independent count/build
threads; this does not replace the GPU regression for reductions/barriers.
"""
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'applications/gasUGKP/private_backend/GpuResidentStrict.cu'

def function(text, name):
    match = re.search(r'^(?:__\w+__\s+)*(?:inline\s+)?(?:void|int|double|PressureProjectionCell) ' + name + r'\s*\(', text, re.M)
    if not match:
        match = re.search(r'^struct ' + name + r'\b', text, re.M)
    assert match, name
    start = match.start()
    if text[:start].rstrip().endswith('template<bool PoissonMode>'):
        start = text.rfind('template<bool PoissonMode>', 0, start)
    prefix = re.search(r'template<bool (?:GatherSurvivors|CompactParticles) = false>\s*$', text[:start])
    if prefix: start = prefix.start()
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    result = text[start:end].replace('asm("trap;");', 'throw std::runtime_error("trap");')
    if text[match.start():].startswith('struct '): result += ';'
    dependencies = {
        'finalizeCsrSegmentedPoolCellsKernel': 'CsrPoolFinalizeOperation',
        'accumulateCsrSegmentedMomentTasksPersistentKernel': 'CsrMomentOperation',
        'finalizeCsrSegmentedMomentsAndRecoverKernel': 'CsrMomentRecoveryOperation',
        'gatherCsrSegmentedParticlesKernel': 'CsrGatherOperation',
    }
    if name in dependencies:
        queue = (ROOT / 'common/CsrPersistentQueue.cuh').read_text()
        shared = 'template<class State, class Operation>\n' + function(queue, 'runCsrPersistentQueue')
        op = function(text, dependencies[name])
        if name == 'gatherCsrSegmentedParticlesKernel':
            # Its surrounding fixture supplies template<int BlockThreads>.
            result = op + '\n' + result
            result = result.replace('};\n__global__', '};\ntemplate<int BlockThreads>\n__global__')
            # Shared loop is inserted by the fixture before its template.
        else:
            result = shared + '\n' + op + '\n' + result
    return result

def compile_run(tmp_path, body):
    source = tmp_path / 'check.cpp'
    source.write_text(body)
    binary = tmp_path / 'check'
    subprocess.run(['g++', '-std=c++17', '-O2', str(source), '-o', str(binary)], check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

PREAMBLE = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#define __device__
#define __global__
#define __forceinline__ inline
using std::min;
struct Dim { int x=0; } blockIdx, threadIdx, blockDim, gridDim;
template<class T> int atomicAdd(T* p,int v) { int old=*p; *p+=v; return old; }
double finiteOr(double x,double fallback) {return std::isfinite(x)?x:fallback;}
double clampMin(double x,double low) {return std::max(x,low);}
double sqr3(double x,double y,double z) {return x*x+y*y+z*z;}
bool nonFiniteDevice(double x) {return !std::isfinite(x);}
double uniform01Device(unsigned long long& rng) {++rng;return 0.5;}
struct Theta {double value=2;int reads=0;double operator[](int) {++reads;return value;}};
"""

def test_shared_particle_physics_and_no_rejected_theta_load(tmp_path):
    text = SOURCE.read_text()
    names = ['accumulateOnePoolParticle'] if 'void accumulateOnePoolParticle\n' in text else []
    names += ['accumulateOnePoissonPoolParticle', 'accumulateCsrSplitLogicalPoolParticle']
    body = PREAMBLE + r"""
struct DeviceState {
 int particleCapacity=1;int pStatus[1]={1},pCellId[1]={0};
 unsigned long long pRng[1]={10};
 double pm[1]={3},pux[1]={2},puy[1]={3},puz[1]={4},pd[1]={0.5};
 double particleDiameterFallback=1,thetaMin=0.01;Theta pTheta;
};
""" + '\n'.join(function(text,n) for n in names) + r"""
void call(DeviceState& s, bool poisson, int index, double p, std::array<double,8>& a) {
 if(poisson) accumulateCsrSplitLogicalPoolParticle<true>(s,0,index,p,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
 else accumulateCsrSplitLogicalPoolParticle<false>(s,0,index,p,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
}
int main() {
 DeviceState s;std::array<double,8> a{};
 call(s,true,0,0,a);
 if(s.pTheta.reads!=0) {std::cerr<<"Rejected Poisson particle read pTheta "<<s.pTheta.reads<<" times";return 1;}
 assert(s.pRng[0]==11 && s.pStatus[0]==1);
 for(double x:a) assert(x==0);
 call(s,true,-1,1,a);assert(s.pRng[0]==11 && s.pTheta.reads==0);
 call(s,true,0,1,a);
 std::array<double,8> expected={3,6,9,12,52.5,1.5,0.75,1};
 assert(a==expected && s.pStatus[0]==2 && s.pRng[0]==12 && s.pTheta.reads==1);
 DeviceState l1;std::array<double,8> b{};
 accumulateOnePoissonPoolParticle(l1,0,0,1,b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
 assert(b==a && l1.pRng[0]==11 && l1.pTheta.reads==1);
 DeviceState cold;cold.pTheta.value=0.01;std::array<double,8> d{};
 call(cold,false,0,1,d);assert(cold.pRng[0]==10 && cold.pStatus[0]==1 && cold.pTheta.reads==1);
 for(double x:d) assert(x==0);
 DeviceState hot;call(hot,false,0,1,d);assert(d==expected && hot.pRng[0]==10);
}
"""
    compile_run(tmp_path,body)

def test_compact_builder_coverage_and_no_empty_descriptors(tmp_path):
    text = SOURCE.read_text()
    names = ['csrReductionTileParticles'] if 'int csrReductionTileParticles\n' in text else []
    names += ['configureDynamicCsrHeavyPolicyKernel','countCsrReductionTasksKernel','writeCsrReductionTask','materializeCsrReductionTasksKernel']
    body = PREAMBLE + r"""
enum class HeavyDirectoryKind {full=0,splitBaseAndInjection=1,baseOnly=2};
enum class CsrReductionTaskSource {fullIndexed,splitBaseDirect,splitInjectionIndexed,splitLogical};
struct CsrReductionTask {int cell,begin,end,source;};
struct DeviceState {
 int nCells,reductionBlockThreads,multiprocessorCount,lightBlocksPerSm;
 int csrHeavyCellThreshold=0,csrHeavyTileParticles=0,csrHeavyTaskCapacity;
 int *preBaseCellOffset,*cellParticleOffset,*csrCellTaskCount,*csrCellTaskOffset;
 int *csrHeavyCellCount,*csrMultiTaskCellList,*csrHeavyTaskCount;
 CsrReductionTask* csrReductionTasks;int csrReductionDirectoryKind=0;
};
""" + '\n'.join(function(text,n) for n in names) + r"""
int main() {
 std::mt19937 rng(915);
 for(int b:{32,64,128,256}) for(int workers:{1,7,288}) for(int kind:{0,1,2}) for(int trial=0;trial<25;++trial) {
  const int nc=31;std::vector<int> base(nc+1),inj(nc+1),counts(nc+1),offset(nc+1),multi(nc);
  for(int c=0;c<nc;++c) {
   int a=trial==0?0:(trial==1?(c==0?100000:0):rng()%5000);
   int q=trial<2?0:rng()%65;base[c+1]=base[c]+a;inj[c+1]=inj[c]+q;
  }
  int population=kind==0?inj[nc]:(base[nc]+(kind==1?inj[nc]:0));
  int nheavy=0,ntask=0,capacity=population/b+2*nc+1;
  std::vector<CsrReductionTask> tasks(capacity,CsrReductionTask{-99,-99,-99,-99});
  std::fill(offset.begin(),offset.end(),-99);
  DeviceState s{nc,b,1,workers,0,0,capacity,base.data(),inj.data(),counts.data(),offset.data(),&nheavy,multi.data(),&ntask,tasks.data()};
  blockIdx.x=threadIdx.x=0;configureDynamicCsrHeavyPolicyKernel(&s,kind);
  int expectedTile=b*std::max(1,(population+b*workers-1)/(b*workers));
  assert(s.csrHeavyTileParticles==expectedTile);
  // Count must initialize policy/counter itself: the production path no longer
  // launches configure/reset separately. Poison them to catch stale reads.
  s.csrHeavyTileParticles=-1;s.csrHeavyCellThreshold=-1;nheavy=123;ntask=-1;
  blockDim.x=1;gridDim.x=nc+1;
  for(int c=0;c<=nc;++c) {blockIdx.x=c;countCsrReductionTasksKernel(&s,kind);}
  assert(s.csrHeavyTileParticles==expectedTile && s.csrHeavyCellThreshold==expectedTile && nheavy==0);

  offset[0]=0;for(int c=0;c<nc;++c)offset[c+1]=offset[c]+counts[c];
  for(int c=0;c<nc;++c) {blockIdx.x=c;materializeCsrReductionTasksKernel(&s,kind);}
  int expectedTasks=0,expectedHeavy=0;
  for(int c=0;c<nc;++c){expectedTasks+=counts[c];if(counts[c]>1)++expectedHeavy;}
  if(ntask!=expectedTasks) {std::cerr<<"Nonempty queue count "<<ntask<<" expected "<<expectedTasks;return 1;}
  assert(ntask<=2*workers+nc && nheavy==expectedHeavy && nheavy<=std::min(nc,workers));
  for(int task=ntask;task<capacity;++task)assert(tasks[task].cell==-99);
  for(int c=0;c<nc;++c) {
   int begin=kind==0?inj[c]:(kind==2?base[c]:0);
   int end=kind==0?inj[c+1]:(kind==2?base[c+1]:base[c+1]-base[c]+inj[c+1]-inj[c]);
   if(counts[c]==0){assert(offset[c]==offset[c+1]);continue;}
   int cursor=begin;
   for(int t=offset[c];t<offset[c]+counts[c];++t) {
    const auto& d=tasks[t];assert(d.cell==c && d.begin==cursor && d.end>d.begin && d.end-d.begin<=expectedTile);
    assert(d.source==static_cast<int>(kind==0?CsrReductionTaskSource::fullIndexed:(kind==2?CsrReductionTaskSource::splitBaseDirect:CsrReductionTaskSource::splitLogical)));cursor=d.end;
   }
   assert(cursor==end);
   assert((counts[c]>1)==(end-begin>expectedTile));
  }
 }
}
"""
    compile_run(tmp_path,body)



def test_finalizer_grid_stride_covers_more_heavy_cells_than_blocks(tmp_path):
    text = SOURCE.read_text()
    body = PREAMBLE + r"""
#define __shared__
#define __syncthreads() ((void)0)
template<int N> void blockReduceComponentSums(double (&)[N],double*) {}
double warpPartials[256];
struct DeviceState {
 int *csrHeavyCellCount,*csrMultiTaskCellList,*csrCellTaskCount,*csrCellTaskOffset;
 double *csrHeavyPartials,*poissonPoolMass,*poissonPoolMomX,*poissonPoolMomY,*poissonPoolMomZ,*poissonPoolEnergy,*poissonPoolDiameter,*poissonPoolDiameter2;
 int *poolThermalCount,*poissonPoolSampleTargetCount;
 int queueCursor=0; int* csrHeavyTaskCursor=&queueCursor;
};
""" + function(text,'preparePoissonPoolSamplingCell') + '\n' + function(text, 'finalizeCsrSegmentedPoolCellsKernel').replace('extern __shared__ double warpPartials[];', '') + r"""
int main() {
 for(int nc:{0,1,7,24,25,57,433}) for(int blocks:{1,3,24}) {
  std::vector<int> cells(nc),count(nc,2),offset(nc+1),thermal(nc,-1),target(nc,-1);
  std::vector<double> partial(16*nc),mass(nc,-1),mx(nc,-1),my(nc,-1),mz(nc,-1),energy(nc,-1),d(nc,-1),d2(nc,-1);
  for(int c=0;c<nc;++c) {cells[c]=c;offset[c]=2*c;for(int k=0;k<16;++k)partial[16*c+k]=c+1;}
  offset[nc]=2*nc;
  DeviceState s{&nc,cells.data(),count.data(),offset.data(),partial.data(),mass.data(),mx.data(),my.data(),mz.data(),energy.data(),d.data(),d2.data(),thermal.data(),target.data()};
  blockDim.x=1;gridDim.x=blocks;threadIdx.x=0;
  for(int block=0;block<blocks;++block) {blockIdx.x=block;finalizeCsrSegmentedPoolCellsKernel(&s);}
  for(int c=0;c<nc;++c) {
   if(mass[c]!=2*(c+1) || thermal[c]!=2*(c+1) || target[c]!=2*(c+1)) {std::cerr<<"Unprocessed finalizer cell "<<c<<" with grid="<<blocks;return 1;}
  }
 }
}
"""
    compile_run(tmp_path, body)


def test_compact_worker_visits_nonempty_cells_once(tmp_path):
    text = SOURCE.read_text()
    body = PREAMBLE + r"""
#define __shared__
#define __syncthreads() (++schedulerBarriers)
int schedulerBarriers=0;
double warpPartials[256];
struct CsrReductionTask {int cell,begin,end,source;};
struct DeviceState {
 int nCells,*csrHeavyTaskCount,*csrCellTaskCount,*cellParticleCount;
 CsrReductionTask* csrReductionTasks;
 double *V,*momRhoP,*momRhoUPx,*momRhoUPy,*momRhoUPz,*momRhoEP,*momRhoPD,*momRhoHpP,*csrHeavyPartials;
 double rhoMin=1e-12;int* visits;
 int csrReductionDirectoryKind=0;int *cellParticleOffset=nullptr,*preBaseCellOffset=nullptr;
 int queueCursor=0; int* csrHeavyTaskCursor=&queueCursor;
};
template<bool GatherSurvivors=false>
void accumulateCsrHeavyMomentTask(DeviceState& s,int c,int,int,double (&sums)[8],double*) {
 ++s.visits[c];for(int k=0;k<8;++k)sums[k]=c+1;
}
""" + 'enum class HeavyDirectoryKind {full,splitBaseAndInjection,baseOnly};\nenum class CsrReductionTaskSource {fullIndexed,splitBaseDirect,splitInjectionIndexed,splitLogical};\n' + function(text,'accumulateCsrSegmentedMomentTasksPersistentKernel').replace('extern __shared__ double warpPartials[];','') + r"""
int main() {
 for(int nt:{0,1,7,25,57,550}) {
  int blocks=nt+96;
  std::vector<int> counts(nt,1),pc(nt+1,-1),visits(nt,0);std::vector<double> v(nt,1),rho(nt,-1),x(nt),y(nt),z(nt),e(nt),d(nt),h(nt),partial(8*nt);
  std::vector<CsrReductionTask> tasks(nt);for(int c=0;c<nt;++c)tasks[c]={c,c,c+1,0};
  int nonempty=0;schedulerBarriers=0;
  for(int c=0;c<nt;++c) {if(c%3==0)counts[c]=0;else ++nonempty;}
  int heavyTaskCount=0;for(int c=0;c<nt;++c)if(counts[c])tasks[heavyTaskCount++]={c,c,c+1,0};
  DeviceState s{nt,&heavyTaskCount,counts.data(),pc.data(),tasks.data(),v.data(),rho.data(),x.data(),y.data(),z.data(),e.data(),d.data(),h.data(),partial.data(),1e-12,visits.data()};
  std::vector<int> offsets(nt+1);for(int c=0;c<=nt;++c)offsets[c]=c;s.cellParticleOffset=offsets.data();
  blockDim.x=1;gridDim.x=blocks;threadIdx.x=0;
  for(int b=0;b<blocks;++b){blockIdx.x=b;accumulateCsrSegmentedMomentTasksPersistentKernel(&s);}
  assert(schedulerBarriers >= nonempty);
  for(int c=0;c<nt;++c)if(visits[c]!=(c%3!=0) || rho[c]!=(c%3==0?-1:c+1)){std::cerr<<"Task "<<c<<" visited "<<visits[c]<<" times";return 1;}
 }
}
"""
    compile_run(tmp_path,body)






def test_pool_target_cell_handles_empty_and_invalid_mass(tmp_path):
    text=SOURCE.read_text()
    body=PREAMBLE+r"""
struct DeviceState {int *poolThermalCount,*poissonPoolSampleTargetCount;double *poissonPoolMass;};
"""+function(text,'preparePoissonPoolSamplingCell')+r"""
int main() {
 int count[]={4,4,0,4,-1,4,4};int target[7]={-1,-1,-1,-1,-1,-1,-1};
 double mass[]={1,0,1,-1,1,NAN,INFINITY};DeviceState s{count,target,mass};
 for(int c=0;c<7;++c)preparePoissonPoolSamplingCell(s,c);
 assert(target[0]==4);for(int c=1;c<7;++c)assert(target[c]==0);
}
"""
    compile_run(tmp_path,body)

def test_fused_moment_recovery_visits_every_cell_once(tmp_path):
    text=SOURCE.read_text()
    body=PREAMBLE+r"""
#define __shared__
#define __shared__
#define __syncthreads() ((void)0)
double warpPartials[256];
template<int N> void blockReduceComponentSums(double (&)[N],double*) {}
struct DeviceState {
 int nCells,multiprocessorCount,*csrHeavyCellCount,*csrMultiTaskCellList,*csrCellTaskCount,*csrCellTaskOffset,*cellParticleCount;
 double *csrHeavyPartials,*V,*momRhoP,*momRhoUPx,*momRhoUPy,*momRhoUPz,*momRhoEP,*momRhoPD,*momRhoHpP;
 double rhoMin=1e-12;int *visits;
 int cursor=0;int* csrHeavyTaskCursor=&cursor;
};
void solidRecoveryFromParticleMomentsCell(DeviceState& s,int c) {
 ++s.visits[c]; assert(s.momRhoP[c]==(s.csrCellTaskCount[c]>1?double(2*(c+1)):double(c+1)));
}
"""+function(text,'finalizeCsrSegmentedMomentsAndRecoverKernel').replace('extern __shared__ double warpPartials[];','')+r"""
int main() {
 for(int nc:{1,3,24,25,97,433}) for(int sm:{1,3,24}) {
  std::vector<int> cells,count(nc,1),off(nc),pc(nc+1),visits(nc,0);
  std::vector<double> partial(16*nc),v(nc,1),rho(nc),x(nc),y(nc),z(nc),e(nc),d(nc),h(nc);
  for(int c=0;c<nc;++c){rho[c]=c+1;off[c]=2*c;if(c%2==0){cells.push_back(c);count[c]=2;rho[c]=-1;}for(int k=0;k<16;++k)partial[16*c+k]=c+1;}
  int nh=cells.size();DeviceState s{nc,sm,&nh,cells.data(),count.data(),off.data(),pc.data(),partial.data(),v.data(),rho.data(),x.data(),y.data(),z.data(),e.data(),d.data(),h.data(),1e-12,visits.data()};
  blockDim.x=1;threadIdx.x=0;gridDim.x=std::min(sm,nc)+nc;
  for(int b=0;b<gridDim.x;++b){blockIdx.x=b;finalizeCsrSegmentedMomentsAndRecoverKernel(&s);}
  for(int c=0;c<nc;++c)assert(visits[c]==1);
 }
}
"""
    compile_run(tmp_path,body)


def test_moment_fusion_is_opt_in_for_advance_not_restart():
    text=SOURCE.read_text()
    launch=function(text,'launchCsrSegmentedMomentReduction')
    wrapper=function(text,'launchCsrHeavyMomentReduction')
    assert 'const bool deferRecovery = false' in launch
    assert 'if (!deferRecovery)' in launch
    assert 'finalizeCsrSegmentedMomentCellsKernel' in launch
    assert 'const bool deferRecovery = false' in wrapper
    assert 'launchCsrSegmentedMomentReduction(s, block, deferRecovery, gatherSurvivors)' in wrapper
    advance=text[text.index('int ugkwpGpuResidentStrictAdvance'):]
    assert 'launchCsrHeavyMomentReduction(s, block, true, true)' in advance
    assert 'launchCsrHeavyMomentReduction(s, block)' in text[:text.index('int ugkwpGpuResidentStrictAdvance')]
