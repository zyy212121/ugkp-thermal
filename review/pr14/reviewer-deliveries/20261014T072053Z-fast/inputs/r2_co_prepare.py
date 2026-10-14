import difflib
from r2_runner import *
original=D/'inputs/fixtures/courant_reduction.cu';source=D/'inputs/fixtures/courant_fresh_data.cu'
function=r'''
void freshData(CoupledCourantStorage& storage,int count,int round,cudaStream_t stream){
 std::vector<double> h(count);std::vector<int> st(count);
 for(int i=0;i<count;++i){h[i]=double((i*(round+3)+round*13)%109)-77.-double(round)/8;st[i]=(i+round)%19==0?int(ugkwp::GasTransportCode::InvalidStorage):0;}
 if(count && round%7==0){std::uint64_t nan=0x7ff8000000000000ULL+round;std::memcpy(&h[count/2],&nan,8);}
 double*f=nullptr;int*s=nullptr;check(cudaMalloc(&f,std::max(count,1)*sizeof(double)));check(cudaMalloc(&s,std::max(count,1)*sizeof(int)));
 if(count){check(cudaMemcpyAsync(f,h.data(),count*8,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(s,st.data(),count*4,cudaMemcpyHostToDevice,stream));}
 auto first=thrust::make_counting_iterator(0);CoupledCourantValue init{-std::numeric_limits<double>::infinity(),false};
 auto expected=thrust::transform_reduce(thrust::cuda::par.on(stream),first,first+count,CoupledCourantRead{f,s},init,CoupledCourantReduce{});
 auto actual=thrust::transform_reduce(thrust::cuda::par(storage).on(stream),first,first+count,CoupledCourantRead{f,s},init,CoupledCourantReduce{});
 sameBits(actual.value,expected.value);assert(actual.unavailable==expected.unavailable);for(auto b:actual.initializedTail)assert(b==0);
 auto expectedScalar=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(f),thrust::device_pointer_cast(f)+count,init.value,CoupledCourantMaximum{});
 auto scalar=thrust::reduce(thrust::cuda::par(storage).on(stream),thrust::device_pointer_cast(f),thrust::device_pointer_cast(f)+count,init.value,CoupledCourantMaximum{});sameBits(scalar,expectedScalar);
 check(cudaFree(s));check(cudaFree(f));
 std::printf("FRESH_DATA n=%d round=%d result=%a unavailable=%d\n",count,round,actual.value,int(actual.unavailable));
}
'''
changed=original.read_text();at=changed.index('int main(){');changed=changed[:at]+function+changed[at:];changed=changed.replace('    cudaStream_t stream=nullptr;', '    cudaStream_t stream=nullptr;')
old='    check(cudaStreamDestroy(stream));'
new='    { CoupledCourantStorage owner,other; int round=0; for(int n:{1,32,513,32768,513,1,0})for(int repeat=0;repeat<12;++repeat){freshData(owner,n,round++,stream);freshData(other,n,round++,nullptr);} }\n'+old
assert changed.count(old)==1;changed=changed.replace(old,new);source.write_text(changed)
(D/'patches/S205-Co-changing-data.patch').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),changed.splitlines(True),fromfile='a/fixtures/courant_reduction.cu',tofile='b/fixtures/courant_fresh_data.cu')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS205 Co新增证据：原fixtures不断归约同数据，不能证明临时owner不缓存旧Co。追加同owner跨1→32→513→32768→513→1→empty容量周期、改变每轮实际device Co/status/NaNpayload，并交错另一owner/default与nonblockingstream；每轮对真实CUDA无缓存Thrust归约bit比较。生产operator/header完全相同，无容差。Ns2/Ns10仅编译speciesflags分别记录，initcheck单独观察300s。各合法case独立hostinput及device数组。\n')
for ns in [2,10]:
 b=ROOT/'work/fixtures'/f'courant_fresh_{ns}'
 run('BUILD-CO-FRESH-'+str(ns),cmd('nvcc','-std=c++17','-O2','-arch=sm_89','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),source,'-o',b),[9,12,14,26,28],900,'Actualunchangedoperators+cacheowner; changingdata and growth in Ns'+str(ns)+' build.',inputs=[source],kind='BUILD')
for name in ['thermal-resumed','single-full']:
 for p in (OLD/'inputs/native/chmt'/name).rglob('*'):
  if p.is_file() and p.name in ['controlDict','chmtProperties']:print(p,p.read_text())
