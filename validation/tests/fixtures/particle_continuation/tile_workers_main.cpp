void run(std::string name,int length,unsigned threads){
 DeviceState s;double sums[8]={},scratch[8]={};
 expectedBegin=INT_MAX-length;expectedEnd=INT_MAX;s.offset[0]=expectedBegin;s.offset[1]=expectedEnd;s.base[0]=expectedBegin;s.base[1]=expectedEnd;
 seen.clear();blockDim.x=threads;blockIdx.x=0;
 for(threadIdx.x=0;threadIdx.x<threads;++threadIdx.x){
  if(name=="moment")runMoment(s,expectedBegin,expectedEnd);
  else if(name=="pool_direct")accumulateCsrHeavyPoolTask<false>(s,0,expectedBegin,expectedEnd,true,1,sums,scratch);
  else if(name=="pool_indexed")accumulateCsrHeavyPoolTask<false>(s,0,expectedBegin,expectedEnd,false,1,sums,scratch);
  else if(name=="pool_l1")accumulatePoissonPoolParticlesByCellKernel(&s,1);
  else if(name=="split_l1_base")accumulatePoissonPoolSplitSegmentKernel<true,false>(&s,1);
  else if(name=="split_l1_injection")accumulatePoissonPoolSplitSegmentKernel<false,false>(&s,1);
  else if(name=="split_base"){
   s.base[0]=0;s.base[1]=INT_MAX;s.offset[0]=s.offset[1]=0;
   accumulateCsrSplitLogicalPoolTask<false>(s,0,expectedBegin,expectedEnd,1,sums,scratch);
  }else if(name=="split_injection"){
   s.base[0]=s.base[1]=0;s.offset[0]=0;s.offset[1]=INT_MAX;
   // Proxy records directory indices; particle stub i=0 is not a valid logical index, so return same index and check unique positions separately.
   s.sortedParticleIndex.recordIndex=false;
   accumulateCsrSplitLogicalPoolTask<false>(s,0,expectedBegin,expectedEnd,1,sums,scratch);
  }
 }
 std::sort(seen.begin(),seen.end());int multiplier=name=="split_injection"?2:1;assert(seen.size()==static_cast<size_t>(length*multiplier));
 for(int i=0;i<length;++i)for(int k=0;k<multiplier;++k)assert(seen[i*multiplier+k]==expectedBegin+i);
}
int main(int argc,char**argv){
 std::string variant="production";
 std::vector<std::string> names={"moment","pool_direct","pool_indexed","pool_l1","split_l1_base","split_l1_injection","split_base","split_injection"};
 if(argc>1){run(argv[1],2,256);return 0;}
 long long cases=0;for(auto name:names)for(int length:{0,1,2,31,32,255,256,257,511,512,1023,1024,1025})for(unsigned threads:{1,32,64,128,256,512,1024}){run(name,length,threads);++cases;}
 // Exact split injection mapping when the unparenthesized intermediate overflows.
 for(unsigned threads:{1,32,256,1024}) {
  DeviceState s;double sums[8]={},scratch[8]={};
  s.base[0]=0;s.base[1]=INT_MAX-2;s.offset[0]=INT_MAX-7;s.offset[1]=INT_MAX-5;
  s.sortedParticleIndex.recordIndex=false;expectedBegin=INT_MAX-7;expectedEnd=INT_MAX-5;seen.clear();blockDim.x=threads;
  for(threadIdx.x=0;threadIdx.x<threads;++threadIdx.x)
   accumulateCsrSplitLogicalPoolTask<false>(s,0,INT_MAX-2,INT_MAX,1,sums,scratch);
  std::sort(seen.begin(),seen.end());assert(seen.size()==4);
  for(int j=0;j<4;++j)assert(seen[j]==expectedBegin+j/2);
 }
 std::cout<<"variant="<<variant<<" actual_header_worker_cases="<<cases<<" all_positions_exact_once=pass\n";
}
