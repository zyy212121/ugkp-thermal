void runCase(std::vector<int> base,std::vector<int> inj,int tile,int kind) {
 int nc=base.size()-1;std::vector<int> counts(nc+1),offset(nc+1),multi(nc);int nh=0,nt=0;
 DeviceState s;s.nCells=nc;s.csrHeavyTileParticles=tile;s.preBaseCellOffset=base.data();s.cellParticleOffset=inj.data();s.csrCellTaskCount=counts.data();s.csrCellTaskOffset=offset.data();s.csrHeavyCellCount=&nh;s.csrMultiTaskCellList=multi.data();s.csrHeavyTaskCount=&nt;
 blockDim.x=1;gridDim.x=nc+1;threadIdx.x=0;
 for(int c=0;c<=nc;++c){blockIdx.x=c;countCsrReductionTasksKernel(&s,kind);}
 for(int c=0;c<nc;++c)offset[c+1]=offset[c]+counts[c];
 assert(offset.back()<100000);s.csrHeavyTaskCapacity=offset.back()+1;
 std::vector<CsrReductionTask> tasks(s.csrHeavyTaskCapacity,{-99,-99,-99,-99});s.csrReductionTasks=tasks.data();
 for(int c=0;c<nc;++c){blockIdx.x=c;materializeCsrReductionTasksKernel(&s,kind);}
 assert(nt==offset.back());int expectedHeavy=0;
 for(int c=0;c<nc;++c){
  const int begin=kind==0?inj[c]:(kind==2?base[c]:0);
  const int end=kind==0?inj[c+1]:(kind==2?base[c+1]:(base[c+1]-base[c])+(inj[c+1]-inj[c]));
  int cur=begin;expectedHeavy+=counts[c]>1;
  for(int t=offset[c];t<offset[c+1];++t){auto d=tasks[t];assert(d.cell==c&&d.begin==cur&&d.end>d.begin&&d.end-d.begin<=tile);
   assert(d.source==(kind==0?0:kind==2?1:3));cur=d.end;}
  assert(cur==end);assert(counts[c]==(end>begin?1+(end-begin-1)/tile:0));
 }
 assert(nh==expectedHeavy);assert(tasks[offset.back()].cell==-99);
}
int main() {
 for(int tile:{1,3,31,32,255,256,1024,1073741824,INT_MAX}){
  for(int kind=0;kind<(HAS_BASE?3:2);++kind){
   // Keep symbolic descriptor count small, never allocate particle backing.
   if(tile>=1073741824){runCase({0,0,0},{0,1,INT_MAX},tile,kind);runCase({0,1,INT_MAX},{0,0,0},tile,kind);runCase({0,INT_MAX-1},{0,1},tile,kind);}
   runCase({0,0,1,2,31},{0,1,4,20,30},tile,kind);
  }
 }
 std::mt19937 rng(12071);long long cases=0;
 for(int trial=0;trial<10000;++trial){
  int nc=1+rng()%32;std::vector<int>b(nc+1),q(nc+1);
  for(int c=0;c<nc;++c){b[c+1]=b[c]+rng()%400;q[c+1]=q[c]+rng()%300;}
  int tile=1+rng()%5000;for(int kind=0;kind<(HAS_BASE?3:2);++kind){runCase(b,q,tile,kind);++cases;}
 }
 for(int d:{0,1,2,31,255,256,257,511})for(int kind=0;kind<(HAS_BASE?3:2);++kind){
  runCase({0,INT_MAX-d},{0,0},INT_MAX,kind);runCase({0,0},{0,INT_MAX-d},INT_MAX,kind);++cases;
 }
 std::cout<<"materialize_has_base="<<HAS_BASE<<" property_cases="<<cases<<" INT_MAX_offset_and_roundup_cases=pass\n";
}
