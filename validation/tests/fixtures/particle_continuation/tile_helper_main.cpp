int reference(long long p,int b,int m,int r) {
 if(p<0||b<=0||m<=0||r<=0)return 0;
 const unsigned __int128 concurrency=static_cast<unsigned __int128>(b)*m*r;
 const unsigned __int128 shares=p==0 ? 1 : 1+(static_cast<unsigned __int128>(p)-1)/concurrency;
 const unsigned __int128 tile=shares*b;
 return tile>INT_MAX?INT_MAX:static_cast<int>(tile);
}
int main() {
 std::mt19937_64 rng(781423);long long checked=0;
 std::vector<long long> populations={0,1,255,256,257,INT_MAX-512LL,INT_MAX-256LL,INT_MAX-255LL,INT_MAX-1LL,INT_MAX,INT_MAX+1LL,2LL*INT_MAX,LLONG_MAX-1,LLONG_MAX};
 for(long long p:populations)for(int b:{1,3,32,256,1024,INT_MAX})for(int m:{1,7,144,INT_MAX})for(int r:{1,4,INT_MAX}) {
  int got=hardwareReductionTile(p,b,m,r),want=reference(p,b,m,r);assert(got==want);++checked;
 }
 for(int i=0;i<200000;++i) {
  long long p=static_cast<long long>(rng()&LLONG_MAX);int b=1+rng()%INT_MAX,m=1+rng()%INT_MAX,r=1+rng()%INT_MAX;
  assert(hardwareReductionTile(p,b,m,r)==reference(p,b,m,r));++checked;
 }
 std::cout<<"helper_reference_cases="<<checked<<"\n";
 assert(hardwareReductionTile(-1,256,1,1)==0);assert(hardwareReductionTile(1,0,1,1)==0);
 // Saturation preserves partition counts and > threshold decisions for int populations.
 for(int i=0;i<200000;++i) {
  long long raw=static_cast<long long>(INT_MAX)+1+(rng()%INT_MAX);int count=rng()%(static_cast<uint64_t>(INT_MAX)+1);
  int n=count?1+(count-1)/INT_MAX:0;long long nw=count?1+(static_cast<long long>(count)-1)/raw:0;
  assert(n==nw);assert((count>INT_MAX)==(count>raw));
 }
 std::cout<<"per_cell_saturation_equivalence_cases=200000\n";
}
