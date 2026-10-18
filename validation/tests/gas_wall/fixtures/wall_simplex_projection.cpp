#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cfenv>
#include <cstdio>
using namespace ugkwp::gaswall;
template<int N>void check(){unsigned long long rng=4291;for(int trial=0;trial<10000;++trial){double x[N+4]{};long double raw[N],sum=0;for(int j=0;j<N-1;++j){rng=rng*6364136223846793005ULL+1;x[3+j]=6.*double(rng>>11)/9007199254740992.-3;raw[j]=x[3+j];sum+=raw[j];}raw[N-1]=1-sum;assert((detail::feasibleLayerComposition<double,N>(x)));double total=0;for(int j=0;j<N-1;++j){assert(x[3+j]>=0&&x[3+j]<=1);total+=x[3+j];}assert(total<=1);long double multiplier=0;bool found=false;for(int j=0;j<N;++j){long double value=j==N-1?1-total:x[3+j];if(value>1e-13L){multiplier=raw[j]-value;found=true;break;}}assert(found);for(int j=0;j<N;++j){long double value=j==N-1?1-total:x[3+j];if(value>1e-13L)assert(fabsl(raw[j]-value-multiplier)<1e-11L);else assert(raw[j]-multiplier<1e-11L);}
 double same[N+4];for(int j=0;j<N+4;++j)same[j]=x[j];assert((detail::feasibleLayerComposition<double,N>(x)));for(int j=0;j<N+4;++j)assert(same[j]==x[j]);}
 for(int mode=0;mode<3;++mode){double x[N+4]{};for(int j=0;j<N-1;++j)x[3+j]=(mode==0?1:-1)*std::numeric_limits<double>::max()/2;assert((detail::feasibleLayerComposition<double,N>(x)));double sum=0;for(int j=0;j<N-1;++j){assert(x[3+j]>=0&&x[3+j]<=1);sum+=x[3+j];}assert(sum<=1);}
 double x[N+4]{};x[3]=std::numeric_limits<double>::infinity();assert((!detail::feasibleLayerComposition<double,N>(x)));printf("PASS Ns=%d 10000 KKT/simplex/idempotence + range/nonfinite\n",N);}
int main(){feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);check<2>();check<3>();check<10>();check<32>();}
