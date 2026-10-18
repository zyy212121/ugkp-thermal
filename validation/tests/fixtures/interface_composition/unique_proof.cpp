#include "ablation/InterfaceMath.H"
#include <cassert>
#include <random>
#include <cstdio>
using namespace chmt;
int main(){std::mt19937_64 rng(72811);std::uniform_real_distribution<double> sample(0,1);const Real gamma=3*2.22044604925031308085e-16/(1-3*2.22044604925031308085e-16);for(int c=0;c<30000;++c){Real u[Ns],v[Ns],r[Ns],bulk[Ns];bool expected=true;for(int i=0;i<Ns;++i){u[i]=2*sample(rng)-1;v[i]=sample(rng);r[i]=c%2?sample(rng):(i%3+1.)/3;bulk[i]=sample(rng);if(c%113==0)bulk[i]=0;if(c%127==0)u[i]=0;if(c%131==0)v[i]=0;}if(c%3==0&&Ns>1){int row=c%Ns,other=(row+1)%Ns;u[row]=-.5;v[row]=.5*r[other]*(1+((c%7)-3)*2.22044604925031308085e-16);}
 for(int i=0;i<Ns;++i){expected=expected&&bulk[i]>0;for(int j=0;j<Ns;++j)if(i!=j){Real product=0,value=0,magnitude=0;if(!composition_detail::productFinite(u[i],r[j],product)||!composition_detail::sumFinite(product,v[i],value)||!composition_detail::sumFinite(absValue(product),absValue(v[i]),magnitude)||value<=gamma*magnitude)expected=false;}}
 assert(composition_detail::uniquePositiveMatrix(u,v,r,bulk,gamma)==expected);}
 printf("PASS Ns=%d linear proof equals full pair proof on 30000 signed/tied/boundary matrices\n",Ns);}
