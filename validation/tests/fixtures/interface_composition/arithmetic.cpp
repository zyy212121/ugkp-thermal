#include "ablation/InterfaceMath.H"
#include <cfenv>
#include <fenv.h>
#include <random>
#include <limits>
#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace chmt;
static double reference(double a,double b,bool divide){fenv_t saved;feholdexcept(&saved);volatile double av=a,bv=b;volatile double value=divide?av/bv:av*bv;double result=value;fesetenv(&saved);return result;}
int main(int argc,char**argv){int count=argc>1?atoi(argv[1]):10000;feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);int mask=fegetexcept();std::mt19937_64 random(3014);double maximum=std::numeric_limits<double>::max();int rejected=0;
for(int i=0;i<count;++i)for(int kind=0;kind<2;++kind){double b=kind?.5+.5*double(random()>>11)*1.11022302462515654042e-16:1+double(random()>>11)*1.11022302462515654042e-16;double a=kind?maximum*b:maximum/b;double expected=reference(a,b,kind);std::feclearexcept(FE_ALL_EXCEPT);double actual=0;bool ok=kind?composition_detail::quotientFinite(a,b,actual):composition_detail::productFinite(a,b,actual);if(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW)||fegetexcept()!=mask)return 2;if(ok!=std::isfinite(expected)||(ok&&actual!=expected))return 3;rejected+=!ok;}
std::cout<<"safe boundary operations="<<2*count<<" overflowing="<<rejected<<" sizeofPhysicsConfig="<<sizeof(PhysicsConfig)<<'\n';}
