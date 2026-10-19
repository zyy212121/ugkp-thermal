#include <initializer_list>
#include "common/gasTransport/MixtureThermo.H"
#include <cstdio>
#include <cstring>
#include <cassert>
using namespace ugkwp;
bool bits(double a,double b){return std::memcmp(&a,&b,sizeof(a))==0;}
int main(){unsigned cases=0;double coef[28]={};for(int i=0;i<28;++i)coef[i]=(i%7==0?3.5:0.);double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
for(int model=0;model<2;++model)for(double t:{-1.,-0.,0.,1.,299.,300.,301.,1000.,1e100,nan,inf})for(int count=0;count<=28;++count)for(int offset:{-1,0,3,14,30})for(double mass:{0.,.028,-.028,nan,inf})for(int issue=0;issue<4;++issue){
 SpeciesThermoData<double> sp[1]{};sp[0].model=static_cast<SpeciesThermoModel>(model);sp[0].coefficientOffset=offset;sp[0].molarMass=mass;sp[0].midTemperature=300;sp[0].hasEntropyReference=true;sp[0].referenceTemperature=300;
 double coefficients[28];std::memcpy(coefficients,coef,sizeof(coef));if(issue==1)coefficients[0]=nan;if(issue==2)coefficients[2]=inf;if(issue==3)coefficients[5]=inf;
 ugkp::diagnostics::DeviceContext before{},after{};SpeciesThermoView<double,1> v;v.species=sp;v.coefficients=coefficients;v.coefficientCount=count;v.diagnostics=&before;
 double h=speciesH(0,t,v),cp=speciesCp(0,t,v),bothH=-7,bothCp=-9;v.diagnostics=&after;speciesCpH(0,t,v,bothCp,bothH);
 if(!bits(h,bothH)||!bits(cp,bothCp)||std::memcmp(&before,&after,sizeof(before))){std::printf("difference model=%d t=%g count=%d offset=%d mass=%g issue=%d\n",model,t,count,offset,mass,issue);return 1;}
 ++cases;
}
std::printf("CpH exact value + diagnostic comparisons: %u passed\n",cases);
}