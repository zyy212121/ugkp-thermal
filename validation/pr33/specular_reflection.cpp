#include "gpu/SpecularReflection.H"
#include <cassert>
#include <cstdio>
int main(){int cases=0;double worst=0;
 for(int nt:{2,4,8})for(int np:{4,8,16}){
  std::vector<std::array<double,3>> dirs;std::vector<double> weights;
  const double pi=std::acos(-1.);
  for(int t=0;t<nt;++t)for(int p=0;p<np;++p){double theta=(t+.5)*pi/nt,phi=(p+.5)*2*pi/np;
   dirs.push_back({std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta)});
   weights.push_back(std::sin(theta)*pi/nt*2*pi/np);}
  for(int k=0;k<40;++k){std::array<double,3> normal{k==0?1.:std::cos(k*.3),k==0?0.:std::sin(k*.7),k==0?0.:std::cos(k*.9)};
   double length=0;for(double x:normal)length+=x*x;for(auto&x:normal)x/=std::sqrt(length);
   auto rows=ugkp::gpu::specularReflection(dirs,weights,normal);std::vector<double> mu(dirs.size()),cols(dirs.size());
   for(int i=0;i<int(dirs.size());++i)for(int d=0;d<3;++d)mu[i]+=dirs[i][d]*normal[d];
   double out=0,in=0;
   for(int i=0;i<int(dirs.size());++i){double value=0,row=0;
    for(auto e:rows[i]){assert(mu[e.donor]>0&&e.weight>=0);row+=e.weight;cols[e.donor]+=-weights[i]*mu[i]*e.weight;value+=e.weight*(1.+.03*e.donor*e.donor);}
    if(mu[i]<0){assert(std::abs(row-1.)<1e-11);in+=-weights[i]*mu[i]*value;}
    else {assert(rows[i].empty());out+=weights[i]*mu[i]*(1.+.03*i*i);}
   }
   assert(std::abs(in-out)<1e-12*std::max(1.,out));worst=std::max(worst,std::abs(in-out)/std::max(1.,out));
   for(int j=0;j<int(dirs.size());++j)if(mu[j]>0)assert(std::abs(cols[j]-weights[j]*mu[j])<1e-12);
   ++cases;
  }
 }
 printf("%d reflection geometries; max relative net flux %.17g\n",cases,worst);
}
