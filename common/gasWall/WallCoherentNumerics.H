#pragma once
#include "WallLayerMath.H"
#include <algorithm>
#include <cmath>
namespace ugkwp { namespace gaswall { namespace coherent {
// Bounded storage with vector-like active length, not a physical acceptance
// policy. Capacities come from the already selected workspace bucket.
template<class T,int Capacity> struct BoundedVector {
 T data[Capacity];int count=0;
 UGKWP_GAS_HD int size()const{return count;}
 UGKWP_GAS_HD void resize(int n){count=n;}
 UGKWP_GAS_HD void clear(){count=0;}
 UGKWP_GAS_HD T&operator[](int i){return data[i];}
 UGKWP_GAS_HD const T&operator[](int i)const{return data[i];}
 UGKWP_GAS_HD T*begin(){return data;} UGKWP_GAS_HD T*end(){return data+count;}
 UGKWP_GAS_HD const T*begin()const{return data;} UGKWP_GAS_HD const T*end()const{return data+count;}
 UGKWP_GAS_HD void push_back(const T&v){data[count++]=v;}
 UGKWP_GAS_HD void assign(int n,const T&v){count=n;for(int i=0;i<n;++i)data[i]=v;}
 UGKWP_GAS_HD BoundedVector&operator=(const BoundedVector&v){count=v.count;for(int i=0;i<count;++i)data[i]=v.data[i];return *this;}
 UGKWP_GAS_HD bool operator==(const BoundedVector&v)const{if(count!=v.count)return false;for(int i=0;i<count;++i)if(data[i]!=v.data[i])return false;return true;}
 UGKWP_GAS_HD bool operator!=(const BoundedVector&v)const{return !(*this==v);}
};
template<class T> UGKWP_GAS_HD inline void swapValue(T&a,T&b){T t=a;a=b;b=t;}
// Numerical alternatives only; never validates or rejects a physical input.
// Original scalar probes retain their existing (including clipped-Y) behavior.
UGKWP_GAS_HD inline bool fdProbeAdmissible(const WallInput<double,2>& in,
                                          int column,double value){
 if(!std::isfinite(value))return false;
 if(column==2)return in.temperature+value>0;
 if(column==3)return value>=0&&value<=1;
 // a=state[4] is signed: the existing physical coordinate is k=a*a.
 if(column==5)return value>=0;
 return true;
}
template<class Probe> UGKWP_GAS_HD inline int fdSelectDirection(Probe probe){
 if(probe(1))return 1;
 if(probe(-1))return -1;
 return 0;
}
// FD-only compatibility with the actual PCHIP pieces. Does not change F.
// At an exact switching tie, permit a finite one-sided adjacent piece;
// exact zero is intentional (no fitted epsilon or physical admissibility gate).
template<class R> UGKWP_GAS_HD inline bool pchipFdMargin(R base,R trial){return base==0 || (base>0?trial>0:trial<0);}
template<class R> UGKWP_GAS_HD inline int pchipInteriorPiece(R dl,R dr){return ((dl>0&&dr>0)||(dl<0&&dr<0))?1:0;}
template<class R> UGKWP_GAS_HD inline int pchipEndpointPiece(R d0,R d1,R slope){return slope*d0<=0?0:d0*d1<0&&std::abs(slope)>3*std::abs(d0)?1:2;}
template<class R,int N,int M,int P> UGKWP_GAS_HD inline bool pchipFdCompatible(const WallModelConfig<R>&cfg,const WallWorkspace<R,N,M>&base,const WallWorkspace<R,N,P>&trial,int changedNode=-1,int changedField=-1){
 for(int j=N+2;j<N+4;++j){
  if(changedField>=0&&j!=changedField)continue;
  for(int i=changedNode<0?1:std::max(1,changedNode-1);i<(changedNode<0?cfg.nodes-1:std::min(cfg.nodes-1,changedNode+2));++i){R hl=base.y[i]-base.y[i-1],hr=base.y[i+1]-base.y[i];R dl=(base.state[i][j]-base.state[i-1][j])/hl,dr=(base.state[i+1][j]-base.state[i][j])/hr,tl=(trial.state[i][j]-trial.state[i-1][j])/hl,tr=(trial.state[i+1][j]-trial.state[i][j])/hr;
   if(pchipInteriorPiece(dl,dr)!=pchipInteriorPiece(tl,tr)&&(!pchipFdMargin(dl,tl)||!pchipFdMargin(dr,tr)))return false;
  }
  if(cfg.nodes>2&&(changedNode<0||changedNode>=cfg.nodes-3)){int i=cfg.nodes-1;R h0=base.y[i]-base.y[i-1],h1=base.y[i-1]-base.y[i-2],d0=(base.state[i][j]-base.state[i-1][j])/h0,d1=(base.state[i-1][j]-base.state[i-2][j])/h1,t0=(trial.state[i][j]-trial.state[i-1][j])/h0,t1=(trial.state[i-1][j]-trial.state[i-2][j])/h1;R slope=((2*h0+h1)*d0-h0*d1)/(h0+h1),ts=((2*h0+h1)*t0-h0*t1)/(h0+h1);
   if(pchipEndpointPiece(d0,d1,slope)!=pchipEndpointPiece(t0,t1,ts)&&(!pchipFdMargin(d0,t0)||!pchipFdMargin(d1,t1)||!pchipFdMargin(slope,ts)||!pchipFdMargin(slope-3*d0,ts-3*t0)||!pchipFdMargin(slope+3*d0,ts+3*t0)))return false;
  }
 }
 return true;
}
template<int M> struct CachedLU {int n=0;BoundedVector<int,M*6>pivot;BoundedVector<double,M*6*35>upper;BoundedVector<double,M*6*17>lower;bool valid=false;
 template<class Matrix> UGKWP_GAS_HD bool factor(Matrix&a,int nn){n=nn;pivot.resize(n);upper.assign(n*35,0.);lower.assign(n*17,0.);for(int k=0;k<n;++k){int p=k;for(int i=k+1;i<std::min(n,k+18);++i)if(std::abs(a[i*n+k])>std::abs(a[p*n+k]))p=i;if(!std::isfinite(a[p*n+k])||std::abs(a[p*n+k])<std::numeric_limits<double>::min()*100)return false;pivot[k]=p;if(p!=k)for(int j=k;j<std::min(n,k+35);++j)swapValue(a[p*n+j],a[k*n+j]);for(int i=k+1;i<std::min(n,k+18);++i){double t=a[i*n+k]/a[k*n+k];lower[k*17+i-k-1]=t;for(int j=k+1;j<std::min(n,k+35);++j)a[i*n+j]-=t*a[k*n+j];}}for(int i=0;i<n;++i)for(int j=i;j<std::min(n,i+35);++j)upper[i*35+j-i]=a[i*n+j];valid=true;return true;}
 template<class Vector> UGKWP_GAS_HD void solve(Vector&b)const{for(int k=0;k<n;++k){if(pivot[k]!=k)swapValue(b[pivot[k]],b[k]);for(int i=k+1;i<std::min(n,k+18);++i)b[i]-=lower[k*17+i-k-1]*b[k];}for(int i=n-1;i>=0;--i){for(int j=i+1;j<std::min(n,i+35);++j)b[i]-=upper[i*35+j-i]*b[j];b[i]/=upper[i*35];}}
};

}}} // ugkwp::gaswall::coherent
