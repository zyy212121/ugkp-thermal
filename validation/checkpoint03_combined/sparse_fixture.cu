#include "GpuPrecisionTypes.H"
#include "gpu/SparseConjugateGradient.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
using namespace ugkp::gpu;
using R=std::remove_cv_t<std::remove_pointer_t<decltype(SparseDeviceLduValues{}.diagonal)>>;
void ck(bool x,const char*m){if(!x)throw std::runtime_error(m);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct D{
 T*p=nullptr;size_t n;D(const std::vector<T>&v):n(v.size()){cu(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));if(n)cu(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 ~D(){cudaFree(p);}std::vector<T>read(){std::vector<T>v(n);if(n)cu(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Context{int n=0,sums=0,interfaces=0;R k=0;};
void global(void*p,R*,int){++static_cast<Context*>(p)->sums;}
void remote(void*p,const R*x,R*y,int n){auto&c=*static_cast<Context*>(p);++c.interfaces;ck(n==c.n,"interface vector count");y[0]-=c.k*x[n-1];y[n-1]-=c.k*x[0];}
void solve(int n,double scale,bool heterogeneous,int path){
 // Host CSR/LDU entry points deliberately retain their declared double API.
 // Each coefficient/RHS is first rounded to R so independent observation uses
 // the same resident scalar equation, not a hidden double reference equation.
 std::vector<double>mass(n),d(n),upper(n-1),lower(n-1),exact(n),rhs(n),old(n),x(n),rowsum(n);
 std::vector<int>owner(n-1),neighbour(n-1),offsets{0},columns;
 std::vector<double>values;const double k=double(R(.15*scale));
 for(int i=0;i<n;++i){mass[i]=double(R(scale*(heterogeneous?std::pow(10.,i%5-2):1.+.1*(i%5))));
  exact[i]=double(R(300.+std::sin(.17*i)));old[i]=x[i]=double(R(299.+.01*(i%11)));d[i]=mass[i];}
 for(int f=0;f<n-1;++f){owner[f]=f;neighbour[f]=f+1;upper[f]=lower[f]=double(R(-.15*std::sqrt(mass[f]*mass[f+1])));
  d[f]=double(R(d[f]-upper[f]));d[f+1]=double(R(d[f+1]-lower[f]));}
 for(int i=0;i<n;++i){double diag=d[i]+((i==0||i==n-1)?k:0);diag=double(R(diag));
  columns.push_back(i);values.push_back(diag);
  if(i){columns.push_back(i-1);values.push_back(lower[i-1]);}
  if(i+1<n){columns.push_back(i+1);values.push_back(upper[i]);}
  if(path!=1&&(i==0||i==n-1)){columns.push_back(i==0?n-1:0);values.push_back(-k);}
  offsets.push_back(values.size());
  R q=0,rs=0;for(int e=offsets[i];e<offsets[i+1];++e){q+=R(values[e])*R(exact[columns[e]]);rs+=R(values[e]);}
  if(path==1&&(i==0||i==n-1)){q-=R(k)*R(exact[i==0?n-1:0]);rs-=R(k);}
  rhs[i]=q;rowsum[i]=rs;
 }
 ConjugateGradientControls controls;controls.maximumIterations=4000;
 const R oldTolerance=R(std::max(1e-10,32.*std::numeric_limits<R>::epsilon()));
 controls.tolerance=sizeof(R)==4?R(256.*std::numeric_limits<R>::epsilon()):oldTolerance;
 controls.normalizationSmall=R(std::abs(scale)*1e-20);
 Context ctx{n,0,0,R(k)};SparseCommunication comm;
 if(path==1){comm.context=&ctx;comm.globalSum=global;comm.addInterface=remote;}
 SparseConjugateGradient solver;ConjugateGradientReport report;
 bool residentFinite=true,solutionFinite=true;
 std::vector<double>affine(n);
 if(path<2){SparseMatrixView matrix{n,int(values.size()),offsets.data(),columns.data(),values.data(),rowsum.data()};
  report=solver.solve(matrix,rhs.data(),x.data(),controls,comm);
 }else{
  std::vector<int>bo{0,n-1},bn{n-1,0};std::vector<double>bd{k,k},bs{k,k},jump{.125,-.125};
  affine[0]=double(R(k)*R(jump[0]));affine[n-1]=double(R(k)*R(jump[1]));
  rhs[0]=double(R(rhs[0]+affine[0]));rhs[n-1]=double(R(rhs[n-1]+affine[n-1]));
  SparseLduView matrix{n,n-1,2,owner.data(),neighbour.data(),bo.data(),bn.data(),d.data(),upper.data(),lower.data(),rhs.data(),bd.data(),bs.data(),jump.data()};
  auto real=[](const std::vector<double>&a){return std::vector<R>(a.begin(),a.end());};
  D<R>dd(real(d)),du(real(upper)),dl(real(lower)),dr(real(rhs)),dbd(real(bd)),dbs(real(bs)),dj(real(jump)),dx(real(old));
  SparseDeviceLduValues resident{dd.p,du.p,dl.p,dr.p,dbd.p,dbs.p,nullptr,dj.p};
  report=solver.solveResidentLdu(matrix,resident,dx.p,controls);
  const auto residentX=dx.read();x.assign(residentX.begin(),residentX.end());
  D<R>res(std::vector<R>(n,0));solver.residualResidentLdu(matrix,resident,dx.p,res.p);
  const auto measured=res.read();double residentL1=0;for(R a:measured){residentFinite=residentFinite&&std::isfinite(double(a));residentL1+=std::abs(double(a));}
  std::cout<<"RESIDENT_TRUE_RESIDUAL n="<<n<<" L1="<<residentL1<<'\n';
 }
 double mean=0;for(double v:old)mean+=v;mean/=n;
 double norm=controls.normalizationSmall,l1=0,signedResidual=0,sourceL1=0,inventory=0,maxError=0;
 for(int i=0;i<n;++i){double ax=affine[i],ax0=affine[i];
  for(int e=offsets[i];e<offsets[i+1];++e){ax+=values[e]*x[columns[e]];ax0+=values[e]*old[columns[e]];}
  if(path==1&&(i==0||i==n-1)){ax-=k*x[i==0?n-1:0];ax0-=k*old[i==0?n-1:0];}
  double r=rhs[i]-ax;l1+=std::abs(r);signedResidual+=r;sourceL1+=std::abs(rhs[i]);
  norm+=std::abs(ax0-rowsum[i]*mean)+std::abs(rhs[i]-rowsum[i]*mean);
  inventory+=mass[i]*(exact[i]-x[i]);maxError=std::max(maxError,std::abs(exact[i]-x[i]));
  solutionFinite=solutionFinite&&std::isfinite(x[i]);
 }
 const double eps=std::numeric_limits<R>::epsilon();
 const double trueBound=4*double(oldTolerance)*norm+128*eps*sourceL1;
 const double closureBound=256*eps*sourceL1+1e-12*std::abs(scale);
 const double oldSolutionBound=std::max(2e-7,256*eps*301);
 // Predeclared FP32 engineering relative accuracy budget, not an epsilon chase.
 const double solutionReferenceScale=301.;
 const double solutionBound=sizeof(R)==4?1e-4*solutionReferenceScale:oldSolutionBound;
 std::cout<<"SPARSE bits="<<8*sizeof(R)<<" n="<<n<<" scale="<<scale<<" heterogeneous="<<heterogeneous<<" path="<<path
  <<" maximumIterations="<<controls.maximumIterations<<" minimumIterations="<<controls.minimumIterations
  <<" oldTolerance="<<oldTolerance<<" tolerance="<<controls.tolerance<<" relativeTolerance="<<controls.relativeTolerance
  <<" normalizationSmall="<<controls.normalizationSmall<<" singularSmall="<<controls.singularSmall
  <<" initialResidual="<<report.initialResidual<<" singular="<<report.singular
  <<" iterations="<<report.iterations<<" converged="<<report.converged<<" reportedResidual="<<report.finalResidual
  <<" independentL1="<<l1<<" norm="<<norm<<" allowedL1="<<trueBound<<" maxSolutionError="<<maxError
  <<" oldSolutionBound="<<oldSolutionBound<<" solutionBound="<<solutionBound
  <<" solutionReferenceScale="<<solutionReferenceScale<<" actualRelativeSolutionError="<<maxError/solutionReferenceScale
  <<" signedResidual="<<signedResidual<<" inventoryDefect="<<inventory<<" closureBound="<<closureBound
  <<" globalCallbacks="<<ctx.sums<<" interfaceCallbacks="<<ctx.interfaces<<'\n';
 std::cout<<std::flush;
 ck(residentFinite,"resident true residual nonfinite");
 ck(solutionFinite,"solution nonfinite");
 ck(report.converged&&!report.singular,"single-scalar PCG did not converge");
 if(path==1)ck(ctx.sums>0&&ctx.interfaces>0,"external interface/global reduction not exercised");
 ck(l1<=trueBound,"independent b-Ax-affine residual exceeds preset bound");
 ck(std::abs(signedResidual-inventory)<=closureBound,"interior/cyclic energy flux cancellation");
 ck(maxError<=solutionBound,"known solution error");

}
int main(){
 std::cout<<std::setprecision(17);
 int passed=0,failed=0;
 for(int n:{7,513})for(bool heterogeneous:{false,true})for(double scale:{1e-6,1.,1e6})for(int path:{0,1,2}){
  try{solve(n,scale,heterogeneous,path);++passed;}
  catch(const std::exception&e){
   ++failed;
   std::cerr<<"FAIL SPARSE n="<<n<<" scale="<<scale<<" heterogeneous="<<heterogeneous
    <<" path="<<path<<" reason="<<e.what()<<std::endl;
  }
 }
 std::cout<<"SPARSE SUMMARY passed="<<passed<<" failed="<<failed<<" total="<<passed+failed<<std::endl;
 return failed?1:0;
}
