



#include "../../common/GpuPrecisionTypes.H"
#include "../../common/gpu/SparseConjugateGradient.cu"
#include <cstdio>
#include <cstdlib>
#include <numeric>
using R=GpuReal;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
template<class T> struct Buffer{
 T* p=nullptr;int n;
 explicit Buffer(int count):n(count){if(n)ck(cudaMalloc(&p,n*sizeof(T)));}
 ~Buffer(){if(p)cudaFree(p);}
 void put(const std::vector<T>& x){require(int(x.size())==n,"buffer extent");if(n)ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> get(){std::vector<T>x(n);if(n)ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
static double tol(){return sizeof(R)==4?3e-4:3e-10;}



static void residentReplay(int n){
 using namespace ugkp::gpu;
 std::vector<int> owner(n-1),neighbour(n-1);

 for(int f=0;f<n-1;++f){owner[f]=(5*f)%n;neighbour[f]=(5*(f+1))%n;}
 Buffer<R>d(n),u(n-1),l(n-1),b(n),x(n);
 SparseLduView topology;topology.rows=n;topology.faces=n-1;
 topology.owner=owner.data();topology.neighbour=neighbour.data();
 std::vector<double>hd(n),hu(n-1),hl(n-1),hb(n);
 topology.diagonal=hd.data();topology.upper=hu.data();topology.lower=hl.data();topology.source=hb.data();
 SparseDeviceLduValues values;values.diagonal=d.p;values.upper=u.p;values.lower=l.p;values.source=b.p;
 ConjugateGradientControls controls;controls.maximumIterations=4000;
 controls.tolerance=R(sizeof(R)==4?2e-6:1e-12);
 SparseConjugateGradient solver;

 solver.configureResidentLocalLdu(topology,values,controls);
 const auto signature=solver.residentExecutionSignature();
 for(int phase=0;phase<3;++phase){
  std::vector<R>diag(n),upper(n-1),rhs(n),exact(n),zero(n,0);
  for(int i=0;i<n;++i){diag[i]=R(phase==0?.5:phase==1?2.:.1);exact[i]=R(1+.2*std::sin(.17*i+phase));}
  for(int f=0;f<n-1;++f){const R k=R((1+.1*(f%11))*(phase+1));
   upper[f]=-k;diag[owner[f]]+=k;diag[neighbour[f]]+=k;}
  for(int i=0;i<n;++i)rhs[i]=diag[i]*exact[i];
  for(int f=0;f<n-1;++f){rhs[owner[f]]+=upper[f]*exact[neighbour[f]];rhs[neighbour[f]]+=upper[f]*exact[owner[f]];}
  d.put(diag);u.put(upper);l.put(upper);b.put(rhs);x.put(zero);
  solver.enqueueResidentLocalLdu(x.p);
  const auto report=solver.collectResidentLocalReport();auto result=x.get();
  std::fprintf(stderr,"REPORT bits=%d n=%d phase=%d iterations=%d converged=%d singular=%d initial=%.9g final=%.9g\n",int(8*sizeof(R)),n,phase,report.iterations,int(report.converged),int(report.singular),double(report.initialResidual),double(report.finalResidual));

  require(signature==solver.residentExecutionSignature(),"coefficient update unexpectedly invalidated topology graph");
  double maxError=0,residual=0,scale=0;
  for(int i=0;i<n;++i){
   double ax=double(diag[i])*result[i];
   for(int f=0;f<n-1;++f){
    if(owner[f]==i)ax+=double(upper[f])*result[neighbour[f]];
    if(neighbour[f]==i)ax+=double(upper[f])*result[owner[f]];
   }
   residual+=std::abs(double(rhs[i])-ax);scale+=std::abs(double(rhs[i]));
   maxError=std::max(maxError,std::abs(double(result[i])-exact[i]));
  }


  std::printf("bits=%d n=%d update=%d iterations=%d trueResidual=%.9g maxError=%.9g\n",
   int(8*sizeof(R)),n,phase,report.iterations,residual/std::max(scale,1e-30),maxError);
  require(report.converged&&!report.singular,"resident PCG failed");
  require(residual/std::max(scale,1e-30)<tol(),"independent b-Ax check failed");
  require(maxError<10*tol(),"manufactured solution mismatch");
 }
}
int main(int argc,char**argv){
 try{
  const int block=argc>1?std::atoi(argv[1]):128;
  require(block==32||block==64||block==96||block==128||block==256,"fixture widths: 32/64/96/128/256");
  ugkp::gpu::workPlan().sparseBlockThreads=block;

  for(int n:{7,129,513})residentReplay(n);
  std::puts("PASS Jacobi PCG coefficient-update graph replay");
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
