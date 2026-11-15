// Compile this fixture ALONE: it includes the implementation to exercise the
// real device preconditioner kernels without adding a production test API.
// Run separately against materialized fp32/fp64 trees; never link a second
// SparseConjugateGradient.o. No encoded/compensated FP32 arithmetic is used.
#include "GpuPrecisionTypes.H"
#include "SparseConjugateGradient.cu"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <chrono>
using R=GpuReal;
static ugkp::gpu::SparsePreconditioner chosen=ugkp::gpu::SparsePreconditioner::Jacobi;
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

static void embeddedGraph(int n, bool large){
 using namespace ugkp::gpu;
 Buffer<R>d(n),u(n-1),l(n-1),b(n),x(n);
 std::vector<int> own(n-1),nei(n-1);for(int i=0;i<n-1;++i){own[i]=i;nei[i]=i+1;}
 std::vector<double> hd(n),hu(n-1),hl(n-1),hb(n);
 SparseLduView matrix;matrix.rows=n;matrix.owner=own.data();matrix.neighbour=nei.data();
 matrix.diagonal=hd.data();matrix.upper=hu.data();matrix.lower=hl.data();matrix.source=hb.data();
 SparseDeviceLduValues values;values.diagonal=d.p;values.upper=u.p;values.lower=l.p;values.source=b.p;
 ConjugateGradientControls controls;controls.preconditioner=chosen;controls.tolerance=sizeof(R)==4?2e-6:1e-12;
 SparseConjugateGradient solver;ExecutionGraphSignature previous;
 cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;int phase=0;
 for(int connected: (large?std::vector<int>{1,1}:std::vector<int>{0,1,0})){
  // Equal entry count keeps CSR capacities/addresses unchanged. Self edges
  // cancel in the isolated SPD operator but retain the same storage footprint.
  matrix.faces=n-1;std::vector<R> diag(n,R(3+phase)),rhs(n),exact(n),upper(n-1,R(-.8));
  for(int f=0;f<n-1;++f){own[f]=f;nei[f]=connected?f+1:f;diag[own[f]]+=R(.8);diag[nei[f]]+=R(.8);}
  for(int i=0;i<n;++i){exact[i]=R(1+.1*std::sin(i*.1));rhs[i]=diag[i]*exact[i];}
  for(int f=0;f<n-1;++f){rhs[own[f]]-=R(.8)*exact[nei[f]];rhs[nei[f]]-=R(.8)*exact[own[f]];}
  d.put(diag);u.put(upper);l.put(upper);b.put(rhs);x.put(std::vector<R>(n,R(0)));
  solver.configureResidentLocalLdu(matrix,values,controls);
  auto current=solver.residentExecutionSignature();
  if(!large && phase && chosen==SparsePreconditioner::Block8)require(!(current==previous),"topology regrouping signature stale");
  bool rebuild=!executable || !(current==previous) || (!large && phase);
  if(rebuild){
   if(executable)ck(cudaGraphExecDestroy(executable));if(graph)ck(cudaGraphDestroy(graph));
   ck(cudaGraphCreate(&graph,0));solver.appendResidentLocalLdu(graph,nullptr,x.p);
   ck(cudaGraphInstantiate(&executable,graph,0));
  }
  ck(cudaGraphLaunch(executable,0));ck(cudaDeviceSynchronize());
  auto report=solver.collectResidentLocalReport();auto answer=x.get();
  double maxError=0;for(int i=0;i<n;++i)maxError=std::max(maxError,std::abs(double(answer[i]-exact[i])));
  std::printf("EMBEDDED bits=%d B=%d choice=%d n=%d phase=%d rebuild=%d converged=%d singular=%d residual=%.12g maxError=%.12g tail=%.12g\n",int(8*sizeof(R)),workPlan().sparseBlockThreads,int(chosen),n,phase,int(rebuild),int(report.converged),int(report.singular),double(report.finalResidual),maxError,double(answer.back()));
  ++phase;
  for(int i=0;i<n;++i)require(std::abs(double(answer[i]-exact[i]))<10*tol(),"same-size topology regrouping solve");
  require(!report.singular,"topology regrouping singular");previous=current;
 }
 ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
 std::puts("PASS actual appendResidentLocalLdu outer graph");
}
int main(int argc,char**argv){try{
 ugkp::gpu::workPlan().sparseBlockThreads=argc>1?std::atoi(argv[1]):128;
 chosen=argc>2&&std::atoi(argv[2])==8?ugkp::gpu::SparsePreconditioner::Block8:ugkp::gpu::SparsePreconditioner::Jacobi;
 embeddedGraph(193,false);embeddedGraph(131201,true);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
