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

// Exercise the resident PCG API with fresh coefficients and graph replay.
// Independent b-Ax and manufactured-solution checks are separate from the
// strict production convergence flag, which is printed without concealment.
static void residentReplay(int n){
 using namespace ugkp::gpu;
 std::vector<int> owner(n-1),neighbour(n-1);
 // A permutation deliberately destroys contiguous spatial numbering.
 for(int f=0;f<n-1;++f){owner[f]=(5*f)%n;neighbour[f]=(5*(f+1))%n;}
 Buffer<R>d(n),u(n-1),l(n-1),b(n),x(n);
 SparseLduView topology;topology.rows=n;topology.faces=n-1;
 topology.owner=owner.data();topology.neighbour=neighbour.data();
 std::vector<double>hd(n),hu(n-1),hl(n-1),hb(n);
 topology.diagonal=hd.data();topology.upper=hu.data();topology.lower=hl.data();topology.source=hb.data();
 SparseDeviceLduValues values;values.diagonal=d.p;values.upper=u.p;values.lower=l.p;values.source=b.p;
 ConjugateGradientControls controls;controls.preconditioner=chosen;controls.maximumIterations=4000;
 controls.tolerance=R(sizeof(R)==4?2e-6:1e-12);
 SparseConjugateGradient solver;
 auto topologyStart=std::chrono::steady_clock::now();
 // No configure between phases: updates must be observed by graph replay.
 solver.configureResidentLocalLdu(topology,values,controls);
 std::printf("SETUP bits=%d n=%d ms=%.9g\n",int(8*sizeof(R)),n,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-topologyStart).count());
 const auto signature=solver.residentExecutionSignature();
 for(int phase=0;phase<4;++phase){
  std::vector<R>diag(n),upper(n-1),rhs(n),exact(n),zero(n,0);
  for(int i=0;i<n;++i){diag[i]=R(phase==3?1000.:phase==0?.5:phase==1?2.:.1);exact[i]=R(1+.2*std::sin(.17*i+phase));}
  for(int f=0;f<n-1;++f){const R k=R((1+.1*(f%11))*(phase+1));
   upper[f]=-k;diag[owner[f]]+=k;diag[neighbour[f]]+=k;}
  for(int i=0;i<n;++i)rhs[i]=diag[i]*exact[i];
  for(int f=0;f<n-1;++f){rhs[owner[f]]+=upper[f]*exact[neighbour[f]];rhs[neighbour[f]]+=upper[f]*exact[owner[f]];}
  d.put(diag);u.put(upper);l.put(upper);b.put(rhs);x.put(zero);
  for(int warm=0;warm<2;++warm){ck(cudaMemsetAsync(x.p,0,n*sizeof(R)));solver.enqueueResidentLocalLdu(x.p);}
  ck(cudaDeviceSynchronize());
  cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
  ck(cudaEventRecord(begin));constexpr int repeats=10;
  for(int rep=0;rep<repeats;++rep){ck(cudaMemsetAsync(x.p,0,n*sizeof(R)));solver.enqueueResidentLocalLdu(x.p);}
  ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float milliseconds=0;
  ck(cudaEventElapsedTime(&milliseconds,begin,end));ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
  std::printf("PERF bits=%d n=%d phase=%d repeats=%d ms_per_solve=%.9g\n",int(8*sizeof(R)),n,phase,repeats,double(milliseconds)/repeats);

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
  require(!report.singular,"singular PCG");
  require(residual/std::max(scale,1e-30)<tol(),"independent b-Ax check failed");
  require(maxError<10*tol(),"manufactured solution mismatch");
 }
}

static void rawAndHostInterfaces(){
 using namespace ugkp::gpu;
 constexpr int n=193;std::vector<int> off(1,0),col,owner(n-1),neighbour(n-1);
 std::vector<double> val,rhs(n),exact(n),rowSum(n),diag(n),upper(n-1,-.8),lower=upper;
 for(int i=0;i<n;++i)exact[i]=1+.1*std::sin(i*.1);
 for(int i=0;i<n;++i){
  double d=2.+(i>0?.8:0)+(i+1<n?.8:0);diag[i]=d;rowSum[i]=2;
  col.push_back(i);val.push_back(d);
  if(i){col.push_back(i-1);val.push_back(-.8);}
  if(i+1<n){col.push_back(i+1);val.push_back(-.8);}
  off.push_back(col.size());
  rhs[i]=d*exact[i]-(i?.8*exact[i-1]:0)-(i+1<n?.8*exact[i+1]:0);
  if(i+1<n){owner[i]=i;neighbour[i]=i+1;}
 }
 ConjugateGradientControls controls;controls.preconditioner=chosen;controls.tolerance=sizeof(R)==4?2e-6:1e-12;
 SparseConjugateGradient solver;
 for(int route=0;route<3;++route){
  std::vector<double>x(n,0);ConjugateGradientReport report;
  SparseCommunication comm;
  if(route==2){
   comm.globalSum=[](void*,R*,int){};
   comm.addInterface=[](void*,const R*,R*,int){};
  }
  if(route==0){SparseMatrixView m{n,int(col.size()),off.data(),col.data(),val.data(),rowSum.data()};report=solver.solve(m,rhs.data(),x.data(),controls);}
  else {SparseLduView m;m.rows=n;m.faces=n-1;m.owner=owner.data();m.neighbour=neighbour.data();m.diagonal=diag.data();m.upper=upper.data();m.lower=lower.data();m.source=rhs.data();
   report=solver.solveLdu(m,x.data(),controls,comm);}
  double error=0;for(int i=0;i<n;++i)error=std::max(error,std::abs(x[i]-exact[i]));
  require(!report.singular&&error<10*tol(),"raw/host/interface manufactured solution");
  std::printf("API route=%d bits=%d iterations=%d converged=%d error=%.9g\n",route,int(sizeof(R)*8),report.iterations,int(report.converged),error);
 }
}


static void largeUpdateCoverage(){
 constexpr int n=131201;
 Buffer<R>x(n),r(n),p(n),ap(n),recurrence(8);
 x.put(std::vector<R>(n,R(1)));r.put(std::vector<R>(n,R(3)));
 p.put(std::vector<R>(n,R(2)));ap.put(std::vector<R>(n,R(4)));
 std::vector<R> scalars(8,R(0));scalars[3]=R(.5);recurrence.put(scalars);
 const int block=ugkp::gpu::workPlan().sparseBlockThreads;
 ugkp::gpu::update<<<(n+block-1)/block,block>>>(n,recurrence.p,p.p,ap.p,x.p,r.p);
 auto xv=x.get(),rv=r.get();
 for(int i=0;i<n;++i)require(xv[i]==R(2)&&rv[i]==R(1),"uncapped block8 update tail");
 std::puts("PASS update coverage beyond capped reduction grid");
}


static void topologyRegrouping(){
 using namespace ugkp::gpu;constexpr int n=193;
 Buffer<R>d(n),u(n-1),l(n-1),b(n),x(n);
 std::vector<int> own(n-1),nei(n-1);for(int i=0;i<n-1;++i){own[i]=i;nei[i]=i+1;}
 std::vector<double> hd(n),hu(n-1),hl(n-1),hb(n);
 SparseLduView matrix;matrix.rows=n;matrix.owner=own.data();matrix.neighbour=nei.data();
 matrix.diagonal=hd.data();matrix.upper=hu.data();matrix.lower=hl.data();matrix.source=hb.data();
 SparseDeviceLduValues values;values.diagonal=d.p;values.upper=u.p;values.lower=l.p;values.source=b.p;
 ConjugateGradientControls controls;controls.preconditioner=chosen;controls.tolerance=sizeof(R)==4?2e-6:1e-12;
 SparseConjugateGradient solver;ExecutionGraphSignature previous;
 for(int connected:{0,1,0}){
  // Equal entry count keeps CSR capacities/addresses unchanged. Self edges
  // cancel in the isolated SPD operator but retain the same storage footprint.
  matrix.faces=n-1;std::vector<R> diag(n,R(3)),rhs(n),exact(n),upper(n-1,R(-.8));
  for(int f=0;f<n-1;++f){own[f]=f;nei[f]=connected?f+1:f;diag[own[f]]+=R(.8);diag[nei[f]]+=R(.8);}
  for(int i=0;i<n;++i){exact[i]=R(1+.1*std::sin(i*.1));rhs[i]=diag[i]*exact[i];}
  for(int f=0;f<n-1;++f){rhs[own[f]]-=R(.8)*exact[nei[f]];rhs[nei[f]]-=R(.8)*exact[own[f]];}
  d.put(diag);u.put(upper);l.put(upper);b.put(rhs);x.put(std::vector<R>(n,R(0)));
  solver.configureResidentLocalLdu(matrix,values,controls);
  auto current=solver.residentExecutionSignature();
  if(connected && chosen==SparsePreconditioner::Block8)require(!(current==previous),"topology regrouping signature stale");
  solver.enqueueResidentLocalLdu(x.p);auto report=solver.collectResidentLocalReport();auto answer=x.get();
  for(int i=0;i<n;++i)require(std::abs(double(answer[i]-exact[i]))<10*tol(),"same-size topology regrouping solve");
  require(!report.singular,"topology regrouping singular");previous=current;
 }
 std::puts("PASS same-n isolate/connected/isolate topology regrouping");
}

int main(int argc,char**argv){
 try{
  const int block=argc>1?std::atoi(argv[1]):128;
  require(block==32||block==64||block==96||block==128||block==256,"fixture widths: 32/64/96/128/256");
  ugkp::gpu::workPlan().sparseBlockThreads=block;
  chosen=argc>2&&std::atoi(argv[2])==8?ugkp::gpu::SparsePreconditioner::Block8:ugkp::gpu::SparsePreconditioner::Jacobi;
  largeUpdateCoverage();
  rawAndHostInterfaces();
  topologyRegrouping();

  for(int n:{7,129,513,4097})residentReplay(n);
  std::puts("PASS sparse API manufactured-solution and coefficient-update graph replay");
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
