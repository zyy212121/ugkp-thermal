



#include "GpuPrecisionTypes.H"
#include "SparseConjugateGradient.cu"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <chrono>
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
 auto topologyStart=std::chrono::steady_clock::now();

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
 ConjugateGradientControls controls;controls.tolerance=sizeof(R)==4?2e-6:1e-12;
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
 ConjugateGradientControls controls;controls.tolerance=sizeof(R)==4?2e-6:1e-12;
 SparseConjugateGradient solver;ExecutionGraphSignature previous;
 for(int connected:{0,1,0}){


  matrix.faces=n-1;std::vector<R> diag(n,R(3)),rhs(n),exact(n),upper(n-1,R(-.8));
  for(int f=0;f<n-1;++f){own[f]=f;nei[f]=connected?f+1:f;diag[own[f]]+=R(.8);diag[nei[f]]+=R(.8);}
  for(int i=0;i<n;++i){exact[i]=R(1+.1*std::sin(i*.1));rhs[i]=diag[i]*exact[i];}
  for(int f=0;f<n-1;++f){rhs[own[f]]-=R(.8)*exact[nei[f]];rhs[nei[f]]-=R(.8)*exact[own[f]];}
  d.put(diag);u.put(upper);l.put(upper);b.put(rhs);x.put(std::vector<R>(n,R(0)));
  solver.configureResidentLocalLdu(matrix,values,controls);
  auto current=solver.residentExecutionSignature();
  if(connected)require(!(current==previous),"topology regrouping signature stale");
  solver.enqueueResidentLocalLdu(x.p);auto report=solver.collectResidentLocalReport();auto answer=x.get();
  for(int i=0;i<n;++i)require(std::abs(double(answer[i]-exact[i]))<10*tol(),"same-size topology regrouping solve");
  require(!report.singular,"topology regrouping singular");previous=current;
 }
 std::puts("PASS same-n isolate/connected/isolate topology regrouping");
}


static void boundaryCoverage(){
 using namespace ugkp::gpu;
 const int n=4;
 const int own[]={0,1,2},nei[]={1,2,3};
 const int bo[]={0,0,3,2},bn[]={-1,3,0,-2};
 const double diag[]={4,4,4,4},upper[]={-.5,-.5,-.5};
 const double bd[]={.3,.2,.2,.4},bs[]={.7,.2,.2,.4},jump[]={0,.15,-.15,.25};
 const double exact[]={1.1,.9,1.3,.8};
 double rhs[4];
 for(int i=0;i<n;++i)rhs[i]=diag[i]*exact[i];
 for(int f=0;f<3;++f){rhs[own[f]]+=upper[f]*exact[nei[f]];rhs[nei[f]]+=upper[f]*exact[own[f]];}
 for(int f=0;f<4;++f){rhs[bo[f]]+=bd[f]*exact[bo[f]]+bs[f]*jump[f];if(bn[f]>=0)rhs[bo[f]]-=bs[f]*exact[bn[f]];else if(bn[f]==-1)rhs[bo[f]]-=bs[f];else rhs[bo[f]]-=bs[f]*exact[2];}
 SparseLduView m;m.rows=n;m.faces=3;m.boundaryFaces=4;m.owner=own;m.neighbour=nei;m.boundaryOwner=bo;m.boundaryNeighbour=bn;m.diagonal=diag;m.upper=upper;m.lower=upper;m.source=rhs;m.boundaryDiagonal=bd;m.boundarySource=bs;m.boundaryJump=jump;
 SparseCommunication comm;comm.globalSum=[](void*,R*,int){};
 comm.addInterface=[](void*,const R* x,R* ax,int){ax[2]-=R(.4)*x[2];};
 ConjugateGradientControls c;c.maximumIterations=1000;c.tolerance=sizeof(R)==4?2e-6:1e-12;
 SparseConjugateGradient solver;double x[4]={};auto report=solver.solveLdu(m,x,c,comm);
 double residual=0,scale=0,error=0;
 for(int i=0;i<n;++i){double ax=diag[i]*x[i],b=rhs[i];for(int f=0;f<3;++f){if(own[f]==i)ax+=upper[f]*x[nei[f]];if(nei[f]==i)ax+=upper[f]*x[own[f]];}for(int f=0;f<4;++f)if(bo[f]==i){ax+=bd[f]*x[i]+bs[f]*jump[f];if(bn[f]>=0)ax-=bs[f]*x[bn[f]];else if(bn[f]==-1)b+=bs[f];else ax-=bs[f]*x[2];}residual+=std::abs(b-ax);scale+=std::abs(b);error=std::max(error,std::abs(x[i]-exact[i]));}
 std::printf("BOUNDARY bits=%d residual=%.12g report=%.12g error=%.12g converged=%d\n",int(sizeof(R)*8),residual/scale,double(report.finalResidual),error,int(report.converged));
 require(report.converged&&!report.singular&&residual/scale<tol()&&error<10*tol(),"boundary independent residual");
}
static void borrowedAddressRebind(){
 using namespace ugkp::gpu;
 const int own[]={0,1},nei[]={1,2};double hd[]={3,4,3},hu[]={-1,-1},hb[]={1,4,7};
 SparseLduView m;m.rows=3;m.faces=2;m.owner=own;m.neighbour=nei;m.diagonal=hd;m.upper=hu;m.lower=hu;m.source=hb;
 Buffer<R>d1(3),u1(2),b1(3),d2(3),u2(2),b2(3),x(3);
 d1.put({3,4,3});u1.put({-1,-1});b1.put({1,4,7});d2.put({6,8,6});u2.put({-2,-2});b2.put({2,8,14});
 SparseConjugateGradient solver;ConjugateGradientControls c;c.tolerance=sizeof(R)==4?2e-6:1e-12;
 for(int phase=0;phase<3;++phase){SparseDeviceLduValues v;v.diagonal=phase==1?d2.p:d1.p;v.upper=v.lower=phase==1?u2.p:u1.p;v.source=phase==1?b2.p:b1.p;x.put({0,0,0});solver.configureResidentLocalLdu(m,v,c);solver.enqueueResidentLocalLdu(x.p);auto report=solver.collectResidentLocalReport();auto result=x.get();require(report.converged,"address rebind convergence");for(int i=0;i<3;++i)require(std::abs(double(result[i])-(i+1))<10*tol(),"borrowed coefficient address rebind stale");}
 std::puts("PASS borrowed coefficient address rebind");
}


static void configuredIsolationAndWidth(){
 using namespace ugkp::gpu;const int n=200;const int saved=workPlan().sparseBlockThreads;
 std::vector<int> own(n-1),nei(n-1),other(n-1);for(int i=0;i<n-1;++i){own[i]=i;nei[i]=i+1;other[i]=i;}
 std::vector<double> hd(n,4),hu(n-1,-1),hb(n,2);
 Buffer<R>d(n),u(n-1),b(n),db(n),ub(n-1),bb(n),x(n),res(n);
 d.put(std::vector<R>(n,4));u.put(std::vector<R>(n-1,-1));std::vector<R> rhs(n,2);rhs.front()=rhs.back()=3;b.put(rhs);
 db.put(std::vector<R>(n,8));ub.put(std::vector<R>(n-1,-.25));bb.put(std::vector<R>(n,3));
 SparseLduView a;a.rows=n;a.faces=n-1;a.owner=own.data();a.neighbour=nei.data();a.diagonal=hd.data();a.upper=a.lower=hu.data();a.source=hb.data();auto otherMatrix=a;otherMatrix.neighbour=other.data();
 SparseDeviceLduValues av;av.diagonal=d.p;av.upper=av.lower=u.p;av.source=b.p;auto bv=av;bv.diagonal=db.p;bv.upper=bv.lower=ub.p;bv.source=bb.p;
 ConjugateGradientControls c;c.tolerance=sizeof(R)==4?2e-6:1e-12;
 for(int route=0;route<2;++route)for(int width:{256,96}){
  workPlan().sparseBlockThreads=width==256?128:256;SparseConjugateGradient solver;solver.configureResidentLocalLdu(a,av,c);
  x.put(std::vector<R>(n,0));solver.residualResidentLdu(otherMatrix,bv,x.p,res.p);ck(cudaDeviceSynchronize());
  workPlan().sparseBlockThreads=width;
  cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
  if(route){ck(cudaGraphCreate(&graph,0));solver.appendResidentLocalLdu(graph,nullptr,x.p);ck(cudaGraphInstantiate(&exec,graph,0));ck(cudaGraphLaunch(exec,0));ck(cudaDeviceSynchronize());}
  else solver.enqueueResidentLocalLdu(x.p);
  auto report=solver.collectResidentLocalReport();auto answer=x.get();double residual=0,scale=0;
  for(int i=0;i<n;++i){double ax=4*double(answer[i])-(i?double(answer[i-1]):0)-(i+1<n?double(answer[i+1]):0);residual+=std::abs(double(rhs[i])-ax);scale+=rhs[i];require(std::abs(double(answer[i])-1)<10*tol(),"configured A contaminated or dynamic width lost rows");}
  require(report.converged&&residual/scale<tol(),"configured A independent residual");if(exec)ck(cudaGraphExecDestroy(exec));if(graph)ck(cudaGraphDestroy(graph));
  std::printf("PASS configured isolation route=%d width=%d residual=%.12g\n",route,width,residual/scale);
 }
 workPlan().sparseBlockThreads=saved;
}
static void extremeScale(){
 using namespace ugkp::gpu;
 for(int exponent:std::vector<int>{sizeof(R)==4?-130:-1030,sizeof(R)==4?120:1000}){
  R k=R(std::ldexp(1.,exponent));Buffer<int>offsets(3),columns(2),valid(1);Buffer<R>values(2),rhs(2),out(2),smooth(2);
  offsets.put({0,1,2});columns.put({0,1});valid.put({1});values.put({k,k});rhs.put({k,k});
  MatrixOperator matrix{};matrix.offsets=offsets.p;matrix.columns=columns.p;matrix.values=values.p;
  vcSmoother<R><<<1,128,128*sizeof(R)>>>(2,matrix,smooth.p,valid.p);ck(cudaDeviceSynchronize());auto good=valid.get();
  if(exponent<0)require(good[0]==0,"overflowed smoother must select existing fallback");
  precondition<<<1,128>>>(2,matrix,rhs.p,out.p,nullptr);auto result=out.get();
  for(R x:result)require(std::isfinite(double(x))&&std::abs(double(x)-1)<tol(),"extreme-scale diagonal fallback");
  std::printf("PASS scale bits=%d exponent=%d smootherValid=%d\n",int(sizeof(R)*8),exponent,good[0]);
 }
}

int main(int argc,char**argv){
 try{
  const int block=argc>1?std::atoi(argv[1]):128;
  require(block==32||block==64||block==96||block==128||block==256,"fixture widths: 32/64/96/128/256");
  ugkp::gpu::workPlan().sparseBlockThreads=block;
  configuredIsolationAndWidth();
  extremeScale();
  boundaryCoverage();
  borrowedAddressRebind();
  largeUpdateCoverage();
  rawAndHostInterfaces();
  topologyRegrouping();

  for(int n:{7,129,513,4097})residentReplay(n);
  std::puts("PASS sparse API manufactured-solution and coefficient-update graph replay");
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
