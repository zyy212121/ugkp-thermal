// Real production PCG, both native precisions. Compile ALONE against a materialized tree.
#include "../../common/GpuPrecisionTypes.H"
#include "../../common/gpu/SparseConjugateGradient.cu"
#include <cstdio>
#include <cstdlib>
#include <limits>
using R=GpuReal;
using namespace ugkp::gpu;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> struct Buffer{T* p=nullptr;int n;Buffer(int count):n(count){if(n)ck(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)cudaFree(p);}void put(const std::vector<T>& a){need(int(a.size())==n,"extent");if(n)ck(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T> get(){std::vector<T>a(n);if(n)ck(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}};
static R tree(std::vector<R> a){for(int active=int(a.size());active>1;active=(active+1)/2)for(int i=0;i<active/2;++i)a[i]+=a[i+(active+1)/2];return a[0];}
static R norm(const std::vector<R>& terms,int B){
 int blocks=std::max(1,std::min(1024,(int(terms.size())+B-1)/B));std::vector<R> part(blocks);
 for(int block=0;block<blocks;++block){std::vector<R> lane(B,0);for(int t=0;t<B;++t)for(int i=block*B+t;i<int(terms.size());i+=blocks*B)lane[t]+=terms[i];part[block]=tree(lane);}
 std::vector<R> lane(B,0);for(int t=0;t<B;++t)for(int i=t;i<blocks;i+=B)lane[t]+=part[i];return tree(lane);
}
static void globalIdentity(void*,R*,int){}
// Decision-unit coverage is separate from the real-matrix solves below.
__global__ void confirmationDecisionChecks(int* ok){
 if(threadIdx.x)return;LocalSolveParameters p{};p.norm=1;p.initial=1;p.controls.tolerance=R(.01);p.controls.maximumIterations=100;
 R q[12]={};q[4]=R(.1);q[6]=54;q[8]=1;
 bool a=!confirmTrueResidual(&p,q)&&q[6]==54&&q[9]==1&&q[10]==1;
 q[8]=0;bool b=confirmTrueResidual(&p,q)&&q[11]!=0&&q[6]==54;
 q[6]=100;q[8]=1;bool c=confirmTrueResidual(&p,q)&&q[11]==0;
 q[4]=R(.001);q[6]=55;bool d=confirmTrueResidual(&p,q)&&q[11]==0;
 *ok=a&&b&&c&&d;
}
static void exercise(int n,int route,R tolerance,int maxIter){
 const int B=workPlan().sparseBlockThreads;
 std::vector<int> own(n-1),nei(n-1);for(int f=0;f<n-1;++f){own[f]=f;nei[f]=f+1;}
 int bo=0,bn=-1;std::vector<double> hd(n),hu(n-1),hb(n),bd(1,0),bs(1,.125),bj(1,2.0);
 SparseLduView m;m.rows=n;m.faces=n-1;m.owner=own.data();m.neighbour=nei.data();m.diagonal=hd.data();m.upper=hu.data();m.lower=hu.data();m.source=hb.data();m.boundaryFaces=1;m.boundaryOwner=&bo;m.boundaryNeighbour=&bn;m.boundaryDiagonal=bd.data();m.boundarySource=bs.data();m.boundaryJump=bj.data();
 Buffer<R> d(n),u(n-1),b(n),x(n),dd(1),ss(1),jj(1);dd.put({R(0)});ss.put({R(.125)});jj.put({R(2)});
 SparseDeviceLduValues v;v.diagonal=d.p;v.upper=u.p;v.lower=u.p;v.source=b.p;v.boundaryDiagonal=dd.p;v.boundarySource=ss.p;v.boundaryJump=jj.p;
 ConjugateGradientControls c;c.maximumIterations=maxIter;c.tolerance=tolerance;
 SparseConjugateGradient solver;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;ExecutionGraphSignature signature;
 if(route>=2){solver.configureResidentLocalLdu(m,v,c);signature=solver.residentExecutionSignature();if(route==3){ck(cudaGraphCreate(&graph,0));solver.appendResidentLocalLdu(graph,nullptr,x.p);ck(cudaGraphInstantiate(&executable,graph,0));}}
 for(int phase=0;phase<2;++phase){
  std::vector<R> diag(n,R(.1)),upper(n-1),source(n),rhs(n),exact(n),zero(n,0),affine(n,0);affine[0]=R(.25);
  for(int i=0;i<n;++i)exact[i]=R(1+.2*std::sin(.17*i+phase));
  for(int f=0;f<n-1;++f){R k=R((1+.1*(f%11))*(phase+2));upper[f]=-k;diag[f]+=k;diag[f+1]+=k;}
  for(int i=0;i<n;++i){R ax=diag[i]*exact[i];if(i)ax+=upper[i-1]*exact[i-1];if(i+1<n)ax+=upper[i]*exact[i+1];rhs[i]=ax+affine[i];source[i]=rhs[i]-(i==0?R(.125):R(0));hd[i]=diag[i];hb[i]=source[i];}
  for(int f=0;f<n-1;++f)hu[f]=upper[f];d.put(diag);u.put(upper);b.put(source);x.put(zero);
  ConjugateGradientReport report;std::vector<R> answer;
  if(route<2){std::vector<double> result(n,0);SparseCommunication comm;if(route==1)comm.globalSum=globalIdentity;report=solver.solveLdu(m,result.data(),c,comm);answer.assign(result.begin(),result.end());}
  else {if(route==2)solver.enqueueResidentLocalLdu(x.p);else {ck(cudaGraphLaunch(executable,0));ck(cudaDeviceSynchronize());}report=solver.collectResidentLocalReport();answer=x.get();need(signature==solver.residentExecutionSignature(),"coefficient update changed captured topology");}
  std::vector<R> terms(n),normalization(n);double trueDouble=0,scale=0,maxError=0;
  for(int i=0;i<n;++i){R ax=diag[i]*answer[i];if(i)ax+=upper[i-1]*answer[i-1];if(i+1<n)ax+=upper[i]*answer[i+1];ax+=affine[i];R assembled=source[i]+(i==0?R(.125):R(0));terms[i]=std::abs(assembled-ax);normalization[i]=std::abs(affine[i])+std::abs(assembled);double ad=double(diag[i])*answer[i]+double(affine[i]);if(i)ad+=double(upper[i-1])*answer[i-1];if(i+1<n)ad+=double(upper[i])*answer[i+1];trueDouble+=std::abs(double(assembled)-ad);scale+=std::abs(double(assembled));maxError=std::max(maxError,std::abs(double(answer[i])-exact[i]));}
  const R nativeTrue=norm(terms,B)/(norm(normalization,B)+c.normalizationSmall);
  std::printf("TRUE_RESIDUAL bits=%d B=%d n=%d route=%d phase=%d max=%d iterations=%d replacements=%d converged=%d stagnant=%d budget=%d singular=%d target=%.12g report=%.12g native=%.12g doubleRelative=%.12g maxError=%.12g\n",int(8*sizeof(R)),B,n,route,phase,maxIter,report.iterations,report.residualReplacements,int(report.converged),int(report.stagnated),int(report.budgetExhausted),int(report.singular),double(report.targetResidual),double(report.finalResidual),double(nativeTrue),trueDouble/scale,maxError);
  need(std::isfinite(double(report.finalResidual)),"nonfinite true residual");need(report.iterations<=maxIter,"iteration budget reset by restart");need(report.targetResidual==tolerance,"target changed");
  need(std::abs(double(report.finalResidual)-nativeTrue)<=64*std::numeric_limits<R>::epsilon()*std::max(double(nativeTrue),1e-30),"report differs from native b-Ax-affine");
  need(!report.converged||report.finalResidual<report.targetResidual,"false convergence");need(report.converged||report.budgetExhausted||report.stagnated||report.singular,"unclassified early stop");need(!report.budgetExhausted||report.iterations==maxIter,"false budget exhaustion");
 }
 if(executable)ck(cudaGraphExecDestroy(executable));if(graph)ck(cudaGraphDestroy(graph));
}
int main(int argc,char**argv){try{
 workPlan().sparseBlockThreads=argc>1?std::atoi(argv[1]):128;Buffer<int> ok(1);confirmationDecisionChecks<<<1,1>>>(ok.p);need(ok.get()[0],"confirmation decision transition");
 for(int n:{7,129,513})for(int route=0;route<4;++route)exercise(n,route,R(sizeof(R)==4?2e-6:1e-12),4000);
 for(int route=0;route<4;++route){exercise(129,route,R(1e-10),1000);exercise(129,route,R(1e-12),1);}
 std::puts("PASS true-residual closure and truthful stopping reasons; strict false flags retained in records");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
