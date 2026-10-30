#include "gpu/SparseConjugateGradient.H"
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <chrono>
using namespace ugkp::gpu;
struct Interface {int a,b,calls=0;};
void add(void* context,const double* x,double* y,int n){auto& c=*static_cast<Interface*>(context);assert(c.a<n&&c.b<n);y[c.a]-=.1*x[c.b];y[c.b]-=.1*x[c.a];++c.calls;}
int main(){
 SparseConjugateGradient solver;
 for(int n:{0,2,513,100003})for(int scenario=0;scenario<2;++scenario){
  std::vector<int> offsets(1,0),columns;std::vector<double> values,sum(n,3),rhs(n),exact(n),full(n,.5),compact(n,.5);
  Interface reference{0,n?n-1:0},candidate=reference;
  if(scenario&&n>100){reference={37,n-20};candidate=reference;}
  for(int i=0;i<n;++i)exact[i]=1+std::sin(.01*i);
  for(int i=0;i<n;++i){columns.push_back(i);values.push_back(3);rhs[i]=3*exact[i];
   for(int j:{i-1,i+1})if(j>=0&&j<n){columns.push_back(j);values.push_back(-1);rhs[i]-=exact[j];sum[i]-=1;}
   offsets.push_back(columns.size());}
  std::vector<int> cells;
  if(n){cells={reference.b,reference.a};if(reference.a==reference.b)cells.resize(1);
   sum[reference.a]-=.1;sum[reference.b]-=.1;rhs[reference.a]-=.1*exact[reference.b];rhs[reference.b]-=.1*exact[reference.a];}
  SparseMatrixView matrix{n,int(values.size()),offsets.data(),columns.data(),values.data(),sum.data()};
  SparseCommunication a;a.context=&reference;if(n)a.addInterface=add;
  auto b=a;b.context=&candidate;b.interfaceCells=cells.data();b.interfaceCellCount=cells.size();
  ConjugateGradientControls controls;controls.tolerance=1e-13;controls.maximumIterations=1000;
  auto start=std::chrono::steady_clock::now();auto ra=solver.solve(matrix,rhs.data(),full.data(),controls,a);
  double fullSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  start=std::chrono::steady_clock::now();auto rb=solver.solve(matrix,rhs.data(),compact.data(),controls,b);
  double compactSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  assert(ra.converged&&rb.converged&&ra.iterations==rb.iterations&&ra.initialResidual==rb.initialResidual&&ra.finalResidual==rb.finalResidual);
  assert(reference.calls==candidate.calls&&full==compact);
  double maximum=0;for(int i=0;i<n;++i)maximum=std::max(maximum,std::abs(compact[i]-exact[i]));assert(maximum<1e-9);
  std::cout<<"PASS n="<<n<<" interface="<<cells.size()<<" iterations="<<rb.iterations<<" max_error="<<maximum<<" full_seconds="<<fullSeconds<<" compact_seconds="<<compactSeconds<<'\n';
 }
}
