// The oracle links only the original 88bc5434 long-double batch function.
#include "materials/MaterialTransport.H"
#include "materials/MaterialInventoryMath.H"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
namespace chmt {
bool baselineApplyMaterialPacketBatch(const std::vector<SolidQ>&,const std::vector<ExchangePacket>&,
    std::vector<SolidQ>&,std::string&,bool*);
}
using namespace chmt;
void require(bool ok,const std::string&message){if(!ok){std::fprintf(stderr,"%s\n",message.c_str());std::abort();}}
std::uint64_t bits(double x){std::uint64_t u;std::memcpy(&u,&x,sizeof u);return u;}
void same(const SolidQ&a,const SolidQ&b,const std::string&name){
 auto equal=[&](double x,double y){require(bits(x)==bits(y),name+" differs: "+std::to_string(x)+" vs "+std::to_string(y));};
 for(int c=0;c<Nc;++c)equal(a.condensed[c],b.condensed[c]);
 for(int s=0;s<Ns;++s)equal(a.pore[s],b.pore[s]);
 for(int j=0;j<Nr;++j)equal(a.progress[j],b.progress[j]);equal(a.energy,b.energy);equal(a.porosity,b.porosity);
}
SolidQ base(int cell){SolidQ q;for(int c=0;c<Nc;++c)q.condensed[c]=100.+cell+c;
 for(int s=0;s<Ns;++s)q.pore[s]=50.+cell+s;for(int j=0;j<Nr;++j)q.progress[j]=.03125*(cell+j);
 q.energy=12345.+cell;q.porosity=.2+.01*cell;return q;}
ExchangePacket delta(int cell,int component,double value,bool incoming=true){ExchangePacket p;
 p.kind=incoming?ExchangeKind::ParticleWall:ExchangeKind::GasSolid;p.solidCell=cell;
 const double encoded=incoming?value:-value;
 if(component<Nc)p.condensed[component]=encoded;
 else if(component<Nc+Ns)p.pore[component-Nc]=encoded;
 else p.energy=encoded;return p;}
struct Batch {std::string name;std::vector<SolidQ> q;std::vector<ExchangePacket> packets;};
std::vector<Batch> cases(){
 std::vector<Batch> all;auto fresh=[&](const std::string&name,int cells=4)->Batch&{
  all.push_back({name,{},{}});for(int c=0;c<cells;++c)all.back().q.push_back(base(c));return all.back();};
 {
  auto&b=fresh("interleaved");
  for(auto&q:b.q){for(int c=0;c<Nc;++c)q.condensed[c]+=1e10;for(int s=0;s<Ns;++s)q.pore[s]+=1e10;}
  for(int i=0;i<512;++i){const int c=(i*7)%4,k=i%(Nc+Ns+1);const double v=std::ldexp(double(i%13-6),i%21-12);
   b.packets.push_back(delta(c,k,v,i%2));}
  // Packets without a solid consumer neither contribute nor increase counts.
  ExchangePacket ignored;ignored.kind=ExchangeKind::GasFilm;ignored.solidCell=-333;ignored.energy=std::numeric_limits<Real>::quiet_NaN();b.packets.insert(b.packets.begin()+7,ignored);
 }
 {
  auto&b=fresh("zero_packets");for(auto&q:b.q){q.condensed[0]=-0.;q.pore[0]=-0.;q.energy=-0.;}
  b.q[0].porosity=std::numeric_limits<Real>::quiet_NaN();b.q[1].progress[2]=std::numeric_limits<Real>::infinity();
 }
 for(int zeros: {0,2})for(int ulps: {6,7,8}){
  auto&b=fresh("donor_"+std::to_string(zeros)+"_"+std::to_string(ulps));b.q[2].condensed[0]=1.;
  b.packets.push_back(delta(2,0,-(1.+ulps*0x1p-52),false));
  for(int i=0;i<zeros;++i)b.packets.push_back(delta(2,0,0.,i%2));
 }
 {
  auto&b=fresh("wide_intermediate");b.q[0].condensed[0]=b.q[1].pore[Ns-1]=b.q[2].energy=1e308;
  for(int c=0;c<3;++c){const int k=c==0?0:(c==1?Nc+Ns-1:Nc+Ns);
   b.packets.push_back(delta(c,k,1e308));b.packets.push_back(delta(c,k,-1e308,false));}
  b.packets.push_back(delta(2,Nc+Ns,-1e308));b.packets.push_back(delta(2,Nc+Ns,-1e308));
  b.packets.push_back(delta(2,Nc+Ns,1e308));b.packets.push_back(delta(2,Nc+Ns,1e308));
 }
 {
  auto&b=fresh("strong_cancellation");b.q[0].energy=1.;
  for(double v:{1e100,1.,-1e100,0x1p-64,0x1p-63,-0x1p-64,1e308,1e308,-1e308,-1e308})b.packets.push_back(delta(0,Nc+Ns,v));
 }
 for(int mode=0;mode<3;++mode){
  auto&b=fresh("subnormal_"+std::to_string(mode));const double tiny=std::numeric_limits<double>::denorm_min();
  for(auto&q:b.q){for(int c=0;c<Nc;++c)q.condensed[c]=tiny*(c+1);for(int s=0;s<Ns;++s)q.pore[s]=tiny*(s+1);q.energy=-tiny;}
  b.packets.push_back(delta(0,0,mode==0?tiny:-tiny*(mode==1?1:2)));
  b.packets.push_back(delta(1,Nc,-0.));b.packets.push_back(delta(1,Nc+Ns,tiny));
 }
 for(int mode=0;mode<4;++mode){auto&b=fresh("signed_zero_"+std::to_string(mode));
  b.q[0].condensed[0]=b.q[0].pore[0]=b.q[0].energy=mode%2?-0.:0.;
  for(int k: {0,Nc,Nc+Ns})b.packets.push_back(delta(0,k,mode/2?-0.:0.,mode%2));}
 const double bad[]={std::numeric_limits<Real>::quiet_NaN(),std::numeric_limits<Real>::infinity(),-std::numeric_limits<Real>::infinity()};
 for(int kind=0;kind<3;++kind)for(int component:{0,Nc,Nc+Ns}){
  auto&b=fresh("nonfinite_base_"+std::to_string(kind)+"_"+std::to_string(component));
  if(component<Nc)b.q[1].condensed[component]=bad[kind];else if(component<Nc+Ns)b.q[1].pore[component-Nc]=bad[kind];else b.q[1].energy=bad[kind];
  auto&x=fresh("nonfinite_packet_"+std::to_string(kind)+"_"+std::to_string(component));x.packets.push_back(delta(2,component,bad[kind]));
 }
 {
  auto&b=fresh("first_failed_cell");b.q[1].energy=bad[0];b.q[0].condensed[0]=1.;
  b.packets.push_back(delta(1,Nc+Ns,0.));b.packets.push_back(delta(0,0,-2.));
 }
 {
  auto&b=fresh("opposite_infinities");b.q[0].condensed[0]=-bad[1];b.packets.push_back(delta(0,0,bad[1]));
 }
 return all;
}
struct Rows {std::vector<int> offsets,indices;};
Rows rows(const Batch&b){Rows r;r.offsets.push_back(0);for(std::size_t c=0;c<b.q.size();++c){
 for(std::size_t i=0;i<b.packets.size();++i)if((requiredConsumers(b.packets[i].kind)&ConsumeSolid)&&b.packets[i].solidCell==int(c))r.indices.push_back(i);
 r.offsets.push_back(r.indices.size());}return r;}
#if defined(__CUDACC__)
__global__ void apply(const SolidQ*initial,const ExchangePacket*packets,const int*offsets,const int*indices,int cells,SolidQ*output,int*status){
 const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=cells)return;
 status[c]=int(applyMaterialPacketRow(initial[c],packets,indices,offsets[c],offsets[c+1],output[c]));
}
void ck(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
struct Storage {std::vector<void*> pointers;~Storage(){for(void*p:pointers)cudaFree(p);}
 template<class T>T* upload(const std::vector<T>&x){T*p=nullptr;if(x.size()){ck(cudaMalloc(&p,x.size()*sizeof(T)));pointers.push_back(p);ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}return p;}};
#endif
int main(){
 if(LDBL_MANT_DIG!=64)return 77;
#if defined(__CUDACC__)
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
#endif
 int passed=0;
 for(const auto&b:cases()){
  auto csr=rows(b);std::vector<SolidQ> sentinel(b.q.size());for(auto&q:sentinel){q.energy=987.;q.porosity=-.25;}
  auto reference=sentinel;std::string error;bool numerical=false;
  const bool ok=baselineApplyMaterialPacketBatch(b.q,b.packets,reference,error,&numerical);
  if(b.name=="interleaved"||b.name=="wide_intermediate"||b.name=="strong_cancellation"||b.name=="zero_packets")require(ok,b.name+" must exercise complete publication");
  auto host=sentinel;std::vector<int> codes(b.q.size());
  for(std::size_t c=0;c<b.q.size();++c)codes[c]=int(applyMaterialPacketRow(b.q[c],b.packets.data(),csr.indices.data(),csr.offsets[c],csr.offsets[c+1],host[c]));
  int first=-1;for(std::size_t c=0;c<codes.size();++c)if(codes[c]){first=c;break;}
  require(ok==(first<0),b.name+" CPU acceptance changed: "+error);
  if(ok){require(!numerical,b.name+" baseline classification");for(std::size_t c=0;c<b.q.size();++c)same(host[c],reference[c],b.name+" host");}
  else {require(numerical&&error==materialInventoryMessage(MaterialInventoryCode(codes[first])),b.name+" error semantics changed: "+error);
   for(std::size_t c=0;c<b.q.size();++c){same(reference[c],sentinel[c],b.name+" baseline transaction");if(codes[c])same(host[c],sentinel[c],b.name+" failed row write");}}
#if defined(__CUDACC__)
  Storage d;auto actual=sentinel;std::vector<int> status(codes.size());
  const auto*q=d.upload(b.q);const auto*packet=d.upload(b.packets);const auto*offset=d.upload(csr.offsets);const auto*index=d.upload(csr.indices);
  auto*out=d.upload(actual);auto*result=d.upload(status);
  apply<<<1,128>>>(q,packet,offset,index,b.q.size(),out,result);ck(cudaGetLastError());
  ck(cudaMemcpy(actual.data(),out,actual.size()*sizeof(SolidQ),cudaMemcpyDeviceToHost));ck(cudaMemcpy(status.data(),result,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
  require(status==codes,b.name+" CUDA status differs");
  for(std::size_t c=0;c<b.q.size();++c)same(actual[c],host[c],b.name+" CUDA");
#endif
  ++passed;
 }
#if defined(__CUDACC__)
 const char* backend="CUDA and host";
#else
 const char* backend="host";
#endif
 std::printf("PASS packet inventory Ns=%d batches=%d original binary80 vs %s fields and failure semantics\n",Ns,passed,backend);
}
