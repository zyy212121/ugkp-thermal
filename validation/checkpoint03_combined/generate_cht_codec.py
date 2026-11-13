"""Static code generation only. Does not import or execute any numerical test."""
from pathlib import Path
import argparse,re
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
text=(a.root/'applications/CHT/gpu/GpuChtParticleRestart.H').read_text()
body=text.split('struct MutableView',1)[1].split('};',1)[0]
fields=re.findall(r'\b([\w:]+)\*\s+(p\w+)\s*(?:=[^;]*)?;',body)
decl='\n'.join(f' std::vector<{t}> {n}=std::vector<{t}>(N*64);' for t,n in fields)
assign='\n'.join(f'  v.{n}={n}.data();' for t,n in fields)
roundtrip='\n'.join(f' for(size_t i=0;i<N*{64 if n=="pCold2DNodeSpecificEnthalpy" else 8 if n in ("pColdNodeSpecificEnthalpy","pColdRingSolidMass","pCold2DRingContactAge") else 1};++i)ck(current.{n}[i]==original.{n}[i],"current {n} roundtrip");' for ty,n in fields if n!="pThermalEnthalpyResidual")
init='\n'.join(f'  for(size_t i=0;i<{n}.size();++i){n}[i]={t}(1.25+i*.125);' for t,n in fields)
code=r"""
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
#include "common/PrecisionRestart.H"
#ifdef CP3_FP32
#include "common/GpuParticleRestartConversion.H"
#endif
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
using namespace Foam;
void ck(bool x,const char*m){if(!x)throw std::runtime_error(m);}
constexpr size_t N=3;
struct Storage {
FIELDS
 gpuFshRestart::MutableView mutableView(){gpuFshRestart::MutableView v{};v.count=N;
ASSIGN
 return v;}
 gpuFshRestart::ConstView constView()const{gpuFshRestart::ConstView v{};v.count=N;
ASSIGN
 return v;}
 void init(){
INIT
 pT[0]=300.125;pT[1]=2316.;pT[2]=3500.;
 pThermalEnthalpyResidual[0]=.375;pThermalEnthalpyResidual[1]=2.625;pThermalEnthalpyResidual[2]=-1.75;
 pStuck[0]=0;pStuck[1]=1;pStuck[2]=1;
 pContactDuration[0]=.010000000000000002;pColdContactAge[1]=.020000000000000004;
 for(int i=0;i<3;++i){pRng[i]=0x123456789abcdefULL+i;pOrigId[i]=0xfedcba987654321ULL+i;}
 }
};
std::string legacy(Storage&s,int schema){
 std::ostringstream os(std::ios::binary);auto v=s.constView();
 if(schema==9){gpuThermalRestart::writeCurrent<true>(os,"UGKP_CHT_PARTICLES_SCHEMA9_FLOAT_COLD_PROFILES_BIN",v);return os.str();}
 os<<gpuFshRestart::schema8FormatName<<' '<<N<<' '<<gpuThermalRestart::maximumChunkParticles<<'\n';
 std::uint32_t n=N;gpuThermalRestart::writeArray(os,&n,0,1);
 gpuThermalRestart::wireFields<double>(v,0,N,[&](const auto* values,size_t b,size_t n,auto disk){
  using D=decltype(disk);std::vector<D>x(n);for(size_t i=0;i<n;++i)x[i]=D(values[b+i]);
  gpuThermalRestart::writeArray(os,x.data(),0,n);
 });
 gpuThermalRestart::writeArray(os,v.pThermalEnthalpyResidual,0,N);return os.str();
}
void read(const std::string& bytes,Storage&s,int schema){
 std::istringstream is(bytes,std::ios::binary);std::string marker;size_t n;std::uint32_t bound;
 is>>marker>>n>>bound;ck(n==N,"codec count");auto v=s.mutableView();
 if(schema==8)gpuFshRestart::readV8Payload(is,bound,v);
#ifdef CP3_FP32
 else if(schema==9)gpuFshRestart::readLegacyCarryPayload(is,bound,v);
#endif
 else gpuFshRestart::readCurrentPayload(is,bound,v);
 ck(is.peek()==std::char_traits<char>::eof(),"trailing bytes or omitted fields");
}
int main(int argc,char**argv)try{
 ck(argc==2,"scratch directory argument");std::cout<<std::setprecision(17);
 Storage original;original.init();
 for(int schema:{8,9}){
  const auto data=legacy(original,schema);Storage restored;read(data,restored,schema);
  for(int i=0;i<3;++i){
   ck(restored.pm[i]==original.pm[i]&&restored.px[i]==original.px[i]&&restored.pRng[i]==original.pRng[i]&&restored.pOrigId[i]==original.pOrigId[i],"legacy physical mass/position/RNG identity");
   ck(restored.pContactDuration[i]==original.pContactDuration[i],"legacy double time preserved");
   ck(restored.pThermalEnthalpyResidual[i]==original.pThermalEnthalpyResidual[i],"legacy carry read");
  }
#ifdef CP3_FP32
  const ugkwp::legacyRestart::ParticleCaloric oldLaw;
  const double oldH=oldLaw.enthalpy(restored.pT[0])+restored.pThermalEnthalpyResidual[0];
  const float imported=ugkwp::legacyRestart::importMobileTemperature(restored.pT[0],restored.pThermalEnthalpyResidual[0]);
  ck(imported==float(oldLaw.temperature(oldH)),"legacy mobile enthalpy not rounded once");
  ck(ugkwp::legacyRestart::importMobileTemperature(300.125,0)==float(300.125),"zero carry conversion drift");
  // Only maintained 1D cold-wall arithmetic belongs to current acceptance.
  for(int count:{8}){
   float* nodes=count==8?restored.pColdNodeSpecificEnthalpy.data()+8:restored.pCold2DNodeSpecificEnthalpy.data()+128;
   const double carry=count==8?restored.pThermalEnthalpyResidual[1]:restored.pThermalEnthalpyResidual[2];
   std::vector<float> before(nodes,nodes+count);ugkwp::legacyRestart::importProfile(nodes,nodes,count,carry);
   double beforeTotal=0,afterTotal=0;for(int j=0;j<count;++j){ck(nodes[j]==float(double(before[j])+carry),"profile import rounding");beforeTotal+=double(before[j])+carry;afterTotal+=nodes[j];}
   const double bound=std::numeric_limits<float>::epsilon()*std::abs(beforeTotal);
   ck(std::abs(afterTotal-beforeTotal)<=bound,"legacy profile dimensional enthalpy import");
   std::cout<<"IMPORT schema="<<schema<<" nodes="<<count<<" totalHError="<<afterTotal-beforeTotal<<" allowed="<<bound<<'\n';
  }
  std::cout<<"IMPORT schema="<<schema<<" mobileT="<<imported<<" oldTotalH="<<oldH<<'\n';
#else
  ck(restored.pT[0]==original.pT[0],"FP64 legacy temperature changed");
#endif
 }
 std::ostringstream os(std::ios::binary);gpuFshRestart::writeCurrent(os,original.constView());
 Storage current;read(os.str(),current,ugkp::precision::compiledBits==32?10:9);
#ifdef CP3_FP32
 ck(os.str().find("SCHEMA10_FP32_SINGLE_STATE")!=std::string::npos,"FP32 current marker");
 for(double&c:original.pThermalEnthalpyResidual)c+=1234;
 std::ostringstream altered(std::ios::binary);gpuFshRestart::writeCurrent(altered,original.constView());
 ck(altered.str()==os.str(),"schema10 still stores carry");
 for(int i=0;i<3;++i)ck(current.pThermalEnthalpyResidual[i]==0,"schema10 materialized carry");
#else
 ck(os.str().find("SCHEMA9_FLOAT_COLD_PROFILES")!=std::string::npos,"FP64 schema changed");
#endif
 ROUNDTRIP
 // Existing directional precision policy only. No new rejection is introduced.
 const std::string dir=argv[1];std::filesystem::create_directories(dir);std::string error;
 for(int saved:{32,64}){std::ofstream f(ugkp::precision::metadataPath(dir));f<<"schema 1\ncomputePrecision "<<saved<<"\nrestartPrecision "<<saved<<"\n";f.close();
  bool ok=ugkp::precision::checkRestartPrecision(dir,error);
  ck(ok==(saved>=ugkp::precision::compiledBits),"existing precision direction policy");
  std::cout<<"METADATA stored="<<saved<<" compute="<<ugkp::precision::compiledBits<<" accepted="<<ok<<" message="<<error<<'\n';}
 {std::ofstream f(ugkp::precision::metadataPath(dir));f<<"schema 99\ncomputePrecision 32\nrestartPrecision 32\n";}
 ck(ugkp::precision::checkRestartPrecision(dir,error),"unknown metadata newly rejected");
 std::cout<<"PASS CHT schema8/schema9 legacy import, current no-carry codec and precision directions\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
"""
a.output.write_text(code.replace('FIELDS',decl).replace('ASSIGN',assign).replace('INIT',init).replace('ROUNDTRIP',roundtrip))
