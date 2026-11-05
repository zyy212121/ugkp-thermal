#include "restart/Checkpoint.C"
#include <iostream>
#include <iomanip>
#include <cassert>
using namespace chmt;
struct NumericEncoder {
    unsigned configurationVersion=14;
    Bytes data;std::vector<double> numbers;
    void integer(std::uint64_t x,int bytes){for(int i=0;i<bytes;++i){data.push_back(static_cast<unsigned char>(x&255));x>>=8;}}
    void operator()(const Real& x){numbers.push_back(x);integer(0,8);}
    void operator()(const float& x){numbers.push_back(x);integer(0,4);}
    void operator()(const int& x){integer(static_cast<std::uint32_t>(x),4);}
    void operator()(const unsigned& x){integer(x,4);}
    void operator()(const std::uint64_t& x){integer(x,8);}
    void operator()(const unsigned char& x){integer(x,1);}
    void operator()(const bool& x){integer(x?1:0,1);}
    void operator()(const std::string& x){integer(x.size(),8);data.insert(data.end(),x.begin(),x.end());}
    template<class T,std::size_t N>void operator()(const T(&x)[N]){for(auto& v:x)(*this)(v);}
    template<class T,std::size_t N>void operator()(const std::array<T,N>& x){for(auto& v:x)(*this)(v);}
    template<class T>void operator()(const std::vector<T>& x){integer(x.size(),8);for(auto& v:x)(*this)(v);}
    template<class T>typename std::enable_if<std::is_enum<T>::value>::type operator()(const T& x){int value=static_cast<int>(x);(*this)(value);}
    template<class T>typename std::enable_if<!std::is_enum<T>::value>::type operator()(const T& x){visit(*this,x);}
};
HostState load(const char* directory){
 std::ifstream f(std::string(directory)+"/state.chmt",std::ios::binary|std::ios::ate);assert(f);Bytes bytes(size_t(f.tellg()));f.seekg(0);f.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
 Decoder d(bytes);d.offset=72;ModelConfig model;d(model);assert(d.integer(8)==0x31474643504b4755ULL);ugkp::thermal::visitParticlePhaseConfiguration(d,model.particlePhase);ugkp::thermal::visitParticleContactConfiguration(d,model.particleContact);
 assert(!model.physics.enableFilm);HostState state;std::string error;if(!readCheckpoint(directory,model,state,error))throw std::runtime_error(error);state.ledger.clear();return state;
}
int main(int argc,char** argv){if(argc!=3)return 2;auto a=load(argv[1]),b=load(argv[2]);
 // Dry CHMT deliberately marks the inactive legacy film auxiliary temperature
 // unavailable. Only this paired sentinel is excluded from the finite check.
 assert(a.filmAux.size()==b.filmAux.size());
 for(size_t f=0;f<a.filmAux.size();++f)
  if(std::isnan(a.filmAux[f].temperature)&&std::isnan(b.filmAux[f].temperature))
   a.filmAux[f].temperature=b.filmAux[f].temperature=0;
 if(a.budget.radiation!=b.budget.radiation)
  std::cout<<std::setprecision(17)<<"RADIATION_BUDGET_J "<<a.budget.radiation<<" "<<b.budget.radiation<<"\n";
 NumericEncoder x,y;x(a);y(b);x(a.gasTemperature);y(b.gasTemperature);
 for(auto& p:a.particles)ugkp::thermal::visitParticleThermalHistory(x,p.thermal);for(auto& p:b.particles)ugkp::thermal::visitParticleThermalHistory(y,p.thermal);
 assert(x.data==y.data&&x.numbers.size()==y.numbers.size());double relative=0,absolute=0;int differences=0;
 for(size_t i=0;i<x.numbers.size();++i){const double l=x.numbers[i],r=y.numbers[i];assert(std::isfinite(l)&&std::isfinite(r));if(l==r)continue;
  const double difference=std::abs(l-r),scale=std::max(std::abs(l),std::abs(r));++differences;relative=std::max(relative,difference/std::max(1.,scale));absolute=std::max(absolute,difference);
  if(differences<20)std::cout<<std::setprecision(17)<<"DIFF "<<i<<" "<<l<<" "<<r<<" "<<difference<<"\n";
 }
 std::cout<<std::setprecision(17)<<"COMPARE count="<<differences<<" maximumAbsolute="<<absolute<<" scaled="<<relative<<"\n";
 return relative>1e-10?3:0;
}
