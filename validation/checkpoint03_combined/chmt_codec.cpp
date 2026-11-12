#include "restart/Checkpoint.H"
#include "chmt_seed.H"
#include "chmt_support.H"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>
extern "C" int chmt_write_cp2(const char*,int);
extern "C" int chmt_write_cp3(const char*,int);
extern "C" int chmt_write_cp3_inspect(const char*,double*);
namespace {
using namespace chmt;using namespace chmt_accept;
// Physical inventory types follow selected precision; only explicit geometry/time exceptions stay binary64.
#define CHECK_ENERGY_TYPE(field,unit,value) static_assert(sizeof(std::remove_reference_t<decltype(std::declval<HostState>().field)>)==sizeof(E),"energy width: " #field);
CHMT_ENERGY_FIELDS(CHECK_ENERGY_TYPE)
CHMT_PHYSICAL_FIELDS(CHECK_ENERGY_TYPE)
#undef CHECK_ENERGY_TYPE
#define CHECK_INDEPENDENT_TYPE(field,unit,value) static_assert(sizeof(std::remove_reference_t<decltype(std::declval<HostState>().field)>)==8,"independent width: " #field);
CHMT_INDEPENDENT_FIELDS(CHECK_INDEPENDENT_TYPE)
#undef CHECK_INDEPENDENT_TYPE
static_assert(sizeof(decltype(ParticleQ{}.thermal.temperature))==sizeof(E),"particle temperature selected width");
static_assert(sizeof(std::remove_reference_t<decltype(ParticleQ{}.thermal.coldRingSolidMass[0])>)==sizeof(E),"particle ring mass selected width");
static_assert(sizeof(std::remove_reference_t<decltype(ParticleQ{}.thermal.coldNodeSpecificEnthalpy[0])>)==4,"historical profile wire remains f32");
static_assert(sizeof(std::remove_reference_t<decltype(ParticleQ{}.thermal.cold2DRingContactAge[0])>)==8,"ring contact time stays f64");
static_assert(bits!=32||!hasCarry<ugkp::thermal::ParticleThermalHistory>::value,"FP32 physical history must not retain carry");
std::string bytes(const std::string& path){std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
void compare(const HostState& actual,const HostState& expected,const std::string& label){
 #define CHECK_FIELD(field,unit,value) exact(label+"." #field,unit,actual.field,expected.field);
 CHMT_ENERGY_FIELDS(CHECK_FIELD)
 CHMT_PHYSICAL_FIELDS(CHECK_FIELD)
 CHMT_INDEPENDENT_FIELDS(CHECK_FIELD)
 #undef CHECK_FIELD
 require(actual.acceptedSteps==17&&actual.commitSequence==17&&actual.rejectedSteps==3,label+" counters");
 for(int k=0;k<8;++k){
 exact(label+".profile8."+std::to_string(k),"J/kg",actual.particles[0].thermal.coldNodeSpecificEnthalpy[k],expected.particles[0].thermal.coldNodeSpecificEnthalpy[k]);
 exact(label+".ring_mass."+std::to_string(k),"kg",actual.particles[0].thermal.coldRingSolidMass[k],expected.particles[0].thermal.coldRingSolidMass[k]);
 exact(label+".profile_time."+std::to_string(k),"s",actual.particles[0].thermal.cold2DRingContactAge[k],expected.particles[0].thermal.cold2DRingContactAge[k]);}
 for(int k=0;k<64;++k)exact(label+".profile64."+std::to_string(k),"J/kg",actual.particles[0].thermal.cold2DNodeSpecificEnthalpy[k],expected.particles[0].thermal.cold2DNodeSpecificEnthalpy[k]);
 exact(label+".particle_mass","kg",actual.particles[0].mass,expected.particles[0].mass);
 exact(label+".particle_position","m",actual.particles[0].position.x,expected.particles[0].position.x);
 exact(label+".particle_age","s",actual.particles[0].thermal.coldContactAge,expected.particles[0].thermal.coldContactAge);
}
template<class H> double carried(const H& h){if constexpr(hasCarry<H>::value)return h.specificEnthalpyResidual;else return 0;}
}
int main(){
 try{
  ModelConfig model;HostState expected;chmtSeed(model,expected,false);
  const char* tmp=std::getenv("TMPDIR");std::string pattern=std::string(tmp?tmp:"/tmp")+"/chmt-combined-codec.XXXXXX";
  std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);require(mkdtemp(name.data()),"scratch directory");
  const std::string root(name.data());std::string error;
  // One process owns both old writers; old namespace/type/source closures are independent.
  for(int history=0;history<2;++history){
   const std::string label=history?"legacy_cp3":"legacy_cp2",dir=root+"/"+label;
   require((history?chmt_write_cp3:chmt_write_cp2)(dir.c_str(),1)==0,label+" historical writer");
   const auto legacyBytes=bytes(dir+"/state.chmt");require(!legacyBytes.empty(),label+" bytes");
   HostState actual;require(readCheckpoint(dir,model,actual,error),label+" "+error);
   compare(actual,expected,label);
   // The ordinary mobile caloric branch is linear below 2327 K. This independent
   // double observer reconstructs OLD physical h; it is never written into state.
   const double oldT=701.123456789,carry=0.125,cp=153.5/0.101961276;
   const double actualH=double(actual.particles[0].thermal.temperature)*cp+carried(actual.particles[0].thermal);
   const double expectedH=oldT*cp+carry;
   near(label+".legacy_particle_total_h","J/kg",actualH,expectedH,std::abs(expectedH),2e-9,4);
   if(bits==32){
    const float rounded=float(oldT+carry/cp);
    exact(label+".one_round_temperature","K",float(actual.particles[0].thermal.temperature),rounded);
   }
   const auto converted=root+"/"+label+"_converted";
   require(writeCheckpoint(converted,model,actual,error),error);
   require(bytes(converted+"/state.chmt").size()==legacyBytes.size(),label+" preserved doublewire widths/order");
  }
  const std::string native=root+"/native";
  require(writeCheckpoint(native,model,expected,error),error);
  const auto first=bytes(native+"/state.chmt");HostState roundtrip;
  require(readCheckpoint(native,model,roundtrip,error),error);compare(roundtrip,expected,"native_roundtrip");
  require(writeCheckpoint(native,model,roundtrip,error),error);
  require(bytes(native+"/state.chmt")==first,"native selected-state full wire roundtrip");
  // Only this unknown-metadata behavior is tested here. Known 32->64 error and
  // 64->32 warning belong to the planner's shared codec family, not extra nodes.
  std::ofstream metadata(native+"/ugkpPrecision");metadata<<"schema 997\ncomputePrecision 17\nrestartPrecision 17\n";metadata.close();
  HostState unknown;require(readCheckpoint(native,model,unknown,error),"unknown metadata keeps existing reader: "+error);
  compare(unknown,expected,"unknown_metadata");
  double oldWireCarry=-1;
  require(chmt_write_cp3_inspect(native.c_str(),&oldWireCarry)==0,"historical reader decodes native wire");
  exact("native_roundtrip.removed_carry_wire_slot","J/kg",oldWireCarry,0.0);
  std::filesystem::remove_all(root);finish("chmt_codec");return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
