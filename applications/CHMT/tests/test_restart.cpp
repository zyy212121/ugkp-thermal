// Actual checkpoint serialization, bounded geometry carry and fsync failures.
#define main existing_restart_geometry_fixture_main
#include "../devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "tests/TestSupport.H"
#include "restart/Checkpoint.H"
#include "gpu/StateValidation.H"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <unistd.h>
namespace {
int syncCalls=0,failSync=0;
bool sameBits(const std::vector<Real>& a,const std::vector<Real>& b) {
 return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0);
}
std::string bytes(const std::string& path) {
 std::ifstream in(path,std::ios::binary);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
void writeBytes(const std::string& path,const std::string& data) {
 std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write(data.data(),data.size());require(bool(out),"test byte write");
}
std::uint64_t getInteger(const std::string& data,std::size_t offset,int count=8) {
 require(offset+count<=data.size(),"test integer read range");std::uint64_t value=0;
 for(int i=0;i<count;++i)value|=std::uint64_t(static_cast<unsigned char>(data[offset+i]))<<(8*i);
 return value;
}
void setInteger(std::string& data,std::size_t offset,std::uint64_t value,int count=8) {
 require(offset+count<=data.size(),"test integer write range");
 for(int i=0;i<count;++i){data[offset+i]=static_cast<char>(value&255);value>>=8;}
}
void setReal(std::string& data,std::size_t offset,Real value) {
 std::uint64_t bits=0;std::memcpy(&bits,&value,sizeof(bits));setInteger(data,offset,bits);
}
void refreshDigest(std::string& data) {
 // Explicit wire header: magic/schema (16), six uint32 values (24),
 // field-schema/config-size/payload-size/digest (32), then model/state bytes.
 constexpr std::size_t body=72;std::uint64_t digest=14695981039346656037ULL;
 for(std::size_t i=body;i<data.size();++i){digest^=static_cast<unsigned char>(data[i]);digest*=1099511628211ULL;}
 setInteger(data,64,digest);
}
void populateSurface(HostState& h) {
 auto& m=h.solidMesh;auto& s=h.surface;
 for(std::size_t c=0;c<m.volumes.size();++c) {
  int face=-1;for(std::size_t f=0;f<m.owner.size();++f)
   if(m.owner[f]==static_cast<int>(c)&&m.faceCentres[f].z==1&&m.areaVectors[f].z>0)face=static_cast<int>(f);
  require(face>=0,"material surface fixture");m.boundaryKind[face]=BoundaryKind::Interface;
  s.solidFace.push_back(face);s.solidCell.push_back(c);s.gasFace.push_back(-1);s.persistentId.push_back(c);
  s.area.push_back(mag(m.areaVectors[face]));s.centre.push_back(m.faceCentres[face]);
  s.normal.push_back(m.areaVectors[face]/s.area.back());s.baseVelocity.push_back({});s.solidDistance.push_back(.5);
  FilmAux aux;aux.area=s.area.back();aux.normal=s.normal.back();aux.pressure=100000;aux.temperature=500;h.filmAux.push_back(aux);
 }
 std::string error;require(rebuildGeometry(m,error),error);
 s.oldArea=s.area;s.meshVelocity.resize(s.area.size());
}
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd){++syncCalls;if(failSync&&syncCalls==failSync){errno=EIO;return -1;}return __real_fsync(fd);}
int main(){
 ModelConfig model;model.physics=chmt_test::physics();model.physics.enableGas=false;model.physics.modelFingerprint=17;
 HostState h;h.solidMesh=twoHexahedra(0);auto& m=h.solidMesh;m.oldPoints=m.referencePoints=m.points;m.oldVolumes=m.volumes;m.boundaryPrimitive.resize(m.owner.size());m.boundarySst.resize(m.owner.size());h.solid.resize(m.volumes.size());
 for(size_t c=0;c<h.solid.size();++c){h.solid[c].condensed[0]=model.physics.condensed[0].rho*m.volumes[c];h.solid[c].energy=h.solid[c].condensed[0]*condensedE(model.physics.condensed[0],500+10*c);}
 populateSurface(h);
 const Real epsilon=std::numeric_limits<Real>::epsilon();h.solidSweepRemainder={1.125*epsilon,-1.375*epsilon};
 h.time=.5;h.acceptedSteps=h.commitSequence=1;h.rejectedSteps=4;h.lastAcceptedDt=.5;h.nextDt=.05;
 for(int stage=0;stage<2;++stage){auto& g=h.solidStages[stage];g.interval=stage?.5:.25;g.geometryVersion=m.geometryVersion;g.topologyHash=m.topologyHash;g.oldVolume=g.newVolume=g.evaluationVolume=m.volumes;g.sweptVolume.assign(m.owner.size(),0);g.areaVector=m.areaVectors;g.cellCentre=m.cellCentres;g.faceCentre=m.faceCentres;g.oldPoints=g.newPoints=m.points;}
 char root[]="/tmp/chmt-restart-regression.XXXXXX";require(mkdtemp(root)!=nullptr,"temporary directory");const std::string directory=std::string(root)+"/checkpoint",file=directory+"/state.chmt";std::string error;
 require(writeCheckpoint(directory,model,h,error),error);const auto original=bytes(file);require(!original.empty(),"checkpoint bytes exist");HostState restored;require(readCheckpoint(directory,model,restored,error),error);
 require(restored.time==h.time&&restored.rejectedSteps==4&&restored.solid[1].energy==h.solid[1].energy,"bitwise inventories and counters restored");
 require(sameBits(restored.solidSweepRemainder,h.solidSweepRemainder),"signed numerical sweep remainder bitwise roundtrip");
 require(writeCheckpoint(directory,model,restored,error)&&bytes(file)==original,"entire dry checkpoint bitwise reserialization");
 // Wet and phase-boundary inventory states retain the geometry carry exactly.
 auto filmModel=model;filmModel.physics.enableFilm=true;auto phase=h;phase.film.resize(h.surface.area.size());
 for(int phaseCase=0;phaseCase<3;++phaseCase) {
  for(std::size_t f=0;f<phase.film.size();++f) {
   auto& q=phase.film[f];q={};q.mass=phaseCase==0?0:(phaseCase==1?.1:(f==0?0:.1));q.species[0]=q.mass;
   q.enthalpy=q.mass*condensedE(filmModel.physics.liquid,500+10*f);
   require(recoverFilm(q,phase.surface.area[f],phase.filmAux[f].pressure,filmModel.physics,phase.filmAux[f]),"phase checkpoint fixture EOS");
  }
  require(writeCheckpoint(directory,filmModel,phase,error),error);const auto phaseBytes=bytes(file);
  require(readCheckpoint(directory,filmModel,restored,error),error);
  require(sameBits(restored.solidSweepRemainder,phase.solidSweepRemainder),"dry/wet phase remainder preserved bitwise");
  require(writeCheckpoint(directory,filmModel,restored,error)&&bytes(file)==phaseBytes,"phase checkpoint full bitwise reserialization");
 }
 auto empty=h;empty.solidSweepRemainder.clear();require(writeCheckpoint(directory,model,empty,error),error);
 require(readCheckpoint(directory,model,restored,error)&&restored.solidSweepRemainder.empty(),"empty initial geometry carry stays empty");
 auto signedZero=h;signedZero.solidSweepRemainder={0.,-0.};require(writeCheckpoint(directory,model,signedZero,error),error);
 require(readCheckpoint(directory,model,restored,error)&&sameBits(restored.solidSweepRemainder,signedZero.solidSweepRemainder),"signed zero carry is not canonicalized");
 auto atBound=h;atBound.solidSweepRemainder={32*epsilon,-32*epsilon};
 require(writeCheckpoint(directory,model,atBound,error),error);require(readCheckpoint(directory,model,restored,error)&&sameBits(restored.solidSweepRemainder,atBound.solidSweepRemainder),"inclusive signed roundoff boundary roundtrip");
 require(writeCheckpoint(directory,model,h,error)&&bytes(file)==original,"restore accepted dry checkpoint");
 auto changed=h;changed.time=.75;changed.acceptedSteps=changed.commitSequence=2;changed.solidSweepRemainder={-2.5*epsilon,3.25*epsilon};
 for(int failure:{1,2}){syncCalls=0;failSync=failure;require(!writeCheckpoint(directory,model,changed,error),"injected pre-publication fsync failure reports failure");failSync=0;require(bytes(file)==original,"failed pre-publication write preserves accepted checkpoint and carry");require(error.find("sync")!=std::string::npos,"fsync cause retained");}
 // Directory fsync follows atomic rename; failure does not imply rollback.
 syncCalls=0;failSync=3;require(!writeCheckpoint(directory,model,changed,error),"post-rename durability failure reported");failSync=0;
 require(readCheckpoint(directory,model,restored,error)&&restored.time==changed.time&&sameBits(restored.solidSweepRemainder,changed.solidSweepRemainder),"post-rename file remains complete including carry");
 require(writeCheckpoint(directory,model,h,error),"safe retry restores original checkpoint");
 restored=changed;restored.time=999;const auto sentinel=restored.solidSweepRemainder;
 const std::string survivorDirectory=std::string(root)+"/survivor",survivorFile=survivorDirectory+"/state.chmt";
 require(writeCheckpoint(survivorDirectory,model,restored,error),error);const auto survivorBytes=bytes(survivorFile);
 auto expectReadFailure=[&](const std::string& data,const std::string& cause,const std::string& label) {
  writeBytes(file,data);require(!readCheckpoint(directory,model,restored,error),label);
  require(error.find(cause)!=std::string::npos,label+" cause: "+error);
  require(restored.time==999&&sameBits(restored.solidSweepRemainder,sentinel),label+" output unchanged");
  require(writeCheckpoint(survivorDirectory,model,restored,error)&&bytes(survivorFile)==survivorBytes,label+" entire output is bitwise unchanged");
 };
 auto wrong=model;wrong.physics.modelFingerprint++;
 require(!readCheckpoint(directory,wrong,restored,error),"different model rejects");require(restored.time==999&&sameBits(restored.solidSweepRemainder,sentinel),"identity rejection preserves output carry");
 auto corrupt=original;corrupt.back()^=1;expectReadFailure(corrupt,"checksum","payload corruption rejected");
 expectReadFailure(original.substr(0,original.size()-1),"length","truncation rejected");
 require(checkpointSchema==2&&getInteger(original,8)==2,"explicit checkpoint schema bumped for geometry carry");
 auto legacy=original;setInteger(legacy,8,1);expectReadFailure(legacy,"schema","legacy checkpoint cannot silently initialize carry");
 legacy=original;legacy[6]='1';expectReadFailure(legacy,"magic","legacy checkpoint file version refused");
 auto fieldMismatch=original;fieldMismatch[40]^=1;expectReadFailure(fieldMismatch,"field schema","field-schema mismatch refused");
 // Locate the unique encoded vector, then alter payload with a correct digest:
 // these tests exercise semantic validation instead of checksum-only rejection.
 std::string encoded(24,0);setInteger(encoded,0,2);setReal(encoded,8,h.solidSweepRemainder[0]);setReal(encoded,16,h.solidSweepRemainder[1]);
 const auto carryOffset=original.find(encoded,72+getInteger(original,48));
 require(carryOffset!=std::string::npos&&original.find(encoded,carryOffset+1)==std::string::npos,"unique serialized remainder vector");
 for(Real invalid:{std::numeric_limits<Real>::quiet_NaN(),std::numeric_limits<Real>::infinity(),64*epsilon,-64*epsilon}) {
  auto bad=original;setReal(bad,carryOffset+8,invalid);refreshDigest(bad);expectReadFailure(bad,"remainder","checksummed invalid carry rejected");
 }
 auto shortCarry=original;setInteger(shortCarry,carryOffset,1);shortCarry.erase(carryOffset+16,8);setInteger(shortCarry,56,getInteger(shortCarry,56)-8);refreshDigest(shortCarry);
 expectReadFailure(shortCarry,"remainder","checksummed mapped-size mismatch rejected");
 auto hugeCount=original;setInteger(hugeCount,carryOffset,std::numeric_limits<std::uint64_t>::max());refreshDigest(hugeCount);
 expectReadFailure(hugeCount,"vector count","oversized encoded carry count rejected before allocation");
 writeBytes(file,original);
 auto expectInvalidState=[&](HostState invalid,const std::string& label) {
  const auto before=invalid.solidSweepRemainder;
  require(!writeCheckpoint(directory,model,invalid,error),label+" checkpoint rejects");require(bytes(file)==original,label+" preserves prior checkpoint");
  require(!validateRuntimeState(model,invalid,error),label+" runtime rejects");
  require(error.find("remainder")!=std::string::npos,label+" reports carry: "+error);
  require(sameBits(before,invalid.solidSweepRemainder),label+" does not repair input carry");
 };
 auto invalid=h;invalid.solidSweepRemainder.pop_back();expectInvalidState(invalid,"wrong remainder count");
 invalid=h;invalid.solidSweepRemainder.push_back(0);expectInvalidState(invalid,"oversized remainder count");
 invalid=h;invalid.solidSweepRemainder[0]=std::numeric_limits<Real>::quiet_NaN();expectInvalidState(invalid,"nonfinite remainder");
 invalid=h;invalid.solidSweepRemainder[0]=64*epsilon;expectInvalidState(invalid,"out-of-budget remainder");
 invalid=h;invalid.solidSweepRemainder[0]=std::nextafter(32*epsilon,std::numeric_limits<Real>::infinity());expectInvalidState(invalid,"one ulp beyond retained budget");
 invalid=h;invalid.surface.solidFace[0]=-1;expectInvalidState(invalid,"unmapped nonzero remainder");
 invalid=h;invalid.surface.solidFace[0]=static_cast<int>(h.solidMesh.owner.size());expectInvalidState(invalid,"out-of-range remainder mapping");
 auto unmapped=h;unmapped.surface.solidFace[0]=-1;unmapped.solidSweepRemainder[0]=0;
 require(writeCheckpoint(directory,model,unmapped,error),error);require(readCheckpoint(directory,model,restored,error)&&sameBits(restored.solidSweepRemainder,unmapped.solidSweepRemainder),"unmapped zero remainder accepted without mutation");
 require(writeCheckpoint(directory,model,h,error),error);
 auto missing=h;missing.solidStages={};require(!writeCheckpoint(directory,model,missing,error),"missing accepted stage trace rejects");require(bytes(file)==original,"invalid state never replaces checkpoint");
 unlink(file.c_str());rmdir(directory.c_str());unlink(survivorFile.c_str());rmdir(survivorDirectory.c_str());rmdir(root);
 std::cout<<"restart regressions: real 3-D dry/wet state, bounded signed carry, schema/corruption/identity checks, atomicity and fsync failures passed\n";
}
