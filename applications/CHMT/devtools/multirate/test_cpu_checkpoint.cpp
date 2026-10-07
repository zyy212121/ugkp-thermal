#define main chmtExistingSweepFixtureMain
#include "test_sweep_constraints.cpp"
#undef main
#include "restart/Checkpoint.H"
#include "materials/Reaction.H"
#include <unistd.h>
int main(){ModelConfig model;auto& p=model.physics;p.enableGas=false;p.minDt=1e-9;p.maxDt=1;p.cfl=.4;p.modelFingerprint=1;
 for(int c=0;c<Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.conductivity=2;t.Tmin=100;t.Tmax=2000;}
 HostState h;h.solidMesh=twoHexahedra(0);auto& m=h.solidMesh;m.oldPoints=m.points;m.referencePoints=m.points;m.oldVolumes=m.volumes;m.boundaryPrimitive.resize(m.owner.size());m.boundarySst.resize(m.owner.size());
 h.solid.resize(m.volumes.size());for(std::size_t c=0;c<h.solid.size();++c){h.solid[c].condensed[0]=1000*m.volumes[c];h.solid[c].energy=h.solid[c].condensed[0]*condensedE(p.condensed[0],500+10*c);}
 h.time=1;h.acceptedSteps=1;h.commitSequence=1;h.rejectedSteps=3;h.lastAcceptedDt=1;h.nextDt=.1;
 for(int stage=0;stage<2;++stage){auto& g=h.solidStages[stage];g.interval=stage?1:.5;g.geometryVersion=m.geometryVersion;g.topologyHash=m.topologyHash;g.oldVolume=g.newVolume=g.evaluationVolume=m.volumes;g.sweptVolume.assign(m.owner.size(),0);g.areaVector=m.areaVectors;g.cellCentre=m.cellCentres;g.faceCentre=m.faceCentres;g.oldPoints=g.newPoints=m.points;}
 char root[]="/tmp/chmt-checkpoint-owner.XXXXXX";require(mkdtemp(root)!=nullptr,"temporary checkpoint directory");const std::string path=std::string(root)+"/accepted";std::string error;
 require(writeCheckpoint(path,model,h,error),"actual checkpoint write: "+error);HostState restored;require(readCheckpoint(path,model,restored,error),"actual checkpoint read: "+error);
 require(restored.rejectedSteps==3&&restored.acceptedSteps==1&&restored.solidStages[1].newPoints.size()==m.points.size(),"accepted counters/stage geometry restored");
 for(std::size_t c=0;c<h.solid.size();++c)require(restored.solid[c].energy==h.solid[c].energy,"material conservative energy bitwise roundtrip");
 HostState missing=h;missing.solidStages={};require(!writeCheckpoint(std::string(root)+"/missing",model,missing,error),"missing native solid stage history rejected");
 unlink((path+"/state.chmt").c_str());rmdir(path.c_str());rmdir(root);
 std::cout<<"CPU checkpoint: actual 3-D conservative state/stage/counter roundtrip passed\n";
}
