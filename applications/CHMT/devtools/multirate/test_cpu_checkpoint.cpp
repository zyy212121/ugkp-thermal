#define main chmtExistingSweepFixtureMain
#include "test_sweep_constraints.cpp"
#undef main
#include "restart/Checkpoint.H"
#include "materials/Reaction.H"
#include <unistd.h>
int main(){ModelConfig model;auto& p=model.physics;p.enableGas=false;p.minDt=1e-9;p.maxDt=1;p.cfl=.4;
 for(int c=0;c<Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.conductivity=2;t.Tmin=100;t.Tmax=2000;}
 HostState h;h.solidMesh=twoHexahedra(0);auto& m=h.solidMesh;m.oldPoints=m.points;m.referencePoints=m.points;m.oldVolumes=m.volumes;m.boundaryPrimitive.resize(m.owner.size());m.boundarySst.resize(m.owner.size());
 h.solid.resize(m.volumes.size());for(std::size_t c=0;c<h.solid.size();++c){h.solid[c].condensed[0]=1000*m.volumes[c];h.solid[c].energy=h.solid[c].condensed[0]*condensedE(p.condensed[0],500+10*c);}
 h.time=1;h.acceptedSteps=1;h.commitSequence=1;h.rejectedSteps=3;h.lastAcceptedDt=1;h.nextDt=.1;
 for(int stage=0;stage<2;++stage){auto& g=h.solidStages[stage];g.interval=stage?1:.5;g.geometryVersion=m.geometryVersion;g.topologyHash=m.topologyHash;g.oldVolume=g.newVolume=g.evaluationVolume=m.volumes;g.sweptVolume.assign(m.owner.size(),0);g.areaVector=m.areaVectors;g.cellCentre=m.cellCentres;g.faceCentre=m.faceCentres;g.oldPoints=g.newPoints=m.points;}
 char root[]="/tmp/chmt-checkpoint-owner.XXXXXX";require(mkdtemp(root)!=nullptr,"temporary checkpoint directory");const std::string path=std::string(root)+"/accepted";std::string error;
 require(writeCheckpoint(path,model,h,error),"actual checkpoint write: "+error);HostState restored;require(readCheckpoint(path,model,restored,error),"actual checkpoint read: "+error);
 require(restored.rejectedSteps==3&&restored.acceptedSteps==1&&restored.solidStages[1].newPoints.size()==m.points.size(),"accepted counters/stage geometry restored");
 for(std::size_t c=0;c<h.solid.size();++c)require(restored.solid[c].energy==h.solid[c].energy,"material conservative energy bitwise roundtrip");
 // Stage history is serialized as supplied; it is not a writer admission gate.
 HostState missing=h;missing.solidStages={};missing.budget.supportImpulse={NAN,INFINITY,-INFINITY};
 const std::string missingPath=std::string(root)+"/missing";
 require(writeCheckpoint(missingPath,model,missing,error),"checkpoint accepts absent optional stage history: "+error);
 HostState missingRestored;require(readCheckpoint(missingPath,model,missingRestored,error),"read checkpoint without stage history: "+error);
 require(missingRestored.solidStages[0].newPoints.empty()&&missingRestored.solidStages[1].newPoints.empty(),"absent stage history roundtrip");
 for(std::size_t c=0;c<h.solid.size();++c)require(missingRestored.solid[c].energy==h.solid[c].energy,"absent stage history preserves actual conservative energy");
 require(std::isnan(missingRestored.budget.supportImpulse.x)&&missingRestored.budget.supportImpulse.y==INFINITY&&missingRestored.budget.supportImpulse.z==-INFINITY,"legacy support slot bit-pattern classes survive codec without admission");
 unlink((missingPath+"/state.chmt").c_str());rmdir(missingPath.c_str());
 unlink((path+"/state.chmt").c_str());rmdir(path.c_str());rmdir(root);
 std::cout<<"CPU checkpoint: actual 3-D conservative state/stage/counter roundtrip passed\n";
}
