#include "fvCFD.H"
#include "IStringStream.H"
#include "GpuSolidThermalProperties.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuSolidPatchMapper.H"
#include "GpuThermalExchangeState.H"
#include "fixedGradientFvPatchFields.H"
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <iomanip>
extern int t14DictionaryOpens,t14HashCalls,t14FieldScans;
using namespace Foam;
using namespace Foam::gpuThermal;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(scalar actual,scalar expected,scalar tolerance=1e-11){require(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance*std::max(scalar(1),std::abs(expected)),"numeric mismatch");}
dictionary parse(const std::string& text){IStringStream stream(text);return dictionary(stream);}
std::string property(const char* mode="error") {return std::string("rho {type table; outOfBounds ")+mode+"; values ((0 2)(1 3));} Cp {type table; outOfBounds "+mode+"; values ((0 4)(1 6));} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}";}
std::string stateText(){return "formatVersion 99; initialState false; exchangeSequence \"7\"; completedTimeIndex 5; completedSimulationTimeS 3; completedRadiationSimulationTimeS 2; previousExchangeSimulationTimeS 1; fluidMeshTopologySha1 \"changed\"; solidMeshTopologySha1 \"changed\"; auxiliarySolidMeshTopologySha1 \"\"; couplingConfigurationSha1 \"changed\"; wallTemperatureSha1UsedForCompletedInterval \"changed\"; newlyUploadedWallTemperatureSha1 \"changed\"; gasWallLedgerConsumed false; particleWallLedgerConsumed false; particleRadiationApplied false; solidStateUpdated false; wallTemperatureUploaded false; particleMomentsRebuilt false; particleContactEnergyJ -9;";}
void writeText(const fileName& p,const std::string& text){std::ofstream os(p.c_str());os<<text;os.close();require(bool(os),"fixture write");}
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 fvMesh mesh(IOobject(fvMesh::defaultRegion,runTime.timeName(),runTime,IOobject::MUST_READ));
 IOdictionary config(IOobject("testProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 word mode=config.lookup<word>("t14Mode");
 FatalError.throwExceptions(); FatalIOError.throwExceptions();
 try {
 if(mode=="properties"||mode=="clamp"||mode=="nonoverlap"||mode=="signed"||mode=="absorbed"){
  std::string text=property(mode=="clamp"?"clamp":"error");
  if(mode=="nonoverlap")text="rho {type table; outOfBounds error; values ((0 2)(1 3));} Cp {type table; outOfBounds error; values ((2 8)(3 10));} kappa {type table; outOfBounds error; values ((10 6)(11 6));} emissivity {type constant; value 1.2;}";
  if(mode=="signed")text="rho {type constant; value -2;} Cp {type constant; value 4;} kappa {type constant; value -6;} emissivity {type constant; value 1.2;}";
  if(mode=="absorbed")text="rho {type table; outOfBounds error; values ((0 1e17)(1 1)(2 1));} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}";
  GpuSolidThermalProperties p(parse(text));
  if(mode=="clamp")near(p.deltaHv(2,-1),116.0/3);
  else if(mode=="signed"){near(p.deltaHv(2,-1),-24);near(p.Csec(2,-1),-8);near(p.emissivity(3),1.2);}
  else if(mode=="absorbed"){near(p.deltaHv(2,0),p.deltaHv(1,0)+1,0);}
  else {near(p.rho(-1),1);near(p.Cp(-1),2);near(p.rho(2),4);near(p.Cp(2),8);near(p.deltaHv(2,-1),42);near(p.deltaHv(-1,2),-42);near(p.Csec(2,-1),14);near(p.Csec(2,2),32);}
  if(mode=="properties") {GpuSolidThermalProperties q(parse("rho {type table; outOfBounds error; values ((0 2)(1 3)(2 5));} Cp {type constant; value 4;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));near(q.deltaHv(3,-1),56);}
 }
 else if(mode=="unordered"||mode=="reversed_axis"||mode=="duplicate_endpoint"){
  std::string nodes=mode=="unordered"?"((0 2)(2 10)(1 30)(3 50))":mode=="reversed_axis"?"((2 10)(0 2))":"((0 2)(0 3))";
  GpuSolidThermalProperties p(parse("rho {type table; outOfBounds error; values "+nodes+";} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  near(p.rho(mode=="unordered"?1.5:mode=="reversed_axis"?1:0),mode=="unordered"?35:mode=="reversed_axis"?6:3,0);
 }
 else if(mode=="ordinary"||mode=="linear_one"){
  GpuSolidThermalProperties p(parse(mode=="ordinary"?property():"rho {type constant; value 1000;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  if(mode=="ordinary") {for(scalar v:{scalar(0),scalar(.25),scalar(.5),scalar(1)}) {near(p.rho(v),2+v);near(p.Cp(v),4+2*v);std::cout<<"PARITY property "<<std::hexfloat<<p.rho(v)<<' '<<p.Cp(v)<<' '<<p.Hv(v)<<' '<<p.Csec(v,.3)<<'\n';}}
  GpuSolidThermalProperties material(parse("rho {type constant; value 1000;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),mesh);
  label left=mesh.boundaryMesh().findPatchID("left");labelList ids(1,left);List<scalarField> e(1);e[0].setSize(mesh.boundary()[left].size());forAll(e[0],i)e[0][i]=mode=="linear_one"?100*(i+1):100;
  GpuSolidThermalControls c;c.nonlinearMaxIterations=mode=="linear_one"?1:200;
  GpuSolidThermalCandidateSolver solver(mesh,material,ids,c);auto result=solver.solveTemporarySolidCandidate(T,e,.1);
  if(mode=="ordinary")near(result.temperature()[0],300.1);
  else {extern double t14LinearResidual;extern int t14LinearIterations;extern bool t14LinearConverged;require(t14LinearIterations==1&&!t14LinearConverged&&t14LinearResidual>0,"actual finite one-step linear candidate not exercised");require(result.temperature()[0]!=T[0],"old field returned");std::cout<<"LINEAR residual="<<t14LinearResidual<<" iterations="<<t14LinearIterations<<'\n';}
  solver.publishSolidCandidate(result,T);
  forAll(T,i)std::cout<<"PARITY cell "<<i<<' '<<std::hexfloat<<T[i]<<'\n';
  forAll(T.boundaryField(),pi){forAll(T.boundaryField()[pi],f)std::cout<<"PARITY face "<<pi<<' '<<f<<' '<<std::hexfloat<<T.boundaryField()[pi][f]<<'\n';if(isA<fixedGradientFvPatchScalarField>(T.boundaryField()[pi]))for(scalar g:refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[pi]).gradient())std::cout<<"PARITY gradient "<<std::hexfloat<<g<<'\n';}
  std::cout<<"PARITY residual "<<std::hexfloat<<result.integratedEnergyResidualJ<<' '<<result.nonlinearIterations<<'\n';
 }
 else if(mode=="diagnostic_overflow"){
  GpuSolidThermalProperties material(parse("rho {type constant; value 1000;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),mesh);
  label left=mesh.boundaryMesh().findPatchID("left");labelList ids(1,left);List<scalarField> energy(1);energy[0]=scalarField(mesh.boundary()[left].size(),1e308);
  GpuSolidThermalControls c;c.nonlinearMaxIterations=1;
  GpuSolidThermalCandidateSolver solver(mesh,material,ids,c);auto result=solver.solveTemporarySolidCandidate(T,energy,1e307);
  require(!std::isfinite(result.integratedEnergyResidualJ),"diagnostic overflow was hidden");
  forAll(result.temperature(),i)require(std::isfinite(result.temperature()[i]) && result.temperature()[i]!=T[i],"finite physical candidate missing");
  solver.publishSolidCandidate(result,T);
  forAll(T,i)require(T[i]==result.temperature()[i],"computed solid field not published");
  std::cout<<"OVERFLOW physical="<<T[0]<<" diagnostic="<<result.integratedEnergyResidualJ<<'\n';
 }
 else if(mode=="nearest_rounding"){
  fvMesh coarse(IOobject("coarse",runTime.timeName(),runTime,IOobject::MUST_READ,IOobject::NO_WRITE,false));List<SolidPatchPair> pairs(1);pairs[0].fluidPatch="left";pairs[0].solidPatch="right";
  ConservativeNearestSolidPatchMapper mapper(mesh,coarse,pairs);scalarField e(4);e[0]=1e17;e[1]=1;e[2]=-1e17;e[3]=1;auto mapped=mapper.mapIntegratedFaceEnergyToSolid(0,e);require(mapped.size()==2,"actual coarse mapping extent");near(mapped[0],1e17,0);near(mapped[1],-1e17,0);scalar input=0,output=0;for(scalar v:e)input+=v;for(scalar v:mapped)output+=v;near(input,1,0);near(output,0,0);std::cout<<"MAPPING actual input total="<<input<<" output total="<<output<<'\n';
 }
 else if(mode=="candidate_one"||mode=="candidate_zero"||mode=="rebind"){
  fvMesh duplicate(IOobject(fvMesh::defaultRegion,runTime.timeName(),runTime,IOobject::MUST_READ,IOobject::NO_WRITE,false));
  fvMesh& inputMesh=mode=="rebind"?duplicate:mesh;
  volScalarField T(IOobject("T",runTime.timeName(),inputMesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),inputMesh);
  GpuSolidThermalProperties p(parse("rho {type constant; value 1;} Cp {type table; outOfBounds error; values ((100 100)(1000 1000));} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  label left=mesh.boundaryMesh().findPatchID("left"); labelList patches(1,left);List<scalarField> energy(1);energy[0]=scalarField(mesh.boundary()[left].size(),100);
  GpuSolidThermalControls c;c.nonlinearMaxIterations=mode=="candidate_zero"?0:1;c.nonlinearRelativeTolerance=0;c.energyAbsoluteTolerance=0;c.energyRelativeTolerance=0;
  GpuSolidThermalCandidateSolver solver(mesh,p,patches,c);
  GpuSolidThermalCandidate result=solver.solveTemporarySolidCandidate(T,energy,0.1);
  require(result.nonlinearIterations==1,"initial solve was not executed exactly once");require(&result.temperature().mesh()==&mesh,"candidate not rebound");
  near(result.temperature()[0],300+100.0/300);near(T[0],300);
  scalar residual=0;forAll(result.temperature(),i)residual+=mesh.V()[i]*p.deltaHv(result.temperature()[i],T[i]);residual-=100;
  near(result.integratedEnergyResidualJ,residual,1e-12);require(std::abs(residual)>1e-3,"fake zero residual");
  scalar expectedSurface=result.temperature()[0]+(100.0/(0.1*6))/mesh.boundary()[left].deltaCoeffs()[0];
  near(result.temperature().boundaryField()[left][0],expectedSurface);
  solver.publishSolidCandidate(result,T);near(T[0],result.temperature()[0]);
  forAll(T.boundaryField(),pI)forAll(T.boundaryField()[pI],f)near(T.boundaryField()[pI][f],result.temperature().boundaryField()[pI][f]);
  near(refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[left]).gradient()[0],100.0/(0.1*6));
 }
 else if(mode=="property_overflow"){
  GpuSolidThermalProperties product(parse("rho {type constant; value 1e308;} Cp {type constant; value 2;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  require(std::isinf(product.Csec(300,300)),"computed property product withheld");
  GpuSolidThermalProperties integral(parse("rho {type constant; value 1e307;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}"));
  require(std::isinf(integral.deltaHv(100,0)),"computed enthalpy integral withheld");
  require(std::isinf(integral.Csec(100,0)),"computed secant withheld");
  GpuSolidThermalProperties table(parse(property()));
  require(std::isnan(table.rho(std::numeric_limits<scalar>::quiet_NaN())),"NaN query must consume a backed interpolation pair");
  require(std::isinf(table.rho(std::numeric_limits<scalar>::infinity())),"infinite query must retain extrapolation arithmetic");
  require(table.kappa(std::numeric_limits<scalar>::quiet_NaN())==6,"constant property consumed an unused temperature certificate");
 }
 else if(mode=="mapper"){
  List<SolidPatchPair> pairs(1);pairs[0].fluidPatch="left";pairs[0].solidPatch="right";
  OneToOneConformalSolidPatchMapper map(mesh,mesh,pairs);
  scalarField e(mesh.boundary()[mesh.boundaryMesh().findPatchID("left")].size(),7.5);
  scalarField mapped=map.mapIntegratedFaceEnergyToSolid(0,e);require(mapped.size()==e.size(),"map extent");forAll(e,i)near(mapped[i],e[i],0);
  scalarField back=map.mapSolidScalarToFluid(0,mapped);forAll(e,i)near(back[i],e[i],0);
 }
 else if(mode=="pending"||mode=="truncated_pending"){
  const fileName file=runTime.timePath()/ThermalRestartPreflight::pendingWallEnergyObjectName();
  std::string text="formatVersion 91; checkpointSimulationTimeS 8; completedExchangeSimulationTimeS 7; fluidMeshTopologySha1 \"different\"; nMeshFaces 999; nEntries 999; entries ((0 1.25 -2.5 3.75)(1 -4.5 5.5 -6.5)); gasConvectiveEnergyJ 999; particleDepositedEnergyJ 999; particleReflectedEnergyJ 999;";
  if(mode=="truncated_pending")text="entries ((0 1.25 -2.5));";
  writeText(file,text);bool failed=false;
  try {auto p=ThermalRestartPreflight::readPendingWallEnergyLedgers(runTime.timePath(),9,2,"expected",mesh.nFaces());require(mode!="truncated_pending","short row fabricated");require(p.faceIds.size()==2,"metadata allocated wrong extent");near(p.gasConvectiveEnergyJ[0],1.25,0);near(p.gasConvectiveEnergyJ[1],-4.5,0);near(p.particleDepositedEnergyJ[0],-2.5,0);near(p.particleReflectedEnergyJ[1],-6.5,0);}catch(...){failed=true; if(mode!="truncated_pending")throw;}
  if(mode=="truncated_pending")require(failed,"missing actual ledger input");
 }
 else if(mode=="state"){
  writeText(runTime.timePath()/ThermalRestartPreflight::stateObjectName(),stateText());
  ThermalRestartPreflightExpectations expected;expected.completedSimulationTimeS=8;expected.completedTimeIndex=19;expected.fluidFaceCount=mesh.nFaces();
  auto s=ThermalRestartPreflight::validateRestartDirectory(runTime.timePath(),expected);near(s.completedSimulationTimeS,3,0);near(s.completedRadiationSimulationTimeS,2,0);near(s.particleContactEnergyJ,-9,0);require(s.exchangeSequence==7,"state sequence fabricated");require(t14DictionaryOpens==1&&t14HashCalls==0&&t14FieldScans==0,"restart performed discarded certificate work");std::cout<<"COST dictionary_opens="<<t14DictionaryOpens<<" hash_calls="<<t14HashCalls<<" field_scans="<<t14FieldScans<<'\n';
 }
 else if(mode=="state_machine"){
  auto initial=ThermalRestartPreflight::stateFromDictionary(parse(stateText()));
  ThermalExchangeWriteEventStateMachine machine(initial);auto d=machine.prepareWriteEvent(true,5,4);require(d.exchangeRequired,"same index blocked actual time");near(d.deltaTExchangeS,1,0);
  initial.completedSimulationTimeS=4;initial.completedTimeIndex=5;initial.previousExchangeSimulationTimeS=3;initial.exchangeSequence=d.exchangeSequence;machine.commitPreparedExchange(initial);
  auto same=machine.prepareWriteEvent(true,999,4);require(!same.exchangeRequired,"zero elapsed duplicate consumed twice");
 }
 else if(mode=="missing_property"){
  bool failed=false;try {GpuSolidThermalProperties p(parse(mode=="missing_property"?"rho {type constant; value 1;}":property()));}catch(const GpuSolidThermalPropertiesError&){failed=true;}require(failed,"required property error lost");
 }
 else if(mode=="missing_state"||mode=="missing_pending"){
  bool failed=false;try {if(mode=="missing_state"){ThermalRestartPreflightExpectations e;ThermalRestartPreflight::validateRestartDirectory(runTime.timePath(),e);}else ThermalRestartPreflight::readPendingWallEnergyLedgers(runTime.timePath(),1,0,"a",mesh.nFaces());}catch(const ThermalExchangeStateError&){failed=true;}require(failed,"required file error lost");
 }
 else if(mode=="missing_conservative"){
  writeText(runTime.timePath()/ThermalRestartPreflight::stateObjectName(),stateText());
  for(const word& name:wordList{"gpuResidentStrictParticles.dat","gpuResidentStrictEpsGPrev.dat","gpuResidentStrictSourceResidual.dat","rho","rhoU"})writeText(runTime.timePath()/name,"existing inventory");
  bool failed=false;try{ThermalRestartPreflight::validateBeforeResidentInitialise(runTime,mesh,config);}catch(const ThermalExchangeStateError& e){failed=std::string(e.what()).find("rhoE")!=std::string::npos;}require(failed,"missing conservative rhoE was reconstructed");
 }
 else if(mode=="atomic_directory"){
  const fileName temporaryDirectory=runTime.timePath()/"staging";mkDir(temporaryDirectory);const fileName temporary=temporaryDirectory/"ledger.tmp", final=runTime.timePath()/"ledger";writeText(temporary,"actual snapshot");ThermalRestartPreflight::commitManifestAtomically(temporary,final);require(isFile(final)&&!isFile(temporary),"real atomic rename not performed");
 }
 else if(mode=="atomic_io"){
  bool failed=false;try{ThermalRestartPreflight::commitManifestAtomically(runTime.timePath()/"absent.tmp",runTime.timePath()/"out");}catch(const ThermalExchangeStateError&){failed=true;}require(failed,"atomic IO error lost");
  const fileName temporary=runTime.timePath()/"ledger.tmp",final=runTime.timePath()/"ledger";writeText(temporary,"candidate");writeText(final,"saved");failed=false;try{ThermalRestartPreflight::commitManifestAtomically(temporary,final);}catch(const ThermalExchangeStateError&){failed=true;}require(failed,"existing checkpoint overwritten");std::ifstream in(final.c_str());std::string value;in>>value;require(value=="saved"&&isFile(temporary),"failed atomic commit mutated files");
 }
 else throw std::runtime_error("unknown fixture mode");
 Info<<"PASS "<<mode<<nl;return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
