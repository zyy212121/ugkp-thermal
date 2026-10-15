"""Actual backend hooks with sequential host execution of a sparse CUDA worker."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess,runpy
APP=(SOURCE_ROOT / 'applications/CHMT')
def test_device_profile_is_prepared_once_then_split_into_material_packets(tmp_path):
    helpers=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_shared_backend_compile.py')))
    source=helpers['prepare_backend'](tmp_path)
    stub=tmp_path/'cuda_runtime.h'
    stub.write_text(stub.read_text().replace('inline int cudaGetLastError(){return 0;}', 'inline int launchFailure=0;inline int cudaGetLastError(){int result=launchFailure;launchFailure=0;return result;}'))
    fixture=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_reacting_wall_storage.py')))['storage_source']()
    fixture=fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
    body=r'''
 h.solid.resize(1);h.solid[0].condensed[0]=1;h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.surface.normal={{0,0,1}};
 p.minDt=1e-12;p.maxDt=.001;p.cfl=.4;
 std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> b(chmt::createBackend(model,canonical,{},chmt::GasExecutionOptions{},h,error),chmt::destroyBackend);
 assert(b);chmt::WallProgram program;program.interval.end=.01;program.surface=h.surface;
 chmt::WallKnot a,z;a.time=0;z.time=.01;a.gasPoints=z.gasPoints=m.points;
 chmt::WallFaceSample sample;sample.temperature=600;sample.speciesRate[0]=.01;sample.condensedRate[0]=.01;
 a.faces=z.faces={sample};program.knots={a,z};assert(chmt::beginGasWindow(*b,program,error));
 b->microTime=0;b->microDt=.0001;assert(chmt::transport::CoupledTrialPolicy::begin(b.get())==0);
 auto&v=b->storage.hostView();for(int c=0;c<3;++c){v.rho[c]=1;v.p[c]=1e5;v.Tgas[c]=700;v.Ux[c]=5;v.gasSpecies.soundSpeed[c]=400;}
 v.Tgas[1]=900;
 blockIdx.x=0;blockDim.x=1;threadIdx.x=0;
 launchFailure=3;assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)!=0);assert(!b->preparedWallReady);
 assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->preparedMatching[0].state.temperature==900);
 const auto layer=b->preparedWallLayers[0];
 blockIdx.x=wall;chmt::transport::computeGasInternalFaceFluxKernel<false>(b->deviceState,.0001);blockIdx.x=0;
 // A later packet assembly is prohibited from invoking the core again.
 b->model.physics.wallModel.maxIterations=0;b->model.physics.gasViscosity=0;
 assert(chmt::applyCoupledFaces(b.get(),.0001,0)==0);
 assert(b->faceRates[0].layer.conductiveHeatFlux==layer.conductiveHeatFlux);
 assert(std::abs(b->faceRates[0].primary.mass-1e-6)<1e-18);
 assert(std::abs(v.gasPhiRho[wall]+.01)<1e-14);
 // Apply the real positivity publication between packet construction and
 // material ledger accounting, matching the production schedule.
 blockIdx.x=wall;chmt::transport::applyGasFluxPositivityScaleKernel(b->deviceState);blockIdx.x=0;
 assert(std::abs(v.gasPhiRho[wall]+.01)<1e-14);
 assert(chmt::accountCoupledFaces(b.get(),.0001)==0);
 assert(std::abs(b->record.packets[0].mass-1e-6)<1e-18);
 assert(b->record.packets[0].energy==b->faceRates[0].primary.energy);
 assert(b->trial.gasChemistryAudit.empty());
 // A different stage needs the real producer and dependent mechanical flux.
 b->gasStageIndex=1;v.Tgas[1]=1100;v.p[0]=130001;
 assert(chmt::applyCoupledFaces(b.get(),.0001,.0001)==0);
 assert(b->preparedMatching[0].state.temperature==1100);
 assert(b->faceRates[0].layer.conductiveHeatFlux!=layer.conductiveHeatFlux);
 assert(std::abs(b->preparedMatching[0].mechanicalPressure-130001)<1e-8);
 // Supplied physical wall velocity does not invent a mesh sweep/work channel.
 for(auto& knot:b->program.knots)knot.faces[0].normalVelocity=knot.faces[0].solidNormalVelocity=.25;
 assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,.0001)==0);
 blockIdx.x=wall;chmt::transport::computeGasInternalFaceFluxKernel<false>(b->deviceState,.0001);blockIdx.x=0;
 assert(chmt::applyCoupledFaces(b.get(),.0001,.0001)==0);
 assert(b->faceRates[0].primary.pressureWork==0);
 assert(std::abs(b->faceRates[0].trace.velocity.z-(.25+.01/b->faceRates[0].trace.rho))<1e-12);
 for(double sweep:b->gasStage.sweptVolume)assert(sweep==0);
 assert(chmt::rollbackGasWindow(*b,error));assert(chmt::beginGasWindow(*b,program,error));
 // Numeric wall diagnostics complete the actual stage; outer transaction
 // status cleanup belongs to the separate final-consumer integration.
 b->microTime=0;b->microDt=.0001;assert(chmt::transport::CoupledTrialPolicy::begin(b.get())==0);
 v.gasMu=std::numeric_limits<double>::quiet_NaN();assert(b->storage.refreshView());
 blockIdx.x=0;assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->preparedWallReady&&std::isnan(b->preparedWallLayers[0].conductiveHeatFlux));
 blockIdx.x=wall;chmt::transport::computeGasInternalFaceFluxKernel<false>(b->deviceState,.0001);blockIdx.x=0;
 assert(chmt::applyCoupledFaces(b.get(),.0001,0)==0);
 assert(std::isnan(b->faceRates[0].primary.energy)&&std::isnan(b->rawInterface[0].energy));
 assert(b->faceRates[0].primary.species[0]==.0001*.01);

}
'''
    instrumented=source.read_text().replace('transport::evaluateGasBoundaryLayerKernel(', '++profileCalls;transport::evaluateGasBoundaryLayerKernel(')
    body=body.replace('const auto layer=b->preparedWallLayers[0];', 'const auto layer=b->preparedWallLayers[0];const int preparedCalls=profileCalls;')
    body=body.replace('assert(b->faceRates[0].layer.conductiveHeatFlux==layer.conductiveHeatFlux);', 'assert(b->faceRates[0].layer.conductiveHeatFlux==layer.conductiveHeatFlux);assert(profileCalls==preparedCalls);')
    body=body.replace('assert(b->preparedMatching[0].state.temperature==1100);', 'assert(b->preparedMatching[0].state.temperature==1100);assert(profileCalls==preparedCalls+1);')
    source.write_text('int profileCalls=0;\n'+instrumented+fixture+body)
    exe=tmp_path/'stage'
    subprocess.run(helpers['compiler_flags'](tmp_path)+['-I'+str(APP.parents[1]),str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
