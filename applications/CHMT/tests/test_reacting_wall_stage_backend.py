"""Actual backend hooks with sequential host execution of a sparse CUDA worker."""
from pathlib import Path
import subprocess,runpy
APP=Path(__file__).resolve().parents[1]
def test_device_profile_is_prepared_once_then_split_into_material_packets(tmp_path):
    helpers=runpy.run_path(str(APP/'tests/test_shared_backend_compile.py'))
    source=helpers['prepare_backend'](tmp_path)
    fixture=runpy.run_path(str(APP/'tests/test_reacting_wall_storage.py'))['storage_source']()
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
 assert(chmt::prepareCoupledBoundaryLayer(b.get(),.0001,0)==0);
 assert(b->preparedMatching[0].state.temperature==900);
 const auto layer=b->preparedWallLayers[0];
 // A later packet assembly is prohibited from invoking the core again.
 b->model.physics.wallModel.maxIterations=0;b->model.physics.gasViscosity=0;
 assert(chmt::applyCoupledFaces(b.get(),.0001,0)==0);
 assert(b->faceRates[0].layer.conductiveHeatFlux==layer.conductiveHeatFlux);
 assert(std::abs(b->faceRates[0].primary.mass-1e-6)<1e-18);
 assert(std::abs(v.gasPhiRho[wall]+.01)<1e-14);
 assert(chmt::accountCoupledFaces(b.get(),.0001)==0);
 assert(std::abs(b->record.packets[0].mass-1e-6)<1e-18);
 assert(b->record.packets[0].energy==b->faceRates[0].primary.energy);
 assert(b->trial.gasChemistryAudit.empty());
 assert(chmt::applyCoupledFaces(b.get(),.0001,.0001)!=0); // cannot reuse a different physical stage.
 assert(chmt::rollbackGasWindow(*b,error));assert(chmt::beginGasWindow(*b,program,error));
 // A stationary profile failure cannot be repaired by repeatedly halving gas dt.
 v.gasMu=-1;assert(b->storage.refreshView());
 chmt::GasMicroReport report;assert(!chmt::advanceGasMicrostep(*b,.0001,report,error));
 assert(!report.recoverable);assert(report.rejectedTrials==0);assert(error.find("wall code")!=std::string::npos);
 assert(b->trial.time==0);
}
'''
    source.write_text(source.read_text()+fixture+body)
    exe=tmp_path/'stage'
    subprocess.run(helpers['compiler_flags'](tmp_path)+['-I'+str(APP.parents[1]),str(source),str(APP/'mesh/Geometry.C'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
