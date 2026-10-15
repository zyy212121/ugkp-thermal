from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess,runpy
APP=(SOURCE_ROOT / 'applications/CHMT');ROOT=APP.parents[1]
def storage_source():
    fixture=runpy.run_path(str((VALIDATION_ROOT / 'tests/gas_wall/test_wall_geometry.py')))['BASE']
    return fixture+r'''
#include "gpu/SharedGasDeviceStorage.cuh"
#include "configuration/SharedGasModel.H"
int main(){
 auto canonical=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model constant;coefficients(0.01 0.01);})");
 chmt::ModelConfig model;std::string error;assert(chmt::bindSharedGasModel(canonical,model,error));
 auto&p=model.physics;p.gasViscosity=.02;p.gasConductivity=1;p.wallModel.family=WallFamily::BoundaryLayer;
 p.wallModel.model=BoundaryLayerModel::ConstantTransport;p.wallWorkspaceSlots=2;
 const auto fixture=cubes();chmt::HostState h;auto&m=h.gasMesh;m.geometryVersion=1;m.volumes=fixture.volumes;
 for(auto v:fixture.points)m.points.push_back({v[0],v[1],v[2]});
 for(auto v:fixture.cellCentres)m.cellCentres.push_back({v[0],v[1],v[2]});
 std::vector<int> order;for(int f=0;f<int(fixture.owner.size());++f)if(fixture.neighbour[f]>=0)order.push_back(f);
 for(int f=0;f<int(fixture.owner.size());++f)if(fixture.neighbour[f]<0)order.push_back(f);
 int wall=-1;m.faceOffsets={0};m.cellFaceOffsets={0};
 for(int f:order){if(f==0)wall=int(m.owner.size());m.owner.push_back(fixture.owner[f]);m.neighbour.push_back(fixture.neighbour[f]);m.periodicPartner.push_back(-1);
  m.boundaryKind.push_back(f==0?chmt::BoundaryKind::Interface:fixture.neighbour[f]>=0?chmt::BoundaryKind::Internal:chmt::BoundaryKind::Slip);
  V area{},centre{};const int start=fixture.faceOffsets[f],end=fixture.faceOffsets[f+1];
  for(int i=start;i<end;++i){m.facePoints.push_back(fixture.facePoints[i]);const auto a=fixture.points[fixture.facePoints[i]],b=fixture.points[fixture.facePoints[i+1<end?i+1:start]];area=add(area,mul(cross(a,b),.5));centre=add(centre,mul(a,1./(end-start)));}
  m.faceOffsets.push_back(int(m.facePoints.size()));m.areaVectors.push_back({area[0],area[1],area[2]});m.faceCentres.push_back({centre[0],centre[1],centre[2]});}
 for(int c=0;c<3;++c){for(int f=0;f<int(m.owner.size());++f)if(m.owner[f]==c||m.neighbour[f]==c)m.cellFaces.push_back(f);m.cellFaceOffsets.push_back(int(m.cellFaces.size()));}
 m.boundaryPrimitive.resize(m.owner.size());h.surface.gasFace={wall};h.surface.area={1};
 for(int c=0;c<3;++c){chmt::GasPrimitive w;w.rho=1;w.temperature=700;w.Y[0]=1;w.velocity={5,0,0};h.gas.push_back(chmt::conservativeGas(w,1,p));}
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,canonical,{},h,error));auto&v=storage.hostView();
 assert(v.gasBoundaryLayerModel.input[0].quadrature.count==0&&!v.gasBoundaryLayerModel.input[0].quadrature.distance&&!v.gasBoundaryLayerModel.input[0].quadrature.volumeWeight);
 assert(v.gasBoundaryLayer.enabled&&v.gasBoundaryLayer.count==1);assert(v.gasBoundaryLayerModel.workspaceCount==0);assert(!v.gasBoundaryLayerModel.workspace);
 assert(v.gasBoundaryLayer.faceSlot[wall]==0&&v.gasBoundaryLayer.ownerSlot[0]==0&&v.gasBoundaryLayer.ownerSlot[1]==-1);
 chmt::WallKnot knot;knot.faces.resize(1);knot.faces[0].temperature=600;knot.faces[0].speciesRate[0]=.01;
 assert(storage.prepareWallBoundary(knot,h.surface,error));
 for(int c=0;c<3;++c){v.rho[c]=1;v.p[c]=1e5;v.Tgas[c]=700;v.Ux[c]=5;}
 v.Tgas[1]=900;
 assert(ugkwp::evaluateGasBoundaryLayerSlot(v,0,0));assert(v.gasBoundaryLayer.exchange[0].ready);
 assert(v.gasBoundaryLayerModel.input[0].temperature==600);assert(v.gasBoundaryLayerModel.input[0].matching.temperature==900);
 assert(v.gasBoundaryLayer.speciesFlux[0]==-.01);assert(v.gasBoundaryLayer.exchange[0].mass==-.01);
 assert(v.gasBoundaryLayerModel.output[0].reactionIntegral[0]==0);
 // A moving-geometry rebuild retains certified volume but no point allocation/upload.
 ++m.geometryVersion;for(auto& point:m.points)point.z*=1.01;for(auto& centre:m.cellCentres)centre.z*=1.01;for(auto& volume:m.volumes)volume*=1.01;
 assert(storage.prepareWallGeometry(m));const auto& q=v.gasBoundaryLayerModel.input[0].quadrature;
 assert(q.count==0&&!q.distance&&!q.volumeWeight&&near(q.volume,1.01)&&near(q.firstMoment,.51005));
 assert(storage.clearTrialStatus());assert(!v.gasBoundaryLayer.exchange[0].ready);
}
'''
def test_bounded_wall_storage_and_current_device_matching(tmp_path):
    stub=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_shared_device_storage.py')))['STUB']
    (tmp_path/'cuda_runtime.h').write_text(stub)
    source=storage_source()
    src=tmp_path/'storage.cpp';src.write_text(source);exe=tmp_path/'storage'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(tmp_path),'-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_completed_numeric_wall_download_and_real_copy_failures(tmp_path):
    stub=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_shared_device_storage.py')))['STUB']
    stub=stub.replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}',
        'inline int failRead=-1,reads=0;inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){if(kind==cudaMemcpyDeviceToHost){++reads;if(failRead==0){failRead=-1;return 3;}if(failRead>0)--failRead;}std::memcpy(d,s,n);return 0;}')
    (tmp_path/'cuda_runtime.h').write_text(stub)
    source=storage_source();source=source[:source.index(' // A moving-geometry rebuild')]+r'''
 h.surface.area[0]=0;assert(storage.prepareWallBoundary(knot,h.surface,error));
 assert(v.gasBoundaryLayerModel.input[0].massFlux[0]==0);
 h.surface.area[0]=1;assert(storage.prepareWallBoundary(knot,h.surface,error));
 v.Tgas[1]=std::numeric_limits<chmt::Real>::infinity();
 assert(ugkwp::evaluateGasBoundaryLayerSlot(v,0,0));
 assert(v.gasBoundaryLayer.exchange[0].ready&&v.gasBoundaryLayer.status[0]!=0);
 std::vector<ugkwp::gaswall::WallOutput<chmt::Real,chmt::Ns>> outputs;std::vector<chmt::GasWallMatchingSample> matching;
 reads=0;assert(storage.downloadWallOutputs(outputs,matching));assert(reads==3);
 assert(std::isinf(outputs[0].conductiveHeatFlux)&&outputs[0].traceTemperature==600);
 assert(std::isinf(matching[0].state.temperature)&&matching[0].pressure==100000&&matching[0].mechanicalPressure==0);
 for(int failed=0;failed<3;++failed){failRead=failed;assert(!storage.downloadWallOutputs(outputs,matching));}
 failRead=-1;v.gasBoundaryLayerModel.input[0].model.thermo.coefficients=nullptr;
 // evaluateSlot samples the actual current tables again; remove the true source.
 auto table=v.gasSpecies.thermo.coefficients;v.gasSpecies.thermo.coefficients=nullptr;
 assert(!ugkwp::evaluateGasBoundaryLayerSlot(v,0,0));assert(!v.gasBoundaryLayer.exchange[0].ready);
 assert(!storage.downloadWallOutputs(outputs,matching));assert(storage.wallModelFailed());
 v.gasSpecies.thermo.coefficients=table;
 assert(ugkwp::evaluateGasBoundaryLayerSlot(v,0,0));assert(storage.downloadWallOutputs(outputs,matching));
}
'''
    src=tmp_path/'download.cpp';src.write_text(source);exe=tmp_path/'download'
    cmd=['g++','-std=c++17','-O1','-I'+str(tmp_path),'-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd));built=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(built.stdout+built.stderr);assert built.returncode==0,built.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
