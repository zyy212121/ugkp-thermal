#!/usr/bin/env python3
"""Numerical metrics from actual production C++ kernels on CPU, never GPU evidence.

The CHMT boundary adapter uses the existing host CUDA-allocation shim. The wall
mathematics and geometric builder are unmodified production headers. References
are analytic formulas or independently discretized SciPy ODEs; no production
output is used to adjust reference values or tolerances.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import subprocess
from foam import ROOT
from metrics import _norm

class ArtifactBuilt(Exception):
    def __init__(self,metadata):self.metadata=metadata

LIMITS={'slip_heat_W':1e-9,'slip_mass_kg_s':1e-13,'slip_shear_N':1e-12,'outlet_pressure_Pa':1e-9,'outlet_state':1e-10,'constant_wall_relative':1e-10,'bvp_relative':.005,'bvp_species_balance':1e-9,'geometry_absolute':1e-10,'zero_species':1e-12,'ten_species_profile':.006}

def discrepancy(actual,target,scale=1.,limit=1e-10,units='dimensionless'):
    if not isinstance(actual,list):actual=[actual]
    if not isinstance(target,list):target=[target]
    if len(actual)!=len(target) or not actual:raise ValueError('metric shape mismatch')
    if not all(math.isfinite(x) for x in actual+target):raise ValueError('nonfinite production metric')
    norms=_norm([a-b for a,b in zip(actual,target)],scale)
    return dict(actual=actual,target=target,scale=scale,units=units,**norms,threshold=limit,status='PASS' if norms['linf']<=limit else 'FAIL')

def _command(args,log,cwd=None):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True)
    Path(log).write_text('$ '+' '.join(args)+'\n'+result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('command failed '+str(log))
    return result.stdout

def workspace_body(body,workspace_types=(),cuda=False):
    """Expand explicit slot declarations, never infer arbitrary C++ declarations."""
    seen=set()
    def declaration(match):
        index=int(match.group(1));name=match.group(2)
        if index>=len(workspace_types) or index in seen:raise ValueError('invalid or duplicate probe workspace slot')
        seen.add(index)
        if cuda:return f'auto& {name}=resetProbeWorkspace<{workspace_types[index]}>(scratch->workspace_{index})'
        return workspace_types[index]+' '+name
    result=re.sub(r'\bPROBE_WORKSPACE\s*\(\s*(\d+)\s*,\s*([A-Za-z_]\w*)\s*\)',declaration,body)
    if seen!=set(range(len(workspace_types))) or 'PROBE_WORKSPACE' in result:
        raise ValueError('missing or malformed probe workspace declaration')
    return result

def cuda_probe_source(base,body,workspace_types):
    if not workspace_types:raise ValueError('CUDA wall probes require explicit bounded workspace slots')
    device_base=base.replace(' Model(){',' __host__ __device__ Model(){').replace('bool near(', '__host__ __device__ bool near(')
    device_body=workspace_body(body,workspace_types,cuda=True).replace('std::fprintf(stderr,','std::printf(')
    storage='\n'.join(f' alignas({kind}) unsigned char workspace_{i}[sizeof({kind})];' for i,kind in enumerate(workspace_types))
    wrapper=r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <new>
#include <type_traits>
struct ProbeScratch {
STORAGE
};
static_assert(alignof(ProbeScratch)<=256,"cudaMalloc alignment must support probe scratch");
template<class T> __device__ __forceinline__ T& resetProbeWorkspace(void* storage){
 static_assert(std::is_trivially_destructible<T>::value,"probe slot reuse requires trivial destruction");
 return *::new(storage) T;
}
__device__ int productionProbe(ProbeScratch* scratch){BODY
return 0;}
__global__ void launchProbe(ProbeScratch* scratch,int* status){
 ::new(static_cast<void*>(scratch)) ProbeScratch;
 *status=productionProbe(scratch);
}
int main(){ProbeScratch* scratch=nullptr;int* status=nullptr;int h=-1;int result=0;cudaError_t err;
 if(cudaMalloc(&scratch,sizeof(ProbeScratch))!=cudaSuccess)return 11;
 if(cudaMalloc(&status,sizeof(int))!=cudaSuccess){cudaFree(scratch);return 12;}
 launchProbe<<<1,1>>>(scratch,status);
 if(cudaGetLastError()!=cudaSuccess)result=13;
 else if((err=cudaDeviceSynchronize())!=cudaSuccess){std::fprintf(stderr,"CUDA synchronization: %s\n",cudaGetErrorString(err));result=14;}
 else if(cudaMemcpy(&h,status,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)result=15;
 else result=h;
 cudaFree(status);cudaFree(scratch);return result;}
""".replace('STORAGE',storage).replace('BODY',device_body)
    return device_base+'\n'+wrapper

def _build_run(root,work,base,body,helper=None,backend='cpu',workspace_types=()):
    if backend.startswith('cuda') and helper:raise ValueError("native CUDA cannot use the host allocation shim")
    work.mkdir(parents=True)
    host_body=workspace_body(body,workspace_types)
    if helper:
        source=helper['prepare_backend'](work);source.write_text(source.read_text()+base+'\n#include <cstdio>\nint main(){'+host_body+'}\n')
        args=helper['compiler_flags'](work)+[str(source),str(root/'applications/CHMT/mesh/Geometry.C')]
    else:
        source=work/'probe.cpp';source.write_text(base+'\n#include <cstdio>\nint main(){'+host_body+'}\n')
        args=['g++','-std=c++17','-O2','-I'+str(root),'-I'+str(root/'common'),str(source)]
    if backend.startswith('cuda'):
        source=work/'probe.cu'
        source.write_text(cuda_probe_source(base,body,workspace_types))
        args=['nvcc','-std=c++17','-O2','-arch=sm_89','--ptxas-options=-v','-I'+str(root),'-I'+str(root/'common'),str(source)]
    binary=work/'probe';_command(args+['-o',str(binary)],work/'compile.log')
    metadata=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),compile_command=args+['-o',str(binary)],run_command=[str(binary)])
    if backend.startswith('cuda'):
        metadata.update(workspace_types=list(workspace_types),scratch_allocation='one cudaMalloc(sizeof(ProbeScratch)), aligned slots initialized in place at each declaration',cuda_stack_limit_override=False,
            ptxas_resource_report=[line.strip() for line in (work/'compile.log').read_text().splitlines() if 'ptxas info' in line or re.match(r'\s*\d+ bytes stack frame,',line)])
    if backend=='cuda-build':raise ArtifactBuilt(metadata)
    raw=_command([str(binary)],work/'execute.log')
    rows=[list(map(float,line.split())) for line in raw.splitlines() if line.strip()]
    if not rows or any(not row or not all(math.isfinite(v) for v in row) for row in rows):raise ValueError('empty or nonfinite production probe output')
    return rows,metadata

def require_rows(rows,lengths):
    if len(rows)!=len(lengths) or any(len(row)!=size or not all(math.isfinite(v) for v in row) for row,size in zip(rows,lengths)):
        raise ValueError('production probe row shape or finite-value contract failed')


def boundary(root,work,backend='cpu'):
    app=root/'applications/CHMT';helper=runpy.run_path(str(app/'tests/test_shared_backend_compile.py'));fixture=runpy.run_path(str(app/'tests/test_coupled_runtime_transactions.py'))['FIXTURE']
    body=r'''
 for(bool single:{false,true}){
 Fixture f;if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 f.model.physics.gasViscosity=.02;f.h.gas[0].momentum={7,5,-3};
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600+41.5;
 auto&m=f.h.gasMesh;m.boundaryKind.assign(2,BoundaryKind::Slip);m.thermalBoundary.assign(2,ThermalBoundaryKind::ZeroGradient);
 m.thermalBoundary[0]=ThermalBoundaryKind::FixedValue;m.boundaryPrimitive[0].temperature=300;
 SharedGasDeviceStorage storage;if(!storage.configure(f.model,f.gas,{},f.h,f.error))return 3;auto&v=storage.hostView();v.gasFluxScheme=1;v.gasReconstruction=0;blockDim.x=1;threadIdx.x=0;
 transport::recoverGasPrimitivesKernel(&v);transport::updateLegacyGasBoundaryMirrorKernel(&v,0);transport::updateRiemannBoundaryMirrorKernel(&v);
 v.gradUxX[0]=2;v.gradUyX[0]=3;v.gradUzX[0]=-4;
 double mass,mx,my,mz,energy;if(!transport::computeRiemannGasFaceFluxDevice<false>(v,0,mass,mx,my,mz,energy))return 4;
 std::printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",double(single),energy,mass,mx+v.p[0],my,mz);
 }
 for(bool single:{false,true}){
 Fixture f;if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 f.h.gas[0].momentum={7,5,-3};f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600+41.5;
 auto&m=f.h.gasMesh;m.boundaryKind.assign(2,BoundaryKind::Slip);m.boundaryKind[0]=BoundaryKind::Outlet;
 m.thermalBoundary.assign(2,ThermalBoundaryKind::ZeroGradient);m.boundaryPrimitive[0].pressure=90000;
 SharedGasDeviceStorage storage;if(!storage.configure(f.model,f.gas,{},f.h,f.error))return 5;auto&v=storage.hostView();v.gasFluxScheme=1;v.gasReconstruction=0;blockDim.x=1;threadIdx.x=0;
 transport::recoverGasPrimitivesKernel(&v);transport::updateLegacyGasBoundaryMirrorKernel(&v,0);transport::updateRiemannBoundaryMirrorKernel(&v);
 const auto owner=transport::gasCellPrimitive(v,0);const auto outlet=transport::riemannBoundaryState(v,0,owner);
 double mass,mx,my,mz,E,m0,x0,y0,z0,e0;if(!transport::computeRiemannGasFaceFluxDevice<false>(v,0,mass,mx,my,mz,E))return 6;
 std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g ",double(single),v.gasBoundaryP[0],v.riemannBoundaryP[0],outlet.p,outlet.T-owner.T,outlet.ux-owner.ux,outlet.uy-owner.uy);
 v.riemannBoundaryPFix[0]=0;v.riemannBoundaryP[0]=owner.p;if(!transport::computeRiemannGasFaceFluxDevice<false>(v,0,m0,x0,y0,z0,e0))return 7;
 std::printf("%.17g %.17g %.17g %.17g\n",mass,E,m0,e0);
 }
'''
    rows,build=_build_run(root,work,fixture,body,helper,backend);metrics={}
    require_rows(rows,[6,6,11,11])
    if [row[0] for row in rows]!=[0.,1.,0.,1.]:raise ValueError('boundary probe cases missing')
    for row in rows[:2]:
        prefix='single' if row[0] else 'mixture'
        metrics[prefix+'_slip_heat']=discrepancy(row[1],300.,limit=LIMITS['slip_heat_W'],units='W')
        metrics[prefix+'_slip_mass']=discrepancy(row[2],0.,limit=LIMITS['slip_mass_kg_s'],units='kg/s')
        metrics[prefix+'_slip_shear']=discrepancy(row[3:6],[0.]*3,limit=LIMITS['slip_shear_N'],units='N')
    for row in rows[2:]:
        prefix='single' if row[0] else 'mixture'
        metrics[prefix+'_outlet_pressure']=discrepancy(row[1:4],[90000.]*3,limit=LIMITS['outlet_pressure_Pa'],units='Pa')
        metrics[prefix+'_outlet_owner_state']=discrepancy(row[4:7],[0.]*3,limit=LIMITS['outlet_state'])
        mass_delta=row[7]-row[9];energy_delta=row[8]-row[10]
        metrics[prefix+'_pressure_flux_response']={'actual_mass_flux_kg_s':row[7],'owner_pressure_mass_flux_kg_s':row[9],'actual_energy_flux_W':row[8],'owner_pressure_energy_flux_W':row[10],'mass_flux_delta':mass_delta,'energy_flux_delta':energy_delta,'minimum_magnitude':1e-6,'status':'PASS' if abs(mass_delta)>1e-6 and abs(energy_delta)>1e-6 else 'FAIL','reference':'Pressure perturbation must affect the actual Riemann flux; this is a sensitivity check, not an independent exact flux-value reference.'}
    return metrics,build

def wall_constant(root,work,backend='cpu'):
    base=runpy.run_path(str(root/'tests/gas_wall/test_wall_model.py'))['BASE']
    rows,build=_build_run(root,work,base,r'''
 Model m;WallModelConfig<double>c;c.model=BoundaryLayerModel::ConstantTransport;PROBE_WORKSPACE(0,w);WallOutput<double,2>o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s))return 2;std::printf("%.17g %.17g\n",o.traction[0],o.conductiveHeatFlux);
 m.in.matching.velocity[0]=0;m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 if(!evaluateWallModel(m.in,c,w,o,s))return 3;std::printf("%.17g\n",o.conductiveHeatFlux);
''',backend=backend,workspace_types=('WallWorkspace<double,2,32>',))
    require_rows(rows,[2,1])
    targets=[.02,-800.1,-.001*1000*200/math.expm1(.001*1000*.01/.04)]
    values=rows[0]+rows[1]
    return {name:discrepancy(a,b,abs(b),LIMITS['constant_wall_relative'],unit) for name,a,b,unit in zip(['shear','viscous_heating_flux','blowing_heat_flux'],values,targets,['Pa','W/m2','W/m2'])},build

def bvp_reference():
    import numpy as np
    from scipy.integrate import solve_bvp
    # Nondimensionalize before collocation to avoid dimensional 1e5-J/kg terms
    # corrupting residual conditioning. Physical values are restored below.
    scales=np.array([1.,.01,1000.,1000.]);L=.01;gasR=8.31446261815324/.028
    def ode(x,q):
        Y,j,T,F=q*scales[:,None];rho=101325/(gasR*T)
        physical=np.vstack((-j/(rho*(1e-4*(1-Y)+3e-4*Y)),-rho*50*Y,(1e5*j-F)/.04,np.zeros_like(x)))
        return physical*L/scales[:,None]
    def bc(a,b):return np.array([a[1],a[2]-.5,b[0]-.7,b[2]-.7])
    x=np.linspace(0,1,160);guess=np.vstack((np.full_like(x,.5),-.1*x,.5+.2*x,np.full_like(x,-.8)))
    ref=solve_bvp(ode,bc,x,guess,tol=1e-10,max_nodes=20000)
    if not ref.success:raise RuntimeError('independent SciPy BVP did not converge: '+ref.message)
    class PhysicalReference:
        y=ref.y*scales[:,None]
        @staticmethod
        def sol(y):return ref.sol(np.asarray(y)/L)*scales[:,None]
    return PhysicalReference(),{'method':'scipy.integrate.solve_bvp fourth-order collocation with nondimensional residuals','tolerance':1e-10,'nodes':len(ref.x),'max_rms_residual':float(max(ref.rms_residuals)),'equations':'Yprime=-j/(rho*(D_A*(1-Y)+D_B*Y)); jprime=-rho*k*Y; Tprime=(formation*j-F)/lambda; Fprime=0; rho=p/(R*T)','boundary_conditions':'j(0)=0, T(0)=500K, Y(L)=0.7, T(L)=700K; L=0.01m','parameters':{'p_Pa':101325,'W_kg_mol':.028,'D_A_m2_s':1e-4,'D_B_m2_s':3e-4,'rate_s^-1':50,'formation_A_J_kg':1e5,'lambda_W_m_K':.04}}

def wall_bvp(root,work,backend='cpu'):
    ref,meta=bvp_reference();base=runpy.run_path(str(root/'tests/gas_wall/test_wall_model.py'))['BASE']
    rows,build=_build_run(root,work,base,r'''
 Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=0;m.coef[2]=100000;m.rx[0].highRate.preExponential=50;m.in.model.mode=GasMode::MixtureChemistry;m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 for(int nodes:{24,48,96}){WallModelConfig<double>c;c.nodes=nodes;c.maxIterations=100;c.relativeTolerance=1e-10;PROBE_WORKSPACE(0,w);WallOutput<double,2>o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::fprintf(stderr,"wall status %d iteration %d\n",int(s.code),s.iteration);return 2;}
 std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",double(nodes),o.traceMassFraction[0],o.conductiveHeatFlux,o.matchingSpeciesFlux[0],o.reactionIntegral[0],o.speciesBalanceResidual[0],o.matchingSpeciesFlux[1]);
 for(int i=0;i<nodes;++i)std::printf("%.17g %.17g %.17g %.17g\n",double(nodes),w.y[i],reactingWallProfileTemperature(m.in,w,i),1-w.state[i][3]);}
''',backend=backend,workspace_types=('WallWorkspace<double,2,128>',))
    require_rows(rows,[7]+[4]*24+[7]+[4]*48+[7]+[4]*96)
    if [r[0] for r in rows if len(r)==7]!=[24.,48.,96.]:raise ValueError('BVP refinement snapshots missing')
    target=[float(ref.y[0,0]),float(ref.y[3,0]),float(ref.y[1,-1])];metrics={};refinement=[]
    for row in [r for r in rows if len(r)==7]:
        profile=[r for r in rows if len(r)==4 and r[0]==row[0]]
        reference_profile=ref.sol([r[1] for r in profile]);actual_T=[r[2] for r in profile];actual_Y=[r[3] for r in profile]
        errors={name:discrepancy(row[i+1],target[i],abs(target[i]),LIMITS['bvp_relative'],unit) for i,(name,unit) in enumerate(zip(['wall_Y_A','wall_heat_flux','matching_J_A'],['1','W/m2','kg/m2/s']))}
        errors['species_integral_balance']=discrepancy(row[3]-row[4],0.,limit=LIMITS['bvp_species_balance'],units='kg/m2/s')
        errors['net_matching_species_flux']=discrepancy(row[3]+row[6],0.,limit=LIMITS['bvp_species_balance'],units='kg/m2/s')
        errors['T_profile']=discrepancy(actual_T,reference_profile[2].tolist(),700.,LIMITS['bvp_relative'],'K')
        errors['Y_A_profile']=discrepancy(actual_Y,reference_profile[0].tolist(),.7,LIMITS['bvp_relative'],'1')
        energies=[]
        for left,right in zip(profile,profile[1:]):
            dy=right[1]-left[1];rho=.5*(101325/(8.31446261815324/.028*left[2])+101325/(8.31446261815324/.028*right[2]));Y=.5*(left[3]+right[3])
            J=-rho*(1e-4*(1-Y)+3e-4*Y)*(right[3]-left[3])/dy
            energies.append(1e5*J-.04*(right[2]-left[2])/dy)
        errors['formation_inclusive_energy_flux_profile']=discrepancy(energies,[target[1]]*len(energies),abs(target[1]),LIMITS['bvp_relative'],'W/m2')
        errors['total_energy_flux_conservation']=discrepancy(energies,[energies[0]]*len(energies),abs(target[1]),1e-7,'W/m2')
        refinement.append({'nodes':int(row[0]),'production_config_supported':int(row[0])<=128,'role':'production_config' if int(row[0])<=128 else 'diagnostic_template_refinement_only','metrics':errors})
    metrics=next(r['metrics'] for r in refinement if r['nodes']==96);build.update(independent_reference=meta,refinement=refinement,acceptance_nodes=96,production_nodes_limit=128)
    return metrics,build

def geometry(root,work,backend='cpu'):
    base=runpy.run_path(str(root/'tests/gas_wall/test_wall_geometry.py'))['BASE']
    rows,build=_build_run(root,work,base,r'''
 int shape=0;for(auto m:{cubes(),cubes(3,.2),tetra()}){WallGeometryDescriptor<double>d;WallStatus s;if(!buildWallGeometry(m,0,d,s))return 2;
 double V=0,Y=0,Y2=0;std::array<double,3>P={};for(size_t j=0;j<d.distance.size();++j){V+=d.volumeWeight[j];Y+=d.volumeWeight[j]*d.distance[j];Y2+=d.volumeWeight[j]*d.distance[j]*d.distance[j];for(int k=0;k<3;++k)P[k]+=d.volumeWeight[j]*d.quadraturePoints[j][k];}
 std::printf("%d %.17g %.17g %.17g %.17g %.17g %.17g\n",shape++,V,Y,Y2,P[0],P[1],P[2]);}
''',backend=backend)
    require_rows(rows,[7]*3)
    if [r[0] for r in rows]!=[0.,1.,2.]:raise ValueError('geometry probe cases missing')
    targets=[[1,.5,1/3,.5,.5,.5],[1,.5,1/3,.6,.5,.5],[2/3,1/6,1/15,(2/3)*.55,(2/3)*.575,1/6]]
    return {name:discrepancy(row[1:],target,limit=LIMITS['geometry_absolute'],units='mixed: volume m3, first moments m4, second moment m5; unit-scale shapes') for name,row,target in zip(['cube','skew_prism','tetrahedron'],rows,targets)},build

def wall_ten_species(root,work,backend='cpu'):
    base=runpy.run_path(str(root/'tests/gas_wall/test_wall_model.py'))['BASE']
    rows,build=_build_run(root,work,base,r"""
 SpeciesThermoData<double>sp[10];double coef[30]={},elements[10],basis[10]={};GasReactionData<double>rx;GasStoichTerm<double>re,pr;WallInput<double,10>in;
 for(int j=0;j<10;++j){sp[j].coefficientOffset=j*3;sp[j].molarMass=.028;sp[j].minTemperature=200;sp[j].maxTemperature=4000;sp[j].referencePressure=101325;coef[j*3]=1000;elements[j]=1;in.model.diffusivity[j]=1e-4;}
 in.model.thermo.species=sp;in.model.thermo.coefficients=coef;in.model.thermo.coefficientCount=30;in.model.thermo.elementComposition=elements;in.model.thermo.elementCount=1;in.model.thermo.speciesOrderHash=1;
 rx.reactantCount=1;rx.productCount=1;rx.highRate.preExponential=2;re.species=0;re.coefficient=1;pr.species=9;pr.coefficient=1;basis[0]=-1;basis[9]=1;
 auto&m=in.model.mechanism;m.reactions=&rx;m.reactionCount=1;m.reactants=&re;m.reactantTermCount=1;m.products=&pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.pressure=101325;in.temperature=in.matching.temperature=500;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;in.matching.massFraction[0]=1;in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.mode=GasMode::MixtureChemistry;
 WallModelConfig<double>c;c.nodes=48;PROBE_WORKSPACE(0,w);WallOutput<double,10>o;WallStatus st;if(!evaluateWallModel(in,c,w,o,st))return 2;
 for(int j=0;j<10;++j)std::printf("%.17g ",o.traceMassFraction[j]);std::printf("\n%.17g %.17g\n",o.conductiveHeatFlux,o.matchingSpeciesFlux[0]-o.reactionIntegral[0]);
""",backend=backend,workspace_types=('WallWorkspace<double,10,64>',))
    require_rows(rows,[10,2])
    A=1/math.cosh(math.sqrt(2/1e-4)*.01);targets=[A]+[0.]*8+[1-A]
    return {'trace_species':discrepancy(rows[0],targets,limit=LIMITS['ten_species_profile']),
            'inert_species_exact_zero':discrepancy(rows[0][1:9],[0.]*8,limit=LIMITS['zero_species']),
            'species_sum':discrepancy(sum(rows[0]),1.,limit=1e-12),
            'isothermal_heat_flux':discrepancy(rows[1][0],0.,limit=1e-6,units='W/m2'),
            'species_conservation':discrepancy(rows[1][1],0.,limit=1e-9,units='kg/m2/s')},build

def wall_sst_robustness(root,work,backend='cpu'):
    base=runpy.run_path(str(root/'tests/gas_wall/test_wall_model.py'))['BASE']
    rows,build=_build_run(root,work,base,r"""
 for(double mass:{0.,1e-9,.001}){Model m;m.in.matching.velocity[0]=30;m.in.temperature=m.in.matching.temperature=500;m.in.matching.k=.5;m.in.matching.omega=200;m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
 double distances[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature.distance=distances;m.in.quadrature.volumeWeight=weights;m.in.quadrature.count=2;m.in.quadrature.volume=4e-7;m.in.quadrature.firstMoment=1.6e-9;
 WallModelConfig<double>c;c.enableSst=true;c.nodes=32;c.maxIterations=100;c.relativeTolerance=1e-8;PROBE_WORKSPACE(0,w);WallOutput<double,2>o;WallStatus st;bool ok=evaluateWallModel(m.in,c,w,o,st);
 std::printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",mass,double(ok),double(st.code),double(st.iteration),o.ownerOmega,o.traction[0]);}
""",backend=backend,workspace_types=('WallWorkspace<double,2,48>',))
    expected=[0.,1e-9,.001]
    if len(rows)!=3 or any(len(row)!=6 or not all(math.isfinite(v) for v in row) for row in rows):raise ValueError('invalid SST probe shape or nonfinite values')
    if [row[0] for row in rows]!=expected:raise ValueError('missing or reordered SST probe cases')
    result={}
    for row in rows:
        name='mass_flux_'+format(row[0],'.8g');result[name]=discrepancy(row[1],1.,limit=0.)
        result[name].update(wall_status_code=int(row[2]),iterations=int(row[3]),owner_omega=row[4],traction_Pa=row[5],reference='Admissible declared SST matching state must converge; this is robustness only, not an independent SST accuracy solution.')
    return result,build

def wall_profile_quadrature(root,work,backend='cpu'):
    if backend!='cpu':raise ValueError('profile geometry construction is a CPU utility')
    source=root/'tests/gas_wall/test_wall_geometry_profile_probe.cpp'
    work.mkdir(parents=True);binary=work/'probe'
    command=['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(root),'-I'+str(root/'common'),str(source),'-o',str(binary)]
    _command(command,work/'compile.log')
    raw=_command([str(binary)],work/'execute.log');rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    expected={(shape,nodes) for shape in ('cube','skew_prism','tetra') for nodes in (24,48,128)}
    if len(rows)!=9 or {(r['geometry'],r['profile_nodes']) for r in rows}!=expected:raise ValueError('profile quadrature cases missing')
    result={}
    for row in rows:
        if not all(isinstance(v,(float,int)) and math.isfinite(v) for k,v in row.items() if k!='geometry'):raise ValueError('nonfinite quadrature evidence')
        if row['quadrature_points']<=0 or row['quadrature_points']>row['point_bound']:raise ValueError('quadrature cost bound exceeded')
        key=row['geometry']+'_N'+str(row['profile_nodes'])
        result[key]=discrepancy(row['source_integral'],row['independent_exact'],1+abs(row['independent_exact']),2e-11,'unit-scale piecewise-linear source volume integral')
        result[key].update(quadrature_points=row['quadrature_points'],point_bound=row['point_bound'])
    return result,dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),compile_command=command,run_command=[str(binary)],reference='Independent exact polynomial antiderivative of the declared piecewise-linear source times cube/skew/tetra section area; positive weights and volume/xyz moments independently asserted.')


JOBS={'boundary_slip_outlet':boundary,'wall_constant_limit':wall_constant,'wall_finite_rate_bvp':wall_bvp,'wall_polyhedral_geometry':geometry,'wall_ten_species':wall_ten_species,'wall_sst_robustness':wall_sst_robustness,'wall_profile_quadrature':wall_profile_quadrature}

def run(output,root=ROOT,names=None,backend='cpu'):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError('host output directory must be fresh')
    output.mkdir(parents=True);sha=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    report={'category':1,'backend':'CPU_PRODUCTION_KERNEL_HOST_SHIM' if backend=='cpu' else 'CUDA_NATIVE','source_commit':sha,'source_root':str(root),'source_dirty':subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True,check=True).stdout.splitlines(),'precision':'float64','cases':[],'gate':'hard','tolerance_policy':'Versioned constants; never fitted or loosened by candidate outputs.'}
    for name in names or JOBS:
        item={'id':name,'cpu_reference':'NOT_RUN','production_execution':'NOT_RUN','status':'NOT_RUN','native_gpu':'NOT_RUN'}
        try:
            if backend.startswith('cuda') and name in ('boundary_slip_outlet','wall_polyhedral_geometry','wall_profile_quadrature'):item.update(status='UNSUPPORTED',reason='This adapter is CPU-only (host CUDA shim or host geometry builder). Full native solver cases are a distinct GPU entry; no substitution is performed.')
            elif backend=='cuda' and __import__('run').cuda_probe()['status']!='AVAILABLE':item['reason']='CUDA device unavailable; actual native kernel execution NOT_RUN'
            elif not shutil.which('nvcc' if backend.startswith('cuda') else 'g++'):item['reason']='requested compiler unavailable'
            elif name.startswith('wall_') and not (root/'common/gasWall/WallModel.H').exists():item.update(status='UNSUPPORTED',reason='Production wall feature is absent from this source revision')
            else:
                metrics,build=JOBS[name](root,output/name,backend)
                if not metrics:raise ValueError('empty production metrics cannot pass')
                item.update(metrics=metrics,build=build,cpu_reference='COMPLETED',production_execution='COMPLETED',status='FAIL' if any(m['status']=='FAIL' for m in metrics.values()) else 'PASS',native_gpu='COMPLETED' if backend=='cuda' else 'NOT_RUN')
        except ImportError as error:item.update(status='NOT_RUN',reason='Required independent reference dependency unavailable: '+str(error))
        except ArtifactBuilt as artifact:item.update(status='NOT_RUN',reason='Native CUDA probe compiled/linked only; no GPU was executed.',build=artifact.metadata,native_compilation='PASS')
        except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as e:item.update(status='FAIL',reason=str(e))
        report['cases'].append(item)
    report['counts']={status:sum(x['status']==status for x in report['cases']) for status in ('PASS','FAIL','NOT_RUN','UNSUPPORTED')}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--source-root',default=ROOT,type=Path);p.add_argument('--cases',nargs='+',choices=JOBS);p.add_argument('--backend',choices=['cpu','cuda','cuda-build'],default='cpu');p.add_argument('--required',action='store_true');a=p.parse_args()
    result=run(a.output,a.source_root,a.cases,a.backend);print(json.dumps(result,indent=2,sort_keys=True))
    if result['counts']['FAIL']:raise SystemExit(1)
    if result['counts']['NOT_RUN'] or result['counts']['UNSUPPORTED']:raise SystemExit(3)
if __name__=='__main__':main()
