"""Original retry-owned leaves: exact status and publication, real kernels."""
import re
import pytest
from test_native_original_success import original_native_root, source, build_run, ROOT


def owner_call(root,name,args):

    files=list((root/'common/operators').glob('*.cuh'))
    for path in files:
        match=re.search(r'(?:void|bool|ChemistryStatus|GasPrimDevice)\s+'+name+r'\s*\([^)]*\)',path.read_text())
        if match:
            return name+'('+args+(',true' if 'nativeRefinement' in match.group() else '')+');'
    raise AssertionError(name)

@pytest.mark.parametrize('case', ['recovery_negative','recovery_closure','divergence','blend','chemistry_carry','periodic','wall_scale','wall_source','reconstruction','wall_omega','boundary_eos','diffusion_face','publisher','subgrid','boundary_temperature','subgrid_temperature'])
def test_original_retry_status_and_unpublished_values(tmp_path,original_native_root,case):
    def body(root):
        prefix=source(root).split('int main(){for(double dt:')[0]
        setup=''
        if case=='recovery_negative':
            setup='s.gasSpecies.rho[0]=-.1;s.gasSpecies.rho[2]=1.1;'+owner_call(root,'recoverGasPrimitiveCell','s,0')
        elif case=='recovery_closure':
            setup='s.gasSpecies.rho[0]=.8;'+owner_call(root,'recoverGasPrimitiveCell','s,0')
        elif case=='divergence':
            setup='s.gasPhiRho[0]=2000;s.gasSpecies.flux[0]=1400;s.gasSpecies.flux[3]=600;'+owner_call(root,'applyGasFluxDivergenceByCellKernel','&s,.001')
        elif case=='blend':
            setup='for(int k=0;k<4;++k)s.gasSpecies.initial[k]=s.gasSpecies.rho[k];s.rhoNext[0]=1;s.gasSpecies.initial[0]=-.1;s.gasSpecies.initial[2]=1.1;'+owner_call(root,'blendGasConservativeStateKernel','&s,.5,.5')
        elif case=='periodic':
            setup='s.facePeriodicPair[1]=2;s.facePeriodicPair[2]=1;s.faceNeighbour[1]=1;s.faceNeighbour[2]=0;s.gasPhiRho[1]=1;s.gasPhiRho[2]=-1;s.gasSpecies.flux[1]=2;s.gasSpecies.flux[4]=2;s.gasSpecies.flux[2]=-2;s.gasSpecies.flux[5]=-2;threadIdx.x=1;'+owner_call(root,'enforcePeriodicGasFluxAntisymmetryKernel','&s')
        elif case in ('wall_scale','wall_source','wall_omega'):
            from test_boundary_layer_operators import SETUP
            prefix+=SETUP.replace('void wall(State&s)','void wall(DeviceState&s)')
            setup='wall(s);'
            if case=='wall_scale':
                setup+='s.gasPhiRho[1]=1;s.gasSpecies.flux[1]=1;s.gasFluxPositivityScale[0]=.5;s.gasSpecies.positivityScale[0]=.5;threadIdx.x=1;'+owner_call(root,'applyGasFluxPositivityScaleKernel','&s')
            elif case=='wall_omega':
                setup+='s.gasBoundaryLayer.sst[0].ownerOmega=0;'+owner_call(root,'applySstWallFunctionStateCell','s,0,false')
            else:
                setup+='s.sstMaxSourceNumber=0;'+owner_call(root,'applySstFluxAndSourceKernel','&s,.1')
        elif case=='reconstruction':
            setup='s.gasReconstruction=1;s.p[0]=-100;GasPrimDevice result{11,12,13,14,15,16};Real y[2]={17,18};auto ok='+owner_call(root,'reconstructGasMixtureFace','s,0,1,result,y')+'std::printf("RECON %d %a %a %a %a\\n",ok,result.rho,result.T,y[0],y[1]);'
        elif case in ('boundary_eos','boundary_temperature'):
            setup='s.riemannBoundaryKind[1]=0;s.riemannBoundaryRhoFix[1]=s.riemannBoundaryPFix[1]=s.riemannBoundaryTFix[1]=1;s.riemannBoundaryRho[1]=3;s.riemannBoundaryP[1]=1e5;s.riemannBoundaryT[1]=700;auto face='+owner_call(root,'riemannBoundaryState','s,1,gasCellPrimitive(s,0),nullptr')+'std::printf("EOS %a %a\\n",face.rho,face.T);'
            if case=='boundary_temperature':setup=setup.replace('s.riemannBoundaryRhoFix[1]=s.riemannBoundaryPFix[1]=s.riemannBoundaryTFix[1]=1;','s.riemannBoundaryRhoFix[1]=0;s.riemannBoundaryPFix[1]=s.riemannBoundaryTFix[1]=1;').replace('s.riemannBoundaryT[1]=700;','s.riemannBoundaryT[1]=5000;')
        elif case=='diffusion_face':
            setup='s.gasSpecies.rho[0]+=.000001;s.gasSpecies.rho[1]+=.000001;Real m,x,y,z,e;'
            helper=(root/'common/operators/computeRiemannGasFaceFluxDevice.cuh').read_text()
            setup+='auto ok=computeRiemannGasFaceFluxDevice<false>(s,0,m,x,y,z,e'+(',true' if 'nativeRefinement' in helper else '')+');std::printf("DIFF %d %d %a %a\\n",ok,s.gasSpecies.faceStatus[0],m,e);'
        elif case in ('subgrid','subgrid_temperature'):
            setup='s.turbulenceModel=3;s.sstWallTreatment=1;s.gasMu=.001;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=-300;Real mu=7,k=8,q=9;int active=0;'+owner_call(root,'gasFaceSubgridTransportProperties','s,1,0,-1,2,Real(1),mu,k,q,active')+'std::printf("SUBGRID %a %a %a %d\\n",mu,k,q,active);'
            if case=='subgrid_temperature':setup=setup.replace('s.riemannBoundaryT[1]=-300;','s.riemannBoundaryT[1]=5000;')
        elif case=='publisher':
            setup=r'''auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;int status[1]={0};ugkwp::GasBoundaryLayerExchange<Real> ex[1];ugkwp::GasBoundaryLayerSstClosure<Real> cl[1];Real flux[2]={123,456};w.status=status;w.exchange=ex;w.sst=cl;w.speciesFlux=flux;
ugkwp::gaswall::WallInput<Real,2> in;in.pressure=1e-300;in.ownerDistance=1;in.quadrature.volume=1;in.normal[0]=1;in.model.thermo=s.gasSpecies.thermo;auto*coef=const_cast<Real*>(in.model.thermo.coefficients);coef[0]=coef[3]=1000;coef[2]=coef[5]=0;
ugkwp::gaswall::WallOutput<Real,2> out;out.traceTemperature=300;out.wallSpeciesFlux[0]=2;out.wallSpeciesFlux[1]=-2;
'''
            setup+='auto ok=ugkwp::publishGasBoundaryLayerOutput(s,0,in,out,Real(1e308),Real(0));std::printf("PUBLISH %d %d %a %a\\n",ok,ex[0].ready,flux[0],flux[1]);'
        else:
            setup='s.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;s.gasSpecies.cellStatus[0]=int(ugkwp::GasTransportCode::InvalidThermodynamics);'+ 'auto chemical='+owner_call(root,'advanceGasChemistryCell','s,0,0.,1.')+'std::printf("CHEM %d\\n",int(chemical.code));'
        return prefix+'''\nint main(){DeviceState s;setupNative(s,0,0);blockIdx.x=threadIdx.x=0;'''+setup+r'''
std::printf("FACE %d %d %a %a %a %a\n",s.gasSpecies.faceStatus[1],s.gasSpecies.faceStatus[2],s.gasPhiRho[1],s.gasSpecies.flux[1],s.rhoK[0],s.rhoOmega[0]);
std::printf("%d %a %a %a %a %a %a %a\n",s.gasSpecies.cellStatus[0],s.rho[0],s.rhoUx[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.p[0],s.Tgas[0]);}
'''
    before=build_run(tmp_path,'original',original_native_root,body(original_native_root))
    after=build_run(tmp_path,'candidate',ROOT,body(ROOT))
    assert after==before


def test_all_native_stage_launches_forward_owner_argument():
    host=(ROOT/'common/GpuGasAdvance.cuh').read_text()
    for name,count in [('recoverGasPrimitivesKernel',6),('enforcePeriodicGasFluxAntisymmetryKernel',2),('applyGasFluxPositivityScaleKernel',1),('applySstFluxAndSourceKernel',1),('computeGasCourantFieldKernel',1),('computeGasPrimitiveGradientsKernel',2),('computeGasGradientLimiterKernel',2)]:
        calls=re.findall(name+r'<<<[\s\S]*?>>>\s*\(([^;]+)\);',host)
        assert len(calls)==count
        assert all('nativeRefinement' in args for args in calls),name
