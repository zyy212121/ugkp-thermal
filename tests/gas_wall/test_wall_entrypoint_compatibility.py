"""Legacy explicit template arguments share the typed borrowed-view solve."""
import subprocess
import pytest
from test_wall_model import BASE, ROOT


@pytest.mark.parametrize('bits', [32, 64])
def test_old_explicit_and_mixed_view_entrypoints(tmp_path, bits):
    model = BASE.split('bool near(')[0].replace('struct Model {', 'template<class R> struct Model {').replace('double', 'R')
    body = r'''
template<class R> void legacyEntryPoints(){
 Model<R> m;WallModelConfig<R> c;WallStatus s;
 WallWorkspace<R,2,48> w;WallOutput<R,2> inferred,explicitArgs;
 c.model=BoundaryLayerModel::ConstantTransport;
 assert(evaluateConstantTransportWall(m.in,c,inferred,s));
 assert((evaluateConstantTransportWall<R,2>(m.in,c,explicitArgs,s)));
 assert(inferred.conductiveHeatFlux==explicitArgs.conductiveHeatFlux);
 assert(inferred.traction[0]==explicitArgs.traction[0]);
 m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.model.mode=GasMode::MixtureChemistry;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e-4;
 c.model=BoundaryLayerModel::ReactingSst;c.nodes=24;
 assert(evaluateReactingWallLayer(m.in,c,w,inferred,s));
 assert((evaluateReactingWallLayer<R,2,48>(m.in,c,w,explicitArgs,s)));
 assert(inferred.conductiveHeatFlux==explicitArgs.conductiveHeatFlux);
 assert(inferred.reactionIntegral[0]==explicitArgs.reactionIntegral[0]);
 assert(inferred.residual==explicitArgs.residual);
}
int main(){
 legacyEntryPoints<float>();legacyEntryPoints<double>();
 Model<float> m;auto in=detail::wallComputationInput(m.in);
 WallModelConfig<double> c;WallStatus s;WallWorkspace<double,2,48> w;
 WallOutput<double,2> inferred,explicitArgs,tableArgs;
 c.model=BoundaryLayerModel::ConstantTransport;
 assert(evaluateConstantTransportWall(in,c,inferred,s));
 assert((evaluateConstantTransportWall<double,2>(in,c,explicitArgs,s)));
 assert((evaluateConstantTransportWall<double,2,float>(in,c,tableArgs,s)));
 assert(inferred.conductiveHeatFlux==explicitArgs.conductiveHeatFlux);
 assert(inferred.conductiveHeatFlux==tableArgs.conductiveHeatFlux);
 in.temperature=in.matching.temperature=500;in.matching.velocity[0]=0;
 in.model.mode=GasMode::MixtureChemistry;
 in.model.diffusivity[0]=in.model.diffusivity[1]=1e-4;
 c.model=BoundaryLayerModel::ReactingSst;c.nodes=24;
 assert(evaluateReactingWallLayer(in,c,w,inferred,s));
 assert((evaluateReactingWallLayer<double,2,48>(in,c,w,explicitArgs,s)));
 assert((evaluateReactingWallLayer<double,2,48,float>(in,c,w,tableArgs,s)));
 assert(inferred.reactionIntegral[0]==explicitArgs.reactionIntegral[0]);
 assert(inferred.reactionIntegral[0]==tableArgs.reactionIntegral[0]);
 assert(inferred.residual==explicitArgs.residual&&inferred.residual==tableArgs.residual);
}
'''
    source = tmp_path/'entrypoints.cpp'
    source.write_text(model+body)
    exe = source.with_suffix('')
    build = subprocess.run(['g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
                            f'-DUGKWP_GPU_REAL_BITS={bits}', '-I', str(ROOT),
                            '-I', str(ROOT/'common'), str(source), '-o', str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
