"""Disabled owner source diagnostics perform no quadrature access or physics change."""
import subprocess
import pytest
from test_wall_model import BASE, ROOT

@pytest.mark.parametrize('bits',[32,64])
def test_owner_source_integral_is_opt_in_and_physics_is_identical(tmp_path,bits):
    body=r'''
#include <cstring>
int main(){Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=200;
 GpuReal y[2]={.002,.006},weight[2]={2e-7,2e-7};m.in.quadrature={y,weight,2,4e-7,1.6e-9};
 WallModelConfig<GpuReal> c;c.enableSst=true;c.nodes=32;c.maxIterations=100;
 WallWorkspace<GpuReal,2,48> w;WallOutput<GpuReal,2> with,without;WallStatus s;
 m.in.integrateOwnerKSource=true;assert(evaluateWallModel(m.in,c,w,with,s));assert(std::isfinite(with.integratedKSource));
 // No valid quadrature pointers: this path cannot touch even one point.
 m.in.integrateOwnerKSource=false;m.in.quadrature.distance=nullptr;m.in.quadrature.volumeWeight=nullptr;
 assert(evaluateWallModel(m.in,c,w,without,s));assert(std::isnan(without.integratedKSource));
 assert(with.traceTemperature==without.traceTemperature&&with.traceDensity==without.traceDensity&&with.ownerOmega==without.ownerOmega&&with.wallKFlux==without.wallKFlux&&with.conductiveHeatFlux==without.conductiveHeatFlux&&with.residual==without.residual&&with.iterations==without.iterations);
 for(int d=0;d<3;++d)assert(with.traction[d]==without.traction[d]&&with.traceVelocity[d]==without.traceVelocity[d]);
 for(int j=0;j<2;++j)assert(with.traceMassFraction[j]==without.traceMassFraction[j]&&with.wallSpeciesFlux[j]==without.wallSpeciesFlux[j]&&with.matchingSpeciesFlux[j]==without.matchingSpeciesFlux[j]&&with.reactionIntegral[j]==without.reactionIntegral[j]);
 m.in.integrateOwnerKSource=true;assert(!evaluateWallModel(m.in,c,w,without,s));
 WallInput<GpuReal,2> defaults;assert(!defaults.integrateOwnerKSource);
}
'''
    src=tmp_path/'probe.cpp';src.write_text(BASE.replace('double','GpuReal')+body)
    exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
