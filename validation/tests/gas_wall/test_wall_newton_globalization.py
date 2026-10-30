"""Captured wall regression with current SST discretization.

Heat reference is the independent NumPy/SciPy FV solve in
validation/pr33/sst_heat_reference.py, not a refreshed production snapshot.
Historical 653.7676567 predates ffd64367/0039dc68 conductance, work weighting,
near-wall source integration and directed omega cross-diffusion changes.
"""
from pathlib import Path
import subprocess
from validation.tools.paths import SOURCE_ROOT

def run_cpp(tmp_path,body):
    source=tmp_path/'captured_wall.cpp'
    source.write_text('#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"\n#include <cassert>\n#include <cmath>\n#include <iostream>\nint main(){'+body+'}\n')
    executable=tmp_path/'captured_wall'
    subprocess.run(['g++','-std=c++17','-O2','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_captured_sst_input_converges_in_physical_domain(tmp_path):
    run_cpp(tmp_path,r'''
    CapturedSstWall model;ugkp::diagnostics::DeviceContext diagnostics{};
    model.input.diagnostics=&diagnostics;model.input.diagnosticEntity=53683;
    ugkwp::gaswall::WallWorkspace<double,2,24> workspace{};
    ugkwp::gaswall::WallOutput<double,2> output{};ugkwp::gaswall::WallStatus status{};
    bool ok=ugkwp::gaswall::evaluateReactingWallLayer(model.input,model.config,workspace,output,status);
    std::cerr<<"ok="<<ok<<" fatal="<<diagnostics.fatalClaim<<" observed="<<diagnostics.firstFatal.observed<<" iterations="<<output.iterations<<" residual="<<output.residual<<"\n";
    assert(ok&&!diagnostics.fatalClaim);
    assert(output.residual<=model.config.absoluteTolerance+model.config.relativeTolerance);
    assert(output.iterations>0&&output.iterations<model.config.maxIterations);
    assert(std::abs(output.conductiveHeatFlux-674.8259379397244)<1e-6);
    assert(workspace.state[0][5]==0);
    for(int i=1;i<model.config.nodes;++i){assert(workspace.state[i][5]>0);assert(std::isfinite(workspace.state[i][5]));}
    ''')

def test_invalid_physical_input_stays_fatal(tmp_path):
    run_cpp(tmp_path,r'''
    CapturedSstWall model;ugkp::diagnostics::DeviceContext diagnostics{};
    model.input.diagnostics=&diagnostics;model.input.diagnosticEntity=53683;
    model.input.temperature=-1;
    ugkwp::gaswall::WallWorkspace<double,2,24> workspace{};
    ugkwp::gaswall::WallOutput<double,2> output{};ugkwp::gaswall::WallStatus status{};
    assert(!ugkwp::gaswall::evaluateReactingWallLayer(model.input,model.config,workspace,output,status));
    assert(diagnostics.fatalClaim&&diagnostics.firstFatal.observed==-1);
    assert(diagnostics.firstFatal.code==ugkp::diagnostics::Code::NegativeTemperature);
    ''')


def test_rejected_newton_candidates_do_not_publish_fatal_diagnostics(tmp_path):
    run_cpp(tmp_path,r'''
    CapturedSstWall model;ugkp::diagnostics::DeviceContext diagnostics{};
    model.input.diagnostics=&diagnostics;model.input.model.thermo.diagnostics=&diagnostics;
    ugkwp::gaswall::WallWorkspace<double,2,24> workspace{};loadCapturedSstIterate(workspace);
    ugkwp::gaswall::detail::LayerContext<double,2> context;
    assert(ugkwp::gaswall::detail::makeContext(model.input,model.config,context));
    double norm=0,alpha=0;
    assert(ugkwp::gaswall::detail::allResidual(model.input,model.config,context,workspace.state,workspace.y,workspace.residual,norm));
    const double initial=norm;
    assert(ugkwp::gaswall::detail::applyLayerNewtonUpdate(model.input,model.config,context,workspace,norm,alpha));
    assert(alpha==.015625);assert(norm<initial);assert(!diagnostics.fatalClaim);
    for(int i=1;i<model.config.nodes;++i)assert(workspace.state[i][5]>0);
    ''')

def test_exhausted_mathematical_search_restores_current_residual(tmp_path):
    run_cpp(tmp_path,r'''
    CapturedSstWall model;ugkp::diagnostics::DeviceContext diagnostics{};
    model.input.diagnostics=&diagnostics;model.input.model.thermo.diagnostics=&diagnostics;
    ugkwp::gaswall::WallWorkspace<double,2,24> workspace{};loadCapturedSstIterate(workspace);
    ugkwp::gaswall::detail::LayerContext<double,2> context;
    assert(ugkwp::gaswall::detail::makeContext(model.input,model.config,context));
    double norm=0,alpha=-1;
    assert(ugkwp::gaswall::detail::allResidual(model.input,model.config,context,workspace.state,workspace.y,workspace.residual,norm));
    const auto before=workspace;const double initial=norm;
    for(int i=0;i<model.config.nodes;++i)for(int j=0;j<6;++j)workspace.delta[i][j]=0;
    assert(ugkwp::gaswall::detail::applyLayerNewtonUpdate(model.input,model.config,context,workspace,norm,alpha));
    assert(alpha==0&&norm==initial&&!diagnostics.fatalClaim);
    for(int i=0;i<model.config.nodes;++i)for(int j=0;j<6;++j){assert(workspace.state[i][j]==before.state[i][j]);assert(workspace.residual[i][j]==before.residual[i][j]);}
    ''')


def test_exhausted_iteration_budget_rejects_without_publication(tmp_path):
    run_cpp(tmp_path,r'''
    CapturedSstWall model;ugkp::diagnostics::DeviceContext diagnostics{};
    model.input.diagnostics=&diagnostics;model.config.maxIterations=0;
    ugkwp::gaswall::WallWorkspace<double,2,24> workspace{};
    ugkwp::gaswall::WallOutput<double,2> output{};ugkwp::gaswall::WallStatus status{};
    output.conductiveHeatFlux=123;
    assert(!ugkwp::gaswall::evaluateReactingWallLayer(model.input,model.config,workspace,output,status));
    assert(diagnostics.fatalClaim&&diagnostics.firstFatal.sourceSiteId==2422);
    assert(status.code==ugkwp::gaswall::WallCode::NonConvergence);
    assert(status.residual>model.config.absoluteTolerance+model.config.relativeTolerance);
    assert(output.conductiveHeatFlux==123);
    assert(workspace.state[0][5]==0);
    for(int i=1;i<model.config.nodes;++i)assert(workspace.state[i][5]>0);
    ''')
