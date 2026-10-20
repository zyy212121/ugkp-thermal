"""The actual FV geometry and matching adapter must agree on sample position."""
import subprocess
import pytest
import test_wall_geometry


def compile_run(tmp_path, body):
    base = test_wall_geometry.BASE.replace(
        '#include "common/gasWall/WallGeometryBuilder.H"',
        '#include "common/gasWall/WallGeometryBuilder.H"\n'
        '#include "common/gasTransport/GasWallMatching.H"\n'
        'using namespace ugkwp;')
    source = tmp_path / 'matching.cpp'
    source.write_text(base + '\nint main(){\n' + body + '\n}\n')
    executable = tmp_path / 'matching'
    result = subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-pedantic', '-I', str(test_wall_geometry.ROOT), '-I',
        str(test_wall_geometry.ROOT / 'common'), str(source), '-o', str(executable)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('reverse', [False, True])
def test_geometry_to_conserved_matching_reproduces_boundary_linear_fields(tmp_path, reverse):
    compile_run(tmp_path, r'''
    auto mesh=cubes(5);WallGeometryDescriptor<double> descriptor;WallStatus status;
    WallGeometryOptions<double> options;options.reconstructMatching=true;
    options.constructQuadrature=false;
    assert(buildWallGeometry(mesh,0,descriptor,status,options));
    assert(near(descriptor.matchingPoint[2],1));
    struct S:GasStateView<double,double,SstCoefficientsT<double>>{GasSpeciesState<double,2> gasSpecies;};
    S s;s.nCells=5;
    SpeciesThermoData<double> thermo[2];double coefficients[6]={1000,0,0,1000,0,0};
    for(int a=0;a<2;++a){thermo[a].coefficientOffset=3*a;thermo[a].molarMass=.028;
        thermo[a].minTemperature=200;thermo[a].maxTemperature=4000;}
    s.gasSpecies.thermo.species=thermo;s.gasSpecies.thermo.coefficients=coefficients;
    s.gasSpecies.thermo.coefficientCount=6;
    double rho[5],mx[5],my[5]{},mz[5]{},temperature[5],energy[5],rk[5],rw[5],species[10];
    const double cv=1000-universalGasConstant<double>()/.028;
    const double direction=REVERSE?-1:1,k0=REVERSE?10:1;
    for(int i=0;i<5;++i){const double y=mesh.cellCentres[i][2];rho[i]=1;
        mx[i]=10*y;temperature[i]=700+20*y;
        // rhoE is the actual cell average: <U^2> differs from <U>^2.
        energy[i]=cv*temperature[i]+50*(y*y+1./12.);
        rk[i]=k0+direction*y;rw[i]=100-direction*y;
        species[i]=.7;species[5+i]=.3;}
    s.rho=rho;s.rhoUx=mx;s.rhoUy=my;s.rhoUz=mz;s.rhoE=energy;s.Tgas=temperature;
    s.rhoK=rk;s.rhoOmega=rw;s.gasSpecies.rho=species;
    WallMatchingState<double,2> result;double pressure=0;
    assert(recoverWallMatching(s,descriptor.matchingCells.data(),descriptor.matchingWeights.data(),
        int(descriptor.matchingCells.size()),true,result,pressure));
    std::cerr<<"matching U="<<result.velocity[0]<<" k="<<result.k<<" omega="<<result.omega
        <<" T="<<result.temperature<<"\n";
    assert(near(result.velocity[0],10,1e-12));
    assert(near(result.k,k0+direction,1e-12));assert(near(result.omega,100-direction,1e-12));
    assert(near(result.temperature,720,1e-12));
    assert(near(pressure,universalGasConstant<double>()/.028*720,1e-12));
    assert(near(result.massFraction[0],.7));assert(near(result.massFraction[1],.3));
    '''.replace('REVERSE', 'true' if reverse else 'false'))
