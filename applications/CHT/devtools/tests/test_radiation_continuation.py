"""Actual production radiation roots, OpenFOAM DO/table/transaction, and generator.

These are CPU tests. They make no CUDA execution/performance claim.
"""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
THERMAL = Path(os.environ.get('T13_THERMAL_SOURCE', ROOT / 'applications/CHT/thermal'))

@pytest.fixture(scope='module')
def radiation_probe(tmp_path_factory):
    if not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('source the real OpenFOAM environment')
    p = tmp_path_factory.mktemp('radiation_native')
    (p/'Make').mkdir()
    (p/'probe.C').write_text((Path(__file__).parent/'radiationContinuation/TestRadiationContinuation.C').read_text())
    (p/'Make/files').write_text(f'probe.C\n{THERMAL}/MieTable.C\n{THERMAL}/MieDoRadiationSolver.C\n{THERMAL}/GpuParticleRadiationCoupler.C\nEXE = {p}/probe\n')
    omp='-fopenmp' if os.environ.get('T13_OPENMP') else ''
    (p/'Make/options').write_text(f'EXE_INC = {omp} -std=c++17 -I{THERMAL} -I{ROOT}/common -I{ROOT}/applications/CHT/gpu -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = {omp} -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    build=subprocess.run(['wmake'],cwd=p,capture_output=True,text=True)
    (p/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stdout+build.stderr
    return p/'probe'

def table(path,*,version='1',checksum=True,qext=1,phi=1,band=1):
    payload='Tvals 300 4000\nDvals 0.0001 0.0002\nMuvals -1 1\nPlanckBandFractions '+f'{band} {band}\nDATA\n'
    for d in range(2):
        for t in range(2):payload+=f'NODE {d} {t} 0.3 0.7 {qext}\nPHI {phi} {phi}\n'
    payload+='END_MIE_TABLE\n'
    header=f'MIE_TABLE\nformatVersion {version}\nunits diameter_m temperature_K wavelength_m\ninterpolation logDiameter_linearTemperature_linearMu\ngeneratorSha256 fixture\ngenerationArgs {{}}\ndiameter 0.0001 0.0002 2\ntemperature 300 4000 2\nmu -1 1 2\nwavelength 1e-6 2e-6 2\n'
    path.write_text(header+payload+'payloadSha1 '+(hashlib.sha1(payload.encode()).hexdigest() if checksum else 'different')+'\n')
    return path

def run(probe,mode,path=None,case=None):
    env=os.environ.copy();env['T13_MODE']=mode
    if path:env['T13_TABLE']=str(path)
    args=[str(probe)]
    if case:args+=['-case',str(case)]
    return subprocess.run(args,env=env,capture_output=True,text=True)

@pytest.mark.parametrize('mode',['root_ordinary','phase_ordinary','phase_negative_weights','phase_signed','root_one','root_zero','root_same_sign','root_equilibrium','root_signed','root_errors','scale_two','scale_negative','scale_tiny','scale_errors','phase_one','phase_zero','phase_asymmetric','phase_errors'])
def test_actual_math(radiation_probe,mode):
    r=run(radiation_probe,mode);assert r.returncode==0,r.stdout+r.stderr

@pytest.mark.parametrize('change',['ordinary','version','hash','units','phi','qext','band'])
def test_actual_table_values(radiation_probe,tmp_path,change):
    p=table(tmp_path/'table',version='different' if change=='version' else '1',checksum=change!='hash',phi=3 if change=='phi' else 1,qext=2 if change=='qext' else 1,band=1.2 if change=='band' else 1)
    if change=='units':p.write_text(p.read_text().replace('units diameter_m temperature_K wavelength_m','units unfamiliar_label'))
    r=run(radiation_probe,'table',p);assert r.returncode==0,r.stdout+r.stderr
    v=list(map(float,next(x for x in r.stdout.splitlines() if x.startswith('TABLE ')).split()[1:]))
    assert v==[.3,.7,2. if change=='qext' else 1.,1.2 if change=='band' else 1.,(.7*3.)/.7 if change=='phi' else 1.]

@pytest.mark.parametrize('fault',['node','phi'])
def test_actual_table_missing_data(radiation_probe,tmp_path,fault):
    p=table(tmp_path/'table');s=p.read_text()
    if fault=='node':s=s.replace('NODE 1 1 0.3 0.7 1\nPHI 1 1\n','')
    else:s=s.replace('PHI 1 1','PHI 1',1)
    p.write_text(s);r=run(radiation_probe,'table',p);assert r.returncode!=0
    assert ('malformed NODE' if fault=='node' else 'invalid PHI value') in r.stderr

@pytest.fixture
def case(tmp_path):
    p=tmp_path/'case';(p/'system').mkdir(parents=True);(p/'constant').mkdir()
    def header(name):return f'FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n'
    (p/'system/controlDict').write_text(header('controlDict')+'application radiationTest; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .1; writeControl timeStep; writeInterval 1;')
    (p/'system/fvSchemes').write_text(header('fvSchemes')+'ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }')
    (p/'system/fvSolution').write_text(header('fvSolution')+'solvers {}')
    (p/'system/blockMeshDict').write_text(header('blockMeshDict')+'convertToMeters 1; vertices ((0 0 0) (.02 0 0) (.02 .01 0) (0 .01 0) (0 0 .01) (.02 0 .01) (.02 .01 .01) (0 .01 .01)); blocks (hex (0 1 2 3 4 5 6 7) (2 1 1) simpleGrading (1 1 1)); edges (); boundary (left { type wall; faces ((0 4 7 3)); } right { type wall; faces ((1 2 6 5)); } sides { type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2)); }); mergePatchPairs ();')
    r=subprocess.run(['blockMesh','-case',str(p)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
    return p

@pytest.mark.parametrize('mode',['do_one','do_zero','do_negative','do_roundoff','do_phase_one','do_relaxation','do_warm','do_identity','do_signed','transaction','ordinary','boundary_tiny','boundary_negative','boundary_cancellation'])
def test_actual_openfoam_radiation(radiation_probe,tmp_path,case,mode):
    r=run(radiation_probe,mode,table(tmp_path/'table'),case);assert r.returncode==0,r.stdout+r.stderr

@pytest.fixture
def generator():
    spec=importlib.util.spec_from_file_location('t13_generator',THERMAL/'mieTables/make_alumina_mie_table.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_two_mu_actual_generation(tmp_path):
    p=tmp_path/'out.dat';cmd=['python',str(THERMAL/'mieTables/make_alumina_mie_table.py'),'--out',str(p),'--n-t','2','--n-d','2','--n-mu','2','--n-lambda','2','--d-min','1e-7','--d-max','2e-7']
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
    s=p.read_text();assert s.count('NODE ')==4 and s.count('PHI ')==4
    assert all(math.isfinite(float(x)) for line in s.splitlines() if line.startswith('PHI ') for x in line.split()[1:])
    old=p.read_bytes();r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode!=0 and p.read_bytes()==old

def test_tiny_real_planck_weight(generator):
    lam=np.array([1.,1.0001]);T=generator.C2/699.
    actual=generator.compute_node(1e-6,T,np.array([-1.,1.]),lam)
    weights=[.5*x*math.log(lam[-1]/lam[0])*generator.planck_weight_lambda(x,T) for x in lam]
    assert 0<sum(weights)<1e-300
    q=[generator.mie_coefficients(math.pi*1e-6/x,generator.alumina_refractive_index(x,T)) for x in lam]
    for i,qi in enumerate([2,1,0]):assert actual[i]==sum(w*v[qi] for w,v in zip(weights,q))/sum(weights)
    assert np.all(np.isfinite(actual[4]))

def test_actual_zero_planck_weight(generator):
    with pytest.raises((RuntimeError,ZeroDivisionError)):generator.compute_node(1e-6,0.,np.array([-1.,1.]),np.array([1.,2.]))

def test_generator_original_single_node_fallback(generator):
    assert np.array_equal(generator.logspace(1.,2.,0),[1.])
    assert np.array_equal(generator.logspace(1.,2.,-1),[1.])

def test_tail_finite_exhaustion(generator):
    # Negative x is a defined finite continuation witness for the existing series;
    # this is not a physical Planck-accuracy assertion.
    T=-generator.C2/.0015
    def tail(x):
        total=0.
        for n in range(1,100001):
            term=math.exp(-n*x)*(x**3/n+3*x*x/(n*n)+6*x/(n**3)+6/(n**4))
            total+=term
            if abs(term)<1e-15*max(1.,abs(total)):return total,n
        return total,100000
    a,na=tail(generator.C2/(1.*T));b,nb=tail(generator.C2/(.5*T))
    assert max(na,nb)==100000 and math.isfinite(a-b)
    assert generator.planck_band_fraction(T,.5,1.)==(a-b)/(math.pi**4/15.)

def test_planck_ordinary_accuracy(generator):
    # Independent dimensionless Simpson quadrature, unchanged analytic normalization.
    T=1500.;lo=.5e-6;hi=8e-6;a=generator.C2/(hi*T);b=generator.C2/(lo*T)
    x=np.linspace(a,b,20001);f=x**3/np.expm1(x);h=(b-a)/20000
    expected=h/3*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum())/(math.pi**4/15)
    assert abs(generator.planck_band_fraction(T,lo,hi)-expected)<1e-12


@pytest.mark.parametrize('signed_case',['ordinary','negative_scattering','negative_diameter'])
def test_actual_first_sweep_independent_values(radiation_probe,tmp_path,case,signed_case):
    p=table(tmp_path/'table')
    qsca=-.7 if signed_case=='negative_scattering' else .7
    diameter=-1e-4 if signed_case=='negative_diameter' else 1e-4
    if signed_case=='negative_scattering':p.write_text(p.read_text().replace('0.3 0.7 1','0.3 -0.7 -0.4'))
    mode='do_one_negative_diameter' if signed_case=='negative_diameter' else 'do_one'
    r=run(radiation_probe,mode,p,case)
    assert r.returncode==0,r.stdout+r.stderr
    # Independently assemble the two unit-aspect-ratio cells, angular quadrature,
    # isotropic scattering and one existing forward Gauss-Seidel direction sweep.
    directions=[];weights=[]
    for ti in range(2):
        theta=(ti+.5)*math.pi/2
        for pi in range(4):
            phi=(pi+.5)*math.pi/2
            directions.append((math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)))
            weights.append(math.sin(theta)*(math.pi/2)*(math.pi/2))
    total=sum(weights);weights=[w*(4*math.pi)/total for w in weights]
    black=lambda t:5.670374419e-8*t**4/math.pi
    ib=black(1000);wall=black(300);absorption=3*1e-6*.3/(2*diameter);scatter=3*1e-6*qsca/(2*diameter)
    intensity=[[ib,ib] for _ in directions]
    for j,(dx,dy,dz) in enumerate(directions):
        area=1e-4;volume=1e-6
        diagonal=(abs(dx)+abs(dy)+abs(dz))*area+(absorption+scatter)*volume
        source=[]
        for cell in range(2):
            incoming=(abs(dy)+abs(dz))*area
            if (cell==0 and dx>0) or (cell==1 and dx<0):incoming+=abs(dx)*area
            mean=sum(w*vals[cell]/(4*math.pi) for w,vals in zip(weights,intensity))
            source.append(incoming*wall+volume*(absorption*ib+scatter*mean))
        solved=intensity[j].copy()
        for cell in range(2):
            rhs=source[cell]
            if cell==0 and dx<0:rhs-=dx*area*solved[1]
            if cell==1 and dx>0:rhs+=dx*area*solved[0]
            solved[cell]=rhs/diagonal
        intensity[j]=solved
    gs=[sum(w*vals[cell] for w,vals in zip(weights,intensity)) for cell in range(2)]
    rows=[x.split() for x in r.stdout.splitlines() if x.startswith('DO ')]
    for row in rows:assert math.isclose(float(row[2]),gs[int(row[1])],rel_tol=2e-13,abs_tol=1e-10)
    cells=[x.split() for x in r.stdout.splitlines() if x.startswith('RESULT_CELL ')]
    assert len(cells)==2
    sources=[absorption*(g-4*math.pi*ib) for g in gs]
    for row in cells:
        expected=[absorption,scatter,absorption+scatter,0.,gs[int(row[1])],sources[int(row[1])]]
        assert len(row[2:])==len(expected)
        for actual,want in zip(map(float,row[2:]),expected):assert math.isclose(actual,want,rel_tol=2e-13,abs_tol=1e-10)
    boundary=0.;faces=0
    for line in r.stdout.splitlines():
        if not line.startswith('FACE '):continue
        row=line.split();actual=float(row[3]);normal=list(map(float,row[4:7]));cell=int(row[7]);expected=0.
        for w,d,values in zip(weights,directions,intensity):
            cosine=sum(a*b for a,b in zip(d,normal));expected+=w*cosine*(values[cell] if cosine>0 else wall)
        assert math.isclose(actual,expected,rel_tol=2e-13,abs_tol=1e-10)
        boundary+=expected*1e-4;faces+=1
    assert faces==10
    stats=next(x.split() for x in r.stdout.splitlines() if x.startswith('RESULT_STATS '))
    total=sum(sources)*1e-6
    for actual,want in zip(map(float,stats[1:4]),[total,boundary,total+boundary]):assert math.isclose(actual,want,rel_tol=2e-13,abs_tol=1e-10)
    assert list(map(int,stats[4:]))==[1,1]


@pytest.mark.parametrize('mode',['do_nonfinite','do_missing','do_zero_divisor','do_unknown'])
def test_real_do_required_errors(radiation_probe,tmp_path,case,mode):
    r=run(radiation_probe,mode,table(tmp_path/'table'),case)
    assert r.returncode!=0 and 'FAIL ' in r.stderr,r.stdout+r.stderr

def test_real_parallel_sweep(radiation_probe,tmp_path,case):
    if not os.environ.get('T13_OPENMP'):pytest.skip('separate real OpenMP build')
    r=run(radiation_probe,'do_parallel',table(tmp_path/'table'),case)
    assert r.returncode==0,r.stdout+r.stderr

def test_original_implicit_math_fixture(tmp_path):
    if not os.environ.get('WM_PROJECT_DIR'):pytest.skip('source real OpenFOAM')
    (tmp_path/'Make').mkdir()
    source=Path(__file__).parent/'particleRadiationImplicitTransactionMath/TestParticleRadiationImplicitTransactionMath.C'
    (tmp_path/'original.C').write_text(source.read_text())
    (tmp_path/'Make/files').write_text(f'original.C\nEXE = {tmp_path}/original\n')
    (tmp_path/'Make/options').write_text(f'EXE_INC = -std=c++17 -I{THERMAL}\nEXE_LIBS = -lOpenFOAM\n')
    r=subprocess.run(['wmake'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([str(tmp_path/'original')],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr

def test_optical_lookup_endpoint_is_independent(radiation_probe,tmp_path):
    p=table(tmp_path/'table');p.write_text(p.read_text().replace('PlanckBandFractions 1 1','PlanckBandFractions 0.25 0.75'))
    r=run(radiation_probe,'table',p);assert r.returncode==0,r.stdout+r.stderr
    values=next(x for x in r.stdout.splitlines() if x.startswith('TABLE ')).split()
    assert float(values[4])==.75

def test_standalone_optical_accuracy_and_failure_remain():
    import sys
    directory=THERMAL/'mieTables';sys.path.insert(0,str(directory))
    try:
        import mie_sphere
        m=complex(1.5,-.01);x=1e-3;solution=mie_sphere.coefficients(x,m)
        ratio=(m*m-1)/(m*m+2)
        expected_scattering=8/3*x**4*abs(ratio)**2
        expected_absorption=-4*x*ratio.imag
        assert math.isclose(solution.qsca,expected_scattering,rel_tol=1e-5)
        assert math.isclose(solution.qabs,expected_absorption,rel_tol=1e-5)
        with pytest.raises(mie_sphere.MieError,match='did not converge'):
            mie_sphere.coefficients(x,m,initial_padding=15,max_padding=15)
    finally:sys.path.pop(0)

@pytest.mark.parametrize('change',['duplicate_temperature','extra_phi','extra_node'])
def test_read_actual_table_payload_without_unused_layout_veto(radiation_probe,tmp_path,change):
    p=table(tmp_path/'table');s=p.read_text()
    if change=='duplicate_temperature':s=s.replace('Tvals 300 4000','Tvals 300 300')
    if change=='extra_phi':s=s.replace('PHI 1 1','PHI 1 1 77')
    if change=='extra_node':s=s.replace('NODE 0 0 0.3 0.7 1','NODE 0 0 0.3 0.7 1 unusedMetadata')
    p.write_text(s);r=run(radiation_probe,'table',p)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'TABLE 0.29999999999999999 0.69999999999999996 1 1 1' in r.stdout

def test_tiny_original_phase_fallback(generator):
    mu=np.linspace(-1,1,17)
    qext,qsca,qabs,g,a,b=generator.mie_coefficients(.4,complex(1.5,-.01))
    raw=generator.mie_phase_raw(.4,qsca,a,b,mu)*1e-310
    integral=generator.trapezoid(raw,mu)
    assert 0<integral<1e-300
    actual=generator.normalize_phi(raw,mu)
    assert np.array_equal(actual,np.ones_like(raw))

def test_planck_actual_overflow(generator):
    with pytest.raises(OverflowError):generator.planck_band_fraction(-1e-8,.5,1)

def test_finite_signed_scattering_keeps_actual_phase(radiation_probe,tmp_path):
    p=table(tmp_path/'table',phi=3)
    p.write_text(p.read_text().replace('0.3 0.7 1','0.3 -0.7 1'))
    r=run(radiation_probe,'table',p);assert r.returncode==0,r.stdout+r.stderr
    values=list(map(float,next(x for x in r.stdout.splitlines() if x.startswith('TABLE ')).split()[1:]))
    assert values[1]==-.7 and values[4]==(-.7*3)/(-.7)


def test_oversized_accessible_snapshot_preserves_diameter_routing(radiation_probe,tmp_path,case):
    r=run(radiation_probe,'transaction_padded',table(tmp_path/'table'),case)
    assert r.returncode==0,r.stdout+r.stderr


def test_exact_and_oversized_complete_transaction_byte_parity(radiation_probe,tmp_path,case):
    p=table(tmp_path/'table');outputs=[]
    for mode in ['transaction_exact_diameter','transaction_padded']:
        r=run(radiation_probe,mode,p,case);assert r.returncode==0,r.stdout+r.stderr
        outputs.append([line for line in r.stdout.splitlines() if line.startswith(('CELL ','DIAG ','APPLY ','BOUNDARY ','ESCAPED ','BALANCE '))])
    assert outputs[0] and outputs[0]==outputs[1]

def test_transaction_missing_used_field_is_error(radiation_probe,tmp_path,case):
    r=run(radiation_probe,'transaction_missing',table(tmp_path/'table'),case)
    assert r.returncode!=0 and 'cell arrays' in r.stderr,r.stdout+r.stderr


@pytest.mark.parametrize('axis',['temperature','diameter','mu','interior','ordinary_interior','duplicate_interior'])
def test_mie_actual_endpoint_and_selected_interval_without_order_certificate(radiation_probe,tmp_path,axis):
    p=table(tmp_path/'table')
    temperatures=[300,1000,2000,4000] if axis=='ordinary_interior' else [300,1000,1000,4000] if axis=='duplicate_interior' else [300,3000,1000,4000] if axis=='interior' else [4000,300] if axis=='temperature' else [300,4000]
    diameters=[.0002,.0001] if axis=='diameter' else [.0001,.0002]
    mus=[1,-1] if axis=='mu' else [-1,1]
    bands=[.25,.35,.45,.75] if 'interior' in axis else [.25,.75]
    payload='Tvals '+' '.join(map(str,temperatures))+'\nDvals '+' '.join(map(str,diameters))+'\nMuvals '+' '.join(map(str,mus))+'\nPlanckBandFractions '+' '.join(map(str,bands))+'\nDATA\n'
    for d in range(2):
        for t in range(len(temperatures)):
            qa=1+t+10*d
            payload+=f'NODE {d} {t} {qa} .7 {qa+.7}\nPHI {qa+3} {qa+5}\n'
    p.write_text('MIE_TABLE\n'+payload+'END_MIE_TABLE\n')
    r=run(radiation_probe,'table_interior' if 'interior' in axis else 'table',p)
    assert r.returncode==0,r.stdout+r.stderr
    values=list(map(float,next(x for x in r.stdout.splitlines() if x.startswith('TABLE ')).split()[1:]))
    dw=0. if axis=='diameter' else (math.log(1.5e-4)-math.log(1e-4))/(math.log(2e-4)-math.log(1e-4))
    qa=(3.25 if axis=='ordinary_interior' else 3.5 if 'interior' in axis else 2.)+10*dw
    expected=[qa,.7,qa+.7,.525 if axis=='ordinary_interior' else .6 if 'interior' in axis else .75,qa+(3 if axis=='mu' else 4)]
    for actual,want in zip(values,expected):assert math.isclose(actual,want,rel_tol=2e-14,abs_tol=1e-14)


@pytest.mark.parametrize('fault',['missing_axis_node','nonfinite_query'])
def test_mie_true_selected_lookup_error(radiation_probe,tmp_path,fault):
    p=table(tmp_path/'table');s=p.read_text()
    if fault=='missing_axis_node':s=s.replace('Tvals 300 4000','Tvals 300')
    p.write_text(s);r=run(radiation_probe,'table_nonfinite' if fault=='nonfinite_query' else 'table',p)
    assert r.returncode!=0
    assert ('requires at least two values' if fault=='missing_axis_node' else 'no accessible bracket') in r.stderr
