"""Invalid measured evidence must not receive a qualification pass."""
from pathlib import Path
import csv,json,subprocess,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DRIVERS=ROOT/'validation/cuda/wall_coupling'

def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

class NativeQualificationComparators(unittest.TestCase):
    def sequence(self,mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            rows=[dict(owner=owner,step=1,stage=1,ok=1,actualOk=1,fitOk=1,finite=1,residual=1e-10,moment=1e-15,
                seconds=.1,rawQ=5,rawTau=2,fitQ=6,fitTau=3,source0=4,source1=7,checksum=100,
                wallJ0=0,wallJ1=0,matchingJ0=0,matchingJ1=0,heatBalance=0,kBalance=0,profileJ0=.001,profileJ1=-.001)
                for owner in [0,7782]]
            table(d/'frozen.csv',[dict(row) for row in rows])
            if mutation:mutation(rows)
            table(d/'actual.csv',rows)
            return subprocess.run(['python3',str(DRIVERS/'compare_sequence.py'),'--actual',str(d/'actual.csv'),
                '--reference',str(d/'frozen.csv'),'--output',str(d/'result.json'),'--limit','1'],capture_output=True,text=True)

    def test_sequence_accepts_conservative_nonzero_transport(self):
        result=self.sequence();self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_sequence_rejects_missing_physical_flux_columns(self):
        result=self.sequence(lambda rows:[row.pop('profileJ0') for row in rows]);self.assertNotEqual(result.returncode,0)

    def test_sequence_rejects_wall_leak(self):
        result=self.sequence(lambda rows:rows[0].update(wallJ0=1e-8));self.assertNotEqual(result.returncode,0)

    def test_sequence_rejects_zero_transport_control(self):
        result=self.sequence(lambda rows:[row.update(profileJ0=0,profileJ1=0) for row in rows]);self.assertNotEqual(result.returncode,0)

    def test_sequence_rejects_species_flux_nonconservation(self):
        result=self.sequence(lambda rows:rows[0].update(profileJ1=-.002));self.assertNotEqual(result.returncode,0)

    def case(self,leak,closure_error=0,adaptive=False,species_leak=0,steps=5,accumulated_time=False):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);case=d/'case';times=np.cumsum(np.full(steps,2e-7)) if accumulated_time else np.arange(1,steps+1)*2e-7;time=case/format(times[-1],'.17g');time.mkdir(parents=True);cpu=d/'cpu';cpu.mkdir();audit=d/'audit';audit.mkdir()
            (case/'input-manifest.json').write_text(json.dumps(dict(steps=steps,dt=2e-7,integrator='SSPRK2')))
            (audit/'run.json').write_text(json.dumps(dict(exit=0,events=[dict(step=i,time=times[i-1],dt=3e-7 if adaptive else 2e-7) for i in range(1,steps+1)])))
            def scalar(name,value):
                (time/name).write_text('FoamFile { format ascii; }\ninternalField nonuniform List<scalar>\n2\n(\n'+str(value)+'\n'+str(value)+'\n);\n')
            for name,value in [('rho',2),('rhoE',10),('p',100),('T',300),('k',1),('omega',2),('nut',.5),('rhoY_Air',1.5),('rhoY_CO2',.5)]:scalar(name,value)
            for name,value in [('U',1),('rhoU',2)]:
                (time/name).write_text('FoamFile { format ascii; }\ninternalField nonuniform List<vector>\n2\n(\n('+str(value)+' 0 0)\n('+str(value)+' 0 0)\n);\n')
            values={'rho':2,'rhoE':10,'p':100,'Tgas':300,'k':1,'omega':2,'nut':.5,'speciesRhoAir':1.5,'speciesRhoCO2':.5,'Ux':1,'Uy':0,'Uz':0,'rhoUx':2,'rhoUy':0,'rhoUz':0,'rhoK':2,'rhoOmega':4}
            for name,value in values.items():np.array([value,value],dtype='<f8').tofile(cpu/(name+'.bin'))
            if closure_error:scalar('rhoY_Air',1.5+closure_error)
            table(cpu/'walls.csv',[dict(owner=owner,q=5,tau=2) for owner in [0,1]])
            table(audit/'wall-work.csv',[dict(owner=owner,solves=2*steps,failures=0,iterations=2,residuals=4,jacobians=1,history=0,fallbacks=0,active=1,maxResidual=1e-10,momentResidual=1e-15,diffusiveEnergy=5,tractionX=2) for owner in [0,1]])
            # Put the bad flux second: Python max() can mask a later NaN.
            table(audit/'wall-faces.csv',[dict(index=0,mass=0,species0=0,species1=0),dict(index=0,mass=leak,species0=species_leak,species1=-species_leak),dict(index=1,mass=1e-6,species0=7e-7,species1=3e-7)])
            result=subprocess.run(['python3',str(DRIVERS/'compare_math_case.py'),'--case',str(case),'--cpu',str(cpu),'--audit',str(audit),'--output',str(d/'result.json')],capture_output=True,text=True)
            return result

    def test_coupled_accepts_fp64_accumulated_time_for_1000_fixed_steps(self):
        result=self.case(0,steps=1000,accumulated_time=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_coupled_accepts_conservative_species_projection_roundoff(self):
        result=self.case(0,species_leak=5e-25);self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_coupled_rejects_species_leak_just_above_roundoff_limit(self):
        self.assertNotEqual(self.case(0,species_leak=65*np.finfo(float).eps*7e-7).returncode,0)

    def test_coupled_rejects_physical_species_leak(self):
        self.assertNotEqual(self.case(0,species_leak=1e-8).returncode,0)

    def test_coupled_rejects_adaptive_dt_even_with_identical_terminal_fields(self):
        self.assertNotEqual(self.case(0,adaptive=True).returncode,0)

    def test_coupled_rejects_species_inventory_defect_within_field_parity_limit(self):
        self.assertNotEqual(self.case(0,closure_error=1e-6).returncode,0)

    def test_coupled_rejects_nonzero_wall_mass(self):
        self.assertNotEqual(self.case(1e-8).returncode,0)

    def test_coupled_rejects_nonfinite_wall_mass_even_after_finite_row(self):
        self.assertNotEqual(self.case(float('nan')).returncode,0)

    def test_coupled_accepts_zero_wall_mass(self):
        result=self.case(0);self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
