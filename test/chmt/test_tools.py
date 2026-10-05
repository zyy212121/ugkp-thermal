"""CPU checks of reference/comparison plumbing, never CHMT solver tests."""
import csv
import io
import math
from pathlib import Path
import tempfile
import unittest

import reference
import compare


class ReferenceTests(unittest.TestCase):
    def test_stefan_root_and_front(self):
        lam = reference.stefan_lambda(1.0)
        self.assertAlmostEqual(lam, 0.6200626333135955, places=14)
        self.assertAlmostEqual(math.sqrt(math.pi)*lam*math.exp(lam*lam)*math.erf(lam), 1.0, places=14)
        self.assertAlmostEqual(reference.stefan_front(0.0), 0.1, places=14)

    def test_stefan_cell_enthalpy_budget(self):
        for n in (32, 128):
            for t in (0.0, 0.01):
                cells = [reference.stefan_cell(i/n, (i+1)/n, t) for i in range(n)]
                self.assertTrue(all(0 <= T <= 1 and 0 <= H <= 2 for T, H in cells))
                total = math.fsum(H/n for T, H in cells)
                self.assertAlmostEqual(total, reference.stefan_energy(t), places=13)
        self.assertAlmostEqual(reference.stefan_energy(0.01)-reference.stefan_energy(0), reference.stefan_heat(0.01), places=13)

    def test_profile_has_correct_boundary_derivatives_and_mean(self):
        delta, mu, ub, tau, G = 0.01, 0.1, 0.4, 2.0, 3.0
        p = reference.film_profile(delta, mu, ub, tau, G)
        self.assertAlmostEqual(p['u_top'], 0.5985, places=14)
        self.assertAlmostEqual(p['u_mean'], 0.499, places=14)
        self.assertAlmostEqual(p['tau_bottom'], 1.97, places=14)
        self.assertAlmostEqual(p['phi'], p['boundary_work']-delta*p['u_mean']*G, places=14)
        f = lambda z: reference.film_velocity(z, delta, mu, ub, tau, G)
        self.assertEqual(f(0), ub)
        # Simpson is exact for the quadratic velocity profile, independent of mean formula.
        self.assertAlmostEqual((f(0)+4*f(delta/2)+f(delta))/6, p['u_mean'], places=14)

    def test_cell_wave_is_average_not_midpoint(self):
        value = reference.sine_average(0, 0.5, 0)
        self.assertAlmostEqual(value, 2/math.pi, places=14)
        self.assertNotAlmostEqual(value, 1.0, places=5)

    def test_ale_wave_speed_is_two(self):
        self.assertAlmostEqual(reference.ale_rho(0.1, 0.2, 0.125), 1+0.2*math.sin(2*math.pi*0.05), places=14)
        self.assertNotAlmostEqual(reference.ale_rho(0.1, 0.2, 0.125), 1+0.2*math.sin(2*math.pi*0.175), places=5)

    def test_ale_rectangle_average_and_geometry(self):
        # Integrating over a whole periodic direction gives mean density 1 exactly.
        self.assertAlmostEqual(reference.ale_average(0, 1, 0.2, 0.4, 0.125), 1, places=14)
        for i in range(16):
            self.assertLess(reference.mesh_node(i/16, 0.25), reference.mesh_node((i+1)/16, 0.25))

    def test_overlap_remap_conserves(self):
        for n in (8, 16, 32):
            old = [i/n for i in range(n+1)]
            new = [x+0.03*math.sin(2*math.pi*x) for x in old]
            q = [1 if i < 3*n//8 else 2 for i in range(n)]
            masses = reference.overlap_integrals(old, q, new)
            self.assertAlmostEqual(math.fsum(masses), 1.625, places=14)
            self.assertTrue(all(1-1e-13 <= m/(b-a) <= 2+1e-13 for m,a,b in zip(masses,new,new[1:])))

    def test_all_solver_cases_are_unverified(self):
        data = reference.load_cases()
        self.assertTrue(all(c['status'] == 'UNVERIFIED' for c in data['cases']))
        self.assertTrue(all(c['adapter_status'] == 'pendingCHMTadapter' for c in data['cases']))

    def test_reference_rows_unique_finite_and_labelled(self):
        rows = list(reference.generate())
        keys = [reference.row_key(r) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(math.isfinite(float(r['value'])) and float(r['scale']) > 0 for r in rows))
        self.assertTrue(all(r['kind'] == 'ANALYTIC_REFERENCE' for r in rows))


class CompareTests(unittest.TestCase):
    def test_numeric_match_is_not_solver_validation(self):
        rows = list(reference.generate(case_filter='film_profile'))
        result = compare.compare_rows(rows, rows)
        self.assertTrue(result['numeric_match'])
        self.assertEqual(result['solver_validation_status'], 'UNVERIFIED')

    def test_comparison_rejects_nan_missing_extra_duplicate(self):
        rows = list(reference.generate(case_filter='film_profile'))
        for changed in (rows[:-1], rows+[rows[0]], rows+[dict(rows[0], sample='unexpected')], [dict(rows[0], value='NaN')]+rows[1:]):
            with self.assertRaises(ValueError):
                compare.compare_rows(rows, changed)

    def test_mean_top_swap_fails(self):
        rows = list(reference.generate(case_filter='film_profile'))
        changed = [dict(r, value=float(r['value'])*2) if r['quantity']=='u_mean' else r for r in rows]
        self.assertFalse(compare.compare_rows(rows, changed)['numeric_match'])

    def test_same_index_copy_is_not_conservative_remap(self):
        rows = list(reference.generate(case_filter='remap'))
        changed = []
        for r in rows:
            r = dict(r)
            if r['variant']=='step' and r['quantity']=='mass':
                n,i = int(r['n']),int(r['sample'])
                a = i/n+0.03*math.sin(2*math.pi*i/n)
                b = (i+1)/n+0.03*math.sin(2*math.pi*(i+1)/n)
                r['value'] = (1 if i<3*n//8 else 2)*(b-a)
            changed.append(r)
        self.assertFalse(compare.compare_rows(rows,changed)['numeric_match'])

    def test_roundoff_dominated_errors_do_not_prove_convergence(self):
        rows = [{'case':'film_plug','metric':'H_L1','n':n,'error':1e-24/n**2} for n in (32,64,128)]
        self.assertFalse(compare.check_convergence(rows)['numeric_match'])

    def test_reference_cannot_be_claimed_as_solver_output(self):
        with self.assertRaises(ValueError):
            compare.validate_run_metadata({'artifact_kind': 'ANALYTIC_REFERENCE'})

    def test_convergence_requires_two_refinements(self):
        rows = [{'case':'film_plug', 'metric':'H_L1', 'n':n, 'error':1/n**2} for n in (32,64,128)]
        result = compare.check_convergence(rows)
        self.assertTrue(result['numeric_match'])
        self.assertEqual(len(result['orders']), 2)
        with self.assertRaises(ValueError):
            compare.check_convergence(rows[:2])

    def test_zero_errors_do_not_prove_convergence(self):
        rows = [{'case':'film_plug', 'metric':'H_L1', 'n':n, 'error':0} for n in (32,64,128)]
        self.assertFalse(compare.check_convergence(rows)['numeric_match'])

    def test_gcl_check_uses_exported_geometry(self):
        rows = [{'step':1,'stage':1,'cell':0,'V_old':1,'V_new':1.1,'sweep_left':0,'sweep_right':0.1,'sweep_bottom':0,'sweep_top':0}]
        self.assertTrue(compare.check_gcl(rows)['numeric_match'])
        rows[0]['V_new'] = 1.2
        self.assertFalse(compare.check_gcl(rows)['numeric_match'])


class RunnerTests(unittest.TestCase):
    def test_prepare_generates_only_specification_and_reference(self):
        import run_case
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'prepared'
            result = run_case.prepare('film_plug',out)
            self.assertEqual(result['status'],'PREPARED_NOT_RUN')
            self.assertTrue((out/'case-spec.json').exists())
            self.assertFalse((out/'observations.csv').exists())
            with self.assertRaises(ValueError):
                run_case.prepare('film_plug',out)

    def test_external_data_case_cannot_be_run_or_prepared_as_complete(self):
        import run_case
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_case.prepare('tacot1',Path(tmp)/'prepared')

    def test_run_rejects_missing_real_executables_without_generating_output(self):
        import run_case
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_case.execute('film_plug',Path(tmp)/'run',Path(tmp)/'missing-adapter',Path(tmp)/'missing-solver')
            self.assertFalse((Path(tmp)/'run').exists())


if __name__ == '__main__':
    unittest.main()
