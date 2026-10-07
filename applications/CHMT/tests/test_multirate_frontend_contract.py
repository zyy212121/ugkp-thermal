#!/usr/bin/env python3
"""Read-only source/build contract checks, not an OpenFOAM or CUDA run."""
from pathlib import Path
import re
import unittest

APP = Path(__file__).resolve().parents[1]

class MultirateFrontendContract(unittest.TestCase):
    def test_build_identity_accepts_foundation_source_install_path(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        # Metadata-only preflight. No marker header is compiled, and the
        # generated identity must remain explicitly COMPILE_ONLY.
        with tempfile.TemporaryDirectory(prefix='chmt-build-identity-') as temporary:
            root = Path(temporary)
            project = root / 'OpenFOAM-10'
            header = project / 'src/finiteVolume/lnInclude/fvCFD.H'
            header.parent.mkdir(parents=True)
            header.touch()
            env = dict(os.environ, WM_PROJECT_DIR=str(project), WM_PROJECT_VERSION='10',
                       PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, str(APP / 'devtools/build_info.py'), '--compile-only',
                       '--species', 'S0,S1', '--output', str(root / 'generated')]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            identity = json.loads((root / 'generated/build.json').read_text())
            self.assertEqual(identity['artifact_kind'], 'COMPILE_ONLY')
            self.assertEqual(identity['openfoam_directory'], str(project))
            header.unlink()
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)

    def test_openfoam_preflight_uses_version_and_headers_not_path_spelling(self):
        # Execute only the real scripts' preflight predicates. These empty
        # filesystem markers are never compiled or treated as an OF runtime.
        import os
        import subprocess
        import tempfile
        scripts = ('Allwmake', 'devtools/check_frontend.sh',
                   'devtools/multirate/check_cpu_material_of.sh')
        with tempfile.TemporaryDirectory(prefix='chmt-of-preflight-') as temporary:
            root = Path(temporary)
            for script in scripts:
                source = (APP / script).read_text()
                guard = re.search(r'^\[\[.*WM_PROJECT_VERSION.*?\]\]', source, re.M).group(0)
                def accepts(project, version):
                    env = dict(os.environ, WM_PROJECT_DIR=str(project), WM_PROJECT_VERSION=version)
                    return subprocess.run(['bash', '-c', guard], env=env,
                                          capture_output=True).returncode == 0
                for name in ('OpenFOAM-10', 'openfoam10', 'custom-foundation-install'):
                    project = root / name
                    header = project / 'src/finiteVolume/lnInclude/fvCFD.H'
                    header.parent.mkdir(parents=True, exist_ok=True)
                    header.touch()
                    with self.subTest(script=script, project=name):
                        self.assertTrue(accepts(project, '10'))
                        self.assertFalse(accepts(project, '11'))
                    header.unlink()
                    with self.subTest(script=script, missing_header=name):
                        self.assertFalse(accepts(project, '10'))

    def test_distinct_clock_configuration_exists(self):
        path = APP / 'configuration/MultirateIO.H'
        self.assertTrue(path.exists(), 'separate multirate dictionary parser is missing')
        source = path.read_text()
        for key in ('couplingInterval', 'gasMaxDt', 'materialMaxSubstep',
                    'driveRelativeTolerance', 'driveSourceFraction', 'driveTractionScale'):
            self.assertIn('"' + key + '"', source)
        self.assertIn('"Multirate"', source)
        self.assertIn('"LegacyExplicit"', source)
        self.assertIn('"Standalone"', source)
        self.assertIn('multirate.couplingInterval is required', source)

    def test_cpu_library_is_separate_from_cuda_sources(self):
        script = (APP / 'Allwmake').read_text()
        self.assertIn('wmake libso', script)
        cpu_files = APP / 'materials/Make/files'
        self.assertTrue(cpu_files.exists(), 'CPU material wmake target is missing')
        sources = cpu_files.read_text()
        for name in ('CpuMaterialDriver.C', 'MaterialTransport.C', 'CpuFilmDriver.C', 'CpuSurfaceInterface.C'):
            self.assertIn(name, sources)
        self.assertIn('libCHMTCpuMaterial', sources)
        self.assertIn('-lCHMTCpuMaterial', (APP / 'Make/options').read_text())
        cuda_array = re.search(r'cu_sources=\((.*?)\)', script, re.S).group(1)
        self.assertNotRegex(cuda_array, r'Cpu|\.C(?:\s|$)')

    def test_generated_library_headers_do_not_change_source_identity(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('source_fingerprint', APP / 'devtools/source_fingerprint.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn('lnInclude', module.EXCLUDED)
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory(prefix='chmt-source-identity-') as temporary:
            root = Path(temporary)
            app = root / 'applications/CHMT'
            app.mkdir(parents=True)
            header = app / 'Driver.H'
            header.write_text('#pragma once\n')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=CHMT test fixture',
                            '-c', 'user.email=chmt-test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            initial = module.manifest(app)['source_fingerprint']
            generated = app / 'materials/lnInclude'
            generated.mkdir(parents=True)
            (generated / 'Driver.H').symlink_to(header)
            self.assertEqual(initial, module.manifest(app)['source_fingerprint'])
            header.write_text('#pragma once\n// actual source changed\n')
            self.assertNotEqual(initial, module.manifest(app)['source_fingerprint'])

    def test_mass_roundoff_is_exposed_in_window_output(self):
        header = (APP / 'io/MultirateEvolution.H').read_text()
        output = (APP / 'io/MultirateOutput.H').read_text()
        self.assertIn('numericalMassRoundoff', header)
        self.assertIn('numerical_mass_roundoff', output)
        self.assertIn('r.numericalMassRoundoff', output)

    def test_standalone_preserves_frozen_adapter_artifact_kind(self):
        source = (APP / 'CHMT.C').read_text()
        self.assertIn('"CUDA_TIME_EVOLUTION"', source)
        self.assertIn('"execution-mode.json"', source)

    def test_main_uses_macro_controller_without_gas_cap_on_target(self):
        source = (APP / 'CHMT.C').read_text()
        self.assertIn('advanceCoupledWindow(', source)
        self.assertIn('CpuMaterialDriver', source)
        self.assertIn('solidSource.get()', source)
        self.assertIn('requireSynchronizedState', source)
        self.assertIn('windows.csv', source)
        self.assertIn('multirate?state.time<target:', source)
        self.assertIn('multirate?state.time==target:', source)

if __name__ == '__main__':
    unittest.main()
