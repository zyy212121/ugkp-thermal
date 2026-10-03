#!/usr/bin/env python3
"""CPU regressions for mirror protection, field closure and fixture reuse."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(sys.argv.pop(1)).resolve()
PAIR = ROOT.parent


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


class Contracts(unittest.TestCase):
    def test_managed_mirror_check_sync_and_unlisted_protection(self):
        tool = ROOT / 'tools/managed_mirrors.py'
        baseline = run(tool, '--pair-root', PAIR)
        self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
        with tempfile.TemporaryDirectory(prefix='mirror-contract-') as tmp:
            pair = Path(tmp)
            for repo in ('gpu-riemann-gkp-main', 'ugkp-thermal'):
                shutil.copytree(PAIR / repo, pair / repo, symlinks=True,
                    ignore=shutil.ignore_patterns('.git','examples','results','build','bin','lib','build_logs','linux64*','__pycache__','.pytest_cache','*.o','*.a','*.so','*.log','*.dep'))
            # A successful first build must not make the next source gate fail.
            generated = {
                'applications/gasUGKP/Make/linux64GccDPInt32Opt/files': b'generated make inputs',
                'applications/gasUGKP/Make/linux64GccDPInt32Opt/gpu/client.o': b'object',
                'applications/gasUGKP/private_backend/build/linux64GccDPInt32Opt/backend.o': b'object',
                'applications/gasUGKP/private_backend/build_logs/run.log': b'build log',
                'applications/gasUGKP/private_backend/libugkwp_cuda_backend.a': b'archive',
                'applications/gasUGKP/lnInclude/generated.H': b'generated include',
            }
            generated_root = pair / 'gpu-riemann-gkp-main'
            for relative, payload in generated.items():
                artifact = generated_root / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(payload)
            post_build = run(tool, '--pair-root', pair)
            self.assertEqual(post_build.returncode, 0, post_build.stdout + post_build.stderr)
            for relative, payload in generated.items():
                self.assertEqual((generated_root / relative).read_bytes(), payload)
            # Arbitrary files remain protected; do not globally ignore suffixes.
            unexpected = generated_root / 'applications/gasUGKP/unlisted.a'
            unexpected.write_bytes(b'unregistered input')
            rejected = run(tool, '--pair-root', pair)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('unlisted.a', rejected.stdout + rejected.stderr)
            unexpected.unlink()
            rejected = run(tool, '--pair-root', pair, '--register',
                'applications/gasUGKP/private_backend/libugkwp_cuda_backend.a')
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('generated build output', rejected.stdout + rejected.stderr)
            target = pair / 'gpu-riemann-gkp-main/common/GpuParticleFields.cuh'
            original = target.read_bytes()
            target.write_bytes(original + b'\n// injected drift\n')
            sentinel = pair / 'gpu-riemann-gkp-main/LOCAL_ONLY'
            sentinel.write_text('retain me')
            failed = run(tool, '--pair-root', pair)
            self.assertNotEqual(failed.returncode, 0, failed.stdout)
            self.assertIn('GpuParticleFields.cuh', failed.stdout + failed.stderr)
            self.assertTrue(target.read_bytes().endswith(b'// injected drift\n'))
            repaired = run(tool, '--pair-root', pair, '--sync')
            self.assertEqual(repaired.returncode, 0, repaired.stdout + repaired.stderr)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(sentinel.read_text(), 'retain me')
            # A new overlapping input must be registered before it is silently mirrored.
            for repo in ('gpu-riemann-gkp-main', 'ugkp-thermal'):
                (pair / repo / 'common/NewUnmanaged.cuh').write_text('same bytes')
            unlisted = run(tool, '--pair-root', pair)
            self.assertNotEqual(unlisted.returncode, 0)
            self.assertIn('unmanaged', unlisted.stdout + unlisted.stderr)
            registered = run(tool, '--pair-root', pair, '--register', 'common/NewUnmanaged.cuh')
            self.assertEqual(registered.returncode, 0, registered.stdout + registered.stderr)
            escaped = run(tool, '--pair-root', pair, '--register', 'common/../../outside.cuh')
            self.assertNotEqual(escaped.returncode, 0)

    def test_field_lifecycle_and_schema_reject_mutations(self):
        tool = ROOT / 'tools/particle_field_contract.py'
        with tempfile.TemporaryDirectory(prefix='field-contract-') as tmp:
            out = Path(tmp)
            good = run(tool, '--root', ROOT, '--cpu', '--output', out / 'good')
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            copied = out / 'repo'
            shutil.copytree(ROOT, copied, symlinks=True)
            source = copied / 'applications/FSH/private_backend/GpuResidentStrict.cu'
            original = source.read_text()
            mutations = {
                'missing-release': ('    release(s->compactPuxOld);', ''),
                'missing-allocation': ('    rc |= allocate(s->puxOld, np, "cudaMalloc strict particle old ux");', ''),
                'missing-declaration': ('    double* puxOld = nullptr;', ''),
            }
            for label, (before, after) in mutations.items():
                self.assertIn(before, original)
                source.write_text(original.replace(before, after, 1))
                result = run(tool, '--root', copied, '--cpu', '--output', out / label)
                self.assertNotEqual(result.returncode, 0, label + result.stdout)
                source.write_text(original)
            wire = copied / 'common/GpuParticleRestartFields.inl'
            lines = wire.read_text().splitlines(True)
            first = next(i for i, line in enumerate(lines) if 'fn(v.px,' in line)
            lines[first], lines[first + 1] = lines[first + 1], lines[first]
            wire.write_text(''.join(lines))
            result = run(tool, '--root', copied, '--cpu', '--output', out / 'wrong-wire-order')
            self.assertNotEqual(result.returncode, 0, result.stdout)
            # Even changing the metadata and regenerating both sides cannot silently
            # change schema7. The real codec must still match the frozen binary golden.
            import json
            metadata = copied / 'common/ParticleFieldManifest.json'
            manifest = json.loads(metadata.read_text())
            fx = next(f for f in manifest['fields'] if f['name'] == 'px')
            fy = next(f for f in manifest['fields'] if f['name'] == 'py')
            fx['schema7'], fy['schema7'] = fy['schema7'], fx['schema7']
            metadata.write_text(json.dumps(manifest))
            generated = run(tool, '--root', copied, '--generate')
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            result = run(tool, '--root', copied, '--cpu', '--output', out / 'mutated-wire-schema')
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn('schema7 binary golden mismatch', result.stdout + result.stderr)

    def test_contact_fixture_can_reuse_output(self):
        script = ROOT / 'tests/fixtures/shared_operators/contact_age_theta.py'
        with tempfile.TemporaryDirectory(prefix='age-contract-') as tmp:
            out = Path(tmp) / 'fixture'
            for app, bits in [('FSH',64),('CHT',32),('CHT',64)]:
                for _ in range(2):
                    result = run(script, ROOT, out / (app+str(bits)), app, bits, '--prepare-only')
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
