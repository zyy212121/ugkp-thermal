#!/usr/bin/env python3
"""Check the complete declared mirror inventory. Use --sync for one-way repair.

common is maintained in ugkp-thermal; gasUGKP input is maintained in
gpu-riemann-gkp-main. Both remain self-contained checkouts. No deletion,
recursive copy, build, or GPU action is performed by this tool.
"""
from pathlib import Path, PurePosixPath
import argparse
import json
import os
import shutil
import sys

REPOS = ('gpu-riemann-gkp-main', 'ugkp-thermal')


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe manifest path: ' + relative)
    target = root.joinpath(*rel.parts)
    # The leaf itself may be a managed symbolic link. Its parent may not escape.
    if root.resolve() not in (target.parent.resolve(), *target.parent.resolve().parents):
        raise ValueError('manifest parent escapes checkout: ' + relative)
    return target


def is_generated_inventory_path(relative):
    """Exclude known compiler outputs, never arbitrary source-file suffixes."""
    parts = PurePosixPath(relative).parts
    if '__pycache__' in parts or '.pytest_cache' in parts:
        return True
    gas = 'applications/gasUGKP/'
    if relative.startswith(gas + 'Make/') and len(parts) > 4:
        platform = parts[3]
        if platform.startswith(('linux', 'darwin', 'mingw')):
            return True
    return (relative.startswith(tuple(gas + directory for directory in
                ('private_backend/build/', 'private_backend/build_logs/', 'lnInclude/')))
            or relative == gas + 'private_backend/libugkwp_cuda_backend.a')


def signature(path):
    if path.is_symlink():
        return ('symlink', os.readlink(path))
    if path.is_file():
        return ('file', path.read_bytes())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair-root', type=Path, help='Directory containing both named checkouts')
    ap.add_argument('--sync', action='store_true', help='Copy only listed paths in their declared upstream direction')
    ap.add_argument('--register', action='append', default=[], metavar='RELATIVE_PATH',
                    help='Explicitly register one new common/ or applications/gasUGKP/ input in both manifests')
    args = ap.parse_args()
    local = Path(__file__).resolve().parents[1]
    pair = (args.pair_root or Path(os.environ.get('UGKP_MANAGED_MIRROR_ROOT', local.parent))).resolve()
    paired = all((pair / repo).is_dir() for repo in REPOS)
    if args.pair_root and not paired:
        ap.error('--pair-root must contain both named checkouts')
    if args.sync and not paired:
        ap.error('--sync requires both named checkouts')
    if args.register and not paired:
        ap.error('--register requires both named checkouts')
    manifest_path = (pair / 'ugkp-thermal' if paired else local) / 'tools/managed_mirrors.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1:
        raise ValueError('unsupported managed mirror manifest schema')
    if args.register:
        known = {entry['path'] for entry in manifest['mirrors']}
        additions = []
        for relative in sorted(set(args.register)):
            if relative in known:
                raise ValueError('already managed: ' + relative)
            if relative.startswith('common/'):
                upstream, downstream = 'ugkp-thermal', 'gpu-riemann-gkp-main'
            elif relative.startswith('applications/gasUGKP/'):
                upstream, downstream = 'gpu-riemann-gkp-main', 'ugkp-thermal'
            else:
                raise ValueError('--register accepts only common/ or applications/gasUGKP/ inputs')
            if is_generated_inventory_path(relative):
                raise ValueError('cannot register generated build output: ' + relative)
            if signature(safe_path(pair / upstream, relative)) is None:
                raise ValueError('cannot register missing upstream: ' + relative)
            safe_path(pair / downstream, relative)
            additions.append(dict(path=relative, upstream=upstream, downstream=downstream))
        for entry in additions:
            for repo in REPOS:
                manifest['local_only'][repo] = [p for p in manifest['local_only'][repo] if p != entry['path']]
        manifest['mirrors'] = sorted(manifest['mirrors'] + additions, key=lambda entry: entry['path'])
        payload = json.dumps(manifest, indent=2) + '\n'
        for repo in REPOS:
            destination = safe_path(pair / repo, 'tools/managed_mirrors.json')
            destination.write_text(payload)
        print(f'Registered {len(additions)} explicit input paths')
    errors, drift, planned = [], [], []
    available = {repo: pair / repo for repo in REPOS} if paired else {}
    if not paired:
        # Each shipped manifest is identical: discover identity from the supported input roots.
        name = 'ugkp-thermal' if (local / 'applications/CHT').is_dir() else 'gpu-riemann-gkp-main'
        available = {name: local}
    declared = {repo: set(manifest['local_only'][repo]) for repo in REPOS}
    for entry in manifest['mirrors']:
        upstream, downstream, relative = entry['upstream'], entry['downstream'], entry['path']
        if upstream not in REPOS or downstream not in REPOS or upstream == downstream:
            raise ValueError('invalid mirror direction')
        for repo in (upstream, downstream):
            if relative in declared[repo]:
                raise ValueError('duplicate manifest member: ' + repo + '/' + relative)
            declared[repo].add(relative)
        if paired:
            src, dst = safe_path(pair / upstream, relative), safe_path(pair / downstream, relative)
            source = signature(src)
            if source is None:
                errors.append('missing upstream: ' + upstream + '/' + relative)
            elif source != signature(dst):
                drift.append(relative)
                planned.append((src, dst))
    for repo, root in available.items():
        observed = set()
        for scope in manifest['inventory_roots']:
            for path in (root / scope).rglob('*'):
                if path.is_file() or path.is_symlink():
                    relative = path.relative_to(root).as_posix()
                    if not is_generated_inventory_path(relative):
                        observed.add(relative)
        expected = {p for p in declared[repo] if any(p.startswith(s + '/') for s in manifest['inventory_roots'])}
        errors.extend('unmanaged: ' + repo + '/' + p for p in sorted(observed - expected))
        # sync can restore a missing downstream leaf, but cannot invent local/upstream data.
        recoverable = {dst.relative_to(root).as_posix() for _, dst in planned if root in dst.parents} if args.sync else set()
        errors.extend('missing declared input: ' + repo + '/' + p for p in sorted(expected - observed - recoverable))
        for relative in sorted(declared[repo] - expected):
            if signature(safe_path(root, relative)) is None:
                if not (args.sync and any(dst == root / relative for _, dst in planned)):
                    errors.append('missing managed auxiliary: ' + repo + '/' + relative)
    if errors:
        print('\n'.join(errors))
        return 1
    if args.sync:
        # All paths and inventory have been validated before the first mutation.
        for src, dst in planned:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            if src.is_symlink():
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(os.readlink(src))
            else:
                shutil.copy2(src, dst)
        print(f'SYNC {len(planned)} whitelisted paths; local-only inputs preserved')
        return 0
    if drift:
        print('\n'.join('mirror drift: ' + p for p in drift))
        return 1
    print(f'PASS managed inventory ({len(manifest["mirrors"])} mirrors; ' + ('paired comparison)' if paired else 'standalone inputs; peer comparison unavailable)'))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print('managed mirror error:', exc, file=sys.stderr)
        raise SystemExit(1)
