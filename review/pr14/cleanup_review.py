#!/usr/bin/env python3
"""PR14_REVIEW_ONLY: archive and remove explicitly inventoried review records.
No action by default. This tool never deletes a production file or a README.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

IDENTITY = 'PR14_REVIEW_ONLY'
PREFIX = 'review/pr14/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve_file(root, name, owned=False):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or str(p) != name:
        raise ValueError('Noncanonical relative path: ' + name)
    if any(part.lower().startswith('readme') for part in p.parts):
        raise ValueError('README paths are never permitted: ' + name)
    if owned and not name.startswith(PREFIX):
        raise ValueError('Not an owned review path: ' + name)
    target = root / name
    for part in [target, *target.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Symlink is not permitted: ' + name)
    target.resolve().relative_to(root)
    if not target.is_file():
        raise ValueError('Expected existing regular file: ' + name)
    return target


def plan(root, manifest):
    if manifest.get('identity') != IDENTITY:
        raise ValueError('Manifest identity mismatch')
    owned = manifest['owned_paths']
    if not owned or len(owned) != len(set(owned)):
        raise ValueError('Empty/duplicate ownership list')
    files = {p: resolve_file(root, p, True).read_bytes() for p in owned}
    actual = {str(p.relative_to(root)) for p in (root / PREFIX).rglob('*') if p.is_file() or p.is_symlink()}
    if actual != set(owned):
        raise ValueError('Ownership inventory differs from disk; update and review manifest first: ' + repr(sorted(actual ^ set(owned))))
    edits = {}
    markers = manifest.get('source_markers', [])
    for entry in markers:
        name = entry['path']
        if name in files or name.startswith(PREFIX) or name in edits:
            raise ValueError('Invalid/duplicate source marker path: ' + name)
        before = resolve_file(root, name).read_bytes()
        if digest(before) != entry['source_sha256_before_cleanup']:
            raise ValueError('Source has changed: ' + name)
        lines = before.splitlines(keepends=True)
        for encoded in entry['exact_standalone_lines_base64']:
            line = base64.b64decode(encoded, validate=True)
            if not re.fullmatch(rb'[ \t]*// PR14_REVIEW_ONLY R014-[A-Za-z0-9-]+(?:[ \t:][^\r\n]*)?\r?\n', line):
                raise ValueError('Only exact standalone marker comment lines can be removed')
            if lines.count(line) != 1:
                raise ValueError('Marker is absent or duplicated: ' + name)
            lines.remove(line)
        after = b''.join(lines)
        if not after or after == before:
            raise ValueError('Marker edit must leave a nonempty production file')
        if digest(after) != entry['source_sha256_after_cleanup']:
            raise ValueError('Expected production remainder mismatch: ' + name)
        edits[name] = (before, after)
    return files, edits


def execute(root, manifest, archive, approval):
    if not approval.strip():
        raise ValueError('An explicit final-review approval reference is required for --apply')
    files, edits = plan(root, manifest)
    archive = archive.absolute()
    if archive.is_symlink() or any(p.is_symlink() for p in archive.parents):
        raise ValueError('Archive path cannot use symlinks')
    if archive.resolve().is_relative_to((root / PREFIX).resolve()):
        raise ValueError('Archive must be outside the removable review directory')
    if archive.exists():
        raise ValueError('Archive must be new; existing archives are never overwritten')
    if not archive.parent.is_dir():
        raise ValueError('Archive parent must already exist')
    receipt = {'identity': IDENTITY, 'approval_reference': approval,
               'records': {k: digest(v) for k, v in files.items()},
               'source_marker_edits': {k: {'before': digest(v[0]), 'after': digest(v[1])} for k, v in edits.items()}}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)
        for k, v in edits.items():
            z.writestr('source-before-cleanup/' + k, v[0])
        z.writestr('ARCHIVE-RECEIPT.json', json.dumps(receipt, indent=2) + '\n')
    with archive.open('rb') as f:
        os.fsync(f.fileno())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC check failed; no cleanup performed')
        for k, v in files.items():
            if z.read(k) != v:
                raise ValueError('Archive record verification failed; no cleanup performed')
        for k, v in edits.items():
            if z.read('source-before-cleanup/' + k) != v[0]:
                raise ValueError('Source archive verification failed; no cleanup performed')
    # Revalidate all inputs after the archive is complete, before the first mutation.
    files2, edits2 = plan(root, manifest)
    if files2 != files or edits2 != edits:
        raise ValueError('Files changed while archiving; no cleanup performed')
    for k, (before, after) in edits.items():
        target = resolve_file(root, k)
        if target.read_bytes() != before:
            raise ValueError('Source changed before marker removal; stop and inspect archive')
        target.write_bytes(after)
        if target.read_bytes() != after:
            raise ValueError('Marker removal write verification failed; recover from archive')
    for k, before in files.items():
        target = resolve_file(root, k, True)
        if target.read_bytes() != before:
            raise ValueError('Review record changed before deletion; stop and inspect archive')
        target.unlink()
    for d in sorted((root / PREFIX).rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir():
            d.rmdir()
    (root / PREFIX).rmdir()
    return {'identity': IDENTITY, 'applied': True, 'archive': str(archive), 'archive_sha256': digest(archive.read_bytes()), 'removed_review_files': len(files), 'source_comment_edits': len(edits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--apply', action='store_true', help='Explicitly archive then remove only inventoried PR14 review artifacts')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--approval-reference', default='', help='Reference to final review/user cleanup approval, not approval granted by this tool')
    args = parser.parse_args()
    root = args.repo.resolve()
    manifest = json.loads((root / PREFIX / 'manifest.json').read_text())
    files, edits = plan(root, manifest)
    if args.apply:
        if not args.archive:
            parser.error('--apply requires --archive and --approval-reference')
        result = execute(root, manifest, args.archive, args.approval_reference)
    else:
        result = {'identity': IDENTITY, 'mode': 'check/dry-run', 'applied': False, 'owned_records': len(files), 'source_comment_edits': len(edits), 'archive_created': False, 'source_bytes_changed': 0}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
