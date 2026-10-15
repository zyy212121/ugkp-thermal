"""Exercise the actual manifest writer without compiling or running CUDA."""
import ast
import hashlib
import json

from validation.tools.paths import SOURCE_ROOT


def test_manifest_records_relocated_generators_and_preserves_existing_inputs(tmp_path):
    script = SOURCE_ROOT / 'validation/tools/low_level_regression.py'
    tree = ast.parse(script.read_text(), filename=str(script))
    index = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'hashes'
                         for target in node.targets))
    writer = compile(ast.Module(body=tree.body[index:index + 2], type_ignores=[]),
                     str(script), 'exec')
    root = tmp_path / 'source'
    out = tmp_path / 'evidence'
    out.mkdir()
    inputs = {
        'applications/CHT/gpu/sample.cu': b'production input\n',
        'validation/tests/fixtures/thermal_workers/gather_CHT.cu.in': b'template\n',
        'validation/tools/tests/fixtures/shared_operators/payload_commit.py': b'payload generator\n',
        'validation/tools/tests/fixtures/thermal_workers/s1_pipeline.py': b'integration generator\n',
    }
    for relative, content in inputs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ignored = root / 'validation/tools/tests/fixtures/__pycache__/ignored.py'
    ignored.parent.mkdir(parents=True)
    ignored.write_text('cached helper\n')
    namespace = {'ROOT': root, 'out': out, 'hashlib': hashlib, 'json': json}
    exec(writer, namespace)
    manifest = json.loads((out / 'source-sha256.json').read_text())
    assert manifest == {name: hashlib.sha256(content).hexdigest()
                        for name, content in inputs.items()}

    generator = 'validation/tools/tests/fixtures/thermal_workers/s1_pipeline.py'
    updated = b'updated integration generator\n'
    (root / generator).write_bytes(updated)
    exec(writer, namespace)
    refreshed = json.loads((out / 'source-sha256.json').read_text())
    assert refreshed[generator] == hashlib.sha256(updated).hexdigest()
    assert refreshed[generator] != manifest[generator]
