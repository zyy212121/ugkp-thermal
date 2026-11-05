"""Exercise build output paths without compiling CUDA or requiring OpenFOAM."""
import os
import shutil
import subprocess

import pytest

from validation.tools.paths import SOURCE_ROOT


@pytest.mark.parametrize("build_kind", ["default", "relative", "absolute"])
def test_build_archive_path_survives_solver_chdir(tmp_path, build_kind):
    source = tmp_path / "source"
    solver = source / "applications/gasUGKP"
    solver.mkdir(parents=True)
    script = source / "tool/gasUGKP/build_cuda_solver.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE_ROOT / "tool/gasUGKP/build_cuda_solver.sh", script)
    shutil.copyfile(SOURCE_ROOT / "tool/cuda_build_flags.sh", source / "tool/cuda_build_flags.sh")
    caller = tmp_path / "caller"
    caller.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()

    mocks = {
        "nvcc": """#!/bin/bash
set -eu
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then touch "$2"; exit 0; fi
    shift
done
exit 1
""",
        "ar": """#!/bin/bash
set -eu
test "$1" = rcs
test -f "$3"
touch "$2"
""",
        "wmake": """#!/bin/bash
set -eu
test "$PWD" = "$EXPECTED_SOLVER"
archive=${UGKWP_CUDA_EXE_LIBS%% *}
test -f "$archive"
printf '%s\\n' "$archive" > "$LINK_LOG"
touch "$FOAM_USER_APPBIN/gasUGKP"
chmod +x "$FOAM_USER_APPBIN/gasUGKP"
""",
    }
    for name, body in mocks.items():
        tool = tools / name
        tool.write_text(body)
        tool.chmod(0o755)

    link_log = tmp_path / "linked-archive"
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
               WM_PROJECT_DIR="/mock/OpenFOAM", WM_OPTIONS="mock-options",
               FOAM_USER_APPBIN=str(tmp_path / "apps"), NVCC=str(tools / "nvcc"),
               EXPECTED_SOLVER=str(solver), LINK_LOG=str(link_log),
               UGKP_DEVELOPMENT_PROBES="0", GAS_UGKP_FMAD="1")
    env.pop("GAS_UGKP_BUILD", None)
    if build_kind == "default":
        build = solver / "private_backend/build/mock-options"
    elif build_kind == "relative":
        env["GAS_UGKP_BUILD"] = "build/pr25"
        build = caller / "build/pr25"
    else:
        build = tmp_path / "absolute-build"
        env["GAS_UGKP_BUILD"] = str(build)

    result = subprocess.run(["bash", str(script)], cwd=caller, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "GpuResidentStrict.o").is_file()
    assert (build / "libugkwp_cuda_backend.a").is_file()
    assert link_log.read_text().strip() == str(build / "libugkwp_cuda_backend.a")
