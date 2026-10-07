#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/chmt-interval.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
cd "$root"
for test in interval_contracts interval_audit; do
    "${CXX:-g++}" -std=c++14 -O2 -Wall -Wextra -Werror -pedantic \
        -Iapplications/CHMT "applications/CHMT/verification/multirate/$test.cpp" \
        -o "$scratch/$test"
    "$scratch/$test"
done
python3 - <<'PY'
from pathlib import Path
source = Path('applications/CHMT/coupling/IntervalContracts.H').read_text()
body = source.split('bool prepare(const GasIntervalRecord&', 1)[1].split('bool readyToCommit(', 1)[0]
assert 'std::map<std::size_t,Change> changed' in body
for forbidden in ('auto solid=solid_', 'auto film=film_', 'auto accounts=accounts_', 'sampleMaterialDonorPlan(', 'std::vector<IntervalDonorSum> withdrawals(accounts_.size())'):
    assert forbidden not in body, forbidden
gas = Path('applications/CHMT/gpu/GasWindowImplementation.cuh').read_text()
assert 'MaterialDonorReserve reserve=window.reserve' not in gas
assert 'window.reserve.prepare(record,reserveTransaction' in gas
assert 'window.reserve.commitPrepared(std::move(reserveTransaction))' in gas
print('sparse donor transaction source checks: PASS')
PY
