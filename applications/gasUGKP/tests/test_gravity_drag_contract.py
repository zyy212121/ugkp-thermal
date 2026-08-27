#!/usr/bin/env python3
                                                                     

from __future__ import annotations

from math import exp, isclose
from pathlib import Path

import pytest

from source_contract_utils import function_block


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def schiller_naumann_inverse_tau(
    rho_g: float,
    mu_g: float,
    rho_s: float,
    diameter: float,
    relative_speed: float,
) -> float:
    re = rho_g * diameter * relative_speed / max(mu_g, 1.0e-30)
    re_safe = max(re, 1.0e-12)
    if re_safe < 1000.0:
        cd = 24.0 / re_safe * (1.0 + 0.15 * re_safe**0.687)
    else:
        cd = 0.44
    return 0.75 * cd * rho_g * relative_speed / (
        rho_s * diameter + 1.0e-300
    )


def gidaspow_ergun_wen_yu_inverse_tau(
    rho_g: float,
    mu_g: float,
    alpha_g: float,
    rho_s: float,
    diameter: float,
    relative_speed: float,
    residual_re: float = 1.0e-3,
) -> float:
                                                                       
    re = rho_g * diameter * relative_speed / max(mu_g, 1.0e-30)
    if alpha_g >= 0.8:
        re_s = alpha_g * re
        if re_s < 1000.0:
            cds_res = 24.0 * (1.0 + 0.15 * re_s**0.687)
        else:
            cds_res = 0.44 * max(re_s, residual_re)
        cd_re = cds_res * alpha_g**-2.65
    else:
        cd_re = (4.0 / 3.0) * (
            150.0 * (1.0 - alpha_g) / alpha_g + 1.75 * re
        )
    return 0.75 * cd_re * mu_g / (rho_s * diameter * diameter)


def test_host_factory_has_base_and_default_schiller_naumann() -> None:
    host = read("gpu/GpuDragModel.H")
    assert "class GpuDragModel" in host
    assert "virtual label modelId() const = 0" in host
    assert "class GpuSchillerNaumannDragModel final" in host
    assert "GpuDragModel::New" in host
    assert 'lookupOrDefault<word>("dragModel", "SchillerNaumann")' in host
    assert "Unknown GPU dragModel" in host


def test_host_factory_has_explicit_gidaspow_model_and_validated_coeffs() -> None:
    host = read("gpu/GpuDragModel.H")
    assert "class GpuGidaspowErgunWenYuDragModel final" in host
    assert 'modelName_("GidaspowErgunWenYu")' in host
    assert 'subDict("GidaspowErgunWenYuCoeffs")' in host
    assert 'lookup("residualRe")' in host
    assert "std::isfinite(residualRe)" in host
    assert "residualRe <= 0" in host
    assert "return 1;" in host
    assert 'selected == "GidaspowErgunWenYu"' in host
    assert '<< "GidaspowErgunWenYu"' in host


def test_standard_gravity_is_optional_and_defaults_to_zero() -> None:
    fields = read("createFields.H")
    assert "uniformDimensionedVectorField g" in fields
    assert '"g",' in fields
    assert "IOobject::READ_IF_PRESENT" in fields
    assert 'dimensionedVector("g", dimAcceleration, vector::zero)' in fields


def test_protocol_carries_drag_and_gravity_as_pod() -> None:
    protocol = read("gpu/GpuBackendProtocol.H")
    assert "protocolMajor = 7" in protocol
    assert "protocolMinor = 0" in protocol
    for field in (
        "dragModel",
        "dragParameter0",
        "dragParameter1",
        "dragParameter2",
        "dragParameter3",
        "gravityX",
        "gravityY",
        "gravityZ",
    ):
        assert field in protocol


def test_every_backend_boundary_accepts_only_defined_model_ids() -> None:
    for relative in (
        "gpu/GpuBackendClient.C",
        "private_backend/GpuBackendServer.C",
        "private_backend/GpuResidentStrict.cu",
    ):
        source = read(relative)
        assert "dragModel < 0" in source
        assert "dragModel > 2" in source
        assert "dragModel == 1" in source
        assert "dragParameter0 <= 0.0" in source


def test_device_drag_model_is_static_and_shared_by_both_consumers() -> None:
    drag = read("private_backend/GpuDragModels.cuh")
    core = read("private_backend/GpuResidentStrict.cu")
    assert "struct DragInput" in drag
    assert "gasVolumeFraction" in drag
    assert "struct SchillerNaumannDeviceDrag" in drag
    assert "struct GidaspowErgunWenYuDeviceDrag" in drag
    assert "static double inverseResponseTime" in drag
    assert "virtual" not in drag
    assert "snCdDevice" not in core

    coupling = function_block(core, "applyEulerianGasSolidCouplingKernel")
    relax_one = function_block(core, "relaxOneParticleToResidentGas")
    relax_kernel = function_block(core, "relaxParticlesToResidentGasKernel")
    assert "template<class DragModel>" in core[: core.index(coupling)]
    assert "DragModel::inverseResponseTime" in coupling
    assert "DragModel::inverseResponseTime" in relax_one
    assert "relaxOneParticleToResidentGas<DragModel>" in relax_kernel


def test_model_switch_occurs_only_in_host_launch_helpers() -> None:
    core = read("private_backend/GpuResidentStrict.cu")
    coupling_launch = function_block(core, "launchEulerianGasSolidCoupling")
    relax_launch = function_block(core, "launchParticleDragRelaxation")
    assert "switch" in coupling_launch
    assert "switch" in relax_launch
    assert "SchillerNaumannDeviceDrag" in coupling_launch
    assert "SchillerNaumannDeviceDrag" in relax_launch
    assert "GidaspowErgunWenYuDeviceDrag" in coupling_launch
    assert "GidaspowErgunWenYuDeviceDrag" in relax_launch
    assert "case 1:" in coupling_launch
    assert "case 1:" in relax_launch
    assert "dragModel" not in function_block(
        core, "applyEulerianGasSolidCouplingKernel"
    )
    assert "dragModel" not in function_block(
        core, "relaxParticlesToResidentGasKernel"
    )


def test_gravity_uses_exact_gas_energy_and_particle_drag_impulse() -> None:
    core = read("private_backend/GpuResidentStrict.cu")
    gas = function_block(core, "applyGasGravitySourceKernel")
    particle = function_block(core, "relaxOneParticleToResidentGas")
    assert "0.5*rho*dt*dt*gravitySquared" in gas
    assert "momX*gx + momY*gy + momZ*gz" in gas
    assert "s.gravityX" in particle
    assert "s.gravityY" in particle
    assert "s.gravityZ" in particle
    assert "(1.0 - alpha)/(invTauDrag + 1.0e-300)" in particle


def test_schiller_naumann_reference_values_are_continuous_and_finite() -> None:
    low = schiller_naumann_inverse_tau(1.2, 1.8e-5, 2500.0, 1.0e-3, 0.01)
    high = schiller_naumann_inverse_tau(1.2, 1.8e-5, 2500.0, 1.0e-3, 30.0)
    assert low > 0.0
    assert high > low

    speed_at_re_1000 = 1000.0 * 1.8e-5 / (1.2 * 1.0e-3)
    below = schiller_naumann_inverse_tau(
        1.2, 1.8e-5, 2500.0, 1.0e-3, speed_at_re_1000 * (1.0 - 1.0e-8)
    )
    above = schiller_naumann_inverse_tau(
        1.2, 1.8e-5, 2500.0, 1.0e-3, speed_at_re_1000 * (1.0 + 1.0e-8)
    )
    assert abs(above - below) / max(abs(above), abs(below)) < 0.03


def test_gidaspow_reference_covers_openfoam_branch_conventions() -> None:
    args = (1.2, 1.8e-5, 2500.0, 3.0e-4)
    ergun = gidaspow_ergun_wen_yu_inverse_tau(
        args[0], args[1], 0.79, args[2], args[3], 0.5
    )
    wen_yu = gidaspow_ergun_wen_yu_inverse_tau(
        args[0], args[1], 0.8, args[2], args[3], 0.5
    )
    stokes = gidaspow_ergun_wen_yu_inverse_tau(
        args[0], args[1], 1.0, args[2], args[3], 0.0
    )
    assert ergun > 0.0
    assert wen_yu > 0.0
    assert stokes == pytest.approx(18.0 * args[1] / (args[2] * args[3] ** 2))

    alpha_g = 0.9
    speed_at_res_1000 = 1000.0 * args[1] / (
        alpha_g * args[0] * args[3]
    )
    at_transition = gidaspow_ergun_wen_yu_inverse_tau(
        args[0], args[1], alpha_g, args[2], args[3], speed_at_res_1000
    )
    expected = (
        0.75
        * (0.44 * 1000.0 * alpha_g**-2.65)
        * args[1]
        / (args[2] * args[3] ** 2)
    )
    assert at_transition == pytest.approx(expected)


def test_exact_constant_acceleration_preserves_internal_energy() -> None:
    rho = 2.0
    velocity = (3.0, -2.0, 0.5)
    gravity = (-1.0, -9.0, 2.0)
    dt = 0.125
    internal = 40.0
    momentum = tuple(rho * value for value in velocity)
    total = internal + 0.5 * sum(value * value for value in momentum) / rho
    dot = sum(m * g for m, g in zip(momentum, gravity))
    gravity_squared = sum(value * value for value in gravity)
    momentum_new = tuple(
        m + rho * g * dt for m, g in zip(momentum, gravity)
    )
    total_new = total + dt * dot + 0.5 * rho * dt * dt * gravity_squared
    internal_new = total_new - 0.5 * sum(
        value * value for value in momentum_new
    ) / rho
    assert internal_new == pytest.approx(internal)


def test_particle_constant_acceleration_limit_is_exact_when_drag_is_zero() -> None:
    dt = 0.03125
    inv_tau = 0.0
    alpha = exp(-dt * max(inv_tau, 0.0))
    impulse = (
        dt * (1.0 - 0.5 * dt * inv_tau + (dt * inv_tau) ** 2 / 6.0)
        if dt * inv_tau < 1.0e-6
        else (1.0 - alpha) / (inv_tau + 1.0e-300)
    )
    assert isclose(alpha, 1.0)
    assert isclose(impulse, dt)
