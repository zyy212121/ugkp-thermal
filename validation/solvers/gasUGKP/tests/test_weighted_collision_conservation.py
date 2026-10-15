from __future__ import annotations

import numpy as np


def conservative_projection(
    weights: np.ndarray,
    sampled_velocity: np.ndarray,
    target_velocity: np.ndarray,
    target_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.sum(weights, dtype=np.longdouble)
    sample_momentum = np.sum(
        weights[:, None].astype(np.longdouble)
        * sampled_velocity.astype(np.longdouble),
        axis=0,
        dtype=np.longdouble,
    )
    sample_mean = sample_momentum / mass
    fluctuations = sampled_velocity.astype(np.longdouble) - sample_mean
    sample_thermal = np.longdouble(0.5) * np.sum(
        weights.astype(np.longdouble)
        * np.sum(fluctuations * fluctuations, axis=1),
        dtype=np.longdouble,
    )
    target_thermal = np.longdouble(1.5) * mass * np.longdouble(target_theta)
    if target_thermal <= 0:
        return np.repeat(target_velocity[None, :], len(weights), axis=0), np.zeros(len(weights))
    if sample_thermal <= np.finfo(np.float64).tiny:
        return np.repeat(target_velocity[None, :], len(weights), axis=0), np.full(len(weights), target_theta)
    scale = np.sqrt(target_thermal / sample_thermal)
    corrected = target_velocity.astype(np.longdouble) + scale * fluctuations
    return corrected.astype(np.float64), np.zeros(len(weights))


def invariants(weights: np.ndarray, velocity: np.ndarray, theta: np.ndarray):
    w = weights.astype(np.longdouble)
    u = velocity.astype(np.longdouble)
    mass = np.sum(w, dtype=np.longdouble)
    momentum = np.sum(w[:, None] * u, axis=0, dtype=np.longdouble)
    energy = np.sum(
        w * (np.longdouble(0.5) * np.sum(u * u, axis=1) + np.longdouble(1.5) * theta),
        dtype=np.longdouble,
    )
    return mass, momentum, energy


def test_unequal_weight_projection_conserves_target_invariants() -> None:
    weights = np.array([0.125, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    sampled = np.array(
        [[-2.0, 0.5, 1.0], [3.0, -1.0, 0.0], [0.0, 4.0, -2.0],
         [1.0, 1.5, 2.5], [-0.5, -2.0, 3.0]],
        dtype=np.float64,
    )
    target_u = np.array([1.25, -0.75, 0.375], dtype=np.float64)
    target_theta = 2.5
    corrected, theta = conservative_projection(weights, sampled, target_u, target_theta)
    mass, momentum, energy = invariants(weights, corrected, theta)
    target_mass = np.sum(weights, dtype=np.longdouble)
    target_momentum = target_mass * target_u.astype(np.longdouble)
    target_energy = target_mass * (
        np.longdouble(0.5) * np.dot(target_u, target_u)
        + np.longdouble(1.5) * np.longdouble(target_theta)
    )
    np.testing.assert_allclose(momentum, target_momentum, rtol=5e-15, atol=5e-15)
    np.testing.assert_allclose(energy, target_energy, rtol=5e-15, atol=5e-15)
    assert mass == target_mass


def test_zero_variance_pool_retains_energy_as_unresolved_theta() -> None:
    weights = np.array([0.25, 2.0], dtype=np.float64)
    sampled = np.repeat(np.array([[3.0, -1.0, 2.0]]), 2, axis=0)
    target_u = np.array([0.5, 1.0, -0.25])
    target_theta = 1.75
    corrected, theta = conservative_projection(weights, sampled, target_u, target_theta)
    _, momentum, energy = invariants(weights, corrected, theta)
    mass = np.sum(weights, dtype=np.longdouble)
    np.testing.assert_allclose(momentum, mass * target_u, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        energy,
        mass * (0.5 * np.dot(target_u, target_u) + 1.5 * target_theta),
        rtol=0,
        atol=2e-15,
    )
