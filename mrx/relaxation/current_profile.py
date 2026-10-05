"""A prescribed current profile for the resistive drive.

The profile is the net toroidal current ``I(s)`` inside the flux surface ``s``, the toroidal flux it encloses
over the total, positive along the field. :class:`CurrentProfile` has two sources:

- :meth:`~CurrentProfile.vmec`: VMEC's inputs, the total current ``curtor`` and polynomial coefficients ``ac`` in
  ``s``. With ``pcurr_type = "power_series"`` (VMEC's default) they give the current density
  ``I'(s) = sum_i ac_i s^i``, with ``"power_series_I"`` the current ``I(s) = sum_i ac_i s^i`` itself. Either is
  scaled so that ``I(1) = curtor``. ``ac = (1, -1)`` with ``"power_series"`` is ``I'(s) = 1 - s``.
- :meth:`~CurrentProfile.from_equilibrium`: the current stored in the equilibrium file, ``profiles["current"]``
  of :mod:`mrx.equilibria` (a VMEC wout's ``buco``, a current-constrained DESC file's ``current``).

The profile fixes one number per surface, so the drive acts on that number only. With ``nu(s) = mu_0 dI/dPhi``
the toroidal current of the field per toroidal flux between neighbouring surfaces and ``mu(s)`` the same for the
profile, the drive is a loop voltage, an electric field along the toroidal angle with a flux-function strength,

    E = eta (nu(s) - mu(s)) <B_zeta>(s) grad zeta,

``<B_zeta>`` the surface average of the covariant toroidal field. This is the toroidal, surface-averaged part of
``eta (nu - mu) B``. Its curl ``f'(s) grad s x grad zeta`` is tangent to the surfaces, so the surfaces stay, and
its circulation around every poloidal loop on a surface is zero, so the toroidal flux inside every surface stays.
Only the poloidal flux, and with it the rotational transform and the enclosed current, changes. The pressure,
which an incompressible relaxation keeps only through the fluxes and volumes of the surfaces, is left alone. The
drive stops when the enclosed current is the profile's.
MRX's current is ``curl B``, so ``mu_0`` carries amperes into the field's units (tesla, metres). A positive
``curtor`` drives a current along the field, as VMEC's ``curtor`` with a positive ``phiedge``.

The flux surfaces are the level sets of the anisotropic-diffusion label of :mod:`mrx.flux_label`, recomputed from
the field at every step. Across islands and chaotic regions the label is flat, and so are ``nu`` and ``mu``.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from mrx.flux_label import ANISOTROPY, enclosed_flux, flux_label

#: VMEC's ``pcurr_type`` values for a polynomial profile: the coefficients give ``I'(s)`` or ``I(s)``.
PCURR_TYPES = ("power_series", "power_series_I")
#: The number of points in ``s`` at which a profile is tabulated.
N_TABLE = 257


class CurrentProfile(eqx.Module):
    """The current profile ``mu_0 dI/ds`` tabulated at the increasing points ``s`` from 0 to 1, linearly
    interpolated between them. ``anisotropy`` is the ``kperp`` of the flux label. Build it with one of the class
    methods."""
    s: jnp.ndarray
    dI_ds: jnp.ndarray
    anisotropy: float = eqx.field(static=True, default=ANISOTROPY)

    @classmethod
    def tabulated(cls, s, dI_ds, anisotropy=ANISOTROPY):
        """The profile of the current density ``dI/ds`` (amperes, positive along the field) at the increasing
        points ``s`` from 0 to 1."""
        dI_ds = mu_0 * np.asarray(dI_ds, dtype=np.float64)
        return cls(s=jnp.asarray(np.asarray(s, dtype=np.float64)), dI_ds=jnp.asarray(dI_ds), anisotropy=anisotropy)

    @classmethod
    def vmec(cls, ac, curtor, pcurr_type="power_series", anisotropy=ANISOTROPY):
        """The profile of VMEC's inputs ``ac``, ``curtor`` (in amperes) and ``pcurr_type`` (:data:`PCURR_TYPES`),
        scaled so that ``I(1) = curtor``."""
        ac = np.asarray(ac, dtype=np.float64)
        if pcurr_type == "power_series":
            dI = ac
        elif pcurr_type == "power_series_I":
            dI = ac[1:] * np.arange(1, ac.size)
        else:
            raise ValueError(f"pcurr_type must be one of {PCURR_TYPES}, got {pcurr_type!r}")
        s = np.linspace(0.0, 1.0, N_TABLE)
        dI_ds = np.polynomial.polynomial.polyval(s, dI)
        I_edge = np.trapezoid(dI_ds, s)
        if I_edge == 0.0:
            raise ValueError(f"the profile ac = {tuple(ac)} ({pcurr_type}) has no net current, so curtor cannot "
                             f"scale it")
        return cls.tabulated(s, dI_ds * curtor / I_edge, anisotropy)

    @classmethod
    def from_equilibrium(cls, equilibrium, anisotropy=ANISOTROPY):
        """The current stored in the equilibrium file, ``equilibrium["profiles"]["current"]`` (a function of
        ``r = sqrt(s)``). Raises if the file stores no current (a GVEC state, an iota-constrained DESC file)."""
        profiles = equilibrium["profiles"]
        if "current" not in profiles:
            raise ValueError(f"{equilibrium['path']} stores no current profile")
        dI_dr = profiles["current"].derivative()
        s = np.linspace(0.0, 1.0, N_TABLE)
        r = np.sqrt(s)
        # dI/ds = I'(r) / 2r, and I''(0) / 2 on the axis, where I'(0) = 0
        dI_ds = np.concatenate([[0.5 * dI_dr.derivative()(0.0)], dI_dr(r[1:]) / (2.0 * r[1:])])
        return cls.tabulated(s, dI_ds, anisotropy=anisotropy)


def profile_rate(seq, B_jk, s_q, profile: CurrentProfile):
    """``mu = mu_0 dI/dPhi`` of the profile at the quadrature points, from the field's reference components
    ``B_jk`` and the label ``s_q``. ``J* = mu B`` carries the profile's current."""
    # the toroidal flux averaged over zeta, the quadrature weights sum to 1
    flux = jnp.abs(jnp.sum(seq.quad.w * B_jk[:, 2]))
    return jnp.interp(s_q, profile.s.astype(s_q.dtype), profile.dI_ds.astype(s_q.dtype)) / flux


def _toroidal_current(seq, J):
    """The contravariant toroidal density ``J det(DPhi) J^zeta`` of the 1-form current ``J`` at the quadrature
    points."""
    return jnp.einsum('qj,qj->q', seq.metric_inv_jkl[:, 2], seq.odd.evaluate_at_quadrature(J, 1)) * seq.jacobian_j


def current_rate(seq, B_jk, J, s_q):
    """``nu = dI/dPhi`` of the 1-form current ``J`` at the quadrature points: the toroidal current over the toroidal
    flux between neighbouring surfaces of the label ``s_q``.

    Both are summed over groups of about one radial layer of samples, sorted by ``s``, and the ratio is
    interpolated linearly in ``s`` between the group means. For ``J = mu B`` with a flux function ``mu`` it is
    ``mu``.
    """
    # the flux-weighted average of J^zeta / B^zeta is the ratio of the binned current and flux
    return jnp.interp(s_q, *_group_average(seq, _toroidal_current(seq, J) / B_jk[:, 2], s_q, seq.quad.w * B_jk[:, 2]))


def _group_average(seq, f, s_q, weight):
    """``(s_mean, f_mean)``: the means of ``s`` and the ``weight``-averages of ``f`` over groups of the quadrature
    points sorted by ``s``, as many groups as there are radial quadrature layers."""
    n_groups = seq.quad.shape[0]
    order = jnp.argsort(s_q)
    group = jnp.zeros(s_q.size, jnp.int32).at[order].set(
        jnp.asarray(np.arange(s_q.size) * n_groups // s_q.size, jnp.int32))
    f_mean = jax.ops.segment_sum(weight * f, group, n_groups) / jax.ops.segment_sum(weight, group, n_groups)
    s_mean = jax.ops.segment_sum(s_q, group, n_groups) / jax.ops.segment_sum(jnp.ones_like(s_q), group, n_groups)
    return s_mean, f_mean


def toroidal_field(seq, B_jk, s_q):
    """The surface average ``<B_zeta>(s)`` of the covariant toroidal component of the field, at the quadrature
    points."""
    B_zeta = jnp.einsum('qj,qj->q', seq.metric_jkl[:, 2], B_jk) / seq.jacobian_j
    return jnp.interp(s_q, *_group_average(seq, B_zeta, s_q, seq.quad.w * seq.jacobian_j))


def profile_drive(B, seq, profile: CurrentProfile, T_guess=None, J_guess=None):
    """Return ``(J_ref, T, info)`` for :func:`mrx.relaxation.physics.resistive_step`.

    ``J_ref = J - (nu - mu) <B_zeta> grad zeta`` (a 1-form on ``seq.odd``, ``J`` the weak curl of ``B``) makes the
    step's electric field ``eta (J - J_ref)`` the loop voltage of the drive. ``T`` is the flux-label temperature
    (pass it back as ``T_guess``) and ``info`` the signed iteration count of its solve (:func:`mrx.flux_label.flux_label`).
    ``J_guess`` warm-starts the weak curl.
    """
    odd = seq.odd
    B_jk = odd.evaluate_at_quadrature(B, 2)
    T, info = flux_label(seq, B_jk, profile.anisotropy, guess=T_guess)
    s_q = enclosed_flux(seq, T, B_jk)
    J = odd.weak_curl(B, guess=J_guess)
    c = (current_rate(seq, B_jk, J, s_q) - profile_rate(seq, B_jk, s_q, profile)) * toroidal_field(seq, B_jk, s_q)
    E_cov = jnp.zeros_like(B_jk).at[:, 2].set(c)
    J_ref = J - odd.M[1].solve(odd.vector_load_values(E_cov, 1, 1))
    return J_ref, T, info
