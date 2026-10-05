"""The current profile of the resistive drive reproduces the current of a VMEC equilibrium.

On li383 the logical radius is ``sqrt(s)`` of the VMEC file, so the flux label of :mod:`mrx.flux_label` must
recover ``s = r^2`` on the equilibrium's own field. The target current ``J* = mu(s) B`` of the file's own profile
(``ac``, ``ctor``, ``pcurr_type``) must enclose the current ``I(s) = 2 pi B_theta / mu_0`` that VMEC's solution
carries (``buco``, the surface average of the covariant poloidal field), and the drive's measure of the current
per flux, :func:`~mrx.relaxation.current_profile.current_rate`, must return ``mu`` on it. The initial field
carries the file's current, so the profile read from the file
(:meth:`~mrx.relaxation.current_profile.CurrentProfile.from_equilibrium`, sign included) must match its ``nu``.
"""
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0
from scipy.io import netcdf_file

from mrx.flux_label import enclosed_flux, flux_label
from mrx.relaxation.current_profile import CurrentProfile, current_rate, profile_rate
from test.conftest import GEOMETRY

#: The radial bins of the enclosed-current comparison.
EDGES = np.linspace(0.0, 1.0, 6)[1:]
#: |s - r^2| at the quadrature points, mean over the volume (measured 3.2e-3 at (8, 12, 12)).
LABEL_ERROR = 0.01
#: |I_J*(s) - I_VMEC(s)| / |I(1)| in the bins. The label resolves one radial layer of (8, 12, 12).
CURRENT_ERROR = 0.03
#: |nu - mu| / |mu| over the volume for J* = mu B: the binning against the projection of J* on the 1-forms.
RATE_ERROR = 0.05
#: |nu - mu| / |mu| over the volume for the initial field against the file's profile: the discretisation of the
#: field's current on (8, 12, 12).
FILE_ERROR = 0.1


def _enclosed_current(seq, J, B_jk, s_q):
    """The toroidal current of the 1-form ``J`` inside the flux fraction ``s`` at :data:`EDGES`, averaged over zeta:
    the cumulative current against the cumulative flux of the points sorted by the label ``s_q``. The label
    gives the points of one radial layer the same ``s``, which this resolves."""
    J_cov = seq.odd.evaluate_at_quadrature(J, 1)
    J_zeta = jnp.einsum("qj,qj->q", seq.metric_inv_jkl[:, 2], J_cov) * seq.jacobian_j
    order = np.argsort(np.asarray(s_q))
    flux = np.cumsum(np.asarray(seq.quad.w * B_jk[:, 2])[order])
    current = np.cumsum(np.asarray(seq.quad.w * J_zeta)[order])
    return np.interp(EDGES, flux / flux[-1], current)


def test_profile_current_matches_vmec(seq, b0):
    """The flux label is ``r^2``, and the profile's current is the field's own current."""
    with netcdf_file(GEOMETRY, mmap=False) as f:
        ac, ctor = np.array(f.variables["ac"][()]), float(f.variables["ctor"][()])
        pcurr_type = f.variables["pcurr_type"][()].tobytes().decode().strip()
        buco = np.abs(np.array(f.variables["buco"][()]))
    # buco lives on VMEC's half mesh, extended linearly to the edge
    s_half = np.r_[0.0, (np.arange(1, buco.size) - 0.5) / (buco.size - 1), 1.0]
    buco = np.r_[0.0, buco[1:], 1.5 * buco[-1] - 0.5 * buco[-2]]
    I_vmec = 2 * np.pi / mu_0 * np.interp(EDGES, s_half, buco)
    B_jk = seq.odd.evaluate_at_quadrature(b0, 2)
    T, info = flux_label(seq, B_jk)
    s_q = np.asarray(enclosed_flux(seq, T, B_jk))
    label_error = float(np.sum(np.asarray(seq.quad.w) * np.abs(s_q - np.asarray(seq.quad.x[:, 0]) ** 2)))

    mu = profile_rate(seq, B_jk, s_q, CurrentProfile.vmec(ac, abs(ctor), pcurr_type))
    J_star = seq.odd.M[1].solve(seq.odd.vector_load_values(mu[:, None] * B_jk, 2, 1))
    nu = current_rate(seq, B_jk, J_star, s_q)
    rate_error = float(jnp.sum(seq.quad.w * jnp.abs(nu - mu)) / jnp.sum(seq.quad.w * jnp.abs(mu)))
    # a positive curtor drives along the field, so the current through the zeta planes has the flux's sign
    I_star = _enclosed_current(seq, J_star, B_jk, s_q) * np.sign(float(jnp.sum(seq.quad.w * B_jk[:, 2])))
    print(f"\n  flux label: CG {int(info)}, mean |s - r^2| {label_error:.3e},  VMEC ctor {ctor:+.4e} A")
    print(f"  |nu - mu| / |mu| {rate_error:.3e}")
    print("  I(s) / |ctor|   s " + " ".join(f"{v:6.3f}" for v in EDGES))
    print("    VMEC (buco)     " + " ".join(f"{v:6.3f}" for v in I_vmec / abs(ctor)))
    print("    of J*           " + " ".join(f"{v:6.3f}" for v in I_star / (mu_0 * abs(ctor))))
    assert info > 0
    assert label_error < LABEL_ERROR
    assert np.max(np.abs(I_star / mu_0 - I_vmec)) < CURRENT_ERROR * abs(ctor)
    assert rate_error < RATE_ERROR

    def mismatch(a, b):
        return float(jnp.sum(seq.quad.w * jnp.abs(a - b)) / jnp.sum(seq.quad.w * jnp.abs(b)))

    nu_b0 = current_rate(seq, B_jk, seq.odd.weak_curl(b0), s_q)
    file_error = mismatch(nu_b0, profile_rate(seq, B_jk, s_q, CurrentProfile.from_equilibrium(seq.equilibrium)))
    print(f"  initial field against the file's profile {file_error:.3e}")
    assert file_error < FILE_ERROR
