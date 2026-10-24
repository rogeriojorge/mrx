"""energy_seed names the problem when the iota range holds no resonance of the chains it can seed."""
import os

import pytest

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.relaxation.initial_conditions import initial_field
from mrx.relaxation.seeding import energy_seed

WOUT = os.path.join(os.path.dirname(__file__), "..", "data", "wout_li383_low_res_reference.nc")


def test_no_resonance_in_the_iota_range_is_reported():
    seq = build_sequence(WOUT, (4, 8, 8), 2)
    compute_nullspaces(seq)
    B, _ = initial_field(seq)
    # a narrow band of m <= n_theta / 2 = 4 chains: restrict the resonances by asking for a poloidal count of 1
    import mrx.relaxation.seeding as seeding
    original = seeding.resonances
    seeding.resonances = lambda *a, **k: []
    try:
        with pytest.raises(ValueError, match="no island chain"):
            energy_seed(seq, B, iotas=(1.0,), verbose=False)
    finally:
        seeding.resonances = original
