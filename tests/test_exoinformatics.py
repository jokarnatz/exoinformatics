import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "exoinformatics"))

exoinformatics = importlib.import_module("exoinformatics")


def test_basic_helpers():
    assert exoinformatics.heaviside(-1) == 0
    assert exoinformatics.heaviside(2) == 1
    assert exoinformatics.kronecker(3, 3) == 1
    assert exoinformatics.kronecker(3, 4) == 0


def test_m0_and_omega_values():
    assert exoinformatics.M0(5, 3) == 4
    assert exoinformatics.Omega0(7) == 7
    assert exoinformatics.OmegaI(3) == 10
    assert exoinformatics.OmegaC(4) == 6


def test_lookup_functions():
    assert exoinformatics.WI(5, 3, 3) == 22
    assert exoinformatics.WC(5, 2, 2) == 18


def test_randomswap_preserves_length_and_values():
    radii = [1.0, 2.0, 3.0, 4.0]
    swapped = exoinformatics.randomswap(radii)

    assert len(swapped) == len(radii)
    assert sorted(swapped) == sorted(radii)
    assert swapped != radii
