"""Testy membrane/resonance.py - czysta logika koincydencji, bez zaleznosci
od reszty pipeline'u."""
from __future__ import annotations

import numpy as np
import pytest

from membrane.resonance import compute_resonance


def _mask(shape, true_cells):
    m = np.zeros(shape, dtype=bool)
    for (i, j) in true_cells:
        m[i, j] = True
    return m


def test_resonance_flags_only_cells_with_enough_coincidences():
    shape = (5, 5)
    masks = {
        "a": _mask(shape, [(1, 1), (2, 2)]),
        "b": _mask(shape, [(1, 1), (3, 3)]),
        "c": _mask(shape, [(1, 1), (2, 2)]),
    }
    res = compute_resonance(masks, k=2)
    assert res.is_resonance[1, 1]  # a,b,c wszystkie True -> 3 >= 2
    assert res.is_resonance[2, 2]  # a,c True -> 2 >= 2
    assert not res.is_resonance[3, 3]  # tylko b -> 1 < 2
    assert res.n_resonance_cells == 2


def test_resonance_k_equal_to_number_of_channels_requires_all():
    shape = (3, 3)
    masks = {
        "a": _mask(shape, [(0, 0), (1, 1)]),
        "b": _mask(shape, [(0, 0)]),
    }
    res = compute_resonance(masks, k=2)
    assert res.is_resonance[0, 0]
    assert not res.is_resonance[1, 1]
    assert res.n_resonance_cells == 1


def test_resonance_no_coincidence_anywhere():
    shape = (4, 4)
    masks = {
        "a": _mask(shape, [(0, 0)]),
        "b": _mask(shape, [(1, 1)]),
        "c": _mask(shape, [(2, 2)]),
    }
    res = compute_resonance(masks, k=2)
    assert res.n_resonance_cells == 0


def test_resonance_requires_at_least_two_channels():
    with pytest.raises(ValueError):
        compute_resonance({"only_one": np.zeros((3, 3), dtype=bool)}, k=1)


def test_resonance_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_resonance({
            "a": np.zeros((3, 3), dtype=bool),
            "b": np.zeros((4, 4), dtype=bool),
        }, k=1)


def test_resonance_rejects_invalid_k():
    masks = {"a": np.zeros((3, 3), dtype=bool), "b": np.zeros((3, 3), dtype=bool)}
    with pytest.raises(ValueError):
        compute_resonance(masks, k=0)
    with pytest.raises(ValueError):
        compute_resonance(masks, k=3)
