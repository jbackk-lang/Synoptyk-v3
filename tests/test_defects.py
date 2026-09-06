"""Testy membrane/defects.py - kontrola pozytywna (pojedynczy jawny
outlier na tle plaskiego pola musi zostac wykryty) i negatywna (plaskie
pole bez zadnego odchylenia nie zglasza zadnych defektow, mimo ze mediana
i MAD sa liczone na CALYM polu WLACZAJAC testowana komorke - patrz
uzasadnienie w defects.py, dlaczego to bezpieczne przy duzej liczbie
komorek, w odroznieniu od naprawionego gdzie indziej Pattern B)."""
from __future__ import annotations

import numpy as np

from membrane.defects import detect_defects, robust_threshold


def test_robust_threshold_flat_field():
    field = np.full((10, 10), 5.0)
    med, mad_std, threshold = robust_threshold(field)
    assert med == 5.0
    assert mad_std == 0.0


def test_robust_threshold_median_barely_moves_with_one_outlier():
    """1681 komorek (41x41), jedna ekstremalna - mediana powinna zostac
    praktycznie nietknieta (punkt zalamania mediany = 50%, jeden punkt na
    1681 to ~0.06%) - dokladnie ten argument z docstringu defects.py."""
    field = np.zeros((41, 41))
    field[20, 20] = 1000.0
    med, mad_std, threshold = robust_threshold(field)
    assert med == 0.0  # mediana 1680 zer i jednej duzej wartosci to nadal 0


def test_detect_defects_flags_single_outlier_on_flat_background():
    field = np.zeros((41, 41))
    field[10, 15] = 50.0  # jeden jawny "front" na tle spokojnego pola
    result = detect_defects(field, k=3.5)
    assert result.is_defect[10, 15]
    assert result.n_defects == 1


def test_detect_defects_no_false_positives_on_pure_noise_within_bounds():
    """Kontrola negatywna: szum gaussowski (bez zadnej wstrzyknietej
    anomalii) NIE powinien zaflagowac wiecej niz garstki komorek (k=3.5
    dobrane konserwatywnie) - sprawdzamy, ze odsetek flag jest may (<5%),
    nie ze jest dokladnie zero (przy losowym szumie kilka skrajnych
    wartosci moze przekroczyc prog przez przypadek)."""
    rng = np.random.default_rng(42)
    field = rng.normal(loc=0.0, scale=1.0, size=(41, 41))
    result = detect_defects(field, k=3.5)
    fraction_flagged = result.n_defects / field.size
    assert fraction_flagged < 0.05


def test_detect_defects_flat_field_gives_zero_defects():
    field = np.zeros((20, 20))
    result = detect_defects(field)
    assert result.n_defects == 0
    assert not result.is_defect.any()


def test_detect_defects_multiple_separated_fronts():
    field = np.zeros((41, 41))
    field[5, 5] = 80.0
    field[35, 35] = -80.0  # defekt moze byc tez ujemny co do wartosci wejsciowej? nie - detect_defects dziala na magnitude (>=0) w praktyce, ale funkcja sama nie wymusza tego
    result = detect_defects(np.abs(field), k=3.5)
    assert result.is_defect[5, 5]
    assert result.is_defect[35, 35]
    assert result.n_defects == 2
