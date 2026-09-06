"""Testy membrane/spectrum.py na syntetycznych polach ze ZNANA
analitycznie odpowiedzia (kontrole pozytywne/negatywne, nie tylko "kod
sie nie wywala") - dokladnie dyscyplina z protokolu numerologii/
formalizmu (skill timdr-signal-framework), zastosowana tutaj mimo ze to
repo formalnie jest poza zakresem tamtego skilla."""
from __future__ import annotations

import numpy as np
import pytest

from membrane.spectrum import gradient_magnitude, radial_power_spectrum, vorticity


def _grid(n=64, extent=10.0):
    x = np.linspace(0, extent, n)
    y = np.linspace(0, extent, n)
    return np.meshgrid(x, y)


def test_radial_spectrum_high_freq_field_has_more_high_freq_energy_than_low_freq_field():
    n = 64
    dx = 10.0 / (n - 1)
    xx, yy = _grid(n)
    # Pole NISKOCZESTOTLIWOSCIOWE: jeden pelny cykl na calej domenie ->
    # ladowanie w najnizszym pasmie promieniowym (sprawdzone numerycznie
    # przy pisaniu testu, patrz bin edges w spectrum.radial_power_spectrum).
    low = np.sin(2 * np.pi * xx / 10.0)
    # Pole WYSOKOCZESTOTLIWOSCIOWE: 28 cykli (blisko granicy Nyquista dla
    # n=64) - CELOWO nie 16 (16 cykli/10 jednostek wypada dokladnie na
    # granicy pasma low/high przy n_bins=16, co dawalo niejednoznaczny,
    # zalezny-od-zaokraglenia-krawedzi-binu wynik - sprawdzone numerycznie
    # przy debugowaniu tego testu). 28 cykli ląduje jednoznacznie w
    # gornej polowie pasm.
    high = np.sin(2 * np.pi * xx * 28 / 10.0)

    res_low = radial_power_spectrum(low, dx=dx)
    res_high = radial_power_spectrum(high, dx=dx)

    assert res_high.high_freq_fraction > res_low.high_freq_fraction
    # Kontrola negatywna wprost: pole niskoczestotliwosciowe powinno miec
    # WIECEJ energii w dolnych pasmach niz w gornych.
    assert res_low.low_freq_energy > res_low.high_freq_energy
    assert res_high.high_freq_energy > res_high.low_freq_energy


def test_radial_spectrum_constant_field_has_zero_energy():
    n = 32
    field = np.full((n, n), 42.0)
    res = radial_power_spectrum(field, dx=0.35)
    assert res.low_freq_energy == pytest.approx(0.0, abs=1e-6)
    assert res.high_freq_energy == pytest.approx(0.0, abs=1e-6)
    assert res.high_freq_fraction == 0.0


def test_radial_spectrum_rejects_non_square():
    with pytest.raises(ValueError):
        radial_power_spectrum(np.zeros((10, 12)), dx=1.0)


def test_gradient_magnitude_linear_field_matches_analytic():
    n = 21
    dx = dy = 0.5
    x = np.arange(n) * dx
    y = np.arange(n) * dy
    xx, yy = np.meshgrid(x, y)
    a, b = 2.0, -3.0
    field = a * xx + b * yy
    mag = gradient_magnitude(field, dx, dy)
    expected = np.hypot(a, b)
    # Pomin skrajny brzeg (jednostronne roznice moga miec troche wiekszy
    # blad numeryczny niz centralne wewnatrz).
    np.testing.assert_allclose(mag[1:-1, 1:-1], expected, atol=1e-9)


def test_vorticity_solid_body_rotation_matches_analytic():
    """Wirowanie sztywne: u=-omega*y, v=omega*x -> zeta = 2*omega wszedzie
    (klasyczny wynik podrecznikowy dynamiki plynow)."""
    n = 25
    dx = dy = 0.2
    x = np.arange(n) * dx
    y = np.arange(n) * dy
    xx, yy = np.meshgrid(x, y)
    omega = 0.7
    u = -omega * yy
    v = omega * xx
    zeta = vorticity(u, v, dx, dy)
    np.testing.assert_allclose(zeta[1:-1, 1:-1], 2 * omega, atol=1e-9)


def test_vorticity_irrotational_field_is_near_zero():
    """Przeciwny przyklad (kontrola negatywna): pole bez rotacji
    (jednorodny wiatr, u=const, v=const) powinno dac wirowosc ~0 wszedzie."""
    n = 20
    u = np.full((n, n), 5.0)
    v = np.full((n, n), -3.0)
    zeta = vorticity(u, v, dx=0.3, dy=0.3)
    np.testing.assert_allclose(zeta, 0.0, atol=1e-9)
