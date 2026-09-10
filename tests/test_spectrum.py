"""Testy membrane/spectrum.py na syntetycznych polach ze ZNANA
analitycznie odpowiedzia (kontrole pozytywne/negatywne, nie tylko "kod
sie nie wywala") - dokladnie dyscyplina z protokolu numerologii/
formalizmu (skill timdr-signal-framework), zastosowana tutaj mimo ze to
repo formalnie jest poza zakresem tamtego skilla."""
from __future__ import annotations

import numpy as np
import pytest

from membrane.interpolate import wind_to_uv
from membrane.spectrum import gradient_magnitude, radial_power_spectrum, vorticity, wind_direction_coherence


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


# ---------------------------------------------------------------------
# wind_direction_coherence (dodane 2026-09-10) - kontrole analityczne +
# realny scenariusz, ktorym zjawisko zostalo faktycznie odkryte
# (siatka Open-Meteo wokol Gdanska, 2026-09-10 09:00 UTC).
# ---------------------------------------------------------------------

def test_wind_coherence_uniform_direction_is_one_regardless_of_speed():
    """Kontrola pozytywna analityczna: idealnie zgodny kierunek, ale
    RÓŻNE predkosci w kazdej komorce - |Z| musi wyjsc dokladnie 1.0,
    bo predkosc jest ignorowana z definicji (patrz docstring funkcji)."""
    rng = np.random.default_rng(0)
    n = 30
    speeds = rng.uniform(1.0, 50.0, size=(n, n))
    dir_deg = 270.0  # jeden, wspolny kierunek dla calej siatki
    u, v = wind_to_uv(speeds, dir_deg)
    result = wind_direction_coherence(u, v)
    assert result.coherence == pytest.approx(1.0, abs=1e-9)
    # mean_direction_deg uzywa konwencji "kierunek SKAD wieje" (ta sama
    # co wejsciowe dir_deg) - patrz docstring wind_direction_coherence.
    assert result.mean_direction_deg == pytest.approx(dir_deg, abs=1e-6)
    assert result.n_valid == n * n


def test_wind_coherence_random_directions_is_near_zero():
    """Kontrola negatywna: losowe kierunki (duza probka) -> |Z| bliskie
    0, zgodnie z oczekiwaniem statystycznym ~1/sqrt(n) dla sredniej z n
    niezaleznych wektorow jednostkowych o losowej fazie."""
    rng = np.random.default_rng(1)
    n_points = 2000
    dirs = rng.uniform(0, 360, size=n_points)
    speeds = rng.uniform(5.0, 40.0, size=n_points)
    u = np.array([wind_to_uv(s, d)[0] for s, d in zip(speeds, dirs)])
    v = np.array([wind_to_uv(s, d)[1] for s, d in zip(speeds, dirs)])
    result = wind_direction_coherence(u, v)
    assert result.coherence < 3.0 / np.sqrt(n_points)  # hojny margines nad oczekiwanym ~1/sqrt(n)


def test_wind_coherence_ignores_calm_cells():
    """Komorki z zerowa predkoscia (cisza) sa pomijane, nie liczone jako
    wektor zerowy (co bezpodstawnie zanizyloby |Z|)."""
    u = np.array([1.0, 1.0, 1.0, 0.0])
    v = np.array([0.0, 0.0, 0.0, 0.0])
    result = wind_direction_coherence(u, v)
    assert result.coherence == pytest.approx(1.0, abs=1e-9)
    assert result.n_valid == 3


def test_wind_coherence_all_calm_returns_zero_not_nan():
    u = np.zeros((5, 5))
    v = np.zeros((5, 5))
    result = wind_direction_coherence(u, v)
    assert result.coherence == 0.0
    assert result.n_valid == 0


def test_wind_coherence_vs_vorticity_on_real_gdansk_grid_are_genuinely_different():
    """Regresja na REALNYCH danych (Open-Meteo, siatka 5x5 wokol Gdanska,
    2026-09-10 09:00 UTC, pobrane przez przegladarke wewnetrzna - patrz
    historia sesji) - dokladnie ten przypadek, ktory pokazal, ze
    wind_direction_coherence i vorticity mierza NAPRAWDE rozne rzeczy:
    silny gradient PREDKOSCI (14-52 km/h) przy prawie stalym KIERUNKU
    (260-282 stopni) daje jednoczesnie WYSOKA spojnosc kierunkowa I
    DUZA wirowosc (napedzana gradientem predkosci, nie kierunku)."""
    speeds_dirs = [
        (15.6, 278), (14.1, 276), (16.6, 270), (21.6, 267), (18.9, 262),
        (20.1, 279), (21.6, 270), (19.1, 269), (21.7, 263), (23.2, 264),
        (23.3, 279), (22.0, 268), (18.8, 275), (26.6, 271), (26.4, 265),
        (22.7, 280), (21.3, 282), (38.2, 273), (42.5, 267), (43.3, 273),
        (51.1, 278), (51.5, 278), (49.7, 271), (49.3, 261), (49.7, 260),
    ]
    n = 5
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for idx, (speed, deg) in enumerate(speeds_dirs):
        i, j = divmod(idx, n)
        u[i, j], v[i, j] = wind_to_uv(speed, deg)

    coherence_result = wind_direction_coherence(u, v)
    zeta = vorticity(u, v, dx=0.35, dy=0.35)

    # Kierunek bardzo spojny mimo duzego rozrzutu predkosci.
    assert coherence_result.coherence > 0.95
    # A mimo to wirowosc jest wyraznie niezerowa (napedzana gradientem
    # predkosci) - gdyby obie diagnostyki mierzyly "to samo", duza
    # spojnosc kierunkowa implikowalaby wirowosc bliska zeru, a tak nie jest.
    assert np.abs(zeta).max() > 10.0
