"""Testy membrane/interpolate.py - roundtrip wiatru u/v i budowa membrany
z syntetycznych, znanych pol (kontrola pozytywna: pole liniowe powinno
zostac odtworzone niemal dokladnie przez interpolacje kubiczna)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import membrane.interpolate as interpolate_mod
from membrane.interpolate import build_membrane, wind_speed_dir_from_uv, wind_to_uv


def test_wind_uv_roundtrip_cardinal_directions():
    # Wiatr z polnocy (dir_from=0) wieje NA poludnie -> v ujemne, u=0.
    u, v = wind_to_uv(speed=10.0, dir_from_deg=0.0)
    assert math.isclose(u, 0.0, abs_tol=1e-9)
    assert math.isclose(v, -10.0, abs_tol=1e-9)

    # Wiatr ze wschodu (dir_from=90) wieje NA zachod -> u ujemne.
    u, v = wind_to_uv(speed=5.0, dir_from_deg=90.0)
    assert math.isclose(u, -5.0, abs_tol=1e-9)
    assert math.isclose(v, 0.0, abs_tol=1e-9)


@pytest.mark.parametrize("speed,dir_from", [(10.0, 0.0), (5.0, 90.0), (7.5, 200.0), (3.0, 359.0)])
def test_wind_uv_roundtrip_general(speed, dir_from):
    u, v = wind_to_uv(speed, dir_from)
    speed2, dir2 = wind_speed_dir_from_uv(u, v)
    assert math.isclose(speed, speed2, abs_tol=1e-6)
    assert math.isclose(dir_from % 360, dir2 % 360, abs_tol=1e-6)


def test_wind_uv_zero_speed():
    speed, dir_from = wind_speed_dir_from_uv(0.0, 0.0)
    assert speed == 0.0
    assert dir_from == 0.0


def _synthetic_records(center_lat=52.0, center_lon=21.0, n=5, spacing=0.35):
    """5x5 siatka punktow z DOKLADNIE liniowym polem T i P - jesli
    interpolacja dziala poprawnie, membrana powinna odtworzyc te same
    wspolczynniki liniowe wszedzie (kontrola pozytywna: znany wynik
    analityczny, nie tylko 'kod sie nie wywala')."""
    half = n // 2
    records = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            lat = center_lat + i * spacing
            lon = center_lon + j * spacing
            records.append({
                "lat": lat, "lon": lon,
                "temperature_c": 10.0 + 2.0 * (lat - center_lat) - 3.0 * (lon - center_lon),
                "pressure_hpa": 1013.0 + 1.0 * (lat - center_lat),
                "humidity_pct": 70.0,
                "wind_speed_kmh": 10.0,
                "wind_dir_deg": 270.0,
                "precip_mm": 0.0,
            })
    return records


def test_build_membrane_reconstructs_linear_field():
    records = _synthetic_records()
    membrane = build_membrane(records, grid_n=21)
    expected_t = 10.0 + 2.0 * (membrane.lat_grid - 52.0) - 3.0 * (membrane.lon_grid - 21.0)
    # Interpolacja kubiczna pola scisle liniowego powinna byc niemal
    # dokladna w WEWNETRZNEJ czesci membrany (rogi/brzegi moga miec wieksze
    # bledy ekstrapolacji przy fallbacku 'nearest' - patrz docstring
    # _interp_field). Sprawdzamy wiec tylko srodkowy podzbior.
    inner = slice(3, -3)
    np.testing.assert_allclose(membrane.temperature_c[inner, inner], expected_t[inner, inner], atol=0.15)


def test_build_membrane_constant_wind_gives_near_constant_uv():
    records = _synthetic_records()
    membrane = build_membrane(records, grid_n=15)
    # Wiatr identyczny na wszystkich punktach wejsciowych -> u,v powinny
    # byc (niemal) stale na calej membranie.
    assert np.allclose(membrane.u_wind_kmh, membrane.u_wind_kmh[0, 0], atol=0.5)
    assert np.allclose(membrane.v_wind_kmh, membrane.v_wind_kmh[0, 0], atol=0.5)


def test_build_membrane_raises_on_too_few_points():
    records = _synthetic_records()[:3]
    with pytest.raises(ValueError):
        build_membrane(records, grid_n=11)


def test_build_membrane_skips_missing_values_per_field():
    records = _synthetic_records()
    # Usun cisnienie tylko z dwoch punktow - powinno nadal dzialac (>=4
    # punktow zostaje), inne pola nietkniete.
    records[0]["pressure_hpa"] = None
    records[1]["pressure_hpa"] = None
    membrane = build_membrane(records, grid_n=11)
    assert not np.isnan(membrane.pressure_hpa).any()
    assert not np.isnan(membrane.temperature_c).any()


def test_build_membrane_raises_clear_runtime_error_without_scipy(monkeypatch):
    """Regresja na REALNY blad uzytkownika (2026-09-10, Windows Device
    Guard blokuje DLL-e scipy przy starcie webapp/app.py) - symuluje
    brak scipy monkeypatchujac `_HAS_SCIPY`/`griddata` (dokladnie tak,
    jak wygladalby modul po nieudanym imporcie), i sprawdza, ze
    `build_membrane()` rzuca CZYTELNY RuntimeError z konkretna
    wiadomoscia, NIE kryptyczny TypeError/AttributeError z wnetrza
    scipy ani (gorzej) nie wywala calego importu modulu."""
    monkeypatch.setattr(interpolate_mod, "_HAS_SCIPY", False)
    monkeypatch.setattr(interpolate_mod, "griddata", None)
    records = _synthetic_records()
    with pytest.raises(RuntimeError, match="scipy"):
        build_membrane(records, grid_n=11)


def test_interpolate_module_importable_without_scipy_available_flag():
    """Sam import modulu (i wszystkiego co go importuje - spectrum.py,
    analyze.py, webapp/app.py) NIE MOZE zalezec od tego, czy scipy sie
    zaimportowalo - `_HAS_SCIPY` musi istniec jako atrybut modulu
    niezaleznie od wyniku importu (patrz UWAGA O IMPORCIE w naglowku
    interpolate.py)."""
    assert hasattr(interpolate_mod, "_HAS_SCIPY")
    assert hasattr(interpolate_mod, "griddata")
