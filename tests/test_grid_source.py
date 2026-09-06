"""Testy membrane/grid_source.py.

build_grid_points() jest czysta funkcja - testowana normalnie.

fetch_grid_snapshot()/fetch_meteogram() NIE moga zostac przetestowane z
prawdziwym polaczeniem sieciowym w tym srodowisku (sandbox bash ma
zablokowany dostep do internetu - patrz duzy docstring na gorze
grid_source.py). Zamiast tego _http_get() jest monkeypatchowane tak, zeby
zwracalo DOKLADNIE odpowiedzi PRAWDZIWE, zlapane recznie z
api.open-meteo.com przez osobny kanal (przegladarke) w trakcie pisania
tego kodu - ponizsze slowniki to zrzuty realnych odpowiedzi JSON, nie
zmyslony ksztalt danych. Test sprawdza wiec parsowanie na prawdziwym
kontrakcie API, tylko bez wykonania samego polaczenia sieciowego z
wnetrza tej appki."""
from __future__ import annotations

import pytest

from membrane import grid_source
from membrane.grid_source import GridPoint, build_grid_points, fetch_grid_snapshot, fetch_meteogram

# Prawdziwa odpowiedz Open-Meteo dla 2 punktow (?latitude=52.23,52.48&
# longitude=21.01,21.01&current=temperature_2m,precipitation), zlapana
# 2026-09-06 przez przegladarke - patrz docstring grid_source.py.
REAL_TWO_POINT_RESPONSE = [
    {
        "latitude": 52.23009, "longitude": 21.017075, "elevation": 113.0,
        "current_units": {"temperature_2m": "°C", "surface_pressure": "hPa", "relative_humidity_2m": "%",
                            "wind_speed_10m": "km/h", "wind_direction_10m": "°", "precipitation": "mm"},
        "current": {"time": "2026-09-06T15:00", "temperature_2m": 17.0, "surface_pressure": 1009.1,
                    "relative_humidity_2m": 71, "wind_speed_10m": 16.9, "wind_direction_10m": 294, "precipitation": 0.0},
    },
    {
        "latitude": 52.48721, "longitude": 21.010727, "elevation": 100.0,
        "current_units": {"temperature_2m": "°C", "surface_pressure": "hPa", "relative_humidity_2m": "%",
                            "wind_speed_10m": "km/h", "wind_direction_10m": "°", "precipitation": "mm"},
        "current": {"time": "2026-09-06T15:00", "temperature_2m": 16.5, "surface_pressure": 1010.4,
                    "relative_humidity_2m": 68, "wind_speed_10m": 12.1, "wind_direction_10m": 280, "precipitation": 0.0},
    },
]

# Prawdziwa odpowiedz Open-Meteo dla JEDNEGO punktu z daily=... (3 dni),
# zlapana tym samym sposobem - potwierdza, ze pojedynczy punkt daje
# POJEDYNCZY obiekt (nie liste), patrz docstring grid_source.py.
REAL_SINGLE_POINT_DAILY_RESPONSE = {
    "latitude": 52.23009, "longitude": 21.017075, "elevation": 113.0,
    "daily_units": {"time": "iso8601", "precipitation_sum": "mm", "temperature_2m_max": "°C",
                     "temperature_2m_min": "°C", "wind_speed_10m_max": "km/h",
                     "wind_direction_10m_dominant": "°", "surface_pressure_mean": "hPa"},
    "daily": {
        "time": ["2026-09-06", "2026-09-07", "2026-09-08"],
        "precipitation_sum": [0.80, 0.00, 0.00],
        "temperature_2m_max": [17.3, 20.3, 27.1],
        "temperature_2m_min": [12.8, 11.0, 15.9],
        "wind_speed_10m_max": [25.2, 11.2, 11.9],
        "wind_direction_10m_dominant": [276, 232, 175],
        "surface_pressure_mean": [1006.9, 1010.6, 1000.5],
    },
}


def test_build_grid_points_shape_and_center():
    points = build_grid_points(52.0, 21.0, n=5, spacing_deg=0.5)
    assert len(points) == 25
    # Centrum siatki (indeks srodkowy) musi byc dokladnie punktem wejsciowym.
    assert any(p.lat == pytest.approx(52.0) and p.lon == pytest.approx(21.0) for p in points)
    lats = sorted({p.lat for p in points})
    assert lats == pytest.approx([51.0, 51.5, 52.0, 52.5, 53.0])


def test_build_grid_points_rejects_even_n():
    with pytest.raises(ValueError):
        build_grid_points(52.0, 21.0, n=4)


def test_build_grid_points_rejects_too_small_n():
    with pytest.raises(ValueError):
        build_grid_points(52.0, 21.0, n=1)


def test_fetch_grid_snapshot_parses_real_two_point_response(monkeypatch):
    points = [GridPoint(lat=52.23, lon=21.01), GridPoint(lat=52.48, lon=21.01)]
    monkeypatch.setattr(grid_source, "_http_get", lambda url, params, timeout=20.0: REAL_TWO_POINT_RESPONSE)
    records = fetch_grid_snapshot(points)
    assert len(records) == 2
    assert records[0]["temperature_c"] == 17.0
    assert records[0]["pressure_hpa"] == 1009.1
    assert records[0]["wind_dir_deg"] == 294
    assert records[1]["temperature_c"] == 16.5


def test_fetch_grid_snapshot_normalizes_single_object_to_list(monkeypatch):
    """Jeden punkt -> Open-Meteo zwraca POJEDYNCZY obiekt, nie liste -
    sprawdzamy, ze fetch_grid_snapshot normalizuje to poprawnie (patrz
    docstring grid_source.py)."""
    single = REAL_TWO_POINT_RESPONSE[0]
    monkeypatch.setattr(grid_source, "_http_get", lambda url, params, timeout=20.0: single)
    records = fetch_grid_snapshot([GridPoint(lat=52.23, lon=21.01)])
    assert len(records) == 1
    assert records[0]["temperature_c"] == 17.0


def test_fetch_grid_snapshot_raises_on_count_mismatch(monkeypatch):
    monkeypatch.setattr(grid_source, "_http_get", lambda url, params, timeout=20.0: REAL_TWO_POINT_RESPONSE)
    with pytest.raises(ValueError):
        fetch_grid_snapshot([GridPoint(lat=1, lon=1), GridPoint(lat=2, lon=2), GridPoint(lat=3, lon=3)])


def test_fetch_meteogram_parses_real_daily_response(monkeypatch):
    monkeypatch.setattr(grid_source, "_http_get", lambda url, params, timeout=20.0: REAL_SINGLE_POINT_DAILY_RESPONSE)
    result = fetch_meteogram(52.23, 21.01, days=3)
    assert result["days"] == ["2026-09-06", "2026-09-07", "2026-09-08"]
    assert result["precip_mm"] == [0.80, 0.00, 0.00]
    assert result["temp_max_c"] == [17.3, 20.3, 27.1]
    assert result["wind_dir_deg"] == [276, 232, 175]
    assert result["pressure_hpa"] == [1006.9, 1010.6, 1000.5]


def test_fetch_meteogram_rejects_bad_days(monkeypatch):
    monkeypatch.setattr(grid_source, "_http_get", lambda url, params, timeout=20.0: REAL_SINGLE_POINT_DAILY_RESPONSE)
    with pytest.raises(ValueError):
        fetch_meteogram(52.23, 21.01, days=0)
    with pytest.raises(ValueError):
        fetch_meteogram(52.23, 21.01, days=17)
