"""Testy membrane/archive_source.py.

Ten sam wzorzec co tests/test_grid_source.py: _http_get() monkeypatchowane
odpowiedzia PRAWDZIWA, zlapana recznie z archive-api.open-meteo.com przez
przegladarke wewnetrzna (2026-09-07, Warszawa 52.23/21.01, past_days=10) -
patrz docstring archive_source.py po dokladny URL zapytania."""
from __future__ import annotations

from datetime import date

import pytest

from membrane import archive_source
from membrane.archive_source import fetch_archive, fetch_archive_grid
from membrane.grid_source import GridPoint

# Prawdziwa odpowiedz Archive API, zlapana 2026-09-07 (patrz docstring
# modulu) - 11 dni, 2026-08-28..2026-09-07 wlacznie.
REAL_ARCHIVE_RESPONSE = {
    "latitude": 52.26713, "longitude": 20.961182, "elevation": 113.0,
    "daily_units": {"time": "iso8601", "precipitation_sum": "mm", "temperature_2m_max": "°C",
                     "temperature_2m_min": "°C", "wind_speed_10m_max": "km/h",
                     "wind_direction_10m_dominant": "°", "surface_pressure_mean": "hPa"},
    "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01",
                 "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07"],
        "precipitation_sum": [0.00, 5.20, 0.00, 2.40, 3.20, 0.40, 7.00, 2.40, 2.50, 2.20, 0.00],
        "temperature_2m_max": [23.8, 21.9, 25.3, 25.9, 19.9, 21.2, 21.9, 22.1, 19.7, 18.7, 19.3],
        "temperature_2m_min": [13.0, 15.7, 13.8, 16.0, 15.4, 13.6, 12.1, 15.7, 13.1, 12.0, 9.6],
        "wind_speed_10m_max": [18.5, 16.0, 12.1, 11.5, 20.7, 14.4, 12.3, 24.8, 25.6, 24.1, 9.6],
        "wind_direction_10m_dominant": [131, 206, 182, 199, 259, 262, 236, 242, 271, 282, 226],
        "surface_pressure_mean": [1006.2, 999.7, 1001.6, 1000.4, 1001.9, 1005.2, 1003.5, 995.8, 997.9, 1006.9, 1010.7],
    },
}


def test_fetch_archive_parses_real_response_without_trailing_exclusion(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_ARCHIVE_RESPONSE)
    result = fetch_archive(52.23, 21.01, past_days=10, exclude_trailing_days=0)
    assert result["days"] == REAL_ARCHIVE_RESPONSE["daily"]["time"]
    assert result["precip_mm"] == REAL_ARCHIVE_RESPONSE["daily"]["precipitation_sum"]
    assert result["temp_max_c"] == REAL_ARCHIVE_RESPONSE["daily"]["temperature_2m_max"]
    assert result["pressure_hpa"] == REAL_ARCHIVE_RESPONSE["daily"]["surface_pressure_mean"]


def test_fetch_archive_excludes_trailing_days(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_ARCHIVE_RESPONSE)
    # "dzisiaj" = 2026-09-07 -> exclude_trailing_days=2 odcina 2026-09-06 i 2026-09-07.
    result = fetch_archive(
        52.23, 21.01, past_days=10, exclude_trailing_days=2, _today=date(2026, 9, 7),
    )
    assert result["days"] == REAL_ARCHIVE_RESPONSE["daily"]["time"][:-2]
    assert len(result["temp_max_c"]) == len(result["days"])
    assert "2026-09-06" not in result["days"]
    assert "2026-09-07" not in result["days"]


def test_fetch_archive_all_lists_stay_synchronized_after_exclusion(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_ARCHIVE_RESPONSE)
    result = fetch_archive(52.23, 21.01, past_days=10, exclude_trailing_days=3, _today=date(2026, 9, 7))
    lengths = {len(v) for v in result.values()}
    assert len(lengths) == 1  # wszystkie listy tej samej dlugosci po przycieciu


def test_fetch_archive_rejects_bad_past_days():
    with pytest.raises(ValueError):
        fetch_archive(52.23, 21.01, past_days=0)
    with pytest.raises(ValueError):
        fetch_archive(52.23, 21.01, past_days=93)


# ---------------------------------------------------------------------------
# fetch_archive_grid() - podzbior DWOCH punktow z tej samej REALNEJ
# odpowiedzi 3x3 (REAL_GRID_RESPONSE), przechwyconej rownolegle z
# REAL_ARCHIVE_RESPONSE powyzej i uzywanej tez w
# tests/test_meta_adapter.py (patrz docstring archive_source.py po pelny
# URL zapytania siatki 3x3).
# ---------------------------------------------------------------------------
REAL_GRID_RESPONSE_SUBSET = [
    {"latitude": 51.915638, "longitude": 20.604397, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.7, 22.7, 25.9, 26.2, 19.7, 20.9, 22.0, 23.0, 19.9, 17.3],
        "temperature_2m_min": [11.9, 14.8, 12.7, 15.3, 14.7, 13.2, 12.1, 14.5, 12.7, 11.4],
        "precipitation_sum": [0.00, 5.40, 0.20, 2.70, 1.30, 1.70, 1.00, 2.90, 1.20, 2.00],
        "wind_speed_10m_max": [25.4, 20.1, 14.1, 18.0, 27.0, 15.4, 16.5, 30.7, 30.5, 32.8],
        "wind_direction_10m_dominant": [131, 231, 203, 222, 265, 267, 253, 244, 274, 285],
        "surface_pressure_mean": [996.6, 990.8, 992.7, 991.6, 993.4, 996.4, 994.6, 987.6, 989.8, 998.5],
        "relative_humidity_2m_mean": [57, 77, 65, 83, 73, 73, 72, 80, 69, 77],
    }},
    {"latitude": 51.84534, "longitude": 21.06033, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [24.5, 22.9, 26.6, 26.9, 20.8, 21.1, 22.4, 23.5, 20.3, 18.2],
        "temperature_2m_min": [11.9, 14.1, 11.9, 14.6, 15.2, 13.2, 11.8, 15.2, 13.2, 11.7],
        "precipitation_sum": [0.00, 5.00, 0.10, 12.60, 1.20, 3.60, 2.30, 2.40, 0.60, 2.10],
        "wind_speed_10m_max": [24.7, 17.2, 12.8, 23.6, 29.9, 17.8, 12.8, 30.6, 31.3, 35.5],
        "wind_direction_10m_dominant": [129, 220, 201, 208, 263, 264, 247, 241, 272, 285],
        "surface_pressure_mean": [1004.2, 998.1, 999.9, 998.8, 1000.5, 1003.6, 1001.8, 994.7, 996.7, 1005.5],
        "relative_humidity_2m_mean": [57, 77, 67, 83, 74, 78, 75, 77, 68, 76],
    }},
]
_GRID_POINTS_2 = [GridPoint(lat=51.915638, lon=20.604397), GridPoint(lat=51.84534, lon=21.06033)]


def test_fetch_archive_grid_parses_real_response_without_trailing_exclusion(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_GRID_RESPONSE_SUBSET)
    result = fetch_archive_grid(_GRID_POINTS_2, past_days=10, exclude_trailing_days=0)
    assert len(result) == 2
    assert result[0]["daily"]["time"] == REAL_GRID_RESPONSE_SUBSET[0]["daily"]["time"]
    assert result[1]["daily"]["temperature_2m_max"] == REAL_GRID_RESPONSE_SUBSET[1]["daily"]["temperature_2m_max"]


def test_fetch_archive_grid_excludes_trailing_days_for_every_point(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_GRID_RESPONSE_SUBSET)
    result = fetch_archive_grid(
        _GRID_POINTS_2, past_days=10, exclude_trailing_days=2, _today=date(2026, 9, 7),
    )
    # REAL_GRID_RESPONSE_SUBSET konczy sie 2026-09-06 (nie 09-07, w
    # przeciwienstwie do REAL_ARCHIVE_RESPONSE powyzej) - przy
    # _today=2026-09-07/exclude=2 (cutoff=2026-09-06) odcina to TYLKO
    # ostatni dzien (09-06), nie dwa.
    for rec in result:
        assert rec["daily"]["time"] == REAL_GRID_RESPONSE_SUBSET[0]["daily"]["time"][:-1]
        lengths = {len(v) for v in rec["daily"].values()}
        assert len(lengths) == 1  # wszystkie pola tego punktu rownej dlugosci po przycieciu


def test_fetch_archive_grid_rejects_bad_past_days():
    with pytest.raises(ValueError):
        fetch_archive_grid(_GRID_POINTS_2, past_days=0)
    with pytest.raises(ValueError):
        fetch_archive_grid(_GRID_POINTS_2, past_days=93)


def test_fetch_archive_grid_rejects_empty_points():
    with pytest.raises(ValueError):
        fetch_archive_grid([])


def test_fetch_archive_grid_rejects_mismatched_record_count(monkeypatch):
    monkeypatch.setattr(archive_source, "_http_get", lambda url, params, timeout=20.0: REAL_GRID_RESPONSE_SUBSET)
    with pytest.raises(ValueError):
        fetch_archive_grid([GridPoint(lat=1.0, lon=2.0)])  # 1 punkt zadany, 2 rekordy w fixture
