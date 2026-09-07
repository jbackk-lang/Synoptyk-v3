"""Testy membrane/archive_source.py.

Ten sam wzorzec co tests/test_grid_source.py: _http_get() monkeypatchowane
odpowiedzia PRAWDZIWA, zlapana recznie z archive-api.open-meteo.com przez
przegladarke wewnetrzna (2026-09-07, Warszawa 52.23/21.01, past_days=10) -
patrz docstring archive_source.py po dokladny URL zapytania."""
from __future__ import annotations

from datetime import date

import pytest

from membrane import archive_source
from membrane.archive_source import fetch_archive

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
