"""Testy run_collect.py::collect() - idempotentnosc i poprawnosc zapisu
CSV, bez prawdziwego polaczenia sieciowego (fetch_meteogram
monkeypatchowane - patrz uzasadnienie braku sieci w grid_source.py)."""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import pytest

import run_collect as run_collect_mod
from run_collect import collect


FAKE_METEOGRAM = {
    "days": ["2026-09-06", "2026-09-07"],
    "precip_mm": [1.2, 0.0],
    "temp_max_c": [20.0, 22.0],
    "temp_min_c": [10.0, 11.0],
    "wind_speed_kmh": [15.0, 20.0],
    "wind_dir_deg": [270, 280],
    "pressure_hpa": [1012.0, 1010.0],
}


def test_collect_writes_csv_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: FAKE_METEOGRAM)
    csv_path = tmp_path / "out.csv"
    result = collect(city="Warszawa", csv_path=csv_path)
    assert result["n_added"] == 2
    assert not result["skipped"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["city"] == "Warszawa"
    assert rows[0]["target_date"] == "2026-09-06"
    assert rows[1]["lead_days"] == "1"


def test_collect_is_idempotent_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: FAKE_METEOGRAM)
    csv_path = tmp_path / "out.csv"
    first = collect(city="Krakow", csv_path=csv_path)
    second = collect(city="Krakow", csv_path=csv_path)
    assert first["n_added"] == 2
    assert second["n_added"] == 0
    assert second["skipped"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2  # nie 4 - drugie wywolanie nie zdduplikowalo wierszy


def test_collect_unknown_city_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: FAKE_METEOGRAM)
    with pytest.raises(KeyError):
        collect(city="Atlantyda", csv_path=tmp_path / "out.csv")


def test_collect_different_cities_do_not_block_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: FAKE_METEOGRAM)
    csv_path = tmp_path / "out.csv"
    collect(city="Warszawa", csv_path=csv_path)
    result = collect(city="Gdansk", csv_path=csv_path)
    assert result["n_added"] == 2
    assert not result["skipped"]
