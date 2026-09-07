"""Testy run_collect.py::collect() - idempotentnosc i poprawnosc zapisu
CSV, bez prawdziwego polaczenia sieciowego (fetch_meteogram
monkeypatchowane - patrz uzasadnienie braku sieci w grid_source.py)."""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import pytest

from datetime import date

import run_collect as run_collect_mod
from run_collect import collect, collect_archive


FAKE_METEOGRAM = {
    "days": ["2026-09-06", "2026-09-07"],
    "precip_mm": [1.2, 0.0],
    "temp_max_c": [20.0, 22.0],
    "temp_min_c": [10.0, 11.0],
    "wind_speed_kmh": [15.0, 20.0],
    "wind_dir_deg": [270, 280],
    "pressure_hpa": [1012.0, 1010.0],
}

FAKE_ARCHIVE = {
    "days": ["2026-08-28", "2026-08-29"],
    "precip_mm": [0.0, 5.2],
    "temp_max_c": [23.8, 21.9],
    "temp_min_c": [13.0, 15.7],
    "wind_speed_kmh": [18.5, 16.0],
    "wind_dir_deg": [131, 206],
    "pressure_hpa": [1006.2, 999.7],
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
    assert rows[0]["source"] == "prognoza"


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


def test_collect_archive_writes_csv_rows_with_negative_lead(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_archive", lambda lat, lon, past_days=10: FAKE_ARCHIVE)
    csv_path = tmp_path / "out.csv"
    result = collect_archive(
        city="Warszawa", csv_path=csv_path,
    )
    assert result["n_added"] == 2
    assert not result["skipped"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["source"] == "archiwum_openmeteo" for r in rows)
    assert all(int(r["lead_days"]) < 0 for r in rows)  # target_date w przeszlosci wzgledem issue_date=dzis


def test_collect_archive_is_idempotent_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_archive", lambda lat, lon, past_days=10: FAKE_ARCHIVE)
    csv_path = tmp_path / "out.csv"
    first = collect_archive(city="Krakow", csv_path=csv_path)
    second = collect_archive(city="Krakow", csv_path=csv_path)
    assert first["n_added"] == 2
    assert second["n_added"] == 0
    assert second["skipped"]


def test_collect_and_collect_archive_share_same_csv_without_colliding(tmp_path, monkeypatch):
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: FAKE_METEOGRAM)
    monkeypatch.setattr(run_collect_mod, "fetch_archive", lambda lat, lon, past_days=10: FAKE_ARCHIVE)
    csv_path = tmp_path / "out.csv"
    r1 = collect(city="Warszawa", csv_path=csv_path)
    r2 = collect_archive(city="Warszawa", csv_path=csv_path)
    assert r1["n_added"] == 2
    assert r2["n_added"] == 2
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 4
    sources = {r["source"] for r in rows}
    assert sources == {"prognoza", "archiwum_openmeteo"}
