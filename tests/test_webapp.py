"""Testy webapp/app.py przez FastAPI TestClient - siec zawsze
monkeypatchowana (grid_source.fetch_meteogram / fetch_grid_snapshot),
patrz uzasadnienie braku sieci w grid_source.py."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from membrane import grid_source
from webapp.app import app

client = TestClient(app)

FAKE_METEOGRAM = {
    "days": ["2026-09-06", "2026-09-07"],
    "precip_mm": [1.2, 0.0],
    "temp_max_c": [20.0, 22.0],
    "temp_min_c": [10.0, 11.0],
    "wind_speed_kmh": [15.0, 20.0],
    "wind_dir_deg": [270, 280],
    "pressure_hpa": [1012.0, 1010.0],
}


def _fake_grid_snapshot(points):
    rng = np.random.default_rng(0)
    return [
        {
            "lat": p.lat, "lon": p.lon,
            "temperature_c": 15.0 + rng.normal(scale=0.1),
            "pressure_hpa": 1013.0 + rng.normal(scale=0.1),
            "humidity_pct": 60.0,
            "wind_speed_kmh": 10.0,
            "wind_dir_deg": 250.0,
            "precip_mm": 0.0,
        }
        for p in points
    ]


def test_cities_endpoint():
    r = client.get("/api/cities")
    assert r.status_code == 200
    data = r.json()
    assert data["default"] == "Warszawa"
    assert any(c["name"] == "Krakow" for c in data["cities"])


def test_meteogram_endpoint(monkeypatch):
    monkeypatch.setattr("webapp.app.fetch_meteogram", lambda lat, lon, days=7: dict(FAKE_METEOGRAM))
    r = client.get("/api/meteogram?city=Warszawa")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Warszawa"
    assert data["precip_mm"] == [1.2, 0.0]


def test_meteogram_unknown_city_404(monkeypatch):
    monkeypatch.setattr("webapp.app.fetch_meteogram", lambda lat, lon, days=7: dict(FAKE_METEOGRAM))
    r = client.get("/api/meteogram?city=Atlantyda")
    assert r.status_code == 404


def test_collect_endpoint(monkeypatch, tmp_path):
    import run_collect as run_collect_mod
    monkeypatch.setattr(run_collect_mod, "fetch_meteogram", lambda lat, lon, days=7: dict(FAKE_METEOGRAM))
    monkeypatch.setattr("webapp.app.DEFAULT_CSV_PATH", tmp_path / "hist.csv")
    monkeypatch.setattr(run_collect_mod, "DEFAULT_CSV_PATH", tmp_path / "hist.csv")
    r = client.post("/api/collect?city=Gdansk")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Gdansk"
    assert data["n_added"] == 2


def test_analyze_endpoint(monkeypatch):
    monkeypatch.setattr(grid_source, "_http_get", None)  # nigdy nie powinno byc wywolane
    monkeypatch.setattr("webapp.app.run_membrane_analysis",
                         lambda lat, lon: __import__("membrane.analyze", fromlist=["analyze_records"]).analyze_records(_fake_grid_snapshot(grid_source.build_grid_points(lat, lon, n=5, spacing_deg=0.35))))
    r = client.post("/api/analyze?city=Poznan")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Poznan"
    assert "membrane" in data
    assert "resonance" in data


def test_history_endpoint_empty_when_no_csv(monkeypatch, tmp_path):
    monkeypatch.setattr("webapp.app.DEFAULT_CSV_PATH", tmp_path / "does_not_exist.csv")
    r = client.get("/api/history?city=Lublin")
    assert r.status_code == 200
    assert r.json()["rows"] == []


FAKE_ARCHIVE = {
    "days": ["2026-08-28", "2026-08-29"],
    "precip_mm": [0.0, 5.2],
    "temp_max_c": [23.8, 21.9],
    "temp_min_c": [13.0, 15.7],
    "wind_speed_kmh": [18.5, 16.0],
    "wind_dir_deg": [131, 206],
    "pressure_hpa": [1006.2, 999.7],
}


def test_collect_archive_endpoint(monkeypatch, tmp_path):
    import run_collect as run_collect_mod
    monkeypatch.setattr(run_collect_mod, "fetch_archive", lambda lat, lon, past_days=10: dict(FAKE_ARCHIVE))
    monkeypatch.setattr("webapp.app.DEFAULT_CSV_PATH", tmp_path / "hist.csv")
    monkeypatch.setattr(run_collect_mod, "DEFAULT_CSV_PATH", tmp_path / "hist.csv")
    r = client.post("/api/collect_archive?city=Gdansk")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Gdansk"
    assert data["n_added"] == 2


def test_bias_endpoint_insufficient_data_when_no_history(monkeypatch, tmp_path):
    monkeypatch.setattr("webapp.app.DEFAULT_CSV_PATH", tmp_path / "does_not_exist.csv")
    r = client.get("/api/bias?city=Warszawa")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "insufficient_data"
    assert data["by_lead"] == {}


def test_bias_endpoint_ok_with_enough_paired_history(monkeypatch, tmp_path):
    import csv as _csv
    csv_path = tmp_path / "hist.csv"
    fields = [
        "city", "issue_date", "target_date", "lead_days", "source",
        "precip_mm", "temp_max_c", "temp_min_c",
        "wind_speed_kmh", "wind_dir_deg", "pressure_hpa",
    ]
    rows = []
    for i in range(6):
        issue = f"2026-09-{i+1:02d}"
        target = f"2026-09-{i+2:02d}"
        rows.append({"city": "Warszawa", "issue_date": issue, "target_date": target, "lead_days": 1,
                     "source": "prognoza", "precip_mm": "", "temp_max_c": 20.0, "temp_min_c": "",
                     "wind_speed_kmh": "", "wind_dir_deg": "", "pressure_hpa": ""})
        rows.append({"city": "Warszawa", "issue_date": target, "target_date": target, "lead_days": 0,
                     "source": "archiwum_openmeteo", "precip_mm": "", "temp_max_c": 22.0, "temp_min_c": "",
                     "wind_speed_kmh": "", "wind_dir_deg": "", "pressure_hpa": ""})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    monkeypatch.setattr("webapp.app.DEFAULT_CSV_PATH", csv_path)
    r = client.get("/api/bias?city=Warszawa")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["by_lead"]["1"]["n"] == 6
    assert data["by_lead"]["1"]["bias"] == 2.0


def _fake_archive_grid_response(points, n_days=4):
    """Syntetyczny (NIE realny, w przeciwienstwie do REAL_GRID_RESPONSE w
    tests/test_meta_adapter.py) ksztalt odpowiedzi siatki archiwalnej -
    wystarczajacy do sprawdzenia OKABLOWANIA /api/meta (fetch_archive_grid
    -> archive_grid_response_to_daily_records -> build_meta_series_from_
    daily_records -> JSON), nie do weryfikacji samej nauki (to juz robi
    test_meta_adapter.py na PRAWDZIWYCH danych)."""
    dates = [f"2026-09-{i+1:02d}" for i in range(n_days)]
    out = []
    for i, p in enumerate(points):
        out.append({"latitude": p.lat, "longitude": p.lon, "daily": {
            "time": dates,
            "temperature_2m_max": [20.0 + i + 0.5 * d for d in range(n_days)],
            "temperature_2m_min": [10.0 + i for _ in range(n_days)],
            "precipitation_sum": [0.0 for _ in range(n_days)],
            "wind_speed_10m_max": [15.0 for _ in range(n_days)],
            "wind_direction_10m_dominant": [180 for _ in range(n_days)],
            "surface_pressure_mean": [1010.0 for _ in range(n_days)],
            "relative_humidity_2m_mean": [70 for _ in range(n_days)],
        }})
    return out


def test_meta_endpoint(monkeypatch):
    import webapp.app as app_mod
    monkeypatch.setattr(
        app_mod, "fetch_archive_grid",
        lambda points, past_days=10, exclude_trailing_days=2: _fake_archive_grid_response(points),
    )
    r = client.get("/api/meta?city=Warszawa")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Warszawa"
    assert len(data["dates"]) == 4
    assert len(data["states"]) == 4
    assert len(data["phases"]) == 3
    assert len(data["magnitude"]) == 3
    assert all(p in ("stabilna", "przejsciowa", "krytyczna") for p in data["phases"])
    assert "triggered" in data["trigger"]


def test_meta_endpoint_unknown_city_404(monkeypatch):
    import webapp.app as app_mod
    monkeypatch.setattr(
        app_mod, "fetch_archive_grid",
        lambda points, past_days=10, exclude_trailing_days=2: _fake_archive_grid_response(points),
    )
    r = client.get("/api/meta?city=Atlantyda")
    assert r.status_code == 404
