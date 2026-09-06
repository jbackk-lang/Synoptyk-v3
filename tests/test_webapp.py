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
