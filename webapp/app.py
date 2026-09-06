"""
webapp/app.py — lokalna appka (FastAPI) dla Synoptyk-v3.

Ten sam wybor architektoniczny co SYNOPTYK-ARCTIC (patrz tamten
webapp/app.py): prawdziwa mala appka, nie statyczny HTML z wbudowanymi
danymi - endpointy licza/pobieraja na zadanie, dashboard odpytuje je przy
kazdym zaladowaniu/klikniecu.

Endpointy:
- GET  /                — dashboard (statyczny HTML+JS)
- GET  /api/cities       — lista miast/regionow do dropdowna
- GET  /api/meteogram    — ZYWA prognoza dzien-po-dniu (Open-Meteo,
                            fetch_meteogram) dla wybranego miasta - zasila
                            wykres slupkowy opadu + tabele (Krok 6,
                            "Synoptyk klasyczny"). NIE czyta CSV historii
                            - zawsze aktualne dane z API.
- POST /api/collect      — dopisuje dzisiejszy dzien do CSV historii
                            (run_collect.collect()) - osobny przycisk od
                            samego wyswietlania (patrz uzasadnienie w
                            SYNOPTYK-ARCTIC/webapp/app.py, ten sam
                            powod: user musi wiedziec, ze "pokaz" i
                            "zapisz do historii" to dwie rozne rzeczy).
- POST /api/analyze      — Kroki 1-5 (membrane/analyze.py): pobiera
                            siatke punktow wokol miasta, interpoluje w
                            membrane, liczy widmo/gradient/wirowosc,
                            wykrywa defekty i rezonans. Kosztowniejsze niz
                            /api/meteogram (N=25 punktow w 1 zapytaniu +
                            cala analiza numeryczna) - stad osobny
                            przycisk "Analizuj membrane", nie
                            wywolywane automatycznie przy kazdym
                            odswiezeniu strony.
- GET  /api/history      — ostatnie wiersze CSV historii (jesli jakies
                            juz zebrano przyciskiem "Zbierz dane") dla
                            wybranego miasta - widocznosc, ze kolektor
                            dziala, ten sam wzorzec co
                            /api/latest_readings w SYNOPTYK-ARCTIC.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from membrane.analyze import run_membrane_analysis, result_to_json
from membrane.cities import CITIES, DEFAULT_CITY, resolve_city
from membrane.grid_source import fetch_meteogram
from run_collect import DEFAULT_CSV_PATH, collect as _collect

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Synoptyk-v3 — membrana pogodowa")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _resolve_or_404(city: str | None):
    try:
        return resolve_city(city)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/cities")
def cities() -> dict:
    return {
        "cities": [{"name": c.name, "lat": c.lat, "lon": c.lon, "region": c.region} for c in CITIES],
        "default": DEFAULT_CITY,
    }


@app.get("/api/meteogram")
def meteogram(city: str | None = None, days: int = 7) -> dict:
    """Zywa prognoza dzien-po-dniu z Open-Meteo (NIE z CSV) - patrz
    docstring modulu. Blad polaczenia sieciowego -> HTTP 502 z czytelnym
    komunikatem (zamiast HTTP 500 nieprzejrzystego stack trace'u), zeby
    front mogl pokazac uzytkownikowi, ze problem jest siecowy, nie
    wewnetrzny."""
    c = _resolve_or_404(city)
    try:
        data = fetch_meteogram(c.lat, c.lon, days=days)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Blad polaczenia z Open-Meteo: {e}") from e
    data["city"] = c.name
    data["lat"] = c.lat
    data["lon"] = c.lon
    return data


@app.post("/api/collect")
def collect_now(city: str | None = None) -> dict:
    c = _resolve_or_404(city)
    try:
        # csv_path PRZEKAZANE JAWNIE (nie poleganie na domyslnym
        # argumencie collect()) - domyslna wartosc parametru w Pythonie
        # jest wiazana RAZ, przy definicji funkcji, wiec podmiana modulowej
        # stalej DEFAULT_CSV_PATH pozniej (np. w testach przez monkeypatch)
        # NIE zmienilaby juz zwiazanego domyslnego argumentu - czytamy
        # wiec DEFAULT_CSV_PATH z globalnej przestrzeni nazw TEGO modulu
        # przy KAZDYM wywolaniu, zeby monkeypatch("webapp.app.DEFAULT_CSV_PATH")
        # w testach faktycznie mial efekt (patrz tests/test_webapp.py).
        return _collect(city=c.name, csv_path=DEFAULT_CSV_PATH)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Blad polaczenia z Open-Meteo: {e}") from e


@app.get("/api/history")
def history(city: str | None = None, limit: int = 14) -> dict:
    c = _resolve_or_404(city)
    csv_path = DEFAULT_CSV_PATH
    if not csv_path.exists():
        return {"city": c.name, "rows": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in _csv.DictReader(f) if r.get("city") == c.name]
    rows_sorted = sorted(rows, key=lambda r: (r.get("issue_date", ""), r.get("target_date", "")), reverse=True)
    return {"city": c.name, "rows": rows_sorted[:limit], "n_total": len(rows)}


@app.post("/api/analyze")
def analyze(city: str | None = None) -> dict:
    """Kroki 1-5: siatka -> membrana -> widmo -> defekty -> rezonans -
    patrz membrane/analyze.py. Zwykle kilka razy wolniejsze niz
    /api/meteogram (25 punktow zapytania + FFT/gradient/interpolacja na
    41x41 membranie), stad osobny przycisk w dashboardzie."""
    c = _resolve_or_404(city)
    try:
        result = run_membrane_analysis(c.lat, c.lon)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Blad polaczenia z Open-Meteo: {e}") from e
    except ValueError as e:
        # np. za malo punktow / brakujace pola po stronie API - blad
        # danych, nie siecowy, ale rownie czytelny dla uzytkownika jak 502.
        raise HTTPException(status_code=422, detail=str(e)) from e
    payload = result_to_json(result)
    payload["city"] = c.name
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
