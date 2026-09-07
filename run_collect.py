"""
run_collect.py — Krok 6 ("Synoptyk klasyczny", przekroje 1D) - pobiera
prognoze dzien-po-dniu (opad/temp/wiatr/cisnienie) dla WYBRANEGO miasta z
Open-Meteo i dopisuje do CSV historii (idempotentnie - jedno pobranie na
(miasto, dzien), tak jak run_arctic.py w SYNOPTYK-ARCTIC).

To jest OSOBNA sciezka danych od membrany (membrane/analyze.py) - CSV tu
zasila WYLACZNIE panel "meteogram" (wykres slupkowy opadu + tabela) w
dashboardzie, nie analize przestrzenna. Membrana jest liczona na zadanie
(POST /api/analyze), nie zapisywana do historii - to swiadomy wybor:
membrana to duzy obiekt (grid_n x grid_n x 6 pol), zapisywanie kazdej
migawki do CSV szybko urosloby do nieporecznych rozmiarow bez wyraznej
korzysci (nikt jeszcze nie prosil o "trafnosc prognozy membrany w czasie"
- gdyby to sie pojawilo, dopiero wtedy warto dodac osobny format
przechowywania, nie CSV plaski jak tutaj).

ROZSZERZENIE (2026-09-07): kolumna "source" (prognoza/archiwum_openmeteo)
dodana, zeby membrane/bias.py moglo policzyc realna trafnosc prognozy
(rzeczywistosc - prognoza), tym samym wzorcem co SYNOPTYK-ARCTIC
(prognoza.py + fetch.py + bias.py). ISTNIEJACY plik CSV sprzed tej zmiany
(bez kolumny "source") zostal ZRESETOWANY (usuniety) zamiast migrowany -
mial tylko 38 wierszy danych testowych z tej samej sesji, w ktorej
powstal caly modul membrany, nie realna, warta zachowania historie (patrz
README, sekcja "Trafnosc prognozy").
"""
from __future__ import annotations

import csv as _csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from membrane.archive_source import fetch_archive
from membrane.cities import City, resolve_city
from membrane.grid_source import fetch_meteogram

CSV_FIELDS = [
    "city", "issue_date", "target_date", "lead_days", "source",
    "precip_mm", "temp_max_c", "temp_min_c",
    "wind_speed_kmh", "wind_dir_deg", "pressure_hpa",
]

SOURCE_FORECAST = "prognoza"
SOURCE_ARCHIVE = "archiwum_openmeteo"

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "meteogram_snapshots.csv"


def _read_existing_issue_dates(csv_path: Path, city_name: str, source: str) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    return {r["issue_date"] for r in rows if r.get("city") == city_name and r.get("source") == source}


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def collect(city: str | None = None, csv_path: str | Path = DEFAULT_CSV_PATH, days: int = 7) -> dict:
    """Pobiera prognoze `days`-dniowa dla `city` (domyslnie
    membrane.cities.DEFAULT_CITY) i dopisuje do CSV, POMIJAJAC jesli dla
    (city, dzisiejszy issue_date UTC, source=prognoza) juz cos zapisano
    (idempotentnosc - wielokrotne klikniecie przycisku "Zbierz dane" tego
    samego dnia nie duplikuje wierszy, ten sam wzorzec co
    SYNOPTYK-ARCTIC/run_arctic.py).

    Zwraca {"city":.., "issue_date":.., "n_added":.., "skipped": bool}.
    Bledy sieciowe (requests.RequestException) sa PRZEPUSZCZANE w gore
    (nie polykane) - webapp/app.py zamienia je na czytelny komunikat
    HTTP zamiast cichej pustej odpowiedzi."""
    c: City = resolve_city(city)
    csv_path = Path(csv_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    existing = _read_existing_issue_dates(csv_path, c.name, SOURCE_FORECAST)
    if today in existing:
        return {"city": c.name, "issue_date": today, "n_added": 0, "skipped": True}

    meteogram = fetch_meteogram(c.lat, c.lon, days=days)
    rows = []
    for i, target_date in enumerate(meteogram["days"]):
        def _at(key, i=i):
            vals = meteogram.get(key) or []
            return vals[i] if i < len(vals) else ""
        rows.append({
            "city": c.name, "issue_date": today, "target_date": target_date, "lead_days": i,
            "source": SOURCE_FORECAST,
            "precip_mm": _at("precip_mm"), "temp_max_c": _at("temp_max_c"), "temp_min_c": _at("temp_min_c"),
            "wind_speed_kmh": _at("wind_speed_kmh"), "wind_dir_deg": _at("wind_dir_deg"),
            "pressure_hpa": _at("pressure_hpa"),
        })

    _append_rows(csv_path, rows)
    return {"city": c.name, "issue_date": today, "n_added": len(rows), "skipped": False}


def collect_archive(
    city: str | None = None,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    past_days: int = 10,
) -> dict:
    """Pobiera RZECZYWISTA (nie prognozowana) historie pogody dla `city` z
    Open-Meteo Archive API (membrane/archive_source.py) i dopisuje do tego
    samego CSV co collect(), z source=archiwum_openmeteo — to jest
    "rzeczywistosc", wzgledem ktorej membrane/bias.py liczy trafnosc
    prognoz zapisanych przez collect().

    `lead_days` dla wierszy archiwalnych = (target_date - issue_date).dni,
    UJEMNE (bo target_date jest w przeszlosci wzgledem dzisiejszego
    issue_date) — dokladnie ta sama konwencja co SYNOPTYK-ARCTIC
    (arctic_forecast_snapshots.csv, kolumna lead_days dla
    source=archiwum_openmeteo). bias.py nie uzywa lead_days wierszy
    archiwalnych do niczego (parowanie jest po target_date), ale
    zachowujemy konwencje dla spojnosci/czytelnosci CSV.

    Idempotentne jak collect() - jedno pobranie na (city, dzisiejszy
    issue_date UTC, source=archiwum_openmeteo)."""
    c: City = resolve_city(city)
    csv_path = Path(csv_path)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_date = datetime.now(timezone.utc).date()

    existing = _read_existing_issue_dates(csv_path, c.name, SOURCE_ARCHIVE)
    if today_str in existing:
        return {"city": c.name, "issue_date": today_str, "n_added": 0, "skipped": True}

    archive = fetch_archive(c.lat, c.lon, past_days=past_days)
    rows = []
    for target_date in archive["days"]:
        i = archive["days"].index(target_date)

        def _at(key, i=i):
            vals = archive.get(key) or []
            return vals[i] if i < len(vals) else ""

        lead = (datetime.strptime(target_date, "%Y-%m-%d").date() - today_date).days
        rows.append({
            "city": c.name, "issue_date": today_str, "target_date": target_date, "lead_days": lead,
            "source": SOURCE_ARCHIVE,
            "precip_mm": _at("precip_mm"), "temp_max_c": _at("temp_max_c"), "temp_min_c": _at("temp_min_c"),
            "wind_speed_kmh": _at("wind_speed_kmh"), "wind_dir_deg": _at("wind_dir_deg"),
            "pressure_hpa": _at("pressure_hpa"),
        })

    _append_rows(csv_path, rows)
    return {"city": c.name, "issue_date": today_str, "n_added": len(rows), "skipped": False}


if __name__ == "__main__":
    import sys
    city_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect(city_arg)
    print(result)
    print(collect_archive(city_arg))
