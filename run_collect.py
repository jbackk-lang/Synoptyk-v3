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
"""
from __future__ import annotations

import csv as _csv
from datetime import datetime, timezone
from pathlib import Path

from membrane.cities import City, resolve_city
from membrane.grid_source import fetch_meteogram

CSV_FIELDS = [
    "city", "issue_date", "target_date", "lead_days",
    "precip_mm", "temp_max_c", "temp_min_c",
    "wind_speed_kmh", "wind_dir_deg", "pressure_hpa",
]

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "meteogram_snapshots.csv"


def _read_existing_issue_dates(csv_path: Path, city_name: str) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    return {r["issue_date"] for r in rows if r.get("city") == city_name}


def collect(city: str | None = None, csv_path: str | Path = DEFAULT_CSV_PATH, days: int = 7) -> dict:
    """Pobiera prognoze `days`-dniowa dla `city` (domyslnie
    membrane.cities.DEFAULT_CITY) i dopisuje do CSV, POMIJAJAC jesli dla
    (city, dzisiejszy issue_date UTC) juz cos zapisano (idempotentnosc -
    wielokrotne klikniecie przycisku "Zbierz dane" tego samego dnia nie
    duplikuje wierszy, ten sam wzorzec co SYNOPTYK-ARCTIC/run_arctic.py).

    Zwraca {"city":.., "issue_date":.., "n_added":.., "skipped": bool}.
    Bledy sieciowe (requests.RequestException) sa PRZEPUSZCZANE w gore
    (nie polykane) - webapp/app.py zamienia je na czytelny komunikat
    HTTP zamiast cichej pustej odpowiedzi."""
    c: City = resolve_city(city)
    csv_path = Path(csv_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    existing = _read_existing_issue_dates(csv_path, c.name)
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
            "precip_mm": _at("precip_mm"), "temp_max_c": _at("temp_max_c"), "temp_min_c": _at("temp_min_c"),
            "wind_speed_kmh": _at("wind_speed_kmh"), "wind_dir_deg": _at("wind_dir_deg"),
            "pressure_hpa": _at("pressure_hpa"),
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return {"city": c.name, "issue_date": today, "n_added": len(rows), "skipped": False}


if __name__ == "__main__":
    import sys
    city_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect(city_arg)
    print(result)
