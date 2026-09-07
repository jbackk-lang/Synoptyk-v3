"""Testy membrane/bias.py - ten sam wzorzec co
SYNOPTYK-ARCTIC/tests/test_bias.py, na syntetycznym CSV zbudowanym
recznie (nie wymaga sieci)."""
from __future__ import annotations

import csv

from membrane.bias import compute_lead_bias

FIELDS = [
    "city", "issue_date", "target_date", "lead_days", "source",
    "precip_mm", "temp_max_c", "temp_min_c",
    "wind_speed_kmh", "wind_dir_deg", "pressure_hpa",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _row(city, issue, target, lead, source, temp_max):
    return {
        "city": city, "issue_date": issue, "target_date": target, "lead_days": lead,
        "source": source, "precip_mm": "", "temp_max_c": temp_max, "temp_min_c": "",
        "wind_speed_kmh": "", "wind_dir_deg": "", "pressure_hpa": "",
    }


def test_compute_lead_bias_empty_csv_returns_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    _write_csv(csv_path, [])
    assert compute_lead_bias(str(csv_path), "Warszawa") == {}


def test_compute_lead_bias_missing_file_returns_empty(tmp_path):
    assert compute_lead_bias(str(tmp_path / "brak.csv"), "Warszawa") == {}


def test_compute_lead_bias_below_min_samples_returns_empty(tmp_path):
    rows = [
        _row("Warszawa", "2026-09-01", "2026-09-02", 1, "prognoza", 20.0),
        _row("Warszawa", "2026-09-02", "2026-09-02", 0, "archiwum_openmeteo", 21.0),
    ]
    csv_path = tmp_path / "few.csv"
    _write_csv(csv_path, rows)
    # tylko 1 sparowany dzien na lead=1, min_samples domyslnie 5
    assert compute_lead_bias(str(csv_path), "Warszawa") == {}


def test_compute_lead_bias_computes_mae_and_bias_with_enough_samples(tmp_path):
    rows = []
    # 6 dni prognoz lead=1, kazdy o 2.0 nizszy niz "rzeczywistosc"
    for i in range(6):
        issue = f"2026-09-{i+1:02d}"
        target = f"2026-09-{i+2:02d}"
        rows.append(_row("Warszawa", issue, target, 1, "prognoza", 20.0))
        rows.append(_row("Warszawa", target, target, 0, "archiwum_openmeteo", 22.0))
    csv_path = tmp_path / "full.csv"
    _write_csv(csv_path, rows)

    result = compute_lead_bias(str(csv_path), "Warszawa", min_samples=5)
    assert 1 in result
    assert result[1]["n"] == 6
    assert result[1]["bias"] == 2.0  # rzeczywistosc(22) - prognoza(20)
    assert result[1]["mae"] == 2.0


def test_compute_lead_bias_ignores_other_cities(tmp_path):
    rows = []
    for i in range(6):
        issue = f"2026-09-{i+1:02d}"
        target = f"2026-09-{i+2:02d}"
        rows.append(_row("Krakow", issue, target, 1, "prognoza", 20.0))
        rows.append(_row("Krakow", target, target, 0, "archiwum_openmeteo", 22.0))
    csv_path = tmp_path / "other_city.csv"
    _write_csv(csv_path, rows)
    assert compute_lead_bias(str(csv_path), "Warszawa") == {}


def test_compute_lead_bias_ignores_rows_without_matching_source(tmp_path):
    """Wiersze source=prognoza NIGDY nie licza sie jako 'rzeczywistosc' -
    gdyby tak bylo, kazdy dzien parowalby sie sam ze soba (bias=0/mae=0
    zawsze), dokladnie ten blad, ktory SYNOPTYK-ARCTIC/fetch.py naprawia
    przez exclude_trailing_days (patrz jego docstring). Tu: same wiersze
    prognoza, ZERO archiwum_openmeteo -> wynik musi byc pusty, nie
    falszywie doskonaly."""
    rows = []
    for i in range(6):
        issue = f"2026-09-{i+1:02d}"
        target = f"2026-09-{i+2:02d}"
        rows.append(_row("Warszawa", issue, target, 1, "prognoza", 20.0))
    csv_path = tmp_path / "only_forecast.csv"
    _write_csv(csv_path, rows)
    assert compute_lead_bias(str(csv_path), "Warszawa") == {}
