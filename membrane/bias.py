"""
membrane/bias.py — trafnosc prognozy (bias/MAE per lead_days), identyczna
logika co SYNOPTYK-ARCTIC/arctic_synoptyk/bias.py, przeniesiona na schemat
CSV Synoptyk-v3 (run_collect.py: kolumny city/issue_date/target_date/
lead_days/source/temp_max_c/...).

bias = rzeczywistosc - prognoza, per lead_days, liczone TYLKO gdy jest
>= min_samples sparowanych dni (parowanie po target_date, source=prognoza
vs source=archiwum_openmeteo) — brak wpisu w wyniku = brak wystarczajacych
danych, NIGDY zero udawane jako "prognoza idealna".

UCZCIWE OCZEKIWANIE na start: przy jednym-dwoch dniach zbierania (patrz
README, "Trafnosc prognozy") ten modul zwroci PUSTY slownik dla kazdego
miasta — to jest POPRAWNE zachowanie (min_samples domyslnie 5), nie blad,
dokladnie jak SYNOPTYK-ARCTIC/bias.py po pierwszym pobraniu (patrz jego
docstring)."""
from __future__ import annotations

import csv
from collections import defaultdict


def _load_pairs(csv_path: str, city: str, forecast_col: str = "temp_max_c") -> list[dict]:
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return []

    rows = [r for r in rows if r.get("city") == city]
    fc = [r for r in rows if r.get("source") == "prognoza"]
    real = [r for r in rows if r.get("source") == "archiwum_openmeteo"]

    real_by_date: dict[str, float] = {}
    for r in real:
        v = r.get(forecast_col)
        if v not in (None, ""):
            real_by_date[r["target_date"]] = float(v)

    pairs = []
    for r in fc:
        real_val = real_by_date.get(r["target_date"])
        fc_val = r.get(forecast_col)
        lead = r.get("lead_days")
        if real_val is None or fc_val in (None, "") or lead in (None, ""):
            continue
        pairs.append({
            "lead_days": int(float(lead)),
            "forecast": float(fc_val),
            "real": real_val,
        })
    return pairs


def compute_lead_bias(
    csv_path: str,
    city: str,
    min_samples: int = 5,
    forecast_col: str = "temp_max_c",
) -> dict[int, dict]:
    """Zwraca {lead_days: {"bias":.., "mae":.., "n":..}} TYLKO dla
    lead_days z >= min_samples sparowanymi obserwacjami (target_date
    obecny zarowno w wierszach prognoza jak i archiwum_openmeteo dla tego
    miasta) — brak wpisu = brak wystarczajacych danych, nie zero."""
    pairs = _load_pairs(csv_path, city, forecast_col=forecast_col)
    if not pairs:
        return {}

    by_lead: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_lead[p["lead_days"]].append(p)

    result: dict[int, dict] = {}
    for lead, group in by_lead.items():
        n = len(group)
        if n < min_samples:
            continue
        errors = [g["real"] - g["forecast"] for g in group]
        result[lead] = {
            "bias": round(sum(errors) / n, 3),
            "mae": round(sum(abs(e) for e in errors) / n, 3),
            "n": n,
        }
    return result
