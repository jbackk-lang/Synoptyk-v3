"""
membrane/archive_source.py — pobranie RZECZYWISTYCH (nie prognozowanych)
danych dobowych z Open-Meteo Archive API, do liczenia trafnosci prognozy
(patrz membrane/bias.py). Ta sama rola co SYNOPTYK-ARCTIC/fetch.py:fetch_archive,
przeniesiona na schemat danych Synoptyk-v3.

KONTRAKT API zweryfikowany NA ZYWO przez przegladarke wewnetrzna (nie
zgadywany) — 2026-09-07, Warszawa (52.23, 21.01), zapytanie:

    https://archive-api.open-meteo.com/v1/archive?latitude=52.23&longitude=21.01
    &daily=precipitation_sum,temperature_2m_max,temperature_2m_min,
    wind_speed_10m_max,wind_direction_10m_dominant,surface_pressure_mean
    &past_days=10&timezone=UTC

Odpowiedz miala DOKLADNIE ten sam ksztalt co /v1/forecast (grid_source.py) —
te same nazwy dobowych pol, w tym `surface_pressure_mean`, ktore dzialalo
tu bez zmian (niezweryfikowane wczesniej akurat dla TEGO endpointu, tylko
dla /v1/forecast). Zwrocila dane az do "dzisiaj" (2026-09-07) wlacznie.

UCZCIWE ZASTRZEZENIE (ten sam mechanizm i to samo empiryczne odkrycie co
SYNOPTYK-ARCTIC/fetch.py:fetch_archive — patrz jego docstring): ostatnie
~1-2 dni Archive API to jeszcze niesfinalizowana reanaliza, praktycznie
identyczna z modelem prognozy — porownanie prognoza-vs-archiwum na tych
dniach nie jest niezaleznym testem trafnosci, tylko porownaniem modelu
z samym soba. `exclude_trailing_days` (domyslnie 2) odcina je przed
zwroceniem, dokladnie jak w SYNOPTYK-ARCTIC.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from .grid_source import GridPoint, _http_get

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_FIELDS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
]

# Pola dla SIATKI punktow (nie pojedynczego miasta) - jedno pole wiecej
# (relative_humidity_2m_mean) niz DAILY_FIELDS powyzej, bo
# meta_adapter.py::archive_grid_response_to_daily_records() go wymaga
# (humidity_pct - jeden z szesciu kanalow rekordu membrany). Dokladnie te
# same nazwy pol co w REAL_GRID_RESPONSE (tests/test_meta_adapter.py),
# przechwyconym realnym zapytaniu z 2026-09-08.
GRID_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
    "relative_humidity_2m_mean",
]


def fetch_archive(
    lat: float,
    lon: float,
    past_days: int = 10,
    exclude_trailing_days: int = 2,
    timeout: float = 20.0,
    _today: date | None = None,
) -> dict[str, Any]:
    """Pobiera zarejestrowana historie (past_days wstecz od dzis) dla
    JEDNEGO punktu (centrum miasta) — ten sam ksztalt wyniku co
    grid_source.fetch_meteogram(), zeby run_collect.py mogl je zapisywac
    tym samym kodem/CSV_FIELDS:

        {"days": [str,...], "precip_mm":[...], "temp_max_c":[...],
         "temp_min_c":[...], "wind_speed_kmh":[...], "wind_dir_deg":[...],
         "pressure_hpa":[...]}

    `exclude_trailing_days` odcina najswiezsze dni (patrz zastrzezenie w
    docstringu modulu) — dlugosc wszystkich list w wyniku maleje o tyle
    samo dni, zsynchronizowanie miedzy polami jest zachowane (ten sam
    filtr indeksow stosowany do kazdej listy).

    `_today` — tylko do testow (zeby nie zalezec od zegara systemowego).

    Rzuca ValueError, gdy `past_days` poza [1,92] (limit Archive API dla
    tego zakresu w praktyce znacznie wiekszy, ale 92 wystarcza z zapasem
    do liczenia trafnosci lead_days 0-7 i nie testowalismy dalej)."""
    if past_days < 1 or past_days > 92:
        raise ValueError(f"past_days musi byc w [1,92], dostano {past_days}")

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_FIELDS),
        "past_days": past_days,
        "timezone": "UTC",
    }
    data = _http_get(ARCHIVE_URL, params)
    daily = data.get("daily", {})

    result = {
        "days": daily.get("time", []),
        "precip_mm": daily.get("precipitation_sum", []),
        "temp_max_c": daily.get("temperature_2m_max", []),
        "temp_min_c": daily.get("temperature_2m_min", []),
        "wind_speed_kmh": daily.get("wind_speed_10m_max", []),
        "wind_dir_deg": daily.get("wind_direction_10m_dominant", []),
        "pressure_hpa": daily.get("surface_pressure_mean", []),
    }

    if exclude_trailing_days <= 0:
        return result

    today = _today if _today is not None else date.today()
    cutoff = today - timedelta(days=exclude_trailing_days - 1)
    keep_idx = [
        i for i, d in enumerate(result["days"])
        if datetime.strptime(d, "%Y-%m-%d").date() < cutoff
    ]
    return {key: [vals[i] for i in keep_idx] for key, vals in result.items()}


def fetch_archive_grid(
    points: list[GridPoint],
    past_days: int = 10,
    exclude_trailing_days: int = 2,
    timeout: float = 20.0,
    _today: date | None = None,
) -> list[dict[str, Any]]:
    """Pobiera zarejestrowana historie (past_days wstecz od dzis) DLA CALEJ
    SIATKI punktow W JEDNYM zapytaniu (Open-Meteo wspiera liste
    wspolrzednych na endpoincie archiwalnym dokladnie tak samo jak na
    /v1/forecast - potwierdzone REALNYM zapytaniem 3x3, patrz
    REAL_GRID_RESPONSE w tests/test_meta_adapter.py, zlapane niezaleznie od
    tego, kiedy fetch_archive() powyzej bylo weryfikowane dla jednego
    punktu, ale zgadzajace sie z nim liczbowo dla wspolnych dni/wspolrzednych).

    W przeciwienstwie do fetch_archive() (jeden punkt, wynik juz
    "splaszczony" do list per-pole), ta funkcja zwraca SUROWY ksztalt
    Open-Meteo - liste obiektow {"latitude":.., "longitude":..,
    "daily":{...}}, jeden na punkt, W TEJ SAMEJ KOLEJNOSCI co `points" -
    dokladnie to, czego oczekuje
    meta_adapter.archive_grid_response_to_daily_records(). Nie
    splaszczamy tutaj, bo ta funkcja konsumuje wszystkie 9 (albo 25, w
    zaleznosci od `n` siatki) punktow naraz, nie jeden.

    `exclude_trailing_days` przycina NAJSWIEZSZE dni tak samo dla KAZDEGO
    punktu (te same indeksy dni - Open-Meteo zwraca rowna liczbe dni per
    punkt dla jednego zapytania o wspolny zakres, patrz
    archive_grid_response_to_daily_records(), ktore i tak to sprawdza
    jawnie) - identyczne uzasadnienie (niesfinalizowana reanaliza ostatnich
    ~1-2 dni) co w fetch_archive().

    Rzuca ValueError gdy `past_days` poza [1,92] (jak fetch_archive) albo
    `points` jest puste."""
    if not points:
        raise ValueError("points nie moze byc puste")
    if past_days < 1 or past_days > 92:
        raise ValueError(f"past_days musi byc w [1,92], dostano {past_days}")

    lats = ",".join(f"{p.lat:.4f}" for p in points)
    lons = ",".join(f"{p.lon:.4f}" for p in points)
    params = {
        "latitude": lats,
        "longitude": lons,
        "daily": ",".join(GRID_DAILY_FIELDS),
        "past_days": past_days,
        "timezone": "UTC",
    }
    data = _http_get(ARCHIVE_URL, params)
    # Jeden punkt -> pojedynczy obiekt (NIE lista), tak samo jak
    # fetch_grid_snapshot() w grid_source.py - znormalizuj do listy.
    records = data if isinstance(data, list) else [data]
    if len(records) != len(points):
        raise ValueError(
            f"Open-Meteo zwrocilo {len(records)} rekordow dla {len(points)} punktow siatki - "
            "niespodziewana niezgodnosc, parsowanie przerwane zamiast zgadywac dopasowanie."
        )

    if exclude_trailing_days <= 0:
        return records

    today = _today if _today is not None else date.today()
    cutoff = today - timedelta(days=exclude_trailing_days - 1)

    trimmed = []
    for rec in records:
        daily = rec.get("daily", {})
        times = daily.get("time", [])
        keep_idx = [
            i for i, d in enumerate(times)
            if datetime.strptime(d, "%Y-%m-%d").date() < cutoff
        ]
        trimmed_daily = {key: [vals[i] for i in keep_idx] for key, vals in daily.items()}
        trimmed.append({"latitude": rec.get("latitude"), "longitude": rec.get("longitude"), "daily": trimmed_daily})
    return trimmed
