"""
membrane/grid_source.py — pobranie SIATKI punktow pogodowych z Open-Meteo,
surowy material wejsciowy dla interpolate.py (Krok 1 z propozycji
uzytkownika: "naniesienie danych na siatke (GRID)").

Uzyto Open-Meteo (ten sam dostawca co SYNOPTYK-ARCTIC i synoptyk-v2.0 -
darmowe, bez klucza API, ten sam ekosystemowy wybor) zamiast ERA5/GFS
wprost: ERA5 wymaga rejestracji + tokena CDS i pobrania duzych plikow
NetCDF/GRIB, GFS wymaga parsowania plikow GRIB2 - oba niepraktyczne do
zbudowania i przetestowania w jednej sesji. Open-Meteo agreguje dane z
tych samych modeli NWP (w tym GFS/ICON) pod prostym JSON API - siatka
punktow zapytana TUTAJ jest wiec faktycznie prognoza modelu numerycznego
(nie interpolacja stacji), tylko pobierana punkt-po-punkcie zamiast jako
caly plik siatki.

UCZCIWE ZASTRZEZENIE (ten sam wzorzec co TIMDR-Geometry-Formalism/
weingarten.py w skillu timdr-signal-framework, ktory tez nie mogl zostac
uruchomiony we wlasnej sesji): sandbox bash tej sesji ma ZABLOKOWANY
dostep do internetu (kazde zapytanie requests.get() konczy sie
ProxyError/403 - sprawdzone bezposrednio). Kontrakt API PONIZEJ zostal
mimo to zweryfikowany na PRAWDZIWYCH danych - przez osobny kanal
(przegladarke wewnetrzna), ktora dostep do sieci ma - trzema realnymi
zapytaniami do api.open-meteo.com (pojedynczy punkt, dwa punkty, pelny
zestaw pol). Potwierdzone stamtad, dokladnie, i wpisane tu bez zgadywania:
  - JEDEN punkt (?latitude=X&longitude=Y) -> pojedynczy obiekt JSON.
  - WIELE punktow (?latitude=X1,X2&longitude=Y1,Y2) -> LISTA obiektow
    JSON, jeden na punkt, W TEJ SAMEJ KOLEJNOSCI co podane wspolrzedne.
  - Pola current: temperature_2m [C], surface_pressure [hPa],
    relative_humidity_2m [%], wind_speed_10m [km/h],
    wind_direction_10m [stopnie, KIERUNEK SKAD wieje wiatr - konwencja
    meteorologiczna, ta sama co w SYNOPTYK-ARCTIC/Synoptyk-v2.0],
    precipitation [mm].
Testy w tests/test_grid_source.py uzywaja PRAWDZIWEJ odpowiedzi JSON
zlapanej z tych zapytan jako fixture (nie zmyslonego ksztaltu) do
sprawdzenia parsowania - ale sam _http_get() (rzeczywiste polaczenie
sieciowe z wnetrza tej appki) nie zostal uruchomiony w tej sesji. Kiedy
ta appka dziala na normalnym komputerze uzytkownika (nie w tym
sandboxie), dostep do internetu jest zwykly - to ograniczenie jest
specyficzne dla srodowiska, w ktorym appka zostala NAPISANA, nie dla
tego, w ktorym bedzie URUCHOMIONA.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_FIELDS = [
    "temperature_2m",
    "surface_pressure",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lon: float


def build_grid_points(center_lat: float, center_lon: float, n: int = 5, spacing_deg: float = 0.35) -> list[GridPoint]:
    """Buduje kwadratowa siatke n x n punktow wysrodkowana na
    (center_lat, center_lon), rozstaw `spacing_deg` stopni geograficznych
    (domyslnie 0.35 stopnia ~= 25-39 km, w zaleznosci od szerokosci -
    wystarczajaco gesto, zeby front atmosferyczny nad Polska wypadl na
    kilku wewnetrznych punktach, wystarczajaco rzadko, zeby n=5 dalo
    zasieg ~140 km w kazda strone od centrum miasta).

    `n` MUSI byc nieparzyste, zeby centrum siatki pokrywalo sie dokladnie
    z (center_lat, center_lon) - sprawdzone jawnie (ValueError), zamiast
    cicho przesuwac siatke o pol oczka przy n parzystym.
    """
    if n < 3:
        raise ValueError(f"n musi byc >= 3 (potrzebne co najmniej 3x3 do interpolacji/gradientu), dostano n={n}")
    if n % 2 == 0:
        raise ValueError(f"n musi byc nieparzyste, zeby centrum siatki = (center_lat, center_lon) dokladnie, dostano n={n}")
    half = n // 2
    points = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            points.append(GridPoint(lat=center_lat + i * spacing_deg, lon=center_lon + j * spacing_deg))
    return points


def _http_get(url: str, params: dict, timeout: float = 20.0) -> object:
    """Cienka warstwa nad requests.get(), wydzielona wylacznie po to, zeby
    testy mogly ja podmienic (monkeypatch) fixture'em z realnej
    odpowiedzi - patrz docstring modulu. Rzuca requests.RequestException
    jawnie w gore (NIE polyka bledow siecowych), zeby caller
    (fetch_grid_snapshot/fetch_meteogram) i ostatecznie webapp/app.py
    mogly pokazac uzytkownikowi realny powod niepowodzenia (np. brak
    internetu), zamiast cichej pustej odpowiedzi."""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_grid_snapshot(points: list[GridPoint]) -> list[dict]:
    """Pobiera BIEZACE odczyty (current=...) dla calej listy punktow W
    JEDNYM zapytaniu (Open-Meteo wspiera liste wspolrzednych w
    ?latitude=a,b,c&longitude=x,y,z - potwierdzone realnie, patrz
    docstring modulu) - nie N osobnych zapytan, zeby nie zuzywac
    limitu/czasu N-krotnie dla siatki n x n (np. n=5 -> 25 punktow).

    Zwraca liste slownikow, JEDNA POZYCJA NA PUNKT, W TEJ SAMEJ
    KOLEJNOSCI co `points` (Open-Meteo gwarantuje kolejnosc odpowiedzi ==
    kolejnosc wspolrzednych na wejsciu - potwierdzone realnym zapytaniem
    dwupunktowym, patrz docstring modulu):
        {"lat":.., "lon":.., "temperature_c":.., "pressure_hpa":..,
         "humidity_pct":.., "wind_speed_kmh":.., "wind_dir_deg":..,
         "precip_mm":..}
    Brakujace pole w odpowiedzi (np. API czasem nie zwraca jednej zmiennej
    dla konkretnego punktu) -> None w wyniku, NIE 0.0 (patrz interpolate.py,
    ktore musi umiec pominac None przy budowie siatki wejsciowej dla
    scipy.griddata zamiast fałszywie wstawiac zero jako pomiar).
    """
    if not points:
        return []
    lats = ",".join(f"{p.lat:.4f}" for p in points)
    lons = ",".join(f"{p.lon:.4f}" for p in points)
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "UTC",
    }
    data = _http_get(OPEN_METEO_URL, params)
    # Jeden punkt -> pojedynczy obiekt (NIE lista) - potwierdzone realnie,
    # patrz docstring modulu. Znormalizuj do listy, zeby reszta funkcji
    # nie musiala znac tej niespojnosci API.
    records = data if isinstance(data, list) else [data]
    if len(records) != len(points):
        raise ValueError(
            f"Open-Meteo zwrocilo {len(records)} rekordow dla {len(points)} punktow - "
            "niespodziewana niezgodnosc, parsowanie przerwane zamiast zgadywac dopasowanie."
        )
    out = []
    for p, rec in zip(points, records):
        cur = rec.get("current", {})
        out.append({
            "lat": p.lat,
            "lon": p.lon,
            "temperature_c": cur.get("temperature_2m"),
            "pressure_hpa": cur.get("surface_pressure"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "wind_speed_kmh": cur.get("wind_speed_10m"),
            "wind_dir_deg": cur.get("wind_direction_10m"),
            "precip_mm": cur.get("precipitation"),
        })
    return out


DAILY_FIELDS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
]


def fetch_meteogram(lat: float, lon: float, days: int = 7) -> dict:
    """Pobiera prognoze DZIEN-PO-DNIU (daily=...) dla JEDNEGO punktu
    (centrum miasta) - to zasila panel "meteogram" w webapp (wykres
    slupkowy opadu + tabela cisnienie/wiatr/kierunek), NIE siatke/membrane
    (do tego sluzy fetch_grid_snapshot powyzej). Pola potwierdzone
    realnym zapytaniem, patrz docstring modulu.

    Zwraca {"days": [str,...], "precip_mm":[...], "temp_max_c":[...],
    "temp_min_c":[...], "wind_speed_kmh":[...], "wind_dir_deg":[...],
    "pressure_hpa":[...]}, wszystkie listy rownej dlugosci = `days`
    (ograniczone forecast_days, Open-Meteo domyslnie daje do 16 dni)."""
    if days < 1 or days > 16:
        raise ValueError(f"days musi byc w [1,16] (limit Open-Meteo forecast_days), dostano {days}")
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_FIELDS),
        "forecast_days": days,
        "timezone": "UTC",
    }
    data = _http_get(OPEN_METEO_URL, params)
    daily = data.get("daily", {})
    return {
        "days": daily.get("time", []),
        "precip_mm": daily.get("precipitation_sum", []),
        "temp_max_c": daily.get("temperature_2m_max", []),
        "temp_min_c": daily.get("temperature_2m_min", []),
        "wind_speed_kmh": daily.get("wind_speed_10m_max", []),
        "wind_dir_deg": daily.get("wind_direction_10m_dominant", []),
        "pressure_hpa": daily.get("surface_pressure_mean", []),
    }
