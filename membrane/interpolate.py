"""
membrane/interpolate.py — Krok 2: zamiana rzadkiej siatki punktow
(grid_source.py) w jednolita "membrane" - gesta, regularna siatke pol
skalarnych T/P/RH/opad i pola wektorowego wiatru (u,v).

Wiatr NIE jest interpolowany jako (predkosc, kierunek) wprost - kierunek
to wielkosc katowa (0 i 360 to ten sam kierunek), wiec liniowa/kubiczna
interpolacja dwoch "kierunkow" po obu stronach nieciaglosci 0/360 dalaby
bezsensowny wynik (np. interpolacja 350 i 10 stopni "prosto" dalaby 180,
czyli przeciwny kierunek, zamiast poprawnych ~0/360). Standardowe
rozwiazanie meteorologiczne: rozloz na skladowe kartezjanskie u,v
(ciagle, bez nieciaglosci), interpoluj KAZDA oddzielnie, dopiero na
gotowej membranie policz z powrotem predkosc/kierunek jesli potrzebne
(patrz wind_speed_dir_from_uv).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata


def wind_to_uv(speed: float, dir_from_deg: float) -> tuple[float, float]:
    """Konwencja meteorologiczna: dir_from_deg to kierunek, SKAD wieje
    wiatr (0=polnoc, 90=wschod, mierzone zgodnie z ruchem wskazowek
    zegara) - ta sama konwencja co Open-Meteo/SYNOPTYK-ARCTIC/Synoptyk-v2.0
    (patrz WIND_ARROWS w obu appkach). Wektor wiatru sam w sobie wskazuje
    DOKAD wieje, czyli dir_from_deg+180.

    u = skladowa wschodnia (dodatnia -> wieje na wschod)
    v = skladowa polnocna (dodatnia -> wieje na polnoc)

    Standardowy wzor meteorologiczny (np. metpy.calc.wind_components):
        u = -speed * sin(dir_from_rad)
        v = -speed * cos(dir_from_rad)
    (minus, bo dir_from wskazuje SKAD, a u,v to DOKAD)."""
    rad = math.radians(dir_from_deg)
    u = -speed * math.sin(rad)
    v = -speed * math.cos(rad)
    return u, v


def wind_speed_dir_from_uv(u: float, v: float) -> tuple[float, float]:
    """Odwrotnosc wind_to_uv - z powrotem (predkosc, kierunek_skad_stopnie)."""
    speed = math.hypot(u, v)
    if speed == 0:
        return 0.0, 0.0
    dir_from_rad = math.atan2(-u, -v)
    dir_from_deg = math.degrees(dir_from_rad) % 360.0
    return speed, dir_from_deg


@dataclass
class Membrane:
    """Gesta, regularna siatka pol pogodowych - "membrana" z propozycji
    uzytkownika. Wszystkie pola 2D o ksztalcie (grid_n, grid_n).
    lon_grid/lat_grid to wspolrzedne kazdej komorki (z np.meshgrid),
    reszta to interpolowane wartosci fizyczne w tej komorce."""
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    temperature_c: np.ndarray
    pressure_hpa: np.ndarray
    humidity_pct: np.ndarray
    precip_mm: np.ndarray
    u_wind_kmh: np.ndarray
    v_wind_kmh: np.ndarray
    dx_deg: float  # rozstaw siatki w stopniach dlugosci geogr. (do gradientow)
    dy_deg: float  # rozstaw siatki w stopniach szerokosci geogr.


def _interp_field(lons: np.ndarray, lats: np.ndarray, values: np.ndarray,
                   grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    """Interpoluje jedno pole skalarne metoda 'cubic' (gladka membrana -
    uzasadnione, bo pola meteorologiczne sa z natury gladkie poza
    frontami, ktore i tak wykrywamy osobno w defects.py po module
    gradientu, nie po samej interpolacji), z fallbackiem na 'linear' gdy
    'cubic' zwroci same NaN (moze sie zdarzyc przy zdegenerowanym
    ukladzie punktow wejsciowych, np. wszystkie w jednej linii), i
    ostatecznym fallbackiem na 'nearest' dla komorek WCIAZ NaN po obu
    (typowo tylko skrajne rogi membrany poza wypukla otoczka punktow
    wejsciowych - ekstrapolacja, ktorej 'cubic'/'linear' celowo nie robia)."""
    pts = np.column_stack([lons, lats])
    result = griddata(pts, values, (grid_lon, grid_lat), method="cubic")
    if np.isnan(result).all():
        result = griddata(pts, values, (grid_lon, grid_lat), method="linear")
    if np.isnan(result).any():
        nearest = griddata(pts, values, (grid_lon, grid_lat), method="nearest")
        result = np.where(np.isnan(result), nearest, result)
    return result


def build_membrane(records: list[dict], grid_n: int = 41) -> Membrane:
    """Zamienia liste rekordow z grid_source.fetch_grid_snapshot() (rzadka
    siatka, np. 5x5=25 punktow) w gesta membrane grid_n x grid_n
    (domyslnie 41x41 ~= 1681 komorek - gesto wystarczajaco, zeby FFT/
    gradient w spectrum.py mialy sensowna rozdzielczosc czestotliwosciowa/
    przestrzenna).

    Rekordy z brakujacymi wartosciami (None - patrz docstring
    grid_source.fetch_grid_snapshot) sa POMIJANE per pole NIEZALEZNIE
    (jeden punkt moze miec brakujace cisnienie, ale dobra temperature) -
    nie odrzuca calego punktu przez jedno brakujace pole.

    Rzuca ValueError jesli dla ktoregos pola zostaje < 4 uzytecznych
    punktow (scipy.griddata 'cubic' wymaga min. triangulacji Delaunaya w
    2D, faktycznie potrzebne >=4 nie-wspollinowe punkty) - jawnie, zamiast
    cicho zwracac membrane pelna NaN/zer."""
    if len(records) < 4:
        raise ValueError(f"Potrzeba >= 4 punktow wejsciowych do interpolacji 2D, dostano {len(records)}")

    lons_all = np.array([r["lon"] for r in records], dtype=float)
    lats_all = np.array([r["lat"] for r in records], dtype=float)

    lon_min, lon_max = lons_all.min(), lons_all.max()
    lat_min, lat_max = lats_all.min(), lats_all.max()
    grid_lon_1d = np.linspace(lon_min, lon_max, grid_n)
    grid_lat_1d = np.linspace(lat_min, lat_max, grid_n)
    grid_lon, grid_lat = np.meshgrid(grid_lon_1d, grid_lat_1d)
    dx_deg = grid_lon_1d[1] - grid_lon_1d[0]
    dy_deg = grid_lat_1d[1] - grid_lat_1d[0]

    def _field(key: str) -> np.ndarray:
        mask = np.array([r.get(key) is not None for r in records])
        n_valid = int(mask.sum())
        if n_valid < 4:
            raise ValueError(f"Pole {key!r} ma tylko {n_valid} nie-brakujacych punktow (< 4), nie mozna interpolowac 2D")
        vals = np.array([r[key] for r in records if r.get(key) is not None], dtype=float)
        return _interp_field(lons_all[mask], lats_all[mask], vals, grid_lon, grid_lat)

    temperature_c = _field("temperature_c")
    pressure_hpa = _field("pressure_hpa")
    humidity_pct = _field("humidity_pct")
    precip_mm = _field("precip_mm")

    # Wiatr: rozloz na u,v PRZED interpolacja (patrz docstring modulu),
    # pomijajac punkty z brakujaca predkoscia LUB kierunkiem (potrzebne
    # oba, zeby policzyc skladowa).
    uv_mask = np.array([
        r.get("wind_speed_kmh") is not None and r.get("wind_dir_deg") is not None
        for r in records
    ])
    n_uv = int(uv_mask.sum())
    if n_uv < 4:
        raise ValueError(f"Wiatr ma tylko {n_uv} kompletnych punktow (predkosc+kierunek, < 4), nie mozna interpolowac 2D")
    u_vals, v_vals = [], []
    for r in records:
        if r.get("wind_speed_kmh") is not None and r.get("wind_dir_deg") is not None:
            u, v = wind_to_uv(r["wind_speed_kmh"], r["wind_dir_deg"])
            u_vals.append(u)
            v_vals.append(v)
    u_wind_kmh = _interp_field(lons_all[uv_mask], lats_all[uv_mask], np.array(u_vals), grid_lon, grid_lat)
    v_wind_kmh = _interp_field(lons_all[uv_mask], lats_all[uv_mask], np.array(v_vals), grid_lon, grid_lat)

    return Membrane(
        lat_grid=grid_lat, lon_grid=grid_lon,
        temperature_c=temperature_c, pressure_hpa=pressure_hpa,
        humidity_pct=humidity_pct, precip_mm=precip_mm,
        u_wind_kmh=u_wind_kmh, v_wind_kmh=v_wind_kmh,
        dx_deg=float(dx_deg), dy_deg=float(dy_deg),
    )
