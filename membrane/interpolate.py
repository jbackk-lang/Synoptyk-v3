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

BEZPIECZENSTWO IMPORTU SCIPY (2026-09-10, naprawa po realnym bledzie
uzytkownika): `scipy.interpolate.griddata` byl wczesniej JEDYNYM
sposobem 2D interpolacji w tym module, importowanym NA SZTYWNO na
poziomie modulu, co oznaczalo, ze CALY webapp/app.py (a wiec i
endpointy w ogole nieuzywajace interpolacji, np. /api/cities,
/api/meteogram, /api/bias) odmawial startu na maszynie, gdzie sam
IMPORT scipy jest zablokowany (Windows Device Guard - dokladnie ten
sam, juz wczesniej zdiagnozowany problem co w
TIMDR-Earthquake-Core/precursor_validation.py, patrz HISTORIA_I_TESTY.md
tamtego repo). Naprawione w dwoch krokach:

1. Import scipy owiniety w try/except (ten sam wzorzec co `_HAS_SCIPY`
   w TIMDR-Math-Formalism/timdr_formalism/pipeline.py) - caly modul
   (i wszystko co go importuje, w tym spectrum.py/analyze.py/
   webapp/app.py) da sie zaimportowac bez scipy.
2. CZYSTO-NUMPY FALLBACK (2026-09-10, druga runda naprawy - user
   poprosil o realny dzialajacy fallback, nie tylko czytelny blad):
   gdy scipy niedostepne, `_interp_field()` uzywa
   `_thin_plate_spline_interp()` zamiast rzucac RuntimeError - patrz
   docstring tej funkcji dla pelnego uzasadnienia matematycznego i
   UCZCIWEGO zastrzezenia, ze to INNY algorytm niz scipy 'cubic'
   (nie numeryczny odpowiednik, tylko rownowazny CEL: gladka
   interpolacja scattered data w 2D). `Membrane.interpolation_method`
   ("scipy_griddata_cubic" | "numpy_tps_fallback") jest jawnie
   wystawiane w kazdym wyniku (i w JSON API - patrz analyze.py), zeby
   nikt nie pomylil wyniku fallbacku z wynikiem scipy przy porownaniach
   pomiedzy maszynami/wdrozeniami - zgodnie z zasada tego ekosystemu
   "etykietuj metode, nie ukrywaj, ktora sciezka faktycznie policzyla
   wynik" (patrz skill timdr-signal-framework, dyscyplina
   post-hoc/etykietowania).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - dokladnie scenariusz Device Guard
    griddata = None
    _HAS_SCIPY = False


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
    interpolation_method: str = "scipy_griddata_cubic"  # albo "numpy_tps_fallback" - patrz UWAGA O IMPORCIE


def _thin_plate_spline_interp(lons: np.ndarray, lats: np.ndarray, values: np.ndarray,
                               grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    """Czysto-numpowy fallback interpolacji 2D scattered data, uzywany
    TYLKO gdy scipy nie jest dostepne (patrz UWAGA O IMPORCIE w naglowku
    modulu). Implementuje klasyczna interpolacje radialnymi funkcjami
    bazowymi typu "thin-plate spline" (TPS; Duchon 1977; standardowa,
    dobrze udokumentowana metoda interpolacji rozproszonych danych 2D,
    NIE nowa konstrukcja tego projektu) - wymaga tylko
    `np.linalg.solve`/`lstsq`, zero zaleznosci od scipy.

    Model: f(x,y) = a0 + a1*x + a2*y + sum_i w_i * phi(|P - P_i|),
    gdzie phi(r) = r^2*log(r) (jadro TPS, phi(0):=0 z ciaglosci), przy
    warunkach ubocznych sum(w)=0, sum(w*x)=0, sum(w*y)=0 (standardowy
    uklad TPS - patrz np. Bookstein 1989). Skladnik afiniczny (a0,a1,a2)
    gwarantuje, ze pola SCISLE liniowe sa odtwarzane dokladnie (do bledu
    numerycznego) - ta sama wlasciwosc sprawdzana dla scipy 'cubic' w
    test_build_membrane_reconstructs_linear_field, wiec oba backendy
    przechodza ten sam test kontrolny.

    UCZCIWE ZASTRZEZENIE: to NIE jest numeryczny odpowiednik
    scipy.griddata(method='cubic') - inny algorytm (globalna RBF vs.
    lokalna triangulacja Clough-Tocher). Dla gladkich pol meteorologicznych
    da PODOBNE, ale nie identyczne wyniki; w przeciwienstwie do
    griddata('cubic'/'linear') TPS ekstrapoluje analitycznie poza otoczke
    wypukla punktow wejsciowych zamiast dawac NaN (wiec nie potrzeba tu
    fallbacku 'nearest' na rogach) - moze to dawac inne (zwykle gladsze,
    ale przy silnie nieliniowych/szumnych danych czasem mniej stabilne)
    zachowanie na brzegach membrany. Dlatego kazdy Membrane niesie jawna
    etykiete `interpolation_method`, zeby wynik fallbacku nigdy nie byl
    mylony z wynikiem scipy przy porownaniach miedzy wdrozeniami."""
    n = len(values)
    px = lons.reshape(-1, 1)
    py = lats.reshape(-1, 1)
    dx = px - px.T
    dy = py - py.T
    r = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(r > 0, r ** 2 * np.log(r), 0.0)

    P = np.column_stack([np.ones(n), lons, lats])  # n x 3
    top = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    A = np.vstack([top, bottom])
    b = np.concatenate([values, np.zeros(3)])

    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Zdegenerowany uklad (np. punkty wspollinowe) - najmniejsze
        # kwadraty zamiast twardego rozwiazania, zeby nie wywalac
        # calego /api/analyze na trudnym ukladzie wejsciowym.
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    w = coeffs[:n]
    a0, a1, a2 = coeffs[n], coeffs[n + 1], coeffs[n + 2]

    gx = grid_lon.ravel().reshape(-1, 1)
    gy = grid_lat.ravel().reshape(-1, 1)
    gdx = gx - px.T
    gdy = gy - py.T
    gr = np.sqrt(gdx ** 2 + gdy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        gK = np.where(gr > 0, gr ** 2 * np.log(gr), 0.0)

    result = a0 + a1 * gx.ravel() + a2 * gy.ravel() + gK @ w
    return result.reshape(grid_lon.shape)


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
    wejsciowych - ekstrapolacja, ktorej 'cubic'/'linear' celowo nie robia).

    Jesli scipy nie jest dostepne (patrz UWAGA O IMPORCIE w naglowku
    modulu), uzywa `_thin_plate_spline_interp()` - czysto-numpowego
    fallbacku o INNEJ charakterystyce numerycznej, jawnie oznaczonego w
    `Membrane.interpolation_method` (patrz build_membrane)."""
    if not _HAS_SCIPY:
        return _thin_plate_spline_interp(lons, lats, values, grid_lon, grid_lat)
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
        interpolation_method="scipy_griddata_cubic" if _HAS_SCIPY else "numpy_tps_fallback",
    )
