"""membrane/meta_adapter.py -- adapter membrane_state -> MetaState
(integracja z TIMDR-META-DYNAMICS), na wzor meta_dynamics_module.py w
analizator-gieldowy-v3 (ten sam wzorzec: folder-siostra na sys.path,
mapowanie realnych, juz policzonych sygnalow tego repo na Lambda-tau-rho-J,
NIE nowe wymyslanie znaczenia tych liter).

KONTEKST / PO CO TO POWSTALO: TIMDR-META-DYNAMICS ma gotowy formalizm
S_meta(t)={Lambda,tau,rho,J}, M=d/dt(...), classify_phase(M) (progi 0.1/1.0,
JAWNIE oznaczone w kodzie jako arbitralne i zalezne od skali wejscia -
patrz core_meta/meta_operator_M.py::classify_phase docstring). Do tej pory
jedyna realna integracja to analizator-gieldowy-v3 (dane finansowe) -
zero polaczenia z fizyka/pogoda mimo deklarowanego w README
TIMDR-META-DYNAMICS zamiaru "docelowo wspolne definicje Lambda-tau-rho-J".
Ten modul to KONKRETNY, ograniczony krok w tamtym kierunku: nie "dowod ze
Lambda-tau-rho-J to uniwersalny opis pol fizycznych" (to by wymagalo
dowodu, ze M odpowiada prawu fizycznemu - patrz zastrzezenia w skillu
timdr-signal-framework), tylko sprawdzenie, czy podlaczenie REALNYCH,
juz istniejacych, jednostkowych pol membrany Synoptyk-v3 do tego
formalizmu daje cokolwiek sensownego, gdy na wejsciu jest PRAWDZIWY
przebieg synoptyczny (nie synteta).

===========================================================================
PRE-REJESTRACJA MAPOWANIA (zamrozone PRZED policzeniem jakiegokolwiek
realnego M-series lub classify_phase() na prawdziwych danych - patrz
protokol numerologii/formalizmu w skillu timdr-signal-framework, zasada 1:
"zdefiniuj dokladny obiekt/wzorzec PRZED dotknieciem realnych danych lub
zobaczeniem wyniku"):

    Lambda (struktura)      = srednia z high_freq_fraction widma
                              temperatury i cisnienia (obie w [0,1],
                              WSPOLNA jednostka -> usrednienie sensowne
                              bez normalizacji). "Ile z energii pola jest
                              w drobnej skali" = miara STRUKTURY pola w
                              sensie dosc doslownym (rozdzielczosc
                              przestrzenna zjawiska).

    tau (transformacja)     = srednia modulu gradientu temperatury po
                              calej membranie [C/stopien geogr.]. Modul
                              gradientu = doslownie tempo zmiany pola w
                              przestrzeni - najblizszy przestrzenny
                              odpowiednik "tempa transformacji". Wybrano
                              SAM gradient temperatury (nie usrednienie
                              z cisnieniem) celowo - inna jednostki
                              fizyczne (C/stopien vs hPa/stopien) nie da
                              sie usrednic bez arbitralnej normalizacji,
                              a to bylby dokladnie rodzaj ukrytej decyzji,
                              ktora protokol nakazuje jawnie nazwac, nie
                              podjac milczaco.

    rho (anomalia)          = calkowita liczba komorek-defektow zsumowana
                              po WSZYSTKICH 4 kanalach diagnostycznych
                              (gradient_T + gradient_P + wirowosc + opad).
                              Bezposrednia analogia do rho jako "anomalia"
                              w GIA-TIMDR/SYNOPTYK-ARCTIC (M/S) - tam rho
                              tez jest licznikiem, nie wartoscia ciagla.

    J (operator punktowy)   = liczba komorek rezonansowych
                              (coincidence_count >= k, k=3, ta sama
                              wartosc domyslna co RESONANCE_K w analyze.py
                              i K w rezonansie sygnalowym M w GIA-TIMDR).
                              "Punktowy" bo to konkretne, zlokalizowane
                              komorki koincydencji, nie usrednienie.

UCZCIWE ZASTRZEZENIA (musza zostac, nie do "posprzatania" w przyszlej
sesji):
  1. To jest JEDNO z mozliwych mapowan, nie jedyne poprawne - dokladnie
     jak przy mapowaniu financial w analizator-gieldowy-v3 (ktore samo
     zmienilo sie raz, z Lambda=cena na Lambda=trm, gdy pojawily sie
     lepsze realne sygnaly). Gdyby ktos policzyl Lambda/tau/rho/J inaczej
     z tych samych pol membrany, dostalby inne M i inna klasyfikacje faz.
  2. Progi classify_phase() (0.1/1.0) sa PRZENIESIONE bez zmian z
     TIMDR-META-DYNAMICS - NIE skalibrowane na rozkladzie |M| dla tego
     konkretnego mapowania. Wynik "krytyczna"/"przejsciowa"/"stabilna"
     ponizej NIE jest zwalidowana klasyfikacja - jest to demonstracja, ze
     kod dziala end-to-end na prawdziwych danych, analogicznie do
     pojedynczego strzalu z NIEKALIBROWANEJ broni (patrz rozmowa o
     numerologii w tej samej sesji) - pokazuje, ze mechanizm dziala, NIE
     ze jest celny/skalibrowany. Kalibracja wymagalaby wielu niezaleznych
     realnych epizodow frontowych + kontrolki negatywnej (okres bez
     frontu) + testu Manna-Whitneya - dokladnie to, czego ten modul NIE
     robi (patrz plan w README.md#Integracja-z-TIMDR-META-DYNAMICS).
  3. `tau` ma jednostke C/stopien geogr., `Lambda` jest bezwymiarowe
     [0,1], `rho`/`J` to liczby calkowite komorek - MetaOperatorM.magnitude()
     sumuje |Lambda|+|tau|+|rho|+|J| bez normalizacji miedzy nimi (dokladnie
     to samo ostrzezenie co w oryginalnym docstringu classify_phase() -
     "skala Lambda/tau/rho/J zalezy calkowicie od tego, co podlaczysz jako
     wejscie"). rho/J (liczby komorek membrany 41x41=1681) DOMINUJA sume
     nad Lambda (<=1) i tau (typowo <1 C/stopien) - to jest fizyczna
     konsekwencja wyboru "surowa liczba komorek" dla rho/J, NIE blad -
     ale oznacza, ze |M| jest w praktyce napedzane niemal wylacznie przez
     ZMIANE liczby komorek-defektow/rezonansu dzien-do-dnia, nie przez
     Lambda/tau. Nazwane tu jawnie, zeby nikt w przyszlosci nie zdziwil
     sie, czemu klasyfikacja faz "ignoruje" Lambda/tau.
===========================================================================
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from membrane.analyze import AnalyzeResult, analyze_records


def _ensure_timdr_meta_dynamics_on_path() -> None:
    """Dodaje folder-siostre TIMDR-META-DYNAMICS do sys.path, jesli
    jeszcze go tam nie ma - identyczny wzorzec co
    analizator-gieldowy-v3/meta_dynamics_module.py (ten sam ukland
    Downloads\\a\\<repo> jako rodzenstwo katalogow)."""
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.join(here, "..", "..", "TIMDR-META-DYNAMICS")
    sibling = os.path.abspath(sibling)

    if not os.path.isdir(sibling):
        raise ImportError(
            "meta_adapter wymaga folderu 'TIMDR-META-DYNAMICS' jako "
            f"siostry repo Synoptyk-v3 (szukano w: {sibling}). Jesli lezy "
            "gdzie indziej, popraw sciezke w _ensure_timdr_meta_dynamics_on_path()."
        )
    if sibling not in sys.path:
        sys.path.insert(0, sibling)


_ensure_timdr_meta_dynamics_on_path()

from timdr_meta_dynamics import MetaState, MetaOperatorM  # noqa: E402  (import po sys.path.insert - celowo)
from analysis.meta_map import MetaMap  # noqa: E402
from analysis.meta_trigger import MetaTrigger, MetaTriggerResult  # noqa: E402


RESONANCE_K = 3  # ta sama wartosc co analyze.RESONANCE_K - patrz mapowanie J powyzej


@dataclass
class MetaAdapterResult:
    dates: List[str]                       # etykiety dni/snapshotow, dlugosc n
    states: List[MetaState]                # S_meta(t) per dzien, dlugosc n
    analyze_results: List[AnalyzeResult]    # pelny wynik analyze_records per dzien (do inspekcji)
    M_series: List[MetaState]              # M(t) = dS/dt, dlugosc n-1
    phases: List[str]                      # "stabilna"/"przejsciowa"/"krytyczna" per krok M, dlugosc n-1
    trigger: MetaTriggerResult             # pierwszy krok, w ktorym osiagnieto najpowazniejsza faze (jesli w ogole)


def membrane_result_to_meta_state(result: AnalyzeResult) -> MetaState:
    """Mapowanie jednego wyniku analyze_records() -> jeden MetaState.
    Wzory zamrozone w PRE-REJESTRACJI na gorze pliku - NIE zmieniaj tu
    bez dopisania nowej wersji zastrzezenia."""
    Lambda = 0.5 * (
        result.temperature_spectrum.high_freq_fraction
        + result.pressure_spectrum.high_freq_fraction
    )
    tau = float(result.gradient_t.mean())
    rho = float(
        result.defects_t.n_defects
        + result.defects_p.n_defects
        + result.defects_vort.n_defects
        + result.defects_precip.n_defects
    )
    J = float(result.resonance.n_resonance_cells)
    return MetaState(Lambda=Lambda, tau=tau, rho=rho, J=J)


def build_meta_series_from_daily_records(
    daily_records: List[List[dict]],
    dates: Optional[List[str]] = None,
    grid_n_membrane: int = 41,
    dt: float = 1.0,
) -> MetaAdapterResult:
    """Pelny pipeline: lista dni (kazdy dzien = lista rekordow siatki,
    format identyczny jak grid_source.fetch_grid_snapshot()/
    analyze.analyze_records()) -> MetaState per dzien -> M-seria ->
    fazy -> trigger.

    `dt=1.0` = jeden krok miedzy kolejnymi elementami `daily_records"
    reprezentuje "jeden dzien" TYLKO gdy dni sa faktycznie kolejne
    kalendarzowo bez przerw (sprawdzane przez wywolujacego - ten modul
    NIE parsuje dat, traktuje kolejnosc listy jako kolejnosc czasowa,
    zgodnie z tym, jak FieldEvolution.simulate() w TIMDR-META-DYNAMICS
    juz dziala - rowne odstepy czasu miedzy kolejnymi elementami serii).

    Rzuca w gore kazdy blad z analyze_records() (np. za malo punktow) -
    ten sam wzorzec "brak cichych pol-wynikow" co reszta repo."""
    if len(daily_records) < 2:
        raise ValueError(
            f"Potrzeba >= 2 dni do policzenia choc jednego M = dS/dt, dostano {len(daily_records)}"
        )
    if dates is not None and len(dates) != len(daily_records):
        raise ValueError(
            f"len(dates)={len(dates)} != len(daily_records)={len(daily_records)}"
        )
    if dates is None:
        dates = [str(i) for i in range(len(daily_records))]

    analyze_results = [
        analyze_records(records, grid_n_membrane=grid_n_membrane)
        for records in daily_records
    ]
    states = [membrane_result_to_meta_state(r) for r in analyze_results]

    meta_operator = MetaOperatorM()
    M_series: List[MetaState] = []
    for i in range(len(states) - 1):
        M_series.append(meta_operator.compute(states[i], states[i + 1], dt))

    meta_map = MetaMap(meta_operator)
    phases = meta_map.detect_transitions(M_series)

    trigger = MetaTrigger().analyze(phases)

    return MetaAdapterResult(
        dates=dates, states=states, analyze_results=analyze_results,
        M_series=M_series, phases=phases, trigger=trigger,
    )


def archive_grid_response_to_daily_records(raw_points: List[dict]) -> List[List[dict]]:
    """Konwertuje surowa odpowiedz Archive API DLA SIATKI punktow (lista
    obiektow, jeden na punkt, kazdy z kluczem "daily" zawierajacym listy
    rownej dlugosci - dokladnie ksztalt zwracany przez
    archive-api.open-meteo.com dla wielu wspolrzednych, patrz
    tests/test_meta_adapter.py po realny przechwycony przyklad) na liste
    "dni", z ktorych kazdy jest lista rekordow w formacie wymaganym przez
    analyze_records()/build_membrane() (lat/lon/temperature_c/
    pressure_hpa/humidity_pct/precip_mm/wind_speed_kmh/wind_dir_deg).

    Wybor pol (udokumentowany, nie domyslny bez uzasadnienia):
      temperature_c  <- temperature_2m_max (ten sam wybor co
                         archive_source.py::fetch_archive dla trafnosci
                         prognozy - spojnosc w calym repo)
      pressure_hpa   <- surface_pressure_mean
      humidity_pct   <- relative_humidity_2m_mean
      precip_mm      <- precipitation_sum
      wind_speed_kmh <- wind_speed_10m_max
      wind_dir_deg   <- wind_direction_10m_dominant

    Rzuca ValueError, jesli liczba dni rozni sie miedzy punktami (nie
    powinno sie zdarzyc dla jednego zapytania Archive API o wspolny
    zakres dat, ale sprawdzone jawnie zamiast cicho ucinac do najkrotszej)."""
    if not raw_points:
        return []
    n_days_per_point = {len(p["daily"]["time"]) for p in raw_points}
    if len(n_days_per_point) > 1:
        raise ValueError(f"Niezgodna liczba dni miedzy punktami siatki: {n_days_per_point}")
    n_days = n_days_per_point.pop()

    daily_records: List[List[dict]] = []
    for day_idx in range(n_days):
        records = []
        for p in raw_points:
            d = p["daily"]
            records.append({
                "lat": p["latitude"], "lon": p["longitude"],
                "temperature_c": d["temperature_2m_max"][day_idx],
                "pressure_hpa": d["surface_pressure_mean"][day_idx],
                "humidity_pct": d["relative_humidity_2m_mean"][day_idx],
                "precip_mm": d["precipitation_sum"][day_idx],
                "wind_speed_kmh": d["wind_speed_10m_max"][day_idx],
                "wind_dir_deg": d["wind_direction_10m_dominant"][day_idx],
            })
        daily_records.append(records)
    return daily_records
