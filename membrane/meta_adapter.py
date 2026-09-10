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
WERSJA 1 (2026-09-08, PORZUCONA - zachowana tu dla historii, nie jako
zywy kod): Lambda=srednia high_freq_fraction T+P [0,1], tau=SUREY sredni
gradient T [C/stopien], rho=SUROWA suma liczby komorek-defektow (4
kanaly), J=SUROWA liczba komorek rezonansowych. Zweryfikowana end-to-end
na syntetycznych kontrolach (obie przeszly) I na 10 dniach prawdziwych
danych (siatka 3x3 wokol Warszawy, patrz REAL_GRID_RESPONSE w
tests/test_meta_adapter.py) - REALNY WYNIK: wszystkie 9 krokow, dni
spokojne I dzien realnego frontu bez wyjatku, wyszly jako "krytyczna"
(|M| od ~7 do ~143, prog krytyczny=1.0). Przyczyna: rho/J to SUROWE
liczby komorek membrany 41x41=1681 - nawet maly dzien-do-dnia ruch
(dziesiatki komorek) przebija prog 1.0 o dwa rzedy wielkosci, Lambda/tau
(jedyne dwie skladowe w sensownej, malej skali) gina w sumie
|Lambda|+|tau|+|rho|+|J|. Klasyfikacja faz przy V1 miala ZERO mocy
dyskryminujacej (nie rozrozniala dni spokojnych od frontu) - dokladnie
zastrzezenie #3 ponizej, potwierdzone na realnym przykladzie.

WERSJA 2 (2026-09-08, AKTUALNA) - NAPRAWIONO PRZESKALOWANIE, NIE PROGI:
jedyna zmiana wzgledem V1 to sprowadzenie WSZYSTKICH czterech skladowych
do porownywalnej, w wiekszosci przypadkow ograniczonej do [0,1] skali,
zanim wejda do M=d/dt(...) i classify_phase(). Normalizacja jest
zdefiniowana WYLACZNIE przez wielkosci WEWNETRZNE dla kazdego
pojedynczego dnia (calkowita liczba komorek membrany, prog anomalii
danego dnia z defects.py) - NIE przez zadna wartosc podejrzana/dobrana
po zobaczeniu WYNIKU M-serii czy classify_phase() (ktorych V2 jeszcze
nie liczyl w momencie wyboru tych wzorow - to jest pre-rejestracja W
SENSIE PROTOKOLU, nie tylko nazwa). Progi classify_phase() (0.1/1.0)
ZOSTAWIONE BEZ ZMIAN - to nie jest "dostrajanie progu, zeby wyszlo ladnie",
tylko naprawienie WEJSCIA do formuly, ktora te progi zaklada byc w skali
rzedu jednosci (dokladnie tak, jak formalizm TIMDR-META-DYNAMICS byl
uzywany w analizator-gieldowy-v3, gdzie trm/flow/resonance/wolumen sa
tez rzedu jednosci-dziesiatek, nie tysiecy):

    Lambda (struktura)      = srednia z high_freq_fraction widma
                              temperatury i cisnienia. Bez zmian wzgledem
                              V1 - juz bylo bezwymiarowe [0,1].

    tau (transformacja)     = sredni modul gradientu temperatury PODZIELONY
                              przez prog anomalii tego samego dnia
                              (defects_t.threshold, mediana+k*MAD z tego
                              samego pola gradientu - patrz defects.py).
                              Bezwymiarowe: "ile progow-anomalii-tego-dnia
                              stanowi przecietny gradient membrany". Prog
                              jest wielkoscia WEWNETRZNA dla KAZDEGO dnia
                              z osobna (liczony niezaleznie na kazdym
                              snapshocie) - nie parametr dobrany na
                              podstawie calej serii/wyniku koncowego.
                              Typowo << 1, bo prog=mediana+3.5*MAD z
                              definicji lezy w ogonie rozkladu gradientu,
                              wiec SREDNIA calego pola jest z reguly duzo
                              nizsza niz ten prog.

    rho (anomalia)          = calkowita liczba komorek-defektow (4 kanaly
                              zsumowane) PODZIELONA przez (4 * liczba_komorek)
                              - czyli SREDNIA fraction zajetosci defektem
                              na kanal, w [0,1]. Mianownik 4x liczba_komorek
                              (nie 1x), bo licznik sumuje po 4 NIEZALEZNYCH
                              kanalach (jedna komorka moze byc defektem w
                              wiecej niz jednym kanale rownoczesnie) -
                              podzielenie tylko przez liczba_komorek
                              mogloby dac wartosc > 1.

    J (operator punktowy)   = liczba komorek rezonansowych PODZIELONA przez
                              liczbe_komorek - fraction membrany w
                              koincydencji >=k=3 kanalow, w [0,1]
                              (rezonans to JEDNA maska, wiec ten mianownik
                              NIE potrzebuje czynnika 4).

UCZCIWE ZASTRZEZENIA (musza zostac, nie do "posprzatania" w przyszlej
sesji):
  1. To jest JEDNO z mozliwych mapowan (teraz: jedna z mozliwych
     NORMALIZACJI), nie jedyne poprawne - dokladnie jak przy mapowaniu
     financial w analizator-gieldowy-v3 (ktore samo zmienilo sie raz).
     Inny wybor normalizacji (np. Lambda/rho/J wszystkie /1, tau /2-sigma
     zamiast /prog) dalby inne liczbowo M i (byc moze) inna klasyfikacje.
  2. Progi classify_phase() (0.1/1.0) sa NADAL przeniesione bez zmian z
     TIMDR-META-DYNAMICS - teraz przynajmniej DZIALAJACE NA WLASCIWEJ
     SKALI WEJSCIA (rzad jednosci, nie tysiace), ale wciaz NIE
     skalibrowane na rozkladzie |M| dla TEGO KONKRETNEGO mapowania na
     wielu niezaleznych realnych epizodach. Jeden 10-dniowy real-data run
     (patrz README.md#Integracja-z-TIMDR-META-DYNAMICS) pokazuje, ze
     klasyfikacja przy V2 przynajmniej ROZROZNIA dni miedzy soba (rozne
     fazy w roznych krokach) - to NIE jest dowod, ze konkretne progi 0.1/
     1.0 sa dobrze dobrane dla tego zjawiska, tylko ze skala wejscia jest
     juz w rejonie, gdzie te progi moga cokolwiek rozroznic. Prawdziwa
     kalibracja wymagalaby wielu niezaleznych realnych epizodow frontowych
     + kontrolki negatywnej (okres bez frontu) + testu Manna-Whitneya -
     tego ten modul WCIAZ nie robi.
  3. `tau` po normalizacji jest w praktyce zazwyczaj << 1 (patrz
     uzasadnienie w definicji tau powyzej) - oznacza to, ze o ile V1 mial
     odwrotny problem (rho/J dominujace, Lambda/tau ginace), V2 moze miec
     tau SYSTEMATYCZNIE niedowazone wzgledem Lambda/rho/J (wszystkie trzy
     typowo rzedu 0.01-0.5). Nazwane tu jawnie, ten sam wzorzec co
     zastrzezenie #3 z V1 - nie "naprawione i zapomniane", tylko
     przeniesiona, mniejsza wersja tego samego zjawiska.
===========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from membrane.analyze import AnalyzeResult, analyze_records

# ZWENDOROWANE 2026-09-10 (patrz naglowek membrane/_vendor_timdr_meta_dynamics_core.py
# dla pelnego uzasadnienia): wczesniej ten modul ladowal TIMDR-META-DYNAMICS
# przez sys.path sibling-import z folderu-siostry na dysku. Zamienione na
# lokalna, zwendorowana kopie, zeby to repo dzialalo samodzielnie po
# sklonowaniu WYLACZNIE siebie (decyzja na wyrazna prosbe: "repozytoria
# kodu maja byc niezalezne od siebie"). Zachowanie/matematyka bez zmian.
from membrane._vendor_timdr_meta_dynamics_core import MetaState, MetaOperatorM, MetaMap, MetaTrigger, MetaTriggerResult


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
    Wzory V2 (patrz PRE-REJESTRACJA/WERSJA 2 na gorze pliku) - NIE
    zmieniaj tu bez dopisania nowej wersji zastrzezenia i wyjasnienia,
    dlaczego V2 nie wystarcza."""
    n_cells = float(result.grid_n_membrane) ** 2

    Lambda = 0.5 * (
        result.temperature_spectrum.high_freq_fraction
        + result.pressure_spectrum.high_freq_fraction
    )

    # tau: sredni gradient T, znormalizowany do progu anomalii TEGO SAMEGO
    # dnia (wielkosc wewnetrzna per-snapshot, nie dobrana z wyniku serii).
    threshold_t = result.defects_t.threshold
    tau = float(result.gradient_t.mean()) / threshold_t if threshold_t > 0 else 0.0

    n_defects_total = (
        result.defects_t.n_defects
        + result.defects_p.n_defects
        + result.defects_vort.n_defects
        + result.defects_precip.n_defects
    )
    rho = float(n_defects_total) / (4.0 * n_cells)

    J = float(result.resonance.n_resonance_cells) / n_cells

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
