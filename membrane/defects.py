"""
membrane/defects.py — Krok 4: defekty membrany (fronty) jako osobliwosci
pola - komorki, gdzie modul gradientu temperatury/cisnienia jest
nietypowo wysoki wzgledem reszty AKTUALNEJ membrany.

UWAGA O MASKOWANIU PROGU (Pattern B) - dlaczego mediana/MAD tutaj NIE
jest tym samym bledem, ktory zostal naprawiony w tej samej sesji w
FLIGHT-TRACKING-TIMDR/SYNOPTYK-ARCTIC/synoptyk-v2.0 (proz gorzej wprost
w resonance.py sesji: "jedna probka dostarcza zarowno test, jak i wlasny
prog odniesienia"):

  - TAM: SZEREG CZASOWY z ~10-30 probkami, prog liczony jako mean+2*std
    z CALEGO okna WLACZAJAC punkt testowany - jedna anomalia wchodzila do
    wlasnej sredniej/odchylenia z waga ~1/10 do ~1/30, co realnie
    podnosilo prog na tyle, zeby zamaskowac wykrycie (potwierdzone
    empirycznie - patrz TIMDR_Trefoil_RealDataValidation.md i naprawy w
    5 repo). Naprawiono metoda leave-one-out (prog per-punkt liczony BEZ
    tego punktu).

  - TU: POLE PRZESTRZENNE (jedna klatka czasowa) o grid_n=41 -> 1681
    komorek. Mediana/MAD z CALEGO pola WLACZAJAC testowana komorke ma
    ta komorke z waga ~1/1681, nie ~1/10 - o dwa rzedy wielkosci mniejszy
    wplyw pojedynczej komorki na wlasny prog odniesienia niz w przypadku
    szeregu czasowego. Dodatkowo mediana (w odroznieniu od sredniej) ma
    punkt zalamania 50% - pojedyncza ekstremalna komorka NIE MOZE
    przesunac mediany w ogole, dopoki nie stanowi >~50% wszystkich
    komorek (fizycznie niemozliwe dla lokalnego frontu). Z tych dwoch
    powodow LOO nie jest tu potrzebne - ale gdyby ktos w przyszlosci
    zmniejszyl grid_n do bardzo malej siatki (np. 5x5=25 komorek, taki
    sam rzad wielkosci jak problematyczne szeregi czasowe), warto
    ponownie rozwazyc LOO per-komorke. Udokumentowane jawnie tutaj,
    zeby przyszla zmiana grid_n nie przeoczyla tego zalozenia po cichu.

Rezonans-jako-koincydencja (resonance.py) uzywa WYNIKU tej funkcji jako
jednego z kanalow diagnostycznych - patrz tamten modul.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_TO_STD = 1.4826  # wspolczynnik konwersji MAD -> odpowiednik std dla rozkladu normalnego


@dataclass
class DefectResult:
    magnitude: np.ndarray     # modul gradientu (lub inna diagnostyka wejsciowa)
    median: float
    mad_std: float             # MAD przeskalowany do jednostek "jakby std"
    threshold: float
    is_defect: np.ndarray      # maska bool, ksztalt jak magnitude
    n_defects: int


def robust_threshold(values: np.ndarray, k: float = 3.5) -> tuple[float, float, float]:
    """Mediana + k * MAD(przeskalowany) - patrz docstring modulu po
    uzasadnienie, dlaczego mediana/MAD (nie mean/std) i dlaczem liczone
    na calym polu WLACZAJAC testowana komorke jest tu bezpieczne."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * MAD_TO_STD
    return med, mad, med + k * mad


def detect_defects(magnitude: np.ndarray, k: float = 3.5, noise_floor: float = 1e-9) -> DefectResult:
    """Flaguje komorki, gdzie `magnitude` (typowo wyjscie
    spectrum.gradient_magnitude lub abs(spectrum.vorticity)) przekracza
    mediana + k*MAD. k=3.5 dobrane konserwatywnie (odpowiednik ~3.5 sigma
    dla rozkladu normalnego, P~=0.02% szansy przypadkowej flagi na
    komorke pod hipoteza normalnosci - NIE zwalidowane na realnych
    danych, patrz README.md#ograniczenia) - mniejsze k dawaloby wiecej
    (falszywie-pozytywnych) frontow.

    Dwa poprawki znalezione podczas pisania testow (patrz
    tests/test_defects.py i tests/test_analyze.py), obie wynikaja z tej
    samej matematycznej wlasciwosci MAD - punkt zalamania 50% oznacza, ze
    MAD jest DOKLADNIE zero, gdy >=50% wartosci jest scisle rownych
    medianie, NIEZALEZNIE od tego, jak ekstremalna jest reszta:

    1. `noise_floor` (domyslnie 1e-9, dużo mniej niz jakikolwiek fizycznie
       sensowny gradient temperatury [C/stopien] czy cisnienia
       [hPa/stopien], ale duzo wiecej niz szum zaokraglen float64 dla
       wartosci rzedu dziesiatek/setek) - wartosci ponizej progu sa
       przycinane do 0.0 PRZED liczeniem mediany/MAD. Bez tego: pole
       matematycznie stale (np. interpolowane z identycznych punktow
       wejsciowych) generuje niezerowy, ale bezsensowny "gradient" rzedu
       1e-13 przez blad zaokraglenia w cubic-spline + roznicowaniu
       centralnym - MAD na takim "szumie" bywa niezerowy, wiec proste
       `magnitude > threshold` fałszywie flagowalo kilka% komorek jako
       "fronty" na polu, ktore fizycznie jest idealnie plaskie.
    2. Gdy PO przycieciu `mad_std` nadal wynosi 0 (czyli >=50% komorek ma
       IDENTYCZNA wartosc - prawdziwie plaskie tlo, nie artefakt
       zaokraglenia), prog wzgledny (mediana+k*MAD) jest niezdefiniowany
       (MAD=0 nie mowi nic o tym, jak duze odchylenie jest "anomalne").
       Zamiast (poprzedni blad) zwracac "brak defektow" bezwarunkowo -
       co maskowalo NAWET JEDNOZNACZNY pojedynczy outlier na tle
       identycznych zer, bo MAD nie moze zmierzyc rozrzutu tla, ktore
       fizycznie ma zerowy rozrzut - flagujemy KAZDA komorke SCISLE
       rozna od mediany. Uzasadnienie: przy udowodnionym (MAD=0) zerowym
       rozrzucie tla, KAZDE odchylenie jest z definicji anomalia, wiec
       nie trzeba/nie mozna zbudowac progu wzglednego - sam fakt roznicy
       wystarcza."""
    magnitude = np.where(np.abs(magnitude) < noise_floor, 0.0, magnitude)
    med, mad_std, threshold = robust_threshold(magnitude, k=k)
    if mad_std == 0.0:
        is_defect = magnitude != med
    else:
        is_defect = magnitude > threshold
    return DefectResult(magnitude=magnitude, median=med, mad_std=mad_std,
                         threshold=threshold, is_defect=is_defect,
                         n_defects=int(is_defect.sum()))
