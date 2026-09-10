"""
membrane/spectrum.py — Krok 3: widmo deformacji membrany (FFT), gradient
(uskoki/fronty), rotacja (wirowosc pola wiatru), i spojnosc kierunkowa
pola wiatru (zespolony parametr porzadku).

Cztery niezalezne diagnostyki, kazda inny obiekt matematyczny (ten sam
rygor rozroznienia "to samo slowo/podobienstwo, inny obiekt", co reszta
ekosystemu TIMDR - patrz skill timdr-signal-framework, ktory jednak
wprost NIE obejmuje tego repo, bo to case-study domenowy synoptyki, nie
teoria GIA-TIMDR):

1. widmo 2D FFT pola skalarnego -> energia po promieniowych pasmach
   liczby falowej (nisko = duze struktury/wyz-niz, wysoko = lokalne
   uskoki/konwekcja) - to jest DOSLOWNIE ta sama transformata, ktorej
   uzywaja modele spektralne (np. IFS ECMWF rozklada pola na harmoniki
   sferyczne globalnie; tu uzyto plaskiego FFT na malym, lokalnym
   wycinku siatki - uproszczenie geometryczne, nie inny obiekt
   matematyczny).
2. gradient (numpy.gradient, rozniczkowanie centralne) -> modul gradientu
   pokazuje uskoki pola (fronty) - to jest dokladnie mechanizm
   frontogenezy Petterssena (d/dt|grad(theta)|), tu bez skladowej
   czasowej (jedna klatka), czyli sama diagnoza |grad(pole)|.
3. wirowosc (curl) pola wektorowego wiatru: zeta = dv/dx - du/dy - to
   jest STANDARDOWA wirowosc wzgledna z dynamiki atmosfery (skladowa
   pionowa rotora wiatru), nie nowa konstrukcja. WRAZLIWA na gradient
   PREDKOSCI wiatru (nie tylko kierunku) - silny front predkosci przy
   stalym kierunku juz daje duza wirowosc (patrz test ponizej, dane
   realne).
4. spojnosc kierunkowa pola wiatru (`wind_direction_coherence`) -
   ZESPOLONY parametr porzadku (dokladnie ten sam obiekt matematyczny co
   parametr porzadku Kuramoto w TIMDR-Quantum-Lattice:
   Z = (1/N)*sum(exp(i*kierunek)) po calej membranie), CELOWO odrebny od
   wirowosci: ignoruje predkosc, mierzy WYLACZNIE, jak bardzo kierunek
   wiatru jest zgodny w calym obszarze. |Z| w [0,1]: 0 = kierunki
   losowe/przeciwstawne, 1 = idealnie zgodny kierunek. Dodane
   2026-09-10, zweryfikowane na realnych danych Open-Meteo (siatka wokol
   Gdanska, silny gradient PREDKOSCI 14-52 km/h przy prawie stalym
   KIERUNKU 260-282 stopni) - w tym przypadku |Z|=0.996 (bardzo spojny
   kierunek), a wirowosc jednoczesnie duza (do ~107, bo napedzana
   gradientem predkosci) - dwie diagnostyki NAPRAWDE mierza rozne rzeczy
   na tych samych danych, nie sa zdublowane. Kontrola negatywna: losowe
   kierunki (n=1000, ta sama skala predkosci) dajа |Z|~0.03-0.05, zgodne
   z oczekiwaniem statystycznym ~1/sqrt(n) dla braku spojnosci.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindCoherenceResult:
    coherence: float       # |Z| w [0,1] - 0=kierunki losowe, 1=idealnie zgodny kierunek
    mean_direction_deg: float  # kierunek SKAD wieje, stopnie [0,360) - TA SAMA konwencja
                                # co wind_dir_deg/wind_to_uv w interpolate.py (nie "matematyczny"
                                # kat liczby zespolonej wprost - patrz wind_direction_coherence())
    n_valid: int            # liczba komorek z niezerowa predkoscia (wykorzystanych w Z)


@dataclass
class SpectrumResult:
    freqs: np.ndarray          # promieniowe liczby falowe (cykle / stopien)
    power: np.ndarray          # energia widmowa w kazdym pasmie
    low_freq_energy: float     # suma energii w dolnej polowie pasm
    high_freq_energy: float    # suma energii w gornej polowie pasm
    high_freq_fraction: float  # high / (low+high), w [0,1]


def radial_power_spectrum(field: np.ndarray, dx: float, n_bins: int = 16) -> SpectrumResult:
    """2D FFT pola (odjeta srednia, zeby skladowa DC nie zdominowala
    widma), potem binowanie promieniowe |k| -> jedna wartosc energii per
    pasmo (usredniona po kierunku - membrana nie ma wyroznionego kierunku
    "wschod-zachod" vs "polnoc-poludnie" w tej analizie, wiec redukcja do
    1D widma promieniowego upraszcza odczyt bez utraty najwazniejszej
    informacji: JAK BARDZO drobnoskalowe jest pole, nie W KTORA STRONE).

    `dx` to rozstaw siatki (stopnie geogr.) - potrzebny do prawidlowego
    skalowania osi czestotliwosci (np.fft.fftfreq(n, d=dx))."""
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"field musi byc kwadratowa macierza 2D, dostano ksztalt {field.shape}")
    n = field.shape[0]
    centered = field - np.nanmean(field)
    centered = np.nan_to_num(centered, nan=0.0)
    fft2 = np.fft.fftshift(np.fft.fft2(centered))
    power2d = np.abs(fft2) ** 2

    kfreq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    kx, ky = np.meshgrid(kfreq, kfreq)
    kr = np.sqrt(kx ** 2 + ky ** 2)

    k_max = kr.max()
    bin_edges = np.linspace(0, k_max, n_bins + 1)
    freqs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (kr >= bin_edges[i]) & (kr < bin_edges[i + 1])
        power[i] = power2d[mask].sum() if mask.any() else 0.0

    half = n_bins // 2
    low_energy = float(power[:half].sum())
    high_energy = float(power[half:].sum())
    total = low_energy + high_energy
    high_fraction = high_energy / total if total > 0 else 0.0

    return SpectrumResult(freqs=freqs, power=power, low_freq_energy=low_energy,
                           high_freq_energy=high_energy, high_freq_fraction=high_fraction)


def gradient_magnitude(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Modul gradientu |grad(field)| liczony roznicowaniem centralnym
    (np.gradient - jedno dopasowanie roznicowe, NIE potrojne
    roznicowanie/wygladzanie -> roznicowanie, ktore w tym samym
    ekosystemie okazalo sie wzmacniac szum - Wzorzec A z innych repo TIMDR,
    patrz skill timdr-signal-framework; tu unikniete od razu, bo liczymy
    TYLKO pierwsza pochodna, nie lancuch v->a->jerk)."""
    dfield_dy, dfield_dx = np.gradient(field, dy, dx)
    return np.sqrt(dfield_dx ** 2 + dfield_dy ** 2)


def vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Wirowosc wzgledna (skladowa pionowa rotora wiatru):
    zeta = dv/dx - du/dy."""
    _, dv_dx = np.gradient(v, dy, dx)
    du_dy, _ = np.gradient(u, dy, dx)
    return dv_dx - du_dy


def wind_direction_coherence(u: np.ndarray, v: np.ndarray) -> WindCoherenceResult:
    """Zespolony parametr porzadku kierunku wiatru na calej membranie -
    patrz naglowek modulu, punkt 4, dla pelnego uzasadnienia i wyniku
    na realnych danych.

    Kazda komorka z niezerowa predkoscia daje jednostkowy wektor
    kierunku (u,v)/|u,v| = exp(i*kierunek); Z to srednia tych wektorow
    po calej membranie. Komorki z zerowa predkoscia (cisza) sa POMIJANE
    (kierunek ciszy jest niezdefiniowany - wliczenie ich jako wektora
    zerowego zanizyloby |Z| bez fizycznego uzasadnienia, komorki
    aktywne po prostu nie glosuja).

    Celowo NIEZALEZNE od `vorticity()` powyzej - wirowosc jest wrazliwa
    na gradient PREDKOSCI (nawet przy stalym kierunku), ta funkcja
    mierzy WYLACZNIE zgodnosc KIERUNKU, ignorujac predkosc calkowicie.

    `mean_direction_deg` uzywa TEJ SAMEJ konwencji "kierunek SKAD wieje"
    co `wind_speed_dir_from_uv()` w interpolate.py (formula
    `atan2(-Ureal,-Vreal)` ZDUBLOWANA tutaj CELOWO, nie zaimportowana -
    patrz nizej) - NIE surowego matematycznego np.angle(Z) w ukladzie
    wschod-polnoc, ktory dalby inna liczbe (kat dopelniajacy) i
    wprowadzilby DRUGA, niezgodna konwencje kierunku w tym samym pliku.

    UWAGA O IMPORCIE (2026-09-10, naprawa po realnym bledzie): ta
    funkcja CELOWO NIE importuje `wind_speed_dir_from_uv` z
    interpolate.py, mimo ze to 3-linijkowe zdublowanie tej samej formuly
    - `interpolate.py` importuje `scipy.interpolate.griddata` na
    poziomie modulu, wiec import stamtad wciagalby scipy w tranzyt do
    kazdego uzycia `spectrum.py`, NAWET do funkcji ktore scipy w ogole
    nie potrzebuja (ta funkcja liczy sie czystym numpy). Na maszynie
    uzytkownika z zablokowanym przez Device Guard scipy (ten sam
    zdiagnozowany wczesniej problem co w TIMDR-Earthquake-Core) to
    realnie psulo import CALEGO webapp/app.py przy starcie uvicorn -
    naprawione tu przez zdublowanie 3 linii matematyki zamiast
    importu, DOKLADNIE zgodnie z zasada tego ekosystemu "zdubluj mala,
    stabilna formule zamiast wciagac ciezka/kruchą zaleznosc" (patrz
    _vendor_* pliki w innych repo tego ekosystemu)."""
    speed = np.hypot(u, v)
    valid = speed > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return WindCoherenceResult(coherence=0.0, mean_direction_deg=0.0, n_valid=0)
    unit = (u[valid] + 1j * v[valid]) / speed[valid]
    Z = np.mean(unit)
    # Zdublowane z wind_speed_dir_from_uv() w interpolate.py - patrz UWAGA O IMPORCIE wyzej.
    mean_direction_deg = float(np.degrees(np.arctan2(-Z.real, -Z.imag)) % 360.0)
    return WindCoherenceResult(
        coherence=float(np.abs(Z)), mean_direction_deg=mean_direction_deg, n_valid=n_valid,
    )
