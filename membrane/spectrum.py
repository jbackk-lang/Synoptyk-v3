"""
membrane/spectrum.py — Krok 3: widmo deformacji membrany (FFT), gradient
(uskoki/fronty), i rotacja (wirowosc pola wiatru).

Trzy niezalezne diagnostyki, kazda inny obiekt matematyczny (ten sam
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
   pionowa rotora wiatru), nie nowa konstrukcja.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
