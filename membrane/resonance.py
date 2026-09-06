"""
membrane/resonance.py — Krok 5: "rezonans membrany" jako koincydencja
kilku niezaleznych defektow w TEJ SAMEJ komorce siatki jednoczesnie.

Wprost ta sama definicja co rezonans sygnalowy M w GIA-TIMDR (licznik
koincydencji, NIE fizyczny oscylator - patrz skill timdr-signal-framework
§1, ktory jednak NIE obejmuje tego repo: to jest tylko przestrzenne
zastosowanie tej samej idei, zbudowane od zera w tym repo, nie import z
GIA-TIMDR), tylko przeniesiona z osi czasu na siatke przestrzenna: zamiast
"ile kanalow (temp/cisnienie/wilgotnosc/wiatr) przekroczylo prog W TYM
SAMYM MOMENCIE CZASU", liczymy "ile niezaleznych diagnostyk deformacji
(gradient T, gradient P, wirowosc, opad) przekroczylo prog W TEJ SAMEJ
KOMORCE SIATKI".

UCZCIWE ZASTRZEZENIE (ten sam wzorzec dyscypliny co reszta ekosystemu
TIMDR w tej sesji): to jest heurystyka zbudowana na podstawie fizycznej
intuicji ("silne zjawisko synoptyczne = kilka pol odksztalca sie naraz w
tym samym miejscu"), NIE zwalidowana statystycznie na realnych danych
(brak testu Manna-Whitneya z kontrolami pozytywna/negatywna na realnych
frontach/burzach - dokladnie to, co protokol numerologii/formalizmu
skilla timdr-signal-framework wymagalby PRZED uznaniem tego za
"potwierdzony wzorzec", a co jest tu jawnie POMINIETE z braku dostepu do
oznakowanego zbioru realnych zdarzen synoptycznych w tej sesji). Traktuj
`is_resonance` jako WSKAZNIK "tu warto spojrzec", nie jako potwierdzony
detektor zjawisk (frontu/burzy/mezocyklonu).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResonanceResult:
    channel_names: list[str]
    coincidence_count: np.ndarray  # int, ile kanalow=True w kazdej komorce
    is_resonance: np.ndarray       # bool, coincidence_count >= k
    k: int
    n_resonance_cells: int


def compute_resonance(defect_masks: dict[str, np.ndarray], k: int = 3) -> ResonanceResult:
    """`defect_masks`: {"gradient_T": bool_array, "gradient_P": bool_array,
    "wirowosc": bool_array, "opad": bool_array, ...} - wszystkie maski
    MUSZA miec ten sam ksztalt (ta sama siatka membrany). `k` (domyslnie
    3, ta sama wartosc domyslna co K w rezonansie sygnalowym M w
    GIA-TIMDR - wybor spojny z reszta ekosystemu, nie niezalezna
    kalibracja) - ile kanalow musi byc jednoczesnie prawdziwych w
    komorce, zeby uznac ja za "rezonansowa".

    Rzuca ValueError, jesli podano < 2 kanaly (koincydencja wymaga
    >= 2 rzeczy, ktore moga sie zbiec) albo jesli ksztalty masek sie
    nie zgadzaja (zamiast cicho bledn broadcastowac numpy)."""
    if len(defect_masks) < 2:
        raise ValueError(f"Potrzeba >= 2 kanalow defektow do liczenia koincydencji, dostano {len(defect_masks)}")
    names = list(defect_masks.keys())
    shapes = {name: mask.shape for name, mask in defect_masks.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Maski defektow maja rozne ksztalty: {shapes}")
    if k < 1 or k > len(names):
        raise ValueError(f"k musi byc w [1, {len(names)}], dostano k={k}")

    stacked = np.stack([defect_masks[name].astype(int) for name in names], axis=0)
    coincidence_count = stacked.sum(axis=0)
    is_resonance = coincidence_count >= k

    return ResonanceResult(
        channel_names=names,
        coincidence_count=coincidence_count,
        is_resonance=is_resonance,
        k=k,
        n_resonance_cells=int(is_resonance.sum()),
    )
