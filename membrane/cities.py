"""
membrane/cities.py — lista wybieralnych miast/regionów Polski.

Każde miasto to zarazem CENTRUM siatki, wokół którego grid_source.py
buduje NxN siatkę punktów (patrz grid_source.py::build_grid_points) do
zbudowania "membrany" pól pogodowych. Współrzędne z Wikipedii/geonames
(centra miast, zaokrąglone do 2 miejsc po przecinku - wystarczające dla
siatki o rozstawie ~0.25-0.5 stopnia, patrz grid_source.py).

Świadomie NIE stacje IMGW (byłoby bardziej "prawdziwe" jako punkty
pomiarowe), tylko miasta jako centra zapytań do Open-Meteo - ten sam
wybór co SYNOPTYK-ARCTIC (stacje jako nazwane punkty), tylko tu miasto
jest środkiem obszaru siatki, nie pojedynczym punktem pomiaru.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    region: str  # opisowy region PL, do grupowania w dropdownie


CITIES: list[City] = [
    City("Warszawa", 52.23, 21.01, "Mazowsze"),
    City("Krakow", 50.06, 19.94, "Malopolska"),
    City("Gdansk", 54.35, 18.65, "Wybrzeze"),
    City("Wroclaw", 51.11, 17.04, "Dolny Slask"),
    City("Poznan", 52.41, 16.93, "Wielkopolska"),
    City("Lodz", 51.76, 19.46, "Polska Centralna"),
    City("Szczecin", 53.43, 14.55, "Wybrzeze"),
    City("Bydgoszcz", 53.12, 18.00, "Kujawy"),
    City("Lublin", 51.25, 22.57, "Polska Wschodnia"),
    City("Bialystok", 53.13, 23.16, "Polska Wschodnia"),
    City("Katowice", 50.26, 19.02, "Slask"),
    City("Rzeszow", 50.04, 22.00, "Podkarpacie"),
    City("Olsztyn", 53.78, 20.48, "Warmia i Mazury"),
    City("Zielona_Gora", 51.94, 15.51, "Lubuskie"),
    City("Zakopane", 49.30, 19.95, "Tatry/Podhale"),
    City("Suwalki", 54.11, 22.93, "Suwalszczyzna"),
]

CITIES_BY_NAME: dict[str, City] = {c.name: c for c in CITIES}

DEFAULT_CITY = "Warszawa"


def resolve_city(name: str | None) -> City:
    """`?city=` -> City, albo DEFAULT_CITY gdy brak param. Nieznana nazwa
    -> KeyError jawnie (patrz webapp/app.py, ktory zamienia to na HTTP 404
    - ten sam wzorzec co SYNOPTYK-ARCTIC/_resolve_station), zamiast cicho
    spasc na domyslne miasto i ukryc literowke w URL/froncie."""
    key = name or DEFAULT_CITY
    if key not in CITIES_BY_NAME:
        raise KeyError(f"Nieznane miasto/region: {key!r}")
    return CITIES_BY_NAME[key]
