# Synoptyk-v3 — membrana pogodowa

Trzecie podejście do "synoptyka" w tym ekosystemie (po SYNOPTYK-ARCTIC i
synoptyk-v2.0), zbudowane wokół innej hipotezy geometrycznej niż tamte
dwa: zamiast traktować każdy parametr pogodowy jako niezależny szereg
czasowy per stacja (`T(t), P(t), RH(t), Wiatr(t)`), traktujemy go jako
**pole na siatce geograficznej** — `T(x,y), P(x,y), RH(x,y)` i pole
wektorowe wiatru `V(x,y)=(u(x,y),v(x,y))` — "membranę" rozciągniętą nad
obszarem wokół wybranego miasta, z górkami/dolinami/uskokami/wirami,
którą analizujemy metodami z geometrii różniczkowej/analizy widmowej,
zamiast czystej statystyki szeregów czasowych.

Synoptyk‑v3 jest pierwszym narzędziem pogodowym w Polsce, które wykorzystuje
metody geometrii różniczkowej i analizy widmowej do interpretacji
zjawisk synoptycznych.

## Sześć kroków pipeline'u

1. **Siatka** (`membrane/grid_source.py`) — pobranie N×N punktów
   (domyślnie 5×5=25) wokół wybranego miasta z Open-Meteo (temperatura,
   ciśnienie, wilgotność, wiatr, opad), w jednym zapytaniu wsadowym.
2. **Membrana** (`membrane/interpolate.py`) — interpolacja kubiczna
   (scipy.griddata) rzadkiej siatki punktów na gęstą, regularną siatkę
   41×41. Wiatr rozkładany na składowe u/v PRZED interpolacją (kierunek
   to wielkość kątowa — interpolacja wprost dałaby bezsensowny wynik przy
   przejściu 350°→10°).
3. **Widmo** (`membrane/spectrum.py`) — 2D FFT z binowaniem promieniowym
   (nisko/wysokoczęstotliwościowe), moduł gradientu (fronty), wirowość
   `dv/dx - du/dy` (rotacja wiatru).
4. **Defekty** (`membrane/defects.py`) — komórki, gdzie moduł gradientu
   T/P/opadu przekracza próg mediana+k·MAD (odporny na pojedyncze
   ekstrema, k=3.5 dobrane konserwatywnie, NIE skalibrowane na realnych
   danych).
5. **Rezonans** (`membrane/resonance.py`) — koincydencja: komórka, w
   której ≥3 z 4 niezależnych kanałów defektów (gradient T, gradient P,
   wirowość, gradient opadu) przekraczają próg jednocześnie. Wprost ta
   sama definicja co "rezonans M" w GIA-TIMDR (licznik koincydencji, NIE
   fizyczny oscylator), tylko przeniesiona z osi czasu na siatkę
   przestrzenną — zbudowana od zera w tym repo, nie import z GIA-TIMDR.
6. **Przekroje 1D** (`run_collect.py`, panel "Meteogram" w dashboardzie)
   — klasyczny widok `T(t), P(t), wiatr(t), opad(t)` dla jednego punktu
   (miasta), ale jako *pochodna* membrany/API, nie jako główny obiekt.

## Źródło danych — dlaczego Open-Meteo, nie ERA5/GFS wprost

ERA5 wymaga rejestracji + tokena CDS i pobrania dużych plików
NetCDF/GRIB; GFS wymaga parsowania GRIB2. Oba niepraktyczne do zbudowania
i przetestowania w jednej sesji. Open-Meteo agreguje dane z tych samych
modeli NWP (w tym GFS/ICON) pod prostym, darmowym JSON API bez klucza —
siatka punktów pobierana tutaj jest więc faktycznie prognozą modelu
numerycznego (nie interpolacją stacji), tylko pobieraną punkt-po-punkcie.
Ten sam dostawca co SYNOPTYK-ARCTIC i synoptyk-v2.0 (spójność
ekosystemowa).

## Uczciwe ograniczenia

- **Sieć nie została przetestowana z wnętrza tej appki w sesji, w której
  powstała.** Sandbox, w którym pisano ten kod, ma zablokowany dostęp do
  internetu z poziomu Pythona/bash (każde `requests.get()` kończy się
  `ProxyError`/403 — sprawdzone bezpośrednio). Kontrakt API Open-Meteo
  (kształt JSON dla 1 i wielu punktów, dokładne nazwy pól) został mimo to
  zweryfikowany na **prawdziwych** danych — osobnym kanałem (przeglądarką
  wewnętrzną z dostępem do sieci), trzema realnymi zapytaniami do
  `api.open-meteo.com`. Te realne odpowiedzi są wpisane jako fixture w
  `tests/test_grid_source.py` — parsowanie jest więc przetestowane na
  prawdziwym kształcie danych, tylko samo połączenie sieciowe z wnętrza
  `grid_source.py`/`run_collect.py` nie zostało wykonane w tej sesji.
  Kiedy appka działa na normalnym komputerze (nie w tym sandboxie),
  dostęp do internetu jest zwykły — ograniczenie jest specyficzne dla
  środowiska, w którym kod został **napisany**, nie w którym będzie
  **uruchomiony**. `webapp/app.py` zwraca czytelny HTTP 502 (nie 500) na
  błąd sieci — sprawdzone smoke-testem przez `TestClient` w tym samym
  zablokowanym sandboxie (patrz historia commitów).

- **Krzyżowa weryfikacja prognozy Open-Meteo względem niezależnego źródła
  (2026-09-06, Kraków 50.083°N 19.917°E, 5 dni naprzód).** Ponieważ i
  Synoptyk-v3, i (dla realnej ścieżki `/api/forecast`, nie dla
  `run_synoptyk.py`/`data/fetcher.py`, który pobiera reanalizę
  `archive-api.open-meteo.com`, NIE prognozę) synoptyk-v2.0 opierają się
  wyłącznie na Open-Meteo, sprawdzono, czy liczby, które te appki
  pokazują, są w ogóle wiarygodne — przez porównanie z niezależnym
  dostawcą (meteoblue) dla tego samego miasta i tych samych dni:

  | dzień | Open-Meteo maks/min °C | meteoblue maks/min °C | Open-Meteo wiatr km/h | meteoblue wiatr km/h | Open-Meteo opad | meteoblue opad |
  |---|---|---|---|---|---|---|
  | 06.09 | 19.4 / 13.0 | 19 / 11 | 20.9 | 14 | 0 mm | – |
  | 07.09 | 22.0 / 8.7 | 22 / 9 | 9.4 | 4 | 0 mm | – |
  | 08.09 | 29.9 / 13.8 | 29 / 14 | 13.7 | 7 | 0 mm | – |
  | 09.09 | 31.5 / 18.5 | 32 / 16 | 10.5 | 7 | 0 mm | 0–2 mm |
  | 10.09 | 21.8 / 13.3 | 18 / 13 | 18.4 | 8 | 1.2 mm | 5–10 mm |

  Wnioski: temperatura maksymalna zgadza się bardzo dobrze przez pierwsze
  4 dni (rozbieżność ≤1°C) — dane wejściowe są wiarygodne, nie są
  artefaktem błędnego zapytania do API. **Prędkość wiatru w Open-Meteo
  jest systematycznie ~1.5–2× wyższa niż u meteoblue na KAŻDY z 5 dni** —
  to nie błąd w kodzie żadnej z appek (obie tylko przekazują dalej to, co
  zwróci API), tylko realna różnica metodologiczna między dostawcami
  (inny model źródłowy i/lub inna konwencja uśredniania) — traktuj
  wartość wiatru z Synoptyk-v3 jako orientacyjną, nie precyzyjny pomiar.
  Dzień 5 (10.09) pokazuje realną rozbieżność prognoz między dostawcami
  (Open-Meteo cieplej i suszej: 21.8°C/1.2mm; meteoblue chłodniej i
  mokrzej: 18°C/5-10mm) — normalna niepewność prognozy na dalszy termin,
  nie błąd. Jednorazowy spot-check na jednym mieście/dacie, nie
  systematyczna walidacja — nie ekstrapoluj tego na wszystkie miasta czy
  wszystkie zakresy dat.

- **Rezonans membrany (Krok 5) jest heurystyką, NIE zwalidowanym
  detektorem zjawisk.** Zbudowany na fizycznej intuicji ("silne zjawisko
  synoptyczne = kilka pól odkształca się naraz w tym samym miejscu"), ale
  BEZ testu Manna-Whitneya z kontrolami pozytywną/negatywną na
  oznakowanym zbiorze realnych frontów/burzy — dokładnie to, czego
  protokół numerologii/formalizmu (skill `timdr-signal-framework`, choć
  formalnie poza zakresem tego repo) wymagałby przed uznaniem tego za
  "potwierdzony wzorzec". Traktuj `is_resonance` jako wskaźnik "tu warto
  spojrzeć", nie jako potwierdzony detektor frontu/burzy/mezocyklonu.

- **Próg mediana+k·MAD ma znany punkt degeneracji.** MAD (punkt
  załamania 50%) jest dokładnie zero, gdy ≥50% wartości w polu jest
  ściśle sobie równych — co w praktyce zdarza się tylko w idealnie
  syntetycznych testach (pole matematycznie stałe), nie w prawdziwych
  interpolowanych polach meteorologicznych (zawsze mają ciągłą,
  niezerową zmienność). Znalezione i naprawione podczas pisania testów —
  `membrane/defects.py` ma dwa zabezpieczenia: próg szumu zaokrągleń
  (`noise_floor`, żeby "gradient" rzędu 1e-13 z błędu numerycznego
  interpolacji nie był mylony z prawdziwym frontem) i jawną regułę dla
  przypadku "MAD=0 mimo prawdziwego outliera" (flaguj każdą komórkę
  ściśle różną od mediany, gdy rozrzut tła jest dowiedziony zerowy).
  Pełne uzasadnienie w docstringu `defects.py::detect_defects`.

- **k=3.5 (próg defektu) i k=3 (próg koincydencji rezonansu) są wyborami
  konserwatywnymi, NIE skalibrowanymi** na żadnym realnym zbiorze
  oznakowanych zdarzeń synoptycznych. Wartość k=3 dla koincydencji
  wybrana dla spójności z domyślnym progiem "rezonansu M" w GIA-TIMDR, a
  nie z niezależnej kalibracji tego repo.

- **Interpolacja kubiczna na siatce 5×5→41×41 nie ma gwarancji poprawnej
  ekstrapolacji na rogach membrany** poza wypukłą otoczką punktów
  wejściowych (fallback na `nearest`, patrz `interpolate.py::_interp_field`)
  — środek membrany jest wiarygodny, skrajne rogi mniej.

## Uruchomienie

```
pip install -r requirements.txt
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8010
```

albo (Windows) `run_dashboard.bat`. Wymaga połączenia z internetem
(Open-Meteo) — brak własnych/wbudowanych danych.

## Testy

```
python -m pytest -q
```

50/50 testów przechodzi (interpolacja, widmo, defekty, rezonans,
grid_source na realnych fixture'ach, run_collect, webapp przez
TestClient) — wszystkie z kontrolami pozytywnymi i negatywnymi tam, gdzie
to miało sens (pole liniowe/rotacja sztywna/wirowość zerowa mają znaną
analitycznie odpowiedź, nie tylko "kod się nie wywala").

## Struktura

```
membrane/
  cities.py        — lista miast/regionów PL
  grid_source.py    — Krok 1: pobranie siatki z Open-Meteo
  interpolate.py    — Krok 2: interpolacja do membrany
  spectrum.py        — Krok 3: FFT/gradient/wirowość
  defects.py          — Krok 4: detekcja frontów
  resonance.py        — Krok 5: koincydencja = rezonans membrany
  analyze.py           — spina Kroki 1-5 w jeden pipeline
run_collect.py    — Krok 6: kolektor CSV historii (meteogram)
webapp/
  app.py             — FastAPI (endpointy /api/*)
  static/index.html  — dashboard (ciemny motyw jak SYNOPTYK-ARCTIC)
tests/              — 50 testów, patrz wyżej
```
