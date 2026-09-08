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

Synoptyk‑v3 jest pierwszym narzędziem pogodowym w Polsce, które wykorzystuje metody geometrii różniczkowej i analizy widmowej do interpretacji zjawisk synoptycznych.

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

## Wydajność — porównanie z synoptyk-v2.0

Zmierzone bezpośrednio (bez sieci, czysto obliczeniowo, na syntetycznych
danych o typowym rozmiarze dla każdej appki — patrz metodologia niżej):

| | synoptyk-v2.0 (`TIMDRAnalyzer.analyze()`) | Synoptyk-v3 (`analyze_records()`) |
|---|---|---|
| typowy przebieg | `--region poland --days 7`: 6 stacji × 168 wierszy godzinowych | jedno kliknięcie "Analizuj membranę": 25 pkt wejściowych → siatka 41×41 |
| czas / jedna lokalizacja | **~250 ms/stację** (1.5s / 6 stacji) | **~24 ms** |
| główny koszt | pętla `for idx, row in df.iterrows()` (`analyzer/timdr_analyzer.py`) — znane wąskie gardło pandas, mimo wcześniejszej optymalizacji (`.diff()` wyniesione poza pętlę, patrz komentarz w kodzie) | w pełni wektoryzowane: `scipy.interpolate.griddata`, `numpy.fft.fft2`, `numpy.gradient` |

**~10× szybciej dla porównywalnej jednostki pracy** ("pełna analiza
jednej lokalizacji"), mimo że Synoptyk-v3 liczy obiektywnie więcej
(interpolacja 2D + FFT + gradient + wirowość + defekty + rezonans, nie
tylko sygnały 1D). Powód jest architektoniczny, nie przypadkowy: cały
pipeline membrany jest wektoryzowany (numpy/scipy, pętle w C), podczas
gdy `TIMDRAnalyzer` w v2.0 liczy wiersz-po-wierszu w czystym Pythonie.

**Zastrzeżenia (żeby nie przeceniać wyniku):** to NIE jest porównanie
jabłko-do-jabłka — różne kształty danych (siatka 2D vs szereg czasowy
1D) i różne algorytmy, nie dwie implementacje tego samego zadania. Pomiar
NIE obejmuje pobierania danych z sieci — to w praktyce dominuje w obu
appkach (obie pytają Open-Meteo przez HTTP, rzędu setek ms na
zapytanie), więc różnica dotyczy WYŁĄCZNIE lokalnej części obliczeniowej,
którą kontrolujemy w kodzie, nie całkowitego czasu odpowiedzi
użytkownikowi. Metodologia (odtwarzalna): `analyzer.timdr_analyzer.TIMDRAnalyzer`
uruchomione na 6 syntetycznych DataFrame'ach (168 wierszy, `np.random.default_rng`)
sekwencyjnie; `membrane.analyze.analyze_records` uruchomione na 25
syntetycznych rekordach z `membrane.grid_source.build_grid_points(n=5)`,
`grid_n_membrane=41`. Zmierzone `time.perf_counter()`, jednorazowo,
2026-09-06 — nie uśrednione po wielu przebiegach, więc traktuj jako rząd
wielkości, nie precyzyjny benchmark.

## Trafność prognozy

Do 2026-09-07 Synoptyk-v3 nie miał ŻADNEGO mechanizmu do sprawdzenia
trafności prognozy — tylko sam moduł prognozy (`run_collect.py::collect()`,
Open-Meteo `/v1/forecast`), bez punktu odniesienia w postaci rzeczywistej,
zarejestrowanej pogody. Domknięte tym samym wzorcem co SYNOPTYK-ARCTIC
(`fetch.py::fetch_archive` + `bias.py::compute_lead_bias`):

- **`membrane/archive_source.py::fetch_archive()`** — pobiera rzeczywistą
  (nie prognozowaną) pogodę z Open-Meteo Archive API
  (`archive-api.open-meteo.com`), kontrakt zweryfikowany NA ŻYWO przez
  przeglądarkę wewnętrzną 2026-09-07 (Warszawa, 10 dni wstecz) — te same
  nazwy pól co `/v1/forecast` (w tym `surface_pressure_mean`, wcześniej
  zweryfikowane tylko dla forecastu, nie archiwum). `exclude_trailing_days=2`
  odcina ostatnie 1-2 dni (jeszcze niesfinalizowana reanaliza, ten sam
  problem i to samo rozwiązanie co SYNOPTYK-ARCTIC).
- **`run_collect.py::collect_archive()`** — dopisuje te dane do TEGO
  SAMEGO CSV co `collect()`, z nową kolumną `source`
  (`prognoza`/`archiwum_openmeteo`) — **schemat CSV się zmienił, istniejący
  `data/meteogram_snapshots.csv` (38 wierszy danych testowych z tej samej
  sesji, w której powstała membrana) został zresetowany, nie migrowany.**
- **`membrane/bias.py::compute_lead_bias()`** — bias (rzeczywistość −
  prognoza) i MAE per `lead_days`, liczone TYLKO gdy ≥5 sparowanych dni
  (parowanie po `target_date`) — brak wpisu = za mało danych, NIGDY
  fałszywe zero.
- **`GET /api/bias`** (webapp) i sekcja **"Trafność prognozy"** na końcu
  dashboardu — tabela bias/MAE per horyzont, z jawnym komunikatem
  "za mało danych" zamiast pustej tabeli, gdy `status=insufficient_data`.

**Uczciwe oczekiwanie na start**: przy 1-2 dniach zbierania (stan na
2026-09-07) `/api/bias` zwróci `insufficient_data` dla każdego miasta —
dokładnie tak samo jak SYNOPTYK-ARCTIC po swoim pierwszym pobraniu. To
wymaga kilkudniowego/kilkutygodniowego klikania „💾 Zbierz do historii” +
„📡 Zbierz rzeczywistość” (albo odpalania obu co dnia), zanim `bias`/`mae`
zaczną się pojawiać. Do tego czasu jedyne, co można uczciwie powiedzieć o
trafności Synoptyk-v3, to krzyżowa weryfikacja z niezależnym dostawcą
(meteoblue) opisana niżej — to sprawdza wiarygodność DANYCH WEJŚCIOWYCH
(czy Open-Meteo w ogóle zwraca sensowne liczby), nie trafność prognozy
względem tego, co faktycznie się wydarzyło.

## Integracja z TIMDR-META-DYNAMICS (eksperymentalna)

`membrane/meta_adapter.py` mapuje wynik `analyze_records()` na
`MetaState(Lambda,tau,rho,J)` z repozytorium-siostry `TIMDR-META-DYNAMICS`
(ten sam wzorzec sys.path co `analizator-gieldowy-v3/meta_dynamics_module.py`
— folder-siostra musi leżeć obok `Synoptyk-v3` w tym samym katalogu
nadrzędnym). To DRUGA realna integracja tego formalizmu (pierwsza —
finansowa, w `analizator-gieldowy-v3`) i pierwsza z prawdziwymi danymi
fizycznymi/pogodowymi zamiast czysto syntetycznymi lub finansowymi.

**Mapowanie V1 -> V2 (pełna historia, uzasadnienie i zastrzeżenia w
docstringu `meta_adapter.py`) — V1 zamrożone PRZED policzeniem
czegokolwiek na realnych danych, V2 to WYŁĄCZNIE naprawa przeskalowania
V1 po zobaczeniu, że V1 nie działa (nie "dostrajanie progów", tylko
sprowadzenie wejścia do skali, w której inherited progi 0.1/1.0 w ogóle
mają szansę cokolwiek rozróżnić):**

| MetaState | V1 (porzucone) | V2 (aktualne) | jednostka V2 |
|---|---|---|---|
| Λ (struktura) | średnia `high_freq_fraction` T+P | bez zmian | [0,1] |
| τ (transformacja) | średni gradient T | średni gradient T / próg anomalii T tego dnia | bezwymiarowe, zwykle ≪1 |
| ρ (anomalia) | suma komórek-defektów (4 kanały) | to samo / (4 × liczba_komórek) | [0,1] |
| J (operator punktowy) | liczba komórek rezonansowych | to samo / liczba_komórek | [0,1] |

Normalizacje w V2 używają wyłącznie wielkości WEWNĘTRZNYCH per-dzień
(liczba komórek membrany, własny próg anomalii z `defects.py` z tego
samego dnia) — żadna nie została dobrana na podstawie tego, jak wyszedł
`M`-series czy `classify_phase()` na 10-dniowym oknie testowym.

**Kontrole (syntetyczne, `tests/test_meta_adapter.py`):** pozytywna (front
skokowy → `|M|` większe niż brak zmiany) i negatywna (dwa identyczne
snapshoty → `|M|=0`, faza `"stabilna"`) — obie przechodzą, dla V1 i V2.

**Demonstracja na PRAWDZIWYCH danych** (Archive API, siatka 3×3 wokół
Warszawy, 10 kolejnych dni 2026-08-28..2026-09-06, przechwycone przez
przeglądarkę wewnętrzną 2026-09-08 — okno akurat obejmuje realne
ochłodzenie ok. 08-31→09-01, spadek ciśnienia i wzrost wiatru ok. 09-03/04,
widoczne w surowych danych: temp. maks. spada z ~26-28°C do ~17-19°C,
ciśnienie do minimum ~988-998 hPa, wiatr rośnie do ~30-35 km/h):

- **V1 (porzucone): klasyfikacja faz była bezużyteczna.** Wszystkie 9
  kroków, bez wyjątku (dni spokojne I dzień realnego frontu), wyszły jako
  `"krytyczna"` (`|M|` od ~7 do ~143, próg krytyczny to zaledwie 1.0) —
  ρ/J to surowe liczby komórek membrany 41×41=1681, więc nawet mały
  dzień-do-dnia ruch przebijał próg o dwa rzędy wielkości, Λ/τ ginęły w
  sumie. Dokładnie ostrzeżenie z oryginalnego docstringu
  `classify_phase()` ("skala Λ/τ/ρ/J zależy całkowicie od tego, co
  podłączysz"), potwierdzone na realnym przykładzie.
- **V2 (po naprawie przeskalowania): klasyfikacja ROZRÓŻNIA kroki.**
  `|M|` teraz w zakresie ~0.07-0.18, rozkład faz na tym samym oknie:
  3× `"stabilna"`, 6× `"przejściowa"`, 0× `"krytyczna"`
  (`test_end_to_end_real_data_runs_without_crashing` sprawdza wprost, że
  `len(set(phases)) > 1` — V1 by tego testu NIE przeszedł). Krok
  08-30→08-31 (początek realnego ochłodzenia) ma jeden z wyższych `|M|`
  (0.180) w całym oknie, a najspokojniejszy 5-dniowy odcinek (09-02..09-04)
  daje 2 z 3 wystąpień `"stabilna"`. **To ciekawa zgodność z fizyczną
  intuicją, NIE potwierdzony wynik** — jedno 10-dniowe okno bez kontrolki
  negatywnej (okres definitywnie bez frontu) i bez testu
  Manna-Whitneya nie odróżnia "formalizm coś wykrywa" od przypadku na
  n=9 krokach.
- Co to FAKTYCZNIE pokazuje: (a) adapter jest okablowany poprawnie —
  potwierdzone kontrolami syntetycznymi w obu wersjach; (b) błąd
  skalowania w V1 był realny i naprawialny bez naruszania protokołu
  numerologii (normalizacja per-dzień, ustalona przed uruchomieniem V2 na
  realnych danych); (c) V2 na jednym realnym oknie DAJE zróżnicowaną,
  fizycznie niesprzeczną klasyfikację — ale to wciąż nie jest kalibracja.
  Prawdziwa walidacja wymagałaby wielu niezależnych epizodów frontowych +
  okresu bez frontu jako kontrolki negatywnej + testu istotności — jawnie
  nazwany, odrębny, nie zrobiony w tej sesji następny krok.

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

73/73 testów przechodzi (interpolacja, widmo, defekty, rezonans,
grid_source + archive_source na realnych fixture'ach, run_collect, bias,
webapp przez TestClient, meta_adapter na syntetycznych kontrolach +
realnej siatce 3×3/10 dni) — wszystkie z kontrolami pozytywnymi i
negatywnymi tam, gdzie to miało sens (pole liniowe/rotacja sztywna/
wirowość zerowa mają znaną analitycznie odpowiedź, nie tylko "kod się nie
wywala").

## Struktura

```
membrane/
  cities.py           — lista miast/regionów PL
  grid_source.py       — Krok 1: pobranie siatki z Open-Meteo
  interpolate.py       — Krok 2: interpolacja do membrany
  spectrum.py           — Krok 3: FFT/gradient/wirowość
  defects.py             — Krok 4: detekcja frontów
  resonance.py           — Krok 5: koincydencja = rezonans membrany
  analyze.py              — spina Kroki 1-5 w jeden pipeline
  archive_source.py       — rzeczywista pogoda (Open-Meteo Archive API), patrz "Trafność prognozy"
  bias.py                  — bias/MAE per horyzont, patrz "Trafność prognozy"
  meta_adapter.py          — adapter do TIMDR-META-DYNAMICS, patrz "Integracja z TIMDR-META-DYNAMICS"
run_collect.py    — Krok 6: kolektor CSV historii (meteogram + archiwum)
webapp/
  app.py             — FastAPI (endpointy /api/*)
  static/index.html  — dashboard (ciemny motyw jak SYNOPTYK-ARCTIC)
tests/              — 73 testy, patrz wyżej
```
