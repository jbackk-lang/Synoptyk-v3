"""Testy membrane/meta_adapter.py.

Dwie kategorie testow, celowo rozdzielone (patrz docstring meta_adapter.py
"PRE-REJESTRACJA" i skill timdr-signal-framework, protokol numerologii):

1. SYNTETYCZNE kontrolki pozytywna/negatywna (przed-zarejestrowany,
   oczekiwany kierunek efektu: front -> wyzsze |M| niz brak frontu) - to
   sa jedyne testy w tym pliku z ASERCJAMI O KIERUNKU wyniku.

2. Test na PRAWDZIWYCH danych (Archive API, przechwycone przez
   przegladarke wewnetrzna 2026-09-08, siatka 3x3 wokol Warszawy,
   2026-08-28..2026-09-06 - ten sam zakres dat i ten sam punkt centralny
   [52.267,20.961] co juz istniejacy REAL_ARCHIVE_RESPONSE w
   test_archive_source.py, dla ktorego wartosci temperatury/cisnienia z
   dni 08-28..09-06 SIE ZGADZAJA - wzajemna weryfikacja dwoch niezaleznie
   zlapanych odpowiedzi tego samego API) sprawdza WYLACZNIE, ze pipeline
   dziala mechanicznie end-to-end (ksztalty, brak NaN, poprawne typy) -
   CELOWO bez asercji "faza == X w dniu Y", bo progi classify_phase() nie
   sa skalibrowane dla tego mapowania (patrz zastrzezenie #2 w
   meta_adapter.py). Wynik jest zamiast tego wypisany/zinterpretowany w
   README.md#Integracja-z-TIMDR-META-DYNAMICS jako obserwacja, nie
   potwierdzony wynik."""
from __future__ import annotations

from membrane.meta_adapter import (
    archive_grid_response_to_daily_records,
    build_meta_series_from_daily_records,
    membrane_result_to_meta_state,
)
from membrane.analyze import analyze_records


def _grid_records(n=7, spacing=0.35, center_lat=52.0, center_lon=21.0, front=False, front_strength=1.0):
    """Ta sama konstrukcja co tests/test_analyze.py::_grid_records, z
    dodanym `front_strength` (0.0 = pole plaskie, 1.0 = front pelnej
    sily) - zeby moc porownac 'brak zmiany' vs 'silna zmiana' na TEJ
    SAMEJ geometrii siatki."""
    half = n // 2
    records = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            lat = center_lat + i * spacing
            lon = center_lon + j * spacing
            if front and lon >= center_lon:
                temp = 20.0 - 12.0 * front_strength
                pres = 1015.0 - 20.0 * front_strength
            else:
                temp = 20.0
                pres = 1015.0
            records.append({
                "lat": lat, "lon": lon,
                "temperature_c": temp, "pressure_hpa": pres,
                "humidity_pct": 60.0,
                "precip_mm": 5.0 * front_strength if (front and lon >= center_lon) else 0.0,
                "wind_speed_kmh": 20.0 if front else 5.0,
                "wind_dir_deg": 250.0,
            })
    return records


# ---------------------------------------------------------------------------
# 1. Mapowanie: sprawdzenie wzorow z PRE-REJESTRACJI wprost na jednym wyniku.
# ---------------------------------------------------------------------------

def test_mapping_formulas_match_pre_registered_definitions():
    """V2 (patrz meta_adapter.py docstring) - tau/rho/J znormalizowane."""
    records = _grid_records(n=7, front=True, front_strength=1.0)
    result = analyze_records(records, grid_n_membrane=21)
    state = membrane_result_to_meta_state(result)

    n_cells = float(result.grid_n_membrane) ** 2
    expected_lambda = 0.5 * (
        result.temperature_spectrum.high_freq_fraction + result.pressure_spectrum.high_freq_fraction
    )
    expected_tau = float(result.gradient_t.mean()) / result.defects_t.threshold
    expected_rho = float(
        result.defects_t.n_defects + result.defects_p.n_defects
        + result.defects_vort.n_defects + result.defects_precip.n_defects
    ) / (4.0 * n_cells)
    expected_J = float(result.resonance.n_resonance_cells) / n_cells

    assert state.Lambda == expected_lambda
    assert state.tau == expected_tau
    assert state.rho == expected_rho
    assert state.J == expected_J
    # Wszystkie 4 skladowe teraz w porownywalnej skali (nie tysiace vs [0,1]).
    assert 0.0 <= state.Lambda <= 1.0
    assert 0.0 <= state.rho <= 1.0
    assert 0.0 <= state.J <= 1.0


# ---------------------------------------------------------------------------
# 2. Kontrolka POZYTYWNA: silny, realistyczny front -> wieksze |M| niz brak zmiany.
# ---------------------------------------------------------------------------

def test_positive_control_front_gives_larger_M_magnitude_than_no_change():
    calm_a = _grid_records(n=7, front=False)
    calm_b = _grid_records(n=7, front=False)  # identyczne pole, drugi "dzien"
    front = _grid_records(n=7, front=True, front_strength=1.0)

    no_change = build_meta_series_from_daily_records([calm_a, calm_b], grid_n_membrane=21)
    with_front = build_meta_series_from_daily_records([calm_a, front], grid_n_membrane=21)

    from timdr_meta_dynamics import MetaOperatorM
    op = MetaOperatorM()
    mag_no_change = op.magnitude(no_change.M_series[0])
    mag_with_front = op.magnitude(with_front.M_series[0])

    assert mag_with_front > mag_no_change


def test_negative_control_identical_snapshots_give_zero_M_and_stable_phase():
    calm_a = _grid_records(n=7, front=False)
    calm_b = _grid_records(n=7, front=False)
    result = build_meta_series_from_daily_records([calm_a, calm_b], grid_n_membrane=21)

    from timdr_meta_dynamics import MetaOperatorM
    op = MetaOperatorM()
    assert op.magnitude(result.M_series[0]) == 0.0
    assert result.phases[0] == "stabilna"


# ---------------------------------------------------------------------------
# 3. Walidacja wejscia.
# ---------------------------------------------------------------------------

def test_requires_at_least_2_days():
    records = _grid_records(n=7)
    try:
        build_meta_series_from_daily_records([records], grid_n_membrane=21)
        assert False, "oczekiwano ValueError dla < 2 dni"
    except ValueError as e:
        assert "2 dni" in str(e)


def test_dates_length_mismatch_raises():
    records = _grid_records(n=7)
    try:
        build_meta_series_from_daily_records([records, records], dates=["a"], grid_n_membrane=21)
        assert False, "oczekiwano ValueError dla niezgodnej dlugosci dates"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 4. PRAWDZIWE dane: siatka 3x3 wokol Warszawy, Archive API, przechwycone
#    przez przegladarke wewnetrzna 2026-09-08. URL (bez klucza, publiczne
#    API): https://archive-api.open-meteo.com/v1/archive?latitude=51.88,51.88,51.88,52.23,52.23,52.23,52.58,52.58,52.58&longitude=20.66,21.01,21.36,20.66,21.01,21.36,20.66,21.01,21.36&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant,surface_pressure_mean,relative_humidity_2m_mean&start_date=2026-08-28&end_date=2026-09-06&timezone=UTC
#    Punkt srodkowy [52.267,20.961] i zakres dat 08-28..09-06 POKRYWAJA
#    SIE z REAL_ARCHIVE_RESPONSE w test_archive_source.py (zlapane
#    NIEZALEZNIE, dzien wczesniej) - wartosci temperatury/cisnienia dla
#    tych samych dni/wspolrzednych SIE ZGADZAJA (wzajemna weryfikacja
#    dwoch oddzielnych zapytan do tego samego API).
# ---------------------------------------------------------------------------

REAL_GRID_RESPONSE = [
    {"latitude": 51.915638, "longitude": 20.604397, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.7, 22.7, 25.9, 26.2, 19.7, 20.9, 22.0, 23.0, 19.9, 17.3],
        "temperature_2m_min": [11.9, 14.8, 12.7, 15.3, 14.7, 13.2, 12.1, 14.5, 12.7, 11.4],
        "precipitation_sum": [0.00, 5.40, 0.20, 2.70, 1.30, 1.70, 1.00, 2.90, 1.20, 2.00],
        "wind_speed_10m_max": [25.4, 20.1, 14.1, 18.0, 27.0, 15.4, 16.5, 30.7, 30.5, 32.8],
        "wind_direction_10m_dominant": [131, 231, 203, 222, 265, 267, 253, 244, 274, 285],
        "surface_pressure_mean": [996.6, 990.8, 992.7, 991.6, 993.4, 996.4, 994.6, 987.6, 989.8, 998.5],
        "relative_humidity_2m_mean": [57, 77, 65, 83, 73, 73, 72, 80, 69, 77],
    }},
    {"latitude": 51.84534, "longitude": 21.06033, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [24.5, 22.9, 26.6, 26.9, 20.8, 21.1, 22.4, 23.5, 20.3, 18.2],
        "temperature_2m_min": [11.9, 14.1, 11.9, 14.6, 15.2, 13.2, 11.8, 15.2, 13.2, 11.7],
        "precipitation_sum": [0.00, 5.00, 0.10, 12.60, 1.20, 3.60, 2.30, 2.40, 0.60, 2.10],
        "wind_speed_10m_max": [24.7, 17.2, 12.8, 23.6, 29.9, 17.8, 12.8, 30.6, 31.3, 35.5],
        "wind_direction_10m_dominant": [129, 220, 201, 208, 263, 264, 247, 241, 272, 285],
        "surface_pressure_mean": [1004.2, 998.1, 999.9, 998.8, 1000.5, 1003.6, 1001.8, 994.7, 996.7, 1005.5],
        "relative_humidity_2m_mean": [57, 77, 67, 83, 74, 78, 75, 77, 68, 76],
    }},
    {"latitude": 51.84534, "longitude": 21.389397, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [24.3, 22.2, 26.0, 27.6, 21.3, 21.0, 22.8, 23.3, 20.5, 18.0],
        "temperature_2m_min": [12.6, 14.0, 11.6, 14.9, 15.3, 13.1, 11.9, 14.9, 13.5, 11.7],
        "precipitation_sum": [0.00, 4.90, 0.00, 9.40, 0.70, 2.80, 13.60, 3.80, 1.60, 2.40],
        "wind_speed_10m_max": [21.8, 18.0, 12.1, 18.0, 25.1, 15.8, 11.4, 26.7, 28.4, 29.5],
        "wind_direction_10m_dominant": [129, 198, 185, 197, 262, 264, 240, 239, 272, 286],
        "surface_pressure_mean": [1008.0, 1001.6, 1003.5, 1002.2, 1003.8, 1007.0, 1005.2, 998.1, 1000.0, 1008.7],
        "relative_humidity_2m_mean": [56, 78, 66, 81, 75, 78, 74, 78, 67, 78],
    }},
    {"latitude": 52.26713, "longitude": 20.628466, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.9, 22.7, 26.2, 26.3, 20.2, 21.6, 22.1, 22.5, 20.1, 18.2],
        "temperature_2m_min": [12.6, 14.5, 12.7, 15.0, 14.9, 12.8, 11.8, 15.7, 13.1, 11.4],
        "precipitation_sum": [0.00, 2.80, 0.00, 3.20, 0.90, 0.10, 2.80, 2.50, 2.90, 3.30],
        "wind_speed_10m_max": [21.2, 15.9, 13.8, 13.4, 21.8, 16.8, 14.0, 27.7, 27.6, 27.8],
        "wind_direction_10m_dominant": [128, 220, 197, 212, 259, 263, 235, 243, 271, 282],
        "surface_pressure_mean": [1008.5, 1002.2, 1004.0, 1003.0, 1004.7, 1007.9, 1006.2, 998.4, 1000.8, 1009.9],
        "relative_humidity_2m_mean": [59, 78, 67, 85, 78, 77, 75, 77, 72, 80],
    }},
    {"latitude": 52.26713, "longitude": 20.961182, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.8, 21.9, 25.3, 25.9, 19.9, 21.2, 21.9, 22.1, 19.7, 18.7],
        "temperature_2m_min": [13.0, 15.7, 13.8, 16.0, 15.4, 13.6, 12.1, 15.7, 13.1, 12.0],
        "precipitation_sum": [0.00, 5.20, 0.00, 2.40, 3.20, 0.40, 7.00, 2.40, 2.50, 2.20],
        "wind_speed_10m_max": [18.5, 16.0, 12.1, 11.5, 20.7, 14.4, 12.3, 24.8, 25.6, 24.1],
        "wind_direction_10m_dominant": [131, 206, 182, 199, 259, 262, 236, 242, 271, 282],
        "surface_pressure_mean": [1006.2, 999.7, 1001.6, 1000.4, 1001.9, 1005.2, 1003.5, 995.8, 997.9, 1006.9],
        "relative_humidity_2m_mean": [58, 79, 68, 84, 78, 75, 75, 78, 72, 79],
    }},
    {"latitude": 52.196835, "longitude": 21.420664, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.9, 20.6, 25.2, 26.7, 20.4, 20.7, 22.1, 21.8, 19.7, 18.9],
        "temperature_2m_min": [12.5, 13.8, 11.9, 15.2, 14.9, 12.8, 11.9, 14.0, 12.6, 11.5],
        "precipitation_sum": [0.00, 6.70, 0.00, 4.60, 3.60, 1.70, 3.60, 5.10, 2.60, 1.60],
        "wind_speed_10m_max": [21.8, 17.6, 13.5, 13.6, 23.2, 15.4, 12.6, 26.4, 27.4, 25.4],
        "wind_direction_10m_dominant": [131, 186, 182, 191, 254, 257, 240, 234, 271, 283],
        "surface_pressure_mean": [1006.6, 999.8, 1001.8, 1000.5, 1001.9, 1005.1, 1003.5, 995.9, 997.7, 1006.5],
        "relative_humidity_2m_mean": [56, 82, 68, 83, 82, 76, 75, 82, 73, 80],
    }},
    {"latitude": 52.618626, "longitude": 20.652985, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.6, 21.7, 25.5, 25.3, 19.6, 21.3, 22.2, 21.4, 19.8, 18.9],
        "temperature_2m_min": [12.6, 14.5, 12.5, 14.4, 14.7, 13.1, 12.5, 14.9, 12.5, 11.5],
        "precipitation_sum": [0.00, 1.60, 0.00, 1.30, 2.60, 0.70, 0.90, 5.30, 1.40, 2.10],
        "wind_speed_10m_max": [22.7, 17.4, 15.4, 15.0, 23.0, 19.1, 15.2, 31.7, 31.0, 28.1],
        "wind_direction_10m_dominant": [129, 204, 191, 208, 258, 265, 235, 241, 268, 283],
        "surface_pressure_mean": [1007.8, 1001.1, 1002.9, 1001.8, 1003.2, 1006.6, 1004.9, 996.7, 999.0, 1008.3],
        "relative_humidity_2m_mean": [58, 78, 70, 85, 81, 73, 72, 82, 73, 78],
    }},
    {"latitude": 52.618626, "longitude": 20.988806, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.4, 21.4, 25.1, 25.5, 19.3, 21.1, 22.0, 21.4, 19.6, 19.0],
        "temperature_2m_min": [12.3, 14.4, 12.7, 14.7, 14.9, 13.0, 12.5, 14.9, 12.4, 11.6],
        "precipitation_sum": [0.00, 3.90, 0.20, 2.10, 2.70, 0.90, 2.50, 4.50, 2.00, 1.20],
        "wind_speed_10m_max": [22.8, 16.1, 15.1, 18.9, 23.2, 19.5, 14.5, 30.0, 31.3, 29.5],
        "wind_direction_10m_dominant": [129, 198, 189, 206, 256, 264, 236, 241, 268, 283],
        "surface_pressure_mean": [1005.7, 998.6, 1000.5, 999.3, 1000.5, 1004.0, 1002.3, 994.2, 996.2, 1005.3],
        "relative_humidity_2m_mean": [59, 81, 70, 86, 81, 74, 76, 81, 73, 79],
    }},
    {"latitude": 52.618626, "longitude": 21.324627, "daily": {
        "time": ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"],
        "temperature_2m_max": [23.6, 20.5, 24.9, 25.9, 19.6, 21.2, 22.0, 21.6, 19.8, 19.2],
        "temperature_2m_min": [11.9, 13.9, 12.1, 14.6, 15.0, 12.7, 11.7, 14.6, 12.7, 11.6],
        "precipitation_sum": [0.00, 6.20, 0.00, 2.50, 4.40, 0.60, 1.00, 4.80, 1.80, 1.30],
        "wind_speed_10m_max": [21.9, 15.8, 15.0, 15.3, 24.3, 18.0, 14.9, 30.5, 31.3, 29.3],
        "wind_direction_10m_dominant": [129, 186, 188, 200, 252, 261, 236, 237, 268, 283],
        "surface_pressure_mean": [1008.4, 1001.1, 1003.0, 1001.7, 1002.8, 1006.3, 1004.7, 996.5, 998.3, 1007.4],
        "relative_humidity_2m_mean": [59, 83, 70, 84, 83, 75, 76, 81, 72, 78],
    }},
]


def test_archive_grid_response_to_daily_records_shapes():
    daily_records = archive_grid_response_to_daily_records(REAL_GRID_RESPONSE)
    assert len(daily_records) == 10  # 10 dni
    for day_records in daily_records:
        assert len(day_records) == 9  # siatka 3x3
        for r in day_records:
            for key in ("lat", "lon", "temperature_c", "pressure_hpa", "humidity_pct", "precip_mm", "wind_speed_kmh", "wind_dir_deg"):
                assert r[key] is not None

    # Dzien 0 (2026-08-28), punkt centralny [52.267,20.961] (indeks 4) ->
    # zgadza sie z surowa odpowiedzia (kontrola samego parsera, nie API).
    assert daily_records[0][4]["temperature_c"] == 23.8
    assert daily_records[0][4]["pressure_hpa"] == 1006.2


def test_end_to_end_real_data_runs_without_crashing():
    """CELOWO bez asercji o KONKRETNEJ fazie - patrz docstring modulu.
    Sprawdza wylacznie, ze pipeline dziala mechanicznie na prawdziwych
    danych: poprawne ksztalty, brak NaN/Inf, poprawne typy."""
    import math

    daily_records = archive_grid_response_to_daily_records(REAL_GRID_RESPONSE)
    dates = REAL_GRID_RESPONSE[0]["daily"]["time"]

    result = build_meta_series_from_daily_records(daily_records, dates=dates, grid_n_membrane=41, dt=1.0)

    assert len(result.states) == 10
    assert len(result.M_series) == 9
    assert len(result.phases) == 9
    assert all(p in ("stabilna", "przejsciowa", "krytyczna") for p in result.phases)

    for s in result.states:
        assert math.isfinite(s.Lambda)
        assert math.isfinite(s.tau)
        assert math.isfinite(s.rho)
        assert math.isfinite(s.J)
        assert 0.0 <= s.Lambda <= 1.0
        assert 0.0 <= s.rho <= 1.0
        assert 0.0 <= s.J <= 1.0

    assert result.trigger.triggered in (True, False)

    # V2 (normalizacja) powinno przynajmniej ROZROZNIC dni miedzy soba -
    # NIE jest to asercja "faza X jest poprawna" (progi nadal
    # nieskalibrowane, patrz zastrzezenie #2 w meta_adapter.py), tylko
    # sprawdzenie, ze klasyfikacja nie jest STALA przez wszystkie kroki
    # (co bylo dokladnie problemem V1 - wszystkie 9/9 kroki "krytyczna").
    assert len(set(result.phases)) > 1, (
        f"V2 nadal daje jedna, stala faze na wszystkich krokach: {result.phases} - "
        "normalizacja nie naprawila problemu braku mocy dyskryminujacej."
    )
