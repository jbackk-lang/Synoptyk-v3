"""Test end-to-end membrane/analyze.py::analyze_records na syntetycznym
"froncie" (skokowa zmiana temperatury/cisnienia w polowie siatki) -
kontrola pozytywna: pipeline powinien wykryc defekt w pasie frontu.
Kontrola negatywna: gladkie pole bez frontu nie powinno dac defektow."""
from __future__ import annotations

from membrane.analyze import analyze_records, result_to_json


def _grid_records(n=7, spacing=0.35, center_lat=52.0, center_lon=21.0, front=True):
    half = n // 2
    records = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            lat = center_lat + i * spacing
            lon = center_lon + j * spacing
            if front:
                # Front chlodny: skokowa zmiana temp/cisnienia przy lon >= center_lon.
                temp = 20.0 if lon < center_lon else 8.0
                pres = 1015.0 if lon < center_lon else 995.0
            else:
                temp = 15.0
                pres = 1013.0
            records.append({
                "lat": lat, "lon": lon,
                "temperature_c": temp, "pressure_hpa": pres,
                "humidity_pct": 60.0, "precip_mm": 5.0 if (front and lon >= center_lon) else 0.0,
                "wind_speed_kmh": 20.0 if front else 5.0,
                "wind_dir_deg": 250.0,
            })
    return records


def test_analyze_records_detects_front_as_gradient_defect():
    records = _grid_records(n=7, front=True)
    result = analyze_records(records, grid_n_membrane=31)
    assert result.defects_t.n_defects > 0
    assert result.defects_p.n_defects > 0


def test_analyze_records_smooth_field_has_few_or_no_defects():
    records = _grid_records(n=7, front=False)
    result = analyze_records(records, grid_n_membrane=31)
    # Pole calkowicie plaskie (stala temp/cisnienie/wiatr wszedzie) -
    # gradient == 0 wszedzie -> zero defektow (dokladnie kontrola
    # negatywna z test_defects.py::test_detect_defects_flat_field_gives_zero_defects).
    assert result.defects_t.n_defects == 0
    assert result.defects_p.n_defects == 0


def test_analyze_records_resonance_fires_near_front_precip_and_wind():
    """Front z jednoczesnie: silnym gradientem T, silnym gradientem P,
    opadem, i (przez skok predkosci wiatru na granicy interpolacji)
    podwyzszona wirowoscia - powinno dac >=1 komorke rezonansowa (>=3
    kanaly naraz)."""
    records = _grid_records(n=7, front=True)
    result = analyze_records(records, grid_n_membrane=31)
    # Rezonans nie jest gwarantowany matematycznie (zalezy od
    # przestrzennego pokrycia sie akurat 3 z 4 kanalow w tej samej
    # komorce) - ale przy tak silnie zbudowanym syntetycznym froncie ze
    # WSZYSTKIMI czterema sygnalami skupionymi w tym samym pasie
    # oczekujemy przynajmniej jednej komorki rezonansowej. Jesli kiedys
    # ten test zacznie failowac, to sygnal ze albo geometria testu, albo
    # RESONANCE_K wymagaja przegladu - nie nalezy go cicho oslabiac.
    assert result.resonance.n_resonance_cells >= 1


def test_result_to_json_is_serializable_and_consistent():
    records = _grid_records(n=7, front=True)
    result = analyze_records(records, grid_n_membrane=15)
    payload = result_to_json(result)
    assert payload["n_source_points"] == len(records)
    assert payload["defects"]["gradient_T"]["n"] == result.defects_t.n_defects
    assert len(payload["membrane"]["temperature_c"]) == 15
    assert len(payload["membrane"]["temperature_c"][0]) == 15
    assert payload["resonance"]["n_resonance_cells"] == result.resonance.n_resonance_cells
