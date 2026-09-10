"""
membrane/analyze.py — spina Kroki 1-5 (grid_source -> interpolate ->
spectrum -> defects -> resonance) w jedna funkcje wywolywana przez
webapp/app.py (endpoint POST /api/analyze).

Krok 6 (przekroje 1D "Synoptyk klasyczny") NIE jest tutaj - to osobna,
prostsza sciezka (grid_source.fetch_meteogram, jeden punkt, bez
membrany), zasilajaca panel meteogramu w dashboardzie. Ten modul
odpowiada wylacznie za analize PRZESTRZENNA (membrane).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from membrane import defects as defects_mod
from membrane import resonance as resonance_mod
from membrane import spectrum as spectrum_mod
from membrane.grid_source import GridPoint, build_grid_points, fetch_grid_snapshot
from membrane.interpolate import Membrane, build_membrane

DEFAULT_GRID_N_SOURCE = 5   # rzadka siatka pobierana z Open-Meteo (5x5=25 zapytan-punktow w 1 requescie)
DEFAULT_GRID_N_MEMBRANE = 41  # gesta membrana po interpolacji
DEFAULT_SPACING_DEG = 0.35
DEFECT_K = 3.5
RESONANCE_K = 3


@dataclass
class AnalyzeResult:
    n_source_points: int
    grid_n_membrane: int
    temperature_spectrum: spectrum_mod.SpectrumResult
    pressure_spectrum: spectrum_mod.SpectrumResult
    gradient_t: np.ndarray
    gradient_p: np.ndarray
    vorticity: np.ndarray
    wind_coherence: spectrum_mod.WindCoherenceResult
    defects_t: defects_mod.DefectResult
    defects_p: defects_mod.DefectResult
    defects_vort: defects_mod.DefectResult
    defects_precip: defects_mod.DefectResult
    resonance: resonance_mod.ResonanceResult
    membrane: Membrane


def run_membrane_analysis(center_lat: float, center_lon: float,
                           grid_n_source: int = DEFAULT_GRID_N_SOURCE,
                           spacing_deg: float = DEFAULT_SPACING_DEG,
                           grid_n_membrane: int = DEFAULT_GRID_N_MEMBRANE) -> AnalyzeResult:
    """Pelny pipeline: siatka -> pobranie -> interpolacja -> widmo ->
    gradienty/wirowosc -> defekty -> rezonans. Rzuca w gore kazdy blad z
    warstw nizej (siec, za malo punktow do interpolacji, itd.) - webapp/
    app.py zamienia to na czytelny komunikat HTTP zamiast cichego
    pol-wyniku."""
    points: list[GridPoint] = build_grid_points(center_lat, center_lon, n=grid_n_source, spacing_deg=spacing_deg)
    records = fetch_grid_snapshot(points)
    return analyze_records(records, grid_n_membrane=grid_n_membrane)


def analyze_records(records: list[dict], grid_n_membrane: int = DEFAULT_GRID_N_MEMBRANE) -> AnalyzeResult:
    """Czesc pipeline'u NIEZALEZNA od sieci (od `records` w dol) -
    wydzielona osobno, zeby testy mogly ja wywolac na syntetycznych/
    zlapanych-z-fixture rekordach bez potrzeby prawdziwego polaczenia
    sieciowego (patrz tests/test_analyze.py)."""
    membrane = build_membrane(records, grid_n=grid_n_membrane)

    t_spec = spectrum_mod.radial_power_spectrum(membrane.temperature_c, dx=membrane.dx_deg)
    p_spec = spectrum_mod.radial_power_spectrum(membrane.pressure_hpa, dx=membrane.dx_deg)

    grad_t = spectrum_mod.gradient_magnitude(membrane.temperature_c, membrane.dx_deg, membrane.dy_deg)
    grad_p = spectrum_mod.gradient_magnitude(membrane.pressure_hpa, membrane.dx_deg, membrane.dy_deg)
    grad_precip = spectrum_mod.gradient_magnitude(membrane.precip_mm, membrane.dx_deg, membrane.dy_deg)
    vort = spectrum_mod.vorticity(membrane.u_wind_kmh, membrane.v_wind_kmh, membrane.dx_deg, membrane.dy_deg)
    wind_coh = spectrum_mod.wind_direction_coherence(membrane.u_wind_kmh, membrane.v_wind_kmh)

    # Defekt opadu liczony na GRADIENCIE opadu (krawedz strefy opadowej),
    # NIE na samej wartosci opadu - proba progu na wartosci (mediana+k*MAD)
    # zawodzi, gdy "wysoki opad" zajmuje duza (np. polowa) czesc membrany,
    # bo wtedy to nie jest mniejszosciowy outlier w sensie MAD (znalezione
    # przy pisaniu tests/test_analyze.py - patrz tamten plik). Krawedz
    # strefy opadowej JEST punktowa/liniowa anomalia, wiec jej gradient
    # pasuje do tej samej logiki co gradient T/P.
    defects_t = defects_mod.detect_defects(grad_t, k=DEFECT_K)
    defects_p = defects_mod.detect_defects(grad_p, k=DEFECT_K)
    defects_vort = defects_mod.detect_defects(np.abs(vort), k=DEFECT_K)
    defects_precip = defects_mod.detect_defects(grad_precip, k=DEFECT_K)

    res = resonance_mod.compute_resonance({
        "gradient_T": defects_t.is_defect,
        "gradient_P": defects_p.is_defect,
        "wirowosc": defects_vort.is_defect,
        "opad": defects_precip.is_defect,
    }, k=RESONANCE_K)

    return AnalyzeResult(
        n_source_points=len(records), grid_n_membrane=grid_n_membrane,
        temperature_spectrum=t_spec, pressure_spectrum=p_spec,
        gradient_t=grad_t, gradient_p=grad_p, vorticity=vort,
        wind_coherence=wind_coh,
        defects_t=defects_t, defects_p=defects_p, defects_vort=defects_vort,
        defects_precip=defects_precip, resonance=res, membrane=membrane,
    )


def result_to_json(result: AnalyzeResult) -> dict:
    """Serializacja do JSON dla API - siatki 2D sprowadzone do list list
    (zaokraglone, zeby JSON nie pekal na float64 precyzji), macierze
    bool na 0/1 (latwiejsze do narysowania jako heatmapa w JS bez
    specjalnej obslugi bool w kazdej przegladarce)."""
    def _grid(arr: np.ndarray, ndigits: int = 2) -> list:
        return np.round(arr, ndigits).tolist()

    def _bool_grid(arr: np.ndarray) -> list:
        return arr.astype(int).tolist()

    return {
        "n_source_points": result.n_source_points,
        "grid_n_membrane": result.grid_n_membrane,
        "interpolation_method": result.membrane.interpolation_method,
        "membrane": {
            "lat": _grid(result.membrane.lat_grid, 3),
            "lon": _grid(result.membrane.lon_grid, 3),
            "temperature_c": _grid(result.membrane.temperature_c),
            "pressure_hpa": _grid(result.membrane.pressure_hpa),
            "precip_mm": _grid(result.membrane.precip_mm),
        },
        "spectrum": {
            "temperature": {
                "freqs": _grid(result.temperature_spectrum.freqs, 4),
                "power": _grid(result.temperature_spectrum.power, 2),
                "high_freq_fraction": round(result.temperature_spectrum.high_freq_fraction, 4),
            },
            "pressure": {
                "freqs": _grid(result.pressure_spectrum.freqs, 4),
                "power": _grid(result.pressure_spectrum.power, 2),
                "high_freq_fraction": round(result.pressure_spectrum.high_freq_fraction, 4),
            },
        },
        "wind_coherence": {
            "coherence": round(result.wind_coherence.coherence, 4),
            "mean_direction_deg": round(result.wind_coherence.mean_direction_deg, 1),
            "n_valid": result.wind_coherence.n_valid,
        },
        "defects": {
            "gradient_T": {"n": result.defects_t.n_defects, "threshold": round(result.defects_t.threshold, 4), "mask": _bool_grid(result.defects_t.is_defect)},
            "gradient_P": {"n": result.defects_p.n_defects, "threshold": round(result.defects_p.threshold, 4), "mask": _bool_grid(result.defects_p.is_defect)},
            "wirowosc": {"n": result.defects_vort.n_defects, "threshold": round(result.defects_vort.threshold, 4), "mask": _bool_grid(result.defects_vort.is_defect)},
            "opad": {"n": result.defects_precip.n_defects, "threshold": round(result.defects_precip.threshold, 4), "mask": _bool_grid(result.defects_precip.is_defect)},
        },
        "resonance": {
            "k": result.resonance.k,
            "n_resonance_cells": result.resonance.n_resonance_cells,
            "coincidence_count": _bool_grid(result.resonance.coincidence_count),
            "mask": _bool_grid(result.resonance.is_resonance),
        },
    }
