"""NREL SAM (PySAM): PVWatts v8 for a 4 kW system, one clear-sky-like day given as weather data (hourly
GHI from a sine, DNI, DHI, 20 °C, 1 m/s); returns the annual-form AC energy and the peak AC power."""
import math

import PySAM.Pvwattsv8 as pv


def pvwatts(system_capacity=4.0, tilt=20.0, azimuth=180.0, losses=14.0):
    ghi = [max(0.0, 900.0 * math.sin(math.pi * (h - 6) / 12)) for h in range(24)] * 365
    m = pv.default("PVWattsNone")
    m.SolarResource.solar_resource_data = {
        "lat": 40.0, "lon": -105.0, "tz": -7, "elev": 1600.0,
        "year": [2019] * 8760, "month": [mo for mo, n in enumerate([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], 1) for _ in range(24 * n)],
        "day": [d for n in [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] for d in range(1, n + 1) for _ in range(24)],
        "hour": list(range(24)) * 365, "minute": [0] * 8760,
        "gh": ghi, "dn": [0.8 * g for g in ghi], "df": [0.2 * g for g in ghi], "tdry": [20.0] * 8760, "wspd": [1.0] * 8760}
    m.SystemDesign.system_capacity = system_capacity
    m.SystemDesign.tilt = tilt
    m.SystemDesign.azimuth = azimuth
    m.SystemDesign.losses = losses
    m.execute()
    return m.Outputs.ac_annual, max(m.Outputs.ac)
