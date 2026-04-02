import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ZONES = {
    "Adyar": {"lat": 13.0033, "lon": 80.2555, "pop": 250000, "icu": 50, "doc": 100, "o2": 200},
    "Anna Nagar": {"lat": 13.0836, "lon": 80.2110, "pop": 300000, "icu": 80, "doc": 150, "o2": 300},
    "Tondiarpet": {"lat": 13.1251, "lon": 80.2976, "pop": 200000, "icu": 30, "doc": 60, "o2": 150},
    "Mylapore": {"lat": 13.0336, "lon": 80.2743, "pop": 220000, "icu": 45, "doc": 90, "o2": 180},
    "Velachery": {"lat": 12.9750, "lon": 80.2212, "pop": 260000, "icu": 60, "doc": 110, "o2": 250},
    "T. Nagar": {"lat": 13.0405, "lon": 80.2337, "pop": 240000, "icu": 55, "doc": 105, "o2": 220},
}

RESOURCE_TOTALS = {
    "icu": sum(z["icu"] for z in ZONES.values()),
    "doc": sum(z["doc"] for z in ZONES.values()),
    "o2": sum(z["o2"] for z in ZONES.values())
}

def get_zone_metadata(zone):
    return ZONES.get(zone)

def generate_historical_data(days=180):
    np.random.seed(42)  # For reproducibility during demo
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date)
    
    records = []
    # Using consistent base trends for each zone
    zone_base_multiplier = {"Adyar": 1.2, "Anna Nagar": 1.5, "Tondiarpet": 2.5, "Mylapore": 1.0, "Velachery": 1.3, "T. Nagar": 1.1}
    
    for zone, meta in ZONES.items():
        base_cases = np.random.randint(5, 10) * zone_base_multiplier[zone]
        
        # Seasonal trend simulation (peaks mimicking winter/monsoon)
        t = np.arange(days)
        seasonal = 15 * np.sin(2 * np.pi * t / 60)
        noise = np.random.normal(0, 3, days)
        cases = np.maximum(0, base_cases + seasonal + noise).astype(int)
        
        for i, dt in enumerate(dates):
            records.append({
                "date": dt,
                "zone": zone,
                "cases": cases[i],
                "pop": meta["pop"],
                "icu_capacity": meta["icu"],
                "doctors_capacity": meta["doc"],
                "oxygen_capacity": meta["o2"]
            })
    return pd.DataFrame(records)
