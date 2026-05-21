import numpy as np
from typing import Any
from pydantic import BaseModel

class GeoJSONFeature(BaseModel):
    type: str
    geometry: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None

class GeoJSONFeatureCollection(BaseModel):
    type: str
    features: list[GeoJSONFeature]

def _haversine_km_vectorized(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> Any:
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.asin(np.sqrt(a))
    return R * c
