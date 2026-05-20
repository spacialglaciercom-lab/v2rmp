from pydantic import BaseModel
from typing import Any

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any]

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]

def _haversine_km_vectorized(*args, **kwargs):
    pass
