import pytest
from vector_clean import _geom_to_shapely
from shapely.geometry import Point, LineString

def test_geom_to_shapely_valid():
    geom = {'type': 'Point', 'coordinates': [1.0, 2.0]}
    shp = _geom_to_shapely(geom)
    assert isinstance(shp, Point)
    assert shp.x == 1.0
    assert shp.y == 2.0

def test_geom_to_shapely_invalid_no_coords():
    geom = {'type': 'Point'}
    assert _geom_to_shapely(geom) is None

def test_geom_to_shapely_invalid_shape():
    geom = {'type': 'Point', 'coordinates': ["not", "a", "number"]}
    assert _geom_to_shapely(geom) is None
