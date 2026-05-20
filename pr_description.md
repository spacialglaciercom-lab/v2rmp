🧹 [Fix unused local Config class in re_export_onnx.py]

🎯 **What:** Added missing test coverage for `_geom_to_shapely` in `vector_clean.py`. The original implementation had an unchecked generic exception during GeoJSON loading that was completely untested, particularly for invalid coordinates format.
📊 **Coverage:** Covered 3 cases for `_geom_to_shapely`:
  - `test_geom_to_shapely_valid`: tests the happy path with valid coordinates.
  - `test_geom_to_shapely_invalid_no_coords`: tests when coordinates are missing.
  - `test_geom_to_shapely_invalid_shape`: tests when invalid data (like strings instead of numbers) is passed, checking the generic Exception block.
✨ **Result:** Improved reliability of the pipeline by verifying that invalid GeoJSON dicts triggering shapely errors will securely return `None` as intended instead of bubbling up unexpected errors.
