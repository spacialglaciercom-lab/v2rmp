## 2024-05-19 - Test failure in `dem_info_tests::test_dem_info_opens_and_returns_metadata`
**Learning:** `test_dem_info_opens_and_returns_metadata` fails when a `.tif` file is missing, which is an expected behavior in environments where the map data is not pre-downloaded, as mentioned in `.jules/bolt.md`
**Action:** Ignore this specific failure.
