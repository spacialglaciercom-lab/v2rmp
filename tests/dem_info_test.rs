/// Integration test for the `dem_info` MCP tool and underlying `LocalDem::info()`.
/// Uses the existing n45_w074_1arc_v3.tif (Montreal area, 1-arcsecond SRTM).

#[cfg(all(test, feature = "extract"))]
mod dem_info_tests {
    use std::path::Path;
    use v2rmp::core::elevation::local::LocalDem;

    const DEM_PATH: &str = "../rmp.ca/n45_w074_1arc_v3.tif";

    #[test]
    fn test_dem_info_opens_and_returns_metadata() {
        let path = Path::new(DEM_PATH);
        assert!(
            path.exists(),
            "DEM file not found at {}. Download it first.",
            path.display()
        );

        let dem = LocalDem::open(path)
            .expect("Failed to open DEM via GDAL — check GDAL installation");

        let info = dem.info();

        eprintln!("── DEM Info ──");
        eprintln!("  Width:      {}", info.width);
        eprintln!("  Height:     {}", info.height);
        eprintln!("  Pixel X:    {}", info.pixel_size_x);
        eprintln!("  Pixel Y:    {}", info.pixel_size_y);
        eprintln!("  NoData:     {:?}", info.nodata);
        eprintln!("  BBox:");
        eprintln!("    min_lon:  {}", info.bbox.min_lon);
        eprintln!("    min_lat:  {}", info.bbox.min_lat);
        eprintln!("    max_lon:  {}", info.bbox.max_lon);
        eprintln!("    max_lat:  {}", info.bbox.max_lat);

        // Basic sanity checks
        assert!(info.width > 0, "Width should be positive");
        assert!(info.height > 0, "Height should be positive");
        assert!(info.pixel_size_x > 0.0, "Pixel size X should be positive");
        assert!(info.pixel_size_y > 0.0, "Pixel size Y should be positive");
        assert!(info.bbox.min_lon < info.bbox.max_lon, "Longitude range invalid");
        assert!(info.bbox.min_lat < info.bbox.max_lat, "Latitude range invalid");

        // This is a 1-arcsecond DEM (~30m) so pixel size should be ~0.000278°
        assert!(
            info.pixel_size_x < 0.001,
            "Expected sub-degree pixel size for 1-arcsecond DEM, got {}",
            info.pixel_size_x
        );
    }

    #[test]
    fn test_dem_info_bbox_covers_montreal_area() {
        let path = Path::new(DEM_PATH);
        if !path.exists() {
            eprintln!("Skipping: DEM file not found");
            return;
        }

        let dem = LocalDem::open(path).expect("Failed to open DEM");
        let info = dem.info();

        // The file is named n45_w074 — should cover ~45°N, ~74°W (Montreal)
        assert!(
            info.bbox.min_lat <= 45.0 && info.bbox.max_lat >= 45.0,
            "Expected DEM to cover latitude 45° (Montreal), got [{}, {}]",
            info.bbox.min_lat,
            info.bbox.max_lat
        );
        assert!(
            info.bbox.min_lon <= -74.0 && info.bbox.max_lon >= -74.0,
            "Expected DEM to cover longitude -74° (Montreal), got [{}, {}]",
            info.bbox.min_lon,
            info.bbox.max_lon
        );
    }
}
