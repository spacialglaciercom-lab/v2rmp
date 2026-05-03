# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-03

### Added
- Initial release of v2rmp (rmpca)
- Interactive TUI for route optimization workflow
- **Extract road networks from Overture Maps S3 (Parquet format)**
- **Extract road networks from OpenStreetMap PBF files**
- **Cached maps browser - automatically scans for .rmp files in current directory**
- Compile GeoJSON to binary `.rmp` format with CRC32 integrity
- Optimize routes using Chinese Postman Problem algorithm
- Eulerian circuit finding with Hierholzer's algorithm
- Turn penalty support (left, right, u-turn)
- Depot location support for route optimization
- Oneway street handling (ignore/respect/reverse modes)
- Concurrent S3 file processing (up to 10 parallel downloads)
- Node deduplication with 1-meter precision snapping
- Comprehensive keyboard shortcuts for TUI navigation
- Command-line extraction tool (`rmpca-extract`)
- Binary format with ~90% compression vs GeoJSON
- Efficiency metrics and turn statistics in optimization output

### Technical Details
- Built with `ratatui` 0.29 for TUI
- Uses `crossterm` 0.28 for terminal handling
- Parquet/Arrow integration for Overture Maps data
- **OSM PBF parsing with `osmpbf` 0.3 crate**
- Async/await with `tokio` runtime
- GeoJSON and WKB geometry support
- CRC32 checksums for data integrity
- **Filesystem scanning for cached maps**

### Features
- ✅ Overture Maps S3 extraction (fully functional)
- ✅ OpenStreetMap PBF extraction (fully functional)
- ✅ Cached maps browser (fully functional)
- ✅ GeoJSON to .rmp compilation (fully functional)
- ✅ Route optimization with CPP algorithm (fully functional)
- ✅ Interactive TUI (fully functional)

[Unreleased]: https://github.com/yourusername/v2rmp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/v2rmp/releases/tag/v0.1.0