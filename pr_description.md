🧪 [testing improvement description]

🎯 **What:** Added missing test coverage for error paths in `src/core/compile.rs`'s `run_compile` function. Specifically, checking that providing a non-existent input GeoJSON file correctly returns an `anyhow::Result::Err`.
📊 **Coverage:** The error path scenario where the input GeoJSON file fails to open is now explicitly tested.
✨ **Result:** Improved test coverage and validated that the application correctly handles and reports invalid input file paths without panicking.
