import re

with open("src/core/compile.rs", "r") as f:
    content = f.read()

replacement = """    #[test]
    fn test_run_compile_error_on_missing_input() {
        let req = CompileRequest {
            input_geojson: "non_existent_file.geojson".to_string(),
            output_rmp: "output.rmp".to_string(),
            compress: false,
            road_classes: vec![],
            clean_options: None,
            prune_disconnected: false,
        };
        let result = run_compile(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to open input GeoJSON"));
    }

    #[test]
    fn test_is_rmp_file() {
        // Valid scenarios
        assert!(is_rmp_file(b"RMP1"));
        assert!(is_rmp_file(b"RMP1_and_more_data"));
        assert!(is_rmp_file(RMP_MAGIC));

        // Invalid scenarios
        assert!(!is_rmp_file(b""));
        assert!(!is_rmp_file(b"RMP"));
        assert!(!is_rmp_file(b"RMP2"));
        assert!(!is_rmp_file(b"XYZ1"));
    }
}
"""

content = re.sub(r'    #\[test\]\n    fn test_run_compile_error_on_missing_input\(\) \{\n        let req = CompileRequest \{\n            input_geojson: "non_existent_file.geojson".to_string\(\),\n            output_rmp: "output.rmp".to_string\(\),\n            compress: false,\n            road_classes: vec\!\[\],\n            clean_options: None,\n            prune_disconnected: false,\n        \};\n        let result = run_compile\(&req\);\n        assert\!\(result.is_err\(\)\);\n        assert\!\(result.unwrap_err\(\).to_string\(\).contains\("Failed to open input GeoJSON"\)\);\n    \}\n\}', replacement, content)

with open("src/core/compile.rs", "w") as f:
    f.write(content)
