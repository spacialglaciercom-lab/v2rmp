# v2rmp Backward Compatibility Mandates

This document defines the foundational requirements for maintaining backward compatibility across the **v2rmp** codebase. All agents MUST adhere to these rules when modifying the system.

## 1. Machine Learning & Data Integrity
*   **Feature Parity:** Any change to `src/core/ml/features.rs` or feature extraction logic MUST be verified against existing models. If a change breaks the feature vector schema, you must provide a versioning mechanism or a migration path.
*   **Model Compatibility:** Ensure that `src/core/neural_routing.rs` and related inference logic maintain support for existing `.onnx` models located in `models/`.
*   **Data Schema:** The schema for training data (`.jsonl` files in `local/` and `sft_training/`) must remain stable. Do not change existing fields; only add optional ones.

## 2. MCP (Model Context Protocol) Stability
*   **Tool Signatures:** The names and parameter schemas of tools exposed in `src/bin/rmpca-mcp-server.rs` and other MCP servers are public APIs. Do not rename tools or change required parameters without a deprecation cycle.
*   **Configuration:** Maintain the structure of `mcp_config.json`. New configuration options should be optional with sensible defaults.

## 3. CLI & Automation
*   **CLI Flags:** Flags and arguments defined in `src/cli.rs` must remain functional. If a flag is replaced, it should remain as a hidden alias to prevent breaking existing automation scripts (e.g., `scripts/train_models_ci.sh`).
*   **Directory Structure:** The expected locations for models, embeddings, and data files are part of the system's contract. Do not move these without updating all associated scripts and documentation.

## 4. Rust Core API
*   **Public Modules:** Maintain stable interfaces for modules in `src/core/`, particularly `optimize.rs` and `geo_types.rs`.
*   **Trait Stability:** Traits used for VRP solvers and graph embeddings should avoid breaking changes to method signatures.

## Validation Requirement
Before finalizing any change that might impact these areas, agents MUST:
1. Run existing tests (`cargo test`).
2. Verify that existing training data can still be parsed.
3. If possible, run a sample inference using a pre-trained model to ensure feature alignment.
