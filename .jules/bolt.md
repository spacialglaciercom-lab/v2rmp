## 2024-05-14 - Extract matched pattern helper for clean code
**Learning:** We extracted deeply nested Enum `match` in `close_file_browser` into its own helper `apply_file_selection` with earlier `else return` clauses to make `src/app.rs` more readable without altering functionality.
**Action:** Extract repetitive assignment/logging handlers into distinct, semantically-named functions whenever indentation creeps beyond 3 or 4 levels to keep method lengths manageable.
