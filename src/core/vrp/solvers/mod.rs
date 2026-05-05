//! VRP solver modules.

pub mod clarke_wright;
pub mod sweep;
pub mod two_opt;
pub mod or_opt;
pub mod default;
// NOTE: ortools solver requires a Python process or Google OR-Tools C++ bindings.
// For a pure-Rust build, ortools is not included. See the `server` module for
// an HTTP-bridge approach that delegates to the Python backend.