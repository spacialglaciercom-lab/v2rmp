mod fuel;
mod types;

#[cfg(feature = "extract")]
pub mod local;

pub use fuel::FuelCalculator;
pub use types::*;

#[cfg(feature = "extract")]
pub use local::LocalDem;
