mod types;
mod fuel;

#[cfg(feature = "gdal-support")]
pub mod local;

pub use types::*;
pub use fuel::FuelCalculator;

#[cfg(feature = "gdal-support")]
pub use local::LocalDem;
