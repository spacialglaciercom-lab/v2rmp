mod fuel;
mod types;

#[cfg(feature = "gdal-support")]
pub mod local;

pub use fuel::FuelCalculator;
pub use types::*;

#[cfg(feature = "gdal-support")]
pub use local::LocalDem;
