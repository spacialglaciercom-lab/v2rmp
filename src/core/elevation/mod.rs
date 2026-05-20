mod fuel;
mod types;

#[cfg(feature = "gdal-support")]
pub mod local;
mod types;

pub use fuel::FuelCalculator;
pub use types::*;

#[cfg(feature = "gdal-support")]
pub use local::LocalDem;
pub use types::*;
