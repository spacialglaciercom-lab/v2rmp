//! Overture Maps extraction module

pub mod s3_extractor;

pub use s3_extractor::{BBox, Geometry, OvertureExtractor, OvertureSegment};
