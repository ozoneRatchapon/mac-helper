//! Concrete DraftModel implementations.
//!
//! Each draft model provides a different fluency prior for the decode loop.
//! The pruner enforces correctness regardless of which draft model is used.

pub mod ngram;
pub mod uniform;

pub use ngram::NgramDraftModel;
pub use uniform::UniformDraftModel;
