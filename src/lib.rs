//! MTVanityKey: vanity key generation for mesh radios.
//!
//! Two independent search flows share one engine (CPU workers plus an
//! optional Metal or CUDA GPU worker, every hit re-derived on the host with
//! curve25519-dalek before it is reported):
//!
//! * [`meshtastic`]: an X25519 identity key whose Meshtastic node number
//!   (CRC-32 of the public key, firmware 2.8 and later) matches a pattern.
//! * [`meshcore`]: an Ed25519 identity key whose public key starts with a
//!   hex prefix (MeshCore).
//!
//! [`search`] drives either flow to a requested number of hits; [`server`]
//! exposes both behind an HTTP JSON API with a FIFO job queue.

pub mod ed_walk;
pub mod gpu;
pub mod keygen;
pub mod mesh_walk;
pub mod meshcore;
pub mod meshtastic;
pub mod prefix;
pub mod progress;
pub mod search;
pub mod server;
pub mod table;

#[cfg(test)]
mod rng_audit_tests;

use std::sync::atomic::{AtomicU64, Ordering};

/// Attempt counters shared by the workers of one search (CPU and GPU
/// candidates counted separately for the progress line).
#[derive(Default, Debug)]
pub struct Counters {
    pub cpu_attempts: AtomicU64,
    pub gpu_attempts: AtomicU64,
}

impl Counters {
    pub fn total(&self) -> u64 {
        self.cpu_attempts.load(Ordering::Relaxed) + self.gpu_attempts.load(Ordering::Relaxed)
    }
}
