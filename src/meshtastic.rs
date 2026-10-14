//! Meshtastic node-ID vanity mode.
//!
//! Since firmware 2.8 (commit 8267bb22 "Packet Signing via XEdDSA";
//! `NodeDB::createNewIdentity`, `Router::verifyFirstContactNodeInfo`) a
//! Meshtastic node's number is `crc32Buffer(public_key, 32)` of its X25519
//! identity key: ErriezCRC32's standard reflected CRC-32 (polynomial
//! 0xEDB88320, init 0xFFFFFFFF, final xor 0xFFFFFFFF) over the 32 public-key
//! bytes, printed as `!%08x`. Node numbers below `NUM_RESERVED` (4) and the
//! broadcast address 0xFFFFFFFF are rejected by `pickNewNodeNum`, which then
//! picks a random number instead.
//!
//! The private key is a clamped 32-byte X25519 secret; the public key is
//! X25519(secret, 9), the little-endian Montgomery u-coordinate of
//! `secret * B`. The search walks `base + 8k` exactly like the Ed25519 vanity
//! search (the clamp is preserved by `+8` steps) and converts every point's
//! Edwards `y` to `u = (1 + y) / (1 - y)`; see `mesh_walk` (CPU) and the
//! `mesh_kernel`s (GPU). Every hit is re-derived here with curve25519-dalek
//! and a bitwise CRC before it is reported.

#[cfg(feature = "cli")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, Result};
#[cfg(feature = "cli")]
use crossbeam_channel::Sender;
use curve25519_dalek::montgomery::MontgomeryPoint;
#[cfg(feature = "cli")]
use rand_chacha::ChaCha20Rng;

#[cfg(feature = "cli")]
use crate::mesh_walk::UWalk;
#[cfg(feature = "cli")]
use crate::Counters;

/// Node numbers the firmware refuses (`NUM_RESERVED` and `NODENUM_BROADCAST`).
pub fn is_reserved(id: u32) -> bool {
    id < 4 || id == u32::MAX
}

/// `!%08x`, the firmware's `owner.id` rendering of a node number.
pub fn node_id_str(id: u32) -> String {
    format!("!{id:08x}")
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE, reflected). Two implementations: the bitwise one is the
// verifier's independent reference (ErriezCRC32's crc32Update loop
// verbatim), the sliced one is the search path's.
// ---------------------------------------------------------------------------

const CRC32_POLY: u32 = 0xEDB8_8320;

/// Bitwise CRC-32, exactly ErriezCRC32's `crc32Buffer` as used by the firmware.
pub fn crc32_bitwise(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32_POLY & mask);
        }
    }
    !crc
}

/// Slicing-by-4 tables: `[T0 | T1 | T2 | T3]`, 256 words each. `T0` is the
/// bytewise table, `Tk[i] = (Tk-1[i] >> 8) ^ T0[Tk-1[i] & 0xff]`. The GPU
/// kernels receive exactly these words (see `gpu_tables`).
pub fn crc32_tables() -> Vec<u32> {
    let mut t = vec![0u32; 1024];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 { CRC32_POLY ^ (c >> 1) } else { c >> 1 };
        }
        t[i as usize] = c;
    }
    for k in 1..4 {
        for i in 0..256 {
            let prev = t[(k - 1) * 256 + i];
            t[k * 256 + i] = (prev >> 8) ^ t[(prev & 0xFF) as usize];
        }
    }
    t
}

/// CRC-32 of 32 bytes given as eight little-endian words, slicing-by-4
/// (the hot-path form shared with the GPU kernels).
#[inline]
pub fn crc32_words8(words: &[u32; 8], tables: &[u32]) -> u32 {
    debug_assert!(tables.len() >= 1024);
    let mut crc = 0xFFFF_FFFFu32;
    for &w in words {
        crc ^= w;
        crc = tables[768 + (crc & 0xFF) as usize]
            ^ tables[512 + ((crc >> 8) & 0xFF) as usize]
            ^ tables[256 + ((crc >> 16) & 0xFF) as usize]
            ^ tables[(crc >> 24) as usize];
    }
    !crc
}

/// The CPU workers' CRC: the AArch64 CRC32 instructions when the core has
/// them (one `crc32x` per 8 bytes; Apple silicon does), else slicing-by-4.
pub struct Crc32 {
    tables: Vec<u32>,
    /// Whether the AArch64 CRC32 instructions are available (always false
    /// elsewhere, where only the table path exists).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    hw: bool,
}

impl Crc32 {
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        let hw = std::arch::is_aarch64_feature_detected!("crc");
        #[cfg(not(target_arch = "aarch64"))]
        let hw = false;
        Self {
            tables: crc32_tables(),
            hw,
        }
    }

    /// Whether the hardware path is in use (asserted by tests).
    #[cfg(test)]
    pub fn is_hardware(&self) -> bool {
        self.hw
    }

    #[inline]
    pub fn of_bytes32(&self, bytes: &[u8; 32]) -> u32 {
        #[cfg(target_arch = "aarch64")]
        if self.hw {
            // SAFETY: `crc` was detected at construction.
            return unsafe { crc32_bytes32_aarch64(bytes) };
        }
        crc32_words8(&bytes_to_words8(bytes), &self.tables)
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// `crc32x` over the four 8-byte words: the same reflected CRC-32 (the
/// AArch64 CRC32 instructions implement exactly the IEEE polynomial with
/// the reflected bit order).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32_bytes32_aarch64(bytes: &[u8; 32]) -> u32 {
    use std::arch::aarch64::__crc32d;
    let mut crc = 0xFFFF_FFFFu32;
    for chunk in bytes.chunks_exact(8) {
        crc = __crc32d(crc, u64::from_le_bytes(chunk.try_into().expect("8 bytes")));
    }
    !crc
}

/// Eight little-endian words of a 32-byte little-endian encoding.
#[inline]
pub fn bytes_to_words8(bytes: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        w[i] = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
    }
    w
}

// ---------------------------------------------------------------------------
// Pattern: the node-ID targets, mirroring vanity.freq51.net's parsePattern.
// ---------------------------------------------------------------------------

/// Maximum targets per pattern (also the GPU parameter block's capacity).
pub const MAX_TARGETS: usize = 64;

/// Longest accepted pattern text: 64 targets of `!xxxxxxxx,`.
pub const MAX_PATTERN_CHARS: usize = MAX_TARGETS * 10;

/// Words of the GPU membership bitmap (4096 bits).
pub const BITMAP_WORDS: usize = 128;

/// A set of node-ID targets that all fix the same hex digits: `mask` has
/// 0xF in every fixed digit, `targets` are the fixed digits' values (zero
/// elsewhere), deduplicated in input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeIdPattern {
    pub mask: u32,
    pub targets: Vec<u32>,
    text: String,
}

impl NodeIdPattern {
    /// Parse a comma-separated list of up to 64 targets. Each target is up
    /// to 8 hex digits, optionally prefixed with `!`, with `*`, `?`, `.`,
    /// `x` or `X` as wildcards; shorter targets are prefixes (their digits
    /// are the high nibbles). All targets must fix the same digit positions.
    /// When all 8 digits are fixed, node numbers the firmware reserves
    /// (< 4 and 0xFFFFFFFF) are rejected.
    pub fn parse(input: &str) -> Result<Self> {
        let text = input.trim();
        if text.is_empty() {
            return Err(anyhow!("enter a pattern"));
        }
        // The reference site caps its input box at 200 characters; a CLI list of 64
        // full ids with `!` prefixes is 640, so allow exactly that.
        if text.chars().count() > MAX_PATTERN_CHARS {
            return Err(anyhow!("pattern is too long"));
        }
        let items: Vec<&str> = text
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if items.is_empty() {
            return Err(anyhow!("enter a pattern"));
        }
        if items.len() > MAX_TARGETS {
            return Err(anyhow!("at most {MAX_TARGETS} targets, got {}", items.len()));
        }
        let mut mask: Option<u32> = None;
        let mut targets: Vec<u32> = Vec::new();
        for item in &items {
            let digits = item.strip_prefix('!').unwrap_or(item);
            if digits.is_empty() {
                return Err(anyhow!("empty target in the list"));
            }
            let n = digits.chars().count();
            if n > 8 {
                return Err(anyhow!(
                    "\"{item}\" is {n} characters; a node id is 8 hex digits"
                ));
            }
            let mut value = 0u32;
            let mut fixed = 0u32;
            for ch in digits.chars() {
                value <<= 4;
                fixed <<= 4;
                if "*?.xX".contains(ch) {
                    continue;
                }
                let digit = ch.to_digit(16).ok_or_else(|| {
                    anyhow!("\"{ch}\" is not a hex digit. Use 0-9, a-f, or * for any digit")
                })?;
                value |= digit;
                fixed |= 0xF;
            }
            let shift = 4 * (8 - n as u32);
            value <<= shift;
            fixed <<= shift;
            match mask {
                None => mask = Some(fixed),
                Some(m) if m != fixed => {
                    return Err(anyhow!("every target in a set must fix the same digits"))
                }
                Some(_) => {}
            }
            let target = value & fixed;
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        let mask = mask.expect("at least one target");
        if mask == 0 {
            return Err(anyhow!("that matches everything: fix at least one digit"));
        }
        if mask == u32::MAX {
            for &t in &targets {
                if is_reserved(t) {
                    return Err(anyhow!("{} is reserved by the firmware", node_id_str(t)));
                }
            }
        }
        Ok(Self {
            mask,
            targets,
            text: items.join(","),
        })
    }

    /// The normalised pattern text (targets joined by commas).
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Keys expected per hit: 2^(fixed bits) / targets.
    pub fn expected_attempts(&self) -> f64 {
        2f64.powi(self.mask.count_ones() as i32) / self.targets.len() as f64
    }

    pub fn match_probability(&self) -> f64 {
        1.0 / self.expected_attempts()
    }

    /// Whether node number `id` satisfies the pattern and is usable by the
    /// firmware (the reserved numbers are never a hit, whatever the mask).
    pub fn matches(&self, id: u32) -> bool {
        !is_reserved(id) && self.targets.contains(&(id & self.mask))
    }

    /// Bit of the membership bitmap for a masked node number.
    #[inline]
    pub fn bitmap_bit(masked: u32) -> u32 {
        masked.wrapping_mul(0x9E37_79B1) >> 20
    }

    /// 4096-bit bitmap over `bitmap_bit(target)`: a candidate whose bit is
    /// clear is certainly not a target (the exact test runs on the rest).
    pub fn bitmap(&self) -> [u32; BITMAP_WORDS] {
        let mut bits = [0u32; BITMAP_WORDS];
        for &t in &self.targets {
            let b = Self::bitmap_bit(t);
            bits[(b >> 5) as usize] |= 1 << (b & 31);
        }
        bits
    }

    /// Fast-path membership test (bitmap, then exact), as the kernels do it.
    #[inline]
    pub fn quick_matches(&self, id: u32, bitmap: &[u32; BITMAP_WORDS]) -> bool {
        let masked = id & self.mask;
        let b = Self::bitmap_bit(masked);
        if (bitmap[(b >> 5) as usize] >> (b & 31)) & 1 == 0 {
            return false;
        }
        self.matches(id)
    }

    /// The GPU table buffer: the four CRC slicing tables (1024 words) then
    /// the membership bitmap (128 words).
    pub fn gpu_tables(&self) -> Vec<u32> {
        let mut words = crc32_tables();
        words.extend_from_slice(&self.bitmap());
        debug_assert_eq!(words.len(), GPU_TABLE_WORDS);
        words
    }

    /// Target words for the GPU parameter block (zero-padded to 64).
    pub fn gpu_targets(&self) -> [u32; MAX_TARGETS] {
        let mut out = [0u32; MAX_TARGETS];
        out[..self.targets.len()].copy_from_slice(&self.targets);
        out
    }
}

/// Words of `NodeIdPattern::gpu_tables`.
pub const GPU_TABLE_WORDS: usize = 1024 + BITMAP_WORDS;

// ---------------------------------------------------------------------------
// Key derivation and hit verification (the independent reference path).
// ---------------------------------------------------------------------------

/// A verified Meshtastic identity: clamped X25519 secret, public key and
/// the node number the firmware derives from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshKey {
    pub private_key: [u8; 32],
    pub public_key: [u8; 32],
    pub node_id: u32,
    pub source: &'static str,
}

impl MeshKey {
    pub fn node_id_str(&self) -> String {
        node_id_str(self.node_id)
    }

    pub fn private_key_hex(&self) -> String {
        hex::encode(self.private_key)
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(self.public_key)
    }

    pub fn private_key_base64(&self) -> String {
        base64_encode(&self.private_key)
    }

    pub fn public_key_base64(&self) -> String {
        base64_encode(&self.public_key)
    }

    /// The CLI command that installs this identity on a node.
    pub fn meshtastic_command(&self) -> String {
        format!(
            "meshtastic --set security.private_key \"base64:{}\"",
            self.private_key_base64()
        )
    }
}

/// Standard base64 with padding.
pub fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// X25519 clamp (`b[0] &= 248; b[31] &= 127; b[31] |= 64`), the firmware's
/// and the reference implementation's secret-key form.
pub fn is_clamped(secret: &[u8; 32]) -> bool {
    secret[0] & 7 == 0 && secret[31] & 0x80 == 0 && secret[31] & 0x40 != 0
}

/// X25519 public key of a clamped secret: the Montgomery u-coordinate of
/// `secret * B`, little-endian (curve25519-dalek).
pub fn public_key(secret: &[u8; 32]) -> [u8; 32] {
    MontgomeryPoint::mul_base_clamped(*secret).to_bytes()
}

/// The node number the firmware derives from a public key.
pub fn node_id_of(public_key: &[u8; 32]) -> u32 {
    crc32_bitwise(public_key)
}

/// Public key and node number of a secret.
pub fn derive(secret: &[u8; 32]) -> ([u8; 32], u32) {
    let pk = public_key(secret);
    (pk, node_id_of(&pk))
}

/// Re-derive a reported candidate independently (dalek + bitwise CRC) and
/// accept it only if the secret is clamped and the node number matches the
/// pattern; `None` otherwise. Every worker's hits pass through here.
pub fn verify_hit(pattern: &NodeIdPattern, secret: [u8; 32], source: &'static str) -> Option<MeshKey> {
    if !is_clamped(&secret) {
        return None;
    }
    let (public_key, node_id) = derive(&secret);
    if !pattern.matches(node_id) {
        return None;
    }
    Some(MeshKey {
        private_key: secret,
        public_key,
        node_id,
        source,
    })
}

// ---------------------------------------------------------------------------
// CPU workers (the threaded engine; `stepper` runs the same walk on the
// caller's thread).
// ---------------------------------------------------------------------------

#[cfg(feature = "cli")]
const REPORT_EVERY: u64 = 4096;
pub(crate) const LANES: usize = 2;
pub(crate) const WALK_RESTART_ITERS: u64 = 1 << 24;
pub(crate) const RESYNC_WINDOWS: u64 = 1024;

#[cfg(feature = "cli")]
pub fn spawn_cpu_workers(
    threads: usize,
    pattern: Arc<NodeIdPattern>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<MeshKey>,
) -> Vec<JoinHandle<()>> {
    (0..threads)
        .map(|worker_id| {
            let pattern = Arc::clone(&pattern);
            let stop = Arc::clone(&stop);
            let counters = Arc::clone(&counters);
            let tx = tx.clone();
            thread::Builder::new()
                .name(format!("cpu-mesh-{worker_id}"))
                .spawn(move || {
                    let mut rng = crate::keygen::fresh_rng();
                    let walk = UWalk::<LANES>::new(&mut rng);
                    if walk.is_none() {
                        eprintln!("cpu: Montgomery walk disagrees with dalek; using per-candidate derivation");
                    }
                    search(walk, rng, pattern, stop, counters, tx);
                })
                .expect("failed to spawn CPU worker")
        })
        .collect()
}

/// One CPU worker: the u-only walk while it agrees with dalek (checked at
/// every restart and every `RESYNC_WINDOWS` windows), otherwise one dalek
/// derivation per candidate. Runs until `stop`; every hit is re-derived by
/// `verify_hit` before it is sent.
#[cfg(feature = "cli")]
fn search(
    mut walk: Option<UWalk<LANES>>,
    mut rng: ChaCha20Rng,
    pattern: Arc<NodeIdPattern>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    tx: Sender<MeshKey>,
) {
    let crc = Crc32::new();
    let bitmap = pattern.bitmap();
    let mut pending = 0u64;
    let mut hits: Vec<(u64, usize)> = Vec::new();
    let mut windows = 0u64;

    while !stop.load(Ordering::Relaxed) {
        if let Some(w) = walk.as_mut() {
            if w.iters_done() >= WALK_RESTART_ITERS && !w.restart(&mut rng) {
                eprintln!("cpu: Montgomery walk disagrees with dalek; using per-candidate derivation");
                walk = None;
                continue;
            }
            hits.clear();
            w.next_window(|lane, iter, u| {
                let id = crc.of_bytes32(u);
                if pattern.quick_matches(id, &bitmap) {
                    hits.push((iter, lane));
                }
            });
            hits.sort_unstable();
            pending += UWalk::<LANES>::CANDIDATES_PER_WINDOW;
            for &(iter, lane) in &hits {
                if let Some(key) = verify_hit(&pattern, w.scalar_at(lane, iter), "cpu") {
                    let _ = tx.try_send(key);
                } else {
                    eprintln!("cpu: walk reported a candidate that does not re-derive; using per-candidate derivation");
                    walk = None;
                    break;
                }
            }
            windows += 1;
            if walk.is_some() && windows % RESYNC_WINDOWS == 0 && !walk.as_ref().expect("checked").matches_dalek() {
                eprintln!("cpu: Montgomery walk desynchronised from dalek; using per-candidate derivation");
                walk = None;
            }
        } else {
            let secret = crate::keygen::random_base_scalar(&mut rng);
            pending += 1;
            if let Some(key) = verify_hit(&pattern, secret, "cpu") {
                let _ = tx.try_send(key);
            }
        }

        if pending >= REPORT_EVERY {
            counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
            pending = 0;
        }
    }
    if pending > 0 {
        counters.cpu_attempts.fetch_add(pending, Ordering::Relaxed);
    }
}

/// Reference vectors (`secret pubkey nodeid` per line) generated with the
/// reference implementation at vanity.freq51.net: the X25519 public key and
/// the CRC-32 node number of random clamped secrets plus three fixed ones
/// (all-zero, all-ones and 00..1f before clamping).
const ORACLE_VECTORS: &str = include_str!("../tests/meshtastic_oracle_vectors.txt");

/// The oracle vectors as `(secret, public key, node id)` triples.
pub fn oracle_vectors() -> Vec<([u8; 32], [u8; 32], u32)> {
    ORACLE_VECTORS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut f = line.split_whitespace();
            let s: [u8; 32] = hex::decode(f.next().unwrap()).unwrap().try_into().unwrap();
            let p: [u8; 32] = hex::decode(f.next().unwrap()).unwrap().try_into().unwrap();
            let id = u32::from_str_radix(f.next().unwrap(), 16).unwrap();
            (s, p, id)
        })
        .collect()
}

/// Check the host derivation (dalek + bitwise CRC, both CRC paths and the
/// CPU walk's Montgomery conversion) against every oracle vector; the
/// `--self-test` entry point.
pub fn self_test_host() -> Result<()> {
    let vectors = oracle_vectors();
    if vectors.len() < 200 {
        return Err(anyhow!("only {} oracle vectors", vectors.len()));
    }
    let tables = crc32_tables();
    let crc = Crc32::new();
    for (secret, pubkey, id) in &vectors {
        if !is_clamped(secret) {
            return Err(anyhow!("oracle secret {} is not clamped", hex::encode(secret)));
        }
        let (pk, node) = derive(secret);
        if &pk != pubkey || node != *id {
            return Err(anyhow!("derivation of {} disagrees with the oracle", hex::encode(secret)));
        }
        if crc32_words8(&bytes_to_words8(pubkey), &tables) != *id || crc.of_bytes32(pubkey) != *id {
            return Err(anyhow!("CRC-32 paths disagree on {}", hex::encode(pubkey)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cli")]
    use crossbeam_channel::bounded;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    #[cfg(feature = "cli")]
    use std::time::Duration;

    #[test]
    fn host_derivation_matches_oracle_vectors() {
        let vectors = oracle_vectors();
        assert!(vectors.len() >= 200);
        for (secret, pubkey, id) in &vectors {
            assert!(is_clamped(secret));
            let (pk, node) = derive(secret);
            assert_eq!(&pk, pubkey, "pubkey of {}", hex::encode(secret));
            assert_eq!(node, *id, "node id of {}", hex::encode(secret));
            // Both CRC paths agree with the oracle.
            assert_eq!(crc32_words8(&bytes_to_words8(pubkey), &crc32_tables()), *id);
            // And with the Edwards route the walks take: u of s*B via dalek.
            use curve25519_dalek::edwards::EdwardsPoint;
            let u = EdwardsPoint::mul_base_clamped(*secret).to_montgomery().to_bytes();
            assert_eq!(&u, pubkey);
        }
    }

    /// Fresh random vectors from the reference implementation: set
    /// MESHTASTIC_ORACLE_JS to a local copy of vanity.freq51.net's oracle.js
    /// (an ES module exporting clampScalar, x25519Base, crc32 and toHex)
    /// with node installed; skipped silently otherwise. Guards against the
    /// checked-in vectors going stale relative to the reference.
    #[test]
    fn host_derivation_matches_live_oracle() {
        let Ok(oracle) = std::env::var("MESHTASTIC_ORACLE_JS") else {
            return;
        };
        if !std::path::Path::new(&oracle).exists() {
            return;
        }
        let script = format!(
            "import * as o from '{oracle}'; import {{ randomBytes }} from 'node:crypto';\
             for (let i = 0; i < 64; i++) {{ const s = o.clampScalar(randomBytes(32)); const p = o.x25519Base(s);\
             console.log(o.toHex(s), o.toHex(p), o.crc32(p).toString(16).padStart(8, '0')); }}"
        );
        let Ok(out) = std::process::Command::new("node")
            .arg("--input-type=module")
            .arg("-e")
            .arg(&script)
            .output()
        else {
            return;
        };
        if !out.status.success() {
            return;
        }
        let text = String::from_utf8(out.stdout).unwrap();
        let mut n = 0;
        for line in text.lines() {
            let mut f = line.split_whitespace();
            let s: [u8; 32] = hex::decode(f.next().unwrap()).unwrap().try_into().unwrap();
            let p: [u8; 32] = hex::decode(f.next().unwrap()).unwrap().try_into().unwrap();
            let id = u32::from_str_radix(f.next().unwrap(), 16).unwrap();
            assert_eq!(derive(&s), (p, id), "secret {}", hex::encode(s));
            n += 1;
        }
        assert_eq!(n, 64);
    }

    #[test]
    fn crc32_matches_check_value_and_tables_agree() {
        // The IEEE check value.
        assert_eq!(crc32_bitwise(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32_bitwise(b""), 0);
        let tables = crc32_tables();
        let crc = Crc32::new();
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for _ in 0..2000 {
            let mut bytes = [0u8; 32];
            rand_core::RngCore::fill_bytes(&mut rng, &mut bytes);
            assert_eq!(crc32_words8(&bytes_to_words8(&bytes), &tables), crc32_bitwise(&bytes));
            assert_eq!(crc.of_bytes32(&bytes), crc32_bitwise(&bytes));
        }
        for (_, pubkey, id) in oracle_vectors() {
            assert_eq!(crc.of_bytes32(&pubkey), id);
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("crc") {
            assert!(crc.is_hardware());
        }
    }

    #[test]
    fn pattern_parser_mirrors_the_site() {
        let p = NodeIdPattern::parse("dc80").unwrap();
        assert_eq!((p.mask, p.targets.clone()), (0xFFFF_0000, vec![0xDC80_0000]));
        assert_eq!(p.expected_attempts(), 65536.0);
        let p = NodeIdPattern::parse("!DC80xxxx").unwrap();
        assert_eq!((p.mask, p.targets.clone()), (0xFFFF_0000, vec![0xDC80_0000]));
        let p = NodeIdPattern::parse("*?.xX1").unwrap();
        assert_eq!((p.mask, p.targets.clone()), (0x0000_0F00, vec![0x0000_0100]));
        let p = NodeIdPattern::parse(" dead, beef ,!cafe,dead ").unwrap();
        assert_eq!(p.mask, 0xFFFF_0000);
        assert_eq!(p.targets, vec![0xDEAD_0000, 0xBEEF_0000, 0xCAFE_0000]);
        assert_eq!(p.expected_attempts(), 65536.0 / 3.0);
        assert_eq!(p.text(), "dead,beef,!cafe,dead");
        // Trailing wildcards keep the digit positions explicit.
        let p = NodeIdPattern::parse("12**").unwrap();
        assert_eq!((p.mask, p.targets.clone()), (0xFF00_0000, vec![0x1200_0000]));
        // A full id.
        let p = NodeIdPattern::parse("!deadbeef").unwrap();
        assert_eq!((p.mask, p.targets.clone()), (0xFFFF_FFFF, vec![0xDEAD_BEEF]));
        assert_eq!(p.expected_attempts(), 4294967296.0);
        // Empty items between commas are skipped, like the reference parser.
        let p = NodeIdPattern::parse("a,,b").unwrap();
        assert_eq!(p.targets.len(), 2);

        let err = |s: &str| NodeIdPattern::parse(s).unwrap_err().to_string();
        assert_eq!(err(""), "enter a pattern");
        assert_eq!(err("   "), "enter a pattern");
        assert_eq!(err(",,,"), "enter a pattern");
        assert_eq!(err(&"a".repeat(641)), "pattern is too long");
        assert!(NodeIdPattern::parse(&vec!["!deadbeef"; 64].join(",")).is_ok());
        let many = (0..65).map(|i| format!("{i:02x}")).collect::<Vec<_>>().join(",");
        assert_eq!(err(&many), "at most 64 targets, got 65");
        assert_eq!(err("!"), "empty target in the list");
        assert_eq!(err("ab,!"), "empty target in the list");
        assert_eq!(err("123456789"), "\"123456789\" is 9 characters; a node id is 8 hex digits");
        assert_eq!(err("12g4"), "\"g\" is not a hex digit. Use 0-9, a-f, or * for any digit");
        assert_eq!(err("ab,abc"), "every target in a set must fix the same digits");
        assert_eq!(err("a*,*a"), "every target in a set must fix the same digits");
        assert_eq!(err("****"), "that matches everything: fix at least one digit");
        assert_eq!(err("!00000000"), "!00000000 is reserved by the firmware");
        assert_eq!(err("00000003"), "!00000003 is reserved by the firmware");
        assert_eq!(err("ffffffff"), "!ffffffff is reserved by the firmware");
        assert!(NodeIdPattern::parse("00000004").is_ok());
        assert!(NodeIdPattern::parse("fffffffe").is_ok());
        // Reserved numbers are rejected by the matcher under partial masks too.
        let p = NodeIdPattern::parse("0000000*").unwrap();
        assert!(!p.matches(0) && !p.matches(3) && p.matches(4) && p.matches(0xF));
        let p = NodeIdPattern::parse("fffffff*").unwrap();
        assert!(!p.matches(0xFFFF_FFFF) && p.matches(0xFFFF_FFF0));
    }

    #[test]
    fn pattern_matching_and_bitmap() {
        let p = NodeIdPattern::parse("dc80,1234,!ab0d").unwrap();
        let bitmap = p.bitmap();
        assert!(p.matches(0xDC80_1234));
        assert!(p.matches(0x1234_0000));
        assert!(p.matches(0xAB0D_FFFF));
        assert!(!p.matches(0xDC81_0000));
        assert!(!p.matches(0xAB0C_0000));
        let wild = NodeIdPattern::parse("ab*d").unwrap();
        assert!(wild.matches(0xAB0D_FFFF) && wild.matches(0xABFD_0000) && !wild.matches(0xAB0C_0000));
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        for _ in 0..100_000 {
            let id = rand_core::RngCore::next_u32(&mut rng);
            assert_eq!(p.quick_matches(id, &bitmap), p.matches(id));
        }
        for &t in &p.targets {
            assert!(p.quick_matches(t | 0x0000_0005, &bitmap));
        }
        assert_eq!(p.gpu_tables().len(), GPU_TABLE_WORDS);
        assert_eq!(&p.gpu_tables()[1024..], &bitmap[..]);
        assert_eq!(p.gpu_targets()[..3], p.targets[..]);
        assert_eq!(p.gpu_targets()[3..], [0u32; 61]);
    }

    #[test]
    fn verify_hit_rejects_wrong_or_unclamped_secrets() {
        let (secret, pubkey, id) = oracle_vectors()[5];
        let pattern = NodeIdPattern::parse(&format!("{:x}", id >> 16)).unwrap();
        let key = verify_hit(&pattern, secret, "test").expect("true hit");
        assert_eq!((key.public_key, key.node_id), (pubkey, id));
        assert_eq!(key.node_id_str(), format!("!{id:08x}"));
        assert_eq!(key.meshtastic_command(), format!(
            "meshtastic --set security.private_key \"base64:{}\"",
            base64_encode(&secret)
        ));
        let other = NodeIdPattern::parse(&format!("{:x}", (id >> 16) ^ 1)).unwrap();
        assert!(verify_hit(&other, secret, "test").is_none());
        let mut unclamped = secret;
        unclamped[0] |= 1;
        assert!(verify_hit(&pattern, unclamped, "test").is_none());
    }

    #[test]
    fn base64_matches_standard_encoding() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        let (secret, _, _) = oracle_vectors()[0];
        // 32 bytes: 31 zeros then 0x40 (clamped all-zero secret).
        assert_eq!(base64_encode(&secret), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEA=");
    }

    #[cfg(feature = "cli")]
    fn collect_hits(pattern_text: &str, threads: usize, wanted: usize) -> Vec<MeshKey> {
        let pattern = Arc::new(NodeIdPattern::parse(pattern_text).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<MeshKey>(1024);
        let handles = spawn_cpu_workers(threads, Arc::clone(&pattern), Arc::clone(&stop), Arc::clone(&counters), tx);
        let mut hits = Vec::new();
        while hits.len() < wanted {
            hits.push(rx.recv_timeout(Duration::from_secs(120)).expect("CPU workers find short patterns quickly"));
        }
        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        assert!(counters.cpu_attempts.load(Ordering::Relaxed) > 0);
        hits
    }

    #[cfg(feature = "cli")]
    #[test]
    fn cpu_workers_find_keys_that_rederive() {
        for (text, threads, wanted) in [("a", 1usize, 3usize), ("dc8", 2, 2), ("!1234,5678,abcd", 2, 1), ("*a*b", 1, 1)] {
            let pattern = NodeIdPattern::parse(text).unwrap();
            for key in collect_hits(text, threads, wanted) {
                assert_eq!(key.source, "cpu");
                assert!(is_clamped(&key.private_key));
                let (pk, id) = derive(&key.private_key);
                assert_eq!((pk, id), (key.public_key, key.node_id), "pattern {text}");
                assert!(pattern.matches(id));
                assert_eq!(crc32_bitwise(&pk), id);
            }
        }
    }

    /// The fallback path (no walk) finds valid keys too.
    #[cfg(feature = "cli")]
    #[test]
    fn cpu_fallback_path_finds_valid_key() {
        let pattern = Arc::new(NodeIdPattern::parse("7").unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<MeshKey>(16);
        let rng = ChaCha20Rng::from_seed([3u8; 32]);
        let handle = {
            let (pattern, stop, counters) = (Arc::clone(&pattern), Arc::clone(&stop), Arc::clone(&counters));
            thread::spawn(move || search(None, rng, pattern, stop, counters, tx))
        };
        let key = rx.recv_timeout(Duration::from_secs(60)).expect("fallback finds a 1-digit pattern");
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert_eq!(derive(&key.private_key), (key.public_key, key.node_id));
        assert_eq!(key.node_id >> 28, 7);
    }

    /// A desynchronised walk is caught (its first bogus hit fails the dalek
    /// re-derivation, or the periodic check fires) and the worker continues
    /// on the fallback path.
    #[cfg(feature = "cli")]
    #[test]
    fn cpu_worker_recovers_from_desynchronised_walk() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let mut walk = UWalk::<LANES>::new(&mut rng).unwrap();
        walk.desynchronise(0);
        walk.desynchronise(1);
        let pattern = Arc::new(NodeIdPattern::parse("ab").unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let (tx, rx) = bounded::<MeshKey>(16);
        let handle = {
            let (pattern, stop, counters) = (Arc::clone(&pattern), Arc::clone(&stop), Arc::clone(&counters));
            thread::spawn(move || search(Some(walk), rng, pattern, stop, counters, tx))
        };
        let key = rx.recv_timeout(Duration::from_secs(120)).expect("worker recovers and finds a key");
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert_eq!(derive(&key.private_key), (key.public_key, key.node_id));
        assert_eq!(key.node_id >> 24, 0xAB);
    }
}

#[cfg(test)]
mod distribution_tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};

    /// Statistical honesty: the match rate of a 4-digit pattern over
    /// dalek-derived random clamped secrets (no walk involved) and over the
    /// walk's candidates must be 2^-16 within 5 sigma, and the walk's node
    /// ids must be uniform in their top nibble and top byte (chi-square well
    /// under twice the degrees of freedom; a 2^-(4*digits) expectation per
    /// candidate is what the ETA and the GPU/CPU end-to-end runs assume).
    #[test]
    fn node_id_distribution_is_uniform() {
        use rayon::prelude::*;
        let pattern = NodeIdPattern::parse("1234").unwrap();
        let n = 600_000u64;
        let hits: u64 = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xD15C0 + i);
                let secret = crate::keygen::random_base_scalar(&mut rng);
                let (_, id) = derive(&secret);
                pattern.matches(id) as u64
            })
            .sum();
        let expected = n as f64 / 65536.0;
        println!("dalek random secrets: {hits} hits of {n}, expected {expected:.1} (ratio {:.3}, 1 sigma {:.3})", hits as f64 / expected, 1.0 / expected.sqrt());
        assert!((hits as f64 - expected).abs() < 5.0 * expected.sqrt(), "dalek hit rate off: {hits} vs {expected:.1}");

        // The walk's candidates: top-nibble and top-byte histograms.
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(77);
        let mut walk = crate::mesh_walk::UWalk::<2>::new(&mut rng).unwrap();
        let crc = Crc32::new();
        let mut nibble = [0u64; 16];
        let mut byte = [0u64; 256];
        let mut hits4 = 0u64;
        let windows = 4000u64;
        for _ in 0..windows {
            walk.next_window(|_, _, u| {
                let id = crc.of_bytes32(u);
                nibble[(id >> 28) as usize] += 1;
                byte[(id >> 24) as usize] += 1;
                hits4 += pattern.matches(id) as u64;
            });
        }
        let total = windows * 512;
        let exp_n = total as f64 / 16.0;
        let chi_n: f64 = nibble.iter().map(|&c| (c as f64 - exp_n).powi(2) / exp_n).sum();
        let exp_b = total as f64 / 256.0;
        let chi_b: f64 = byte.iter().map(|&c| (c as f64 - exp_b).powi(2) / exp_b).sum();
        let exp4 = total as f64 / 65536.0;
        println!("walk: {total} candidates, chi2 nibble {chi_n:.1} (15 dof), chi2 byte {chi_b:.1} (255 dof), '1234' hits {hits4} expected {exp4:.1}");
        assert!(chi_n < 40.0, "top-nibble chi2 {chi_n}");
        assert!(chi_b < 400.0, "top-byte chi2 {chi_b}");
        assert!((hits4 as f64 - exp4).abs() < 5.0 * exp4.sqrt(), "walk hit rate off: {hits4} vs {exp4:.1}");
        let _ = rng.next_u32();
    }
}
