//! Differential tests of src/kernels/ed25519.cu (the MeshCore and
//! Meshtastic kernels) without an NVIDIA GPU: the kernel source is compiled
//! as host C++ (tests/cuda_host/shim.h) and tests/cuda_host/
//! ed25519_harness.cpp runs it over the reference vectors written here:
//! field multiplications against num-bigint, and full search_kernel /
//! mesh_kernel launches (planted keys at window boundaries, the first-byte
//! rule, the self-check words) against curve25519-dalek and the host's
//! hit -> scalar mapping. The harness additionally checks fe_mul / fe_sq /
//! matches_prefix limb-for-limb against a plain ref10 int64 reference
//! (tests/cuda_host/ed25519_reference.h) and the CRC / encoding helpers of
//! the Meshtastic kernel against bitwise references.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use num_bigint::BigUint;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::keygen::{candidate_from_scalar, random_base_scalar, scalar_add_u64, scalar_to_words};
use crate::meshtastic::NodeIdPattern;
use crate::prefix::Prefix;

/// ref10 radix-2^25.5 limb positions.
const LIMB_POS: [u32; 10] = [0, 26, 51, 77, 102, 128, 153, 179, 204, 230];

fn hex_words(words: &[u32]) -> String {
    words.iter().map(|w| format!("{w:08x}")).collect()
}

fn big_p() -> BigUint {
    (BigUint::from(1u32) << 255u32) - BigUint::from(19u32)
}

/// Integer value of ten signed ref10 limbs, reduced mod p.
fn limbs_to_big(limbs: &[i32; 10]) -> BigUint {
    let p = big_p();
    let mut pos = BigUint::from(0u32);
    let mut neg = BigUint::from(0u32);
    for (i, &l) in limbs.iter().enumerate() {
        if l >= 0 {
            pos += BigUint::from(l as u32) << LIMB_POS[i];
        } else {
            neg += BigUint::from(l.unsigned_abs()) << LIMB_POS[i];
        }
    }
    let neg = neg % &p;
    (pos + &p - neg) % &p
}

fn big_to_le32(v: &BigUint) -> String {
    let mut bytes = v.to_bytes_le();
    bytes.resize(32, 0);
    hex::encode(bytes)
}

fn hex_limbs(limbs: &[i32; 10]) -> String {
    limbs.iter().map(|l| format!("{:08x}", *l as u32)).collect()
}

/// ref10 fe_mul input bound per limb: 1.65*2^26 even, 1.65*2^25 odd.
fn limb_bound(i: usize) -> i64 {
    if i & 1 == 1 {
        (1.65 * 33554432.0) as i64
    } else {
        (1.65 * 67108864.0) as i64
    }
}

/// mode 0: uniform within the bound; 1: every limb at +-bound or 0;
/// 2: canonical reduced; 3: sparse; 4: fe_mul-output magnitudes.
fn random_limbs(rng: &mut ChaCha20Rng, mode: u32) -> [i32; 10] {
    let mut out = [0i32; 10];
    for (i, l) in out.iter_mut().enumerate() {
        let b = limb_bound(i);
        let v: i64 = match mode {
            1 => match rng.next_u32() % 3 {
                0 => b,
                1 => -b,
                _ => 0,
            },
            2 => (rng.next_u64() & if i & 1 == 1 { 0x1FF_FFFF } else { 0x3FF_FFFF }) as i64,
            3 => {
                if rng.next_u32() % 4 == 0 {
                    (rng.next_u64() % (2 * b as u64 + 1)) as i64 - b
                } else {
                    0
                }
            }
            4 => {
                let ob = if i & 1 == 1 { (1.01 * 16777216.0) as i64 } else { (1.01 * 33554432.0) as i64 };
                (rng.next_u64() % (2 * ob as u64 + 1)) as i64 - ob
            }
            _ => (rng.next_u64() % (2 * b as u64 + 1)) as i64 - b,
        };
        *l = v as i32;
    }
    out
}

fn edge_limbs() -> Vec<[i32; 10]> {
    let even = (1i32 << 26) - 1;
    let odd = (1i32 << 25) - 1;
    let mut v = vec![
        [0; 10],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-19, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        // p - 1 canonical.
        [even - 19, odd, even, odd, even, odd, even, odd, even, odd],
        // p itself (non-canonical zero).
        [even - 18, odd, even, odd, even, odd, even, odd, even, odd],
        // 2^255 - 1 = p + 18.
        [even, odd, even, odd, even, odd, even, odd, even, odd],
        // 2^255 - 20 = p - 1 as all-ones minus 19.
        [even - 19, odd, even, odd, even, odd, even, odd, even, odd],
        // Top limb only.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, odd],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -odd],
        // Alternating signs at the bound.
        [0; 10],
        [0; 10],
    ];
    let n = v.len();
    for i in 0..10 {
        let b = limb_bound(i) as i32;
        v[n - 2][i] = if i & 1 == 1 { b } else { -b };
        v[n - 1][i] = if i & 1 == 1 { -b } else { b };
    }
    v
}

fn write_field_vectors(out: &mut String, rng: &mut ChaCha20Rng, random: u32) {
    let p = big_p();
    let edge = edge_limbs();
    let mut pairs: Vec<([i32; 10], [i32; 10])> = Vec::new();
    for a in &edge {
        for b in &edge {
            pairs.push((*a, *b));
        }
    }
    for n in 0..random {
        pairs.push((random_limbs(rng, n % 5), random_limbs(rng, (n / 5) % 5)));
    }
    for a in &edge {
        pairs.push((*a, random_limbs(rng, 0)));
    }
    for (a, b) in &pairs {
        let (ba, bb) = (limbs_to_big(a), limbs_to_big(b));
        writeln!(
            out,
            "F {} {} {} {}",
            hex_limbs(a),
            hex_limbs(b),
            big_to_le32(&((&ba * &bb) % &p)),
            big_to_le32(&((&ba * &ba) % &p)),
        )
        .unwrap();
    }
}

fn pubkey_at(base: &[u8; 32], iter: u32) -> [u8; 32] {
    candidate_from_scalar(scalar_add_u64(*base, 8 * iter as u64), [0u8; 32]).public_key
}

fn debug_words(base: &[u8; 32]) -> String {
    let pk = pubkey_at(base, 0);
    let words: Vec<u32> =
        pk.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    hex_words(&words)
}

/// A search scenario: `threads` bases walked for `iters` candidates each,
/// matched against `prefix`; expected hits are exhaustive (every candidate
/// checked with Prefix::matches, the host's rule), so chance matches of a
/// short prefix are part of the expected set.
fn write_search_scenario(out: &mut String, bases: &[[u8; 32]], iters: u32, prefix: &Prefix) {
    let words = prefix.packed_words();
    writeln!(
        out,
        "W {} {iters} {} {}",
        bases.len(),
        prefix.nibbles(),
        words.iter().map(|w| format!("{w:08x} ")).collect::<String>().trim_end()
    )
    .unwrap();
    for base in bases {
        writeln!(out, "B {}", hex_words(&scalar_to_words(base))).unwrap();
    }
    for (t, base) in bases.iter().enumerate() {
        for iter in 0..iters {
            if prefix.matches(&pubkey_at(base, iter)) {
                writeln!(out, "X {t} {iter}").unwrap();
            }
        }
    }
    writeln!(out, "D {}", debug_words(&bases[0])).unwrap();
    writeln!(out, "E").unwrap();
}

/// A Meshtastic scenario: `threads` bases walked for `iters` candidates,
/// node ids tested against `pattern`; expected hits are exhaustive (every
/// candidate's id derived with dalek + bitwise CRC and tested with the
/// host's `NodeIdPattern::matches`, so chance matches of a short pattern
/// are part of the expected set) and the debug u is bases[0]'s X25519 key.
fn write_mesh_scenario(out: &mut String, bases: &[[u8; 32]], iters: u32, pattern: &NodeIdPattern) {
    let targets = pattern.gpu_targets();
    writeln!(
        out,
        "M {} {iters} {:08x} {} {}",
        bases.len(),
        pattern.mask,
        pattern.targets.len(),
        targets.iter().map(|w| format!("{w:08x} ")).collect::<String>().trim_end()
    )
    .unwrap();
    writeln!(out, "K {}", hex_words(&pattern.gpu_tables())).unwrap();
    for base in bases {
        writeln!(out, "B {}", hex_words(&scalar_to_words(base))).unwrap();
    }
    let mut hits = 0;
    for (t, base) in bases.iter().enumerate() {
        for iter in 0..iters {
            let (_, id) = crate::meshtastic::derive(&scalar_add_u64(*base, 8 * iter as u64));
            if pattern.matches(id) {
                writeln!(out, "X {t} {iter}").unwrap();
                hits += 1;
            }
        }
    }
    assert!(hits <= 64, "mesh scenario hit count {hits}");
    let (pk, _) = crate::meshtastic::derive(&bases[0]);
    writeln!(out, "D {}", hex_words(&crate::meshtastic::bytes_to_words8(&pk))).unwrap();
    writeln!(out, "E").unwrap();
}

/// The 8-digit pattern of candidate `iter` of `base`.
fn mesh_id_of(base: &[u8; 32], iter: u32) -> String {
    format!("{:08x}", crate::meshtastic::derive(&scalar_add_u64(*base, 8 * iter as u64)).1)
}

/// Bases whose planted candidates never start with 00/FF (Prefix::parse
/// rejects such prefixes), drawn from `rng`.
fn bases_for(rng: &mut ChaCha20Rng, n: usize, plant_iters: &[u32]) -> Vec<[u8; 32]> {
    let mut bases = Vec::with_capacity(n);
    while bases.len() < n {
        let b = random_base_scalar(rng);
        let ok = plant_iters.iter().all(|&i| {
            let pk = pubkey_at(&b, i);
            pk[0] != 0x00 && pk[0] != 0xFF
        });
        if ok {
            bases.push(b);
        }
    }
    bases
}

fn prefix_of(pk: &[u8; 32], nibbles: usize) -> Prefix {
    Prefix::parse(&hex::encode_upper(pk)[..nibbles]).unwrap()
}

fn write_scenarios(out: &mut String, rng: &mut ChaCha20Rng, extra_random: usize, window: u32) {
    writeln!(out, "T {}", hex_words(crate::table::basepoint_table())).unwrap();
    // The affine step table for the window the harness is compiled with,
    // and the Meshtastic kernel's d*x*y variant.
    writeln!(out, "P {}", hex_words(&crate::gpu::ed_step_table::step_table(window))).unwrap();
    writeln!(out, "U {}", hex_words(&crate::gpu::ed_step_table::mesh_step_table(window))).unwrap();

    // MeshCore kernel, one planted key per scenario at window-boundary
    // positions of three 300- or 600-candidate walks, every prefix length
    // 1..=8 (odd lengths exercise the half-byte mask).
    let plants: [(usize, u32, u32); 9] = [
        (0, 300, 0),
        (2, 300, 255),
        (1, 300, 256),
        (0, 300, 299),
        (2, 600, 511),
        (1, 600, 512),
        (0, 600, 599),
        (2, 600, 1),
        (1, 1, 0),
    ];
    let mut nib = 1usize;
    for &(t, iters, iter) in &plants {
        let bases = bases_for(rng, 3, &[iter]);
        let prefix = prefix_of(&pubkey_at(&bases[t], iter), nib);
        write_search_scenario(out, &bases, iters, &prefix);
        nib = nib % 8 + 1;
    }
    for n in 0..extra_random {
        let iters = 257 + (rng.next_u32() % 300);
        let t = (rng.next_u32() % 3) as usize;
        let iter = rng.next_u32() % iters;
        let bases = bases_for(rng, 3, &[iter]);
        let prefix = prefix_of(&pubkey_at(&bases[t], iter), 1 + n % 8);
        write_search_scenario(out, &bases, iters, &prefix);
    }

    // First-byte rule: a candidate starting 0x00 sits at (thread 0, iter 0),
    // ahead of every other candidate in emulation order, with prefix "0"
    // (and 0xFF with "F"); it must not be reported.
    for (byte, text) in [(0x00u8, "0"), (0xFFu8, "F")] {
        let mut bases = Vec::new();
        while bases.len() < 2 {
            let b = random_base_scalar(rng);
            if bases.is_empty() {
                if pubkey_at(&b, 0)[0] == byte {
                    bases.push(b);
                }
            } else {
                bases.push(b);
            }
        }
        write_search_scenario(out, &bases, 40, &Prefix::parse(text).unwrap());
    }

    // No match at all: an 8-nibble prefix of a key that is not in the walk.
    {
        let bases = bases_for(rng, 2, &[0]);
        let other = random_base_scalar(rng);
        let prefix = prefix_of(&pubkey_at(&other, 0), 8);
        write_search_scenario(out, &bases, 300, &prefix);
    }

    // Meshtastic kernel: full 8-digit ids planted at window-boundary
    // positions (exact hit sets), short patterns with chance hits (the
    // expected set is exhaustive), multi-target lists with wildcards, and a
    // launch with nothing to find.
    {
        let iters = 300u32;
        let bases: Vec<[u8; 32]> = (0..3).map(|_| random_base_scalar(rng)).collect();
        let ids = [
            mesh_id_of(&bases[0], 0),
            mesh_id_of(&bases[2], 255),
            mesh_id_of(&bases[1], 256),
            mesh_id_of(&bases[0], 299),
            mesh_id_of(&bases[2], 0),
            mesh_id_of(&bases[1], 1),
            mesh_id_of(&bases[1], 127),
            mesh_id_of(&bases[1], 128),
        ];
        write_mesh_scenario(out, &bases, iters, &NodeIdPattern::parse(&ids.join(",")).unwrap());
    }
    {
        let iters = 513u32;
        let bases: Vec<[u8; 32]> = (0..2).map(|_| random_base_scalar(rng)).collect();
        let ids = [mesh_id_of(&bases[1], 511), mesh_id_of(&bases[1], 512), mesh_id_of(&bases[0], 512)];
        write_mesh_scenario(out, &bases, iters, &NodeIdPattern::parse(&ids.join(",")).unwrap());
    }
    {
        // Short patterns: ~1/16 and ~1/256 of 600 candidates match by chance.
        let bases: Vec<[u8; 32]> = (0..2).map(|_| random_base_scalar(rng)).collect();
        write_mesh_scenario(out, &bases, 300, &NodeIdPattern::parse("a").unwrap());
        write_mesh_scenario(out, &bases, 300, &NodeIdPattern::parse("*7").unwrap());
        write_mesh_scenario(out, &bases, 300, &NodeIdPattern::parse("1a,2b,3c,4d").unwrap());
        // Prefix of a planted id with trailing wildcards, plus decoys.
        let id = mesh_id_of(&bases[1], 77);
        let pat = format!("{}**,{},deadbe", &id[..6], &id[..6].replace(|c: char| c.is_ascii_digit(), "0"));
        write_mesh_scenario(out, &bases, 300, &NodeIdPattern::parse(&pat).unwrap());
    }
    {
        // Nothing to find: an id from an unrelated walk.
        let bases: Vec<[u8; 32]> = (0..2).map(|_| random_base_scalar(rng)).collect();
        let other = mesh_id_of(&random_base_scalar(rng), 5);
        write_mesh_scenario(out, &bases, 300, &NodeIdPattern::parse(&other).unwrap());
    }
    for n in 0..extra_random {
        let iters = 257 + (rng.next_u32() % 300);
        let bases: Vec<[u8; 32]> = (0..2).map(|_| random_base_scalar(rng)).collect();
        let t = (rng.next_u32() % 2) as usize;
        let iter = rng.next_u32() % iters;
        let ids = [mesh_id_of(&bases[t], iter), mesh_id_of(&bases[1 - t], (iter + n as u32) % iters)];
        write_mesh_scenario(out, &bases, iters, &NodeIdPattern::parse(&ids.join(",")).unwrap());
    }
}

fn build_vectors(seed: u64, random_fields: u32, extra_random: usize, window: u32) -> (String, usize) {
    let mut out = String::new();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    write_field_vectors(&mut out, &mut rng, random_fields);
    write_scenarios(&mut out, &mut rng, extra_random, window);
    let scenarios = out.lines().filter(|l| *l == "E").count();
    (out, scenarios)
}

/// The kernel's default BATCH_INV_WINDOW (src/kernels/ed25519.cu); the planted
/// positions in write_scenarios sit on this window's boundaries.
const DEFAULT_WINDOW: u32 = 256;

/// A deliberately tiny window: the same scenarios then span dozens of
/// windows each, with short final windows of every length, so the window
/// bookkeeping (slot order, base update, j < w reporting) is exercised far
/// more densely than the default window allows.
const SMALL_WINDOW: u32 = 6;

fn find_compiler() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = std::env::var_os("CXX")
        .map(PathBuf::from)
        .into_iter()
        .chain(["c++", "clang++", "g++"].iter().map(PathBuf::from))
        .collect();
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// A CUDA build host always has a host C++ compiler (nvcc needs one), so a
/// missing compiler fails the test instead of silently skipping the only
/// check the CUDA kernel gets without a GPU; MTVK_ALLOW_NO_CXX=1 opts out.
fn require_compiler() -> Option<PathBuf> {
    let found = find_compiler();
    if found.is_none() {
        if std::env::var_os("MTVK_ALLOW_NO_CXX").is_some() {
            eprintln!("cuda_kernel_host_emulation: no C++ compiler found; skipping (MTVK_ALLOW_NO_CXX)");
            return None;
        }
        panic!("no C++ compiler found for the CUDA host-emulation test: set CXX, or MTVK_ALLOW_NO_CXX=1 to skip");
    }
    found
}

fn compile_harness(cxx: &Path, tag: &str, opt: &str, window: u32) -> (PathBuf, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("mtvanitykey-cuda-host-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("ed25519_harness");
    let compile = Command::new(cxx)
        .args(["-std=c++17", opt, "-Wall", "-Wno-unknown-pragmas", "-Wno-unused-function"])
        .arg(format!("-DBATCH_INV_WINDOW={window}"))
        .arg("-I")
        .arg(root.join("tests/cuda_host"))
        .arg("-I")
        .arg(root.join("src/kernels"))
        .arg(root.join("tests/cuda_host/ed25519_harness.cpp"))
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run C++ compiler");
    assert!(
        compile.status.success(),
        "harness compile failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    (dir, exe)
}

fn run_harness(exe: &Path, dir: &Path, vectors: &str, scenarios: usize) {
    let path = dir.join("vectors.txt");
    std::fs::write(&path, vectors).unwrap();
    let run = Command::new(exe).arg(&path).output().expect("run harness");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    println!("{stdout}");
    assert!(run.status.success(), "harness reported mismatches:\n{stdout}\n{stderr}");
    assert!(stdout.contains(&format!("scenarios {scenarios} failures 0")), "{stdout}");
}

fn cleanup(dir: &Path) {
    if std::env::var_os("MTVK_KEEP_CUDA_HOST_DIR").is_some() {
        eprintln!("keeping {}", dir.display());
    } else {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn cuda_kernel_host_emulation_matches_reference() {
    let Some(cxx) = require_compiler() else {
        return;
    };
    let (dir, exe) = compile_harness(&cxx, "o1", "-O1", DEFAULT_WINDOW);
    let (vectors, scenarios) = build_vectors(0xed25_519c_0de, 1500, 2, DEFAULT_WINDOW);
    run_harness(&exe, &dir, &vectors, scenarios);
    cleanup(&dir);
}

#[test]
fn cuda_kernel_host_emulation_randomized() {
    let Some(cxx) = require_compiler() else {
        return;
    };
    let seed = std::env::var("MTVK_CUDA_HOST_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5eed_ed25_u64);
    // -O2 (the first test uses -O1): a different host codegen of the same
    // carry chains and unrolled field code.
    let (dir, exe) = compile_harness(&cxx, "o2", "-O2", DEFAULT_WINDOW);
    let (vectors, scenarios) = build_vectors(seed, 600, 4, DEFAULT_WINDOW);
    run_harness(&exe, &dir, &vectors, scenarios);
    cleanup(&dir);
}

/// The same scenarios through a kernel compiled with a tiny window (see
/// SMALL_WINDOW), with the matching step table.
#[test]
fn cuda_kernel_host_emulation_small_window() {
    let Some(cxx) = require_compiler() else {
        return;
    };
    let (dir, exe) = compile_harness(&cxx, "w6", "-O2", SMALL_WINDOW);
    let (vectors, scenarios) = build_vectors(0x5a11_0006_u64, 200, 6, SMALL_WINDOW);
    run_harness(&exe, &dir, &vectors, scenarios);
    cleanup(&dir);
}

/// The host-side limb model used for the F vectors must agree with the
/// kernel's canonical encoding on canonical inputs (guards the test itself).
#[test]
fn limb_model_round_trips_canonical_values() {
    let p = big_p();
    let one = limbs_to_big(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(one, BigUint::from(1u32));
    let even = (1i32 << 26) - 1;
    let odd = (1i32 << 25) - 1;
    let pm1 = limbs_to_big(&[even - 19, odd, even, odd, even, odd, even, odd, even, odd]);
    assert_eq!(pm1, &p - BigUint::from(1u32));
    let minus1 = limbs_to_big(&[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(minus1, &p - BigUint::from(1u32));
}

