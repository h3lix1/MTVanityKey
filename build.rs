use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use num_bigint::BigUint;
use num_traits::{Num, One, Zero};

const DIGITS: usize = 43;
const MULTIPLES: usize = 32;
const COMPONENTS: usize = 3;
const LIMBS: usize = 10;
const WORDS: usize = DIGITS * MULTIPLES * COMPONENTS * LIMBS;
const RADIX_BITS: usize = 6;

/// ref10 radix-2^25.5 limb sizes: even limbs 26 bits, odd limbs 25 bits.
const LIMB_BITS: [usize; LIMBS] = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];

fn main() {
    // The generated tables depend only on this file (a change to the pinned
    // build-dependency versions recompiles the script and re-runs it anyway).
    // Without an explicit rerun-if directive cargo falls back to re-running
    // the script whenever ANY file in the package changes, which costs a few
    // seconds of unoptimized bignum arithmetic on every incremental build.
    // The CUDA path registers its own .cu sources and env vars below.
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));

    // Kick off the nvcc compile first so it overlaps the table generation
    // below: the CUDA build script costs max(nvcc, tables), not their sum.
    let cuda = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let config = cuda_config();
        let jobs = CUDA_MODULES
            .iter()
            .map(|(source_name, output_name)| {
                compile_cuda_module(&config, &out_dir, source_name, output_name)
            })
            .collect::<Vec<_>>();
        Some((config, jobs))
    } else {
        None
    };

    let table = generate_table();
    write_table_source(
        &out_dir.join("basepoint_table.rs"),
        "BASEPOINT_TABLE",
        &table,
        "failed to write generated basepoint table",
    );

    if let Some((config, jobs)) = cuda {
        finish_cuda_modules(&config, jobs);
    }
}

/// Emit `pub const {name}_WORDS` and `pub static {name}: [u32; {name}_WORDS]`
/// as Rust source, eight words per line.
fn write_table_source(output: &Path, name: &str, table: &[u32], error: &str) {
    // Each word prints as at most 10 digits plus the "u32, " suffix.
    let mut source = String::with_capacity(table.len() * 16 + 160);
    writeln!(source, "pub const {name}_WORDS: usize = {};", table.len()).unwrap();
    writeln!(source, "pub static {name}: [u32; {name}_WORDS] = [").unwrap();
    for (i, word) in table.iter().enumerate() {
        if i % 8 == 0 {
            source.push_str("    ");
        }
        write!(source, "{word}u32, ").unwrap();
        if i % 8 == 7 {
            source.push('\n');
        }
    }
    source.push_str("];\n");
    fs::write(output, source).expect(error);
}

/// (source file in src/, output file in OUT_DIR) for each CUDA module.
const CUDA_MODULES: [(&str, &str); 1] = [("kernels/ed25519.cu", "mtvanitykey_cuda.cubin")];

/// Everything the nvcc invocation needs: the nvcc lookup and the nvidia-smi
/// / `nvcc --list-gpu-code` arch probes are subprocesses, resolved once.
struct CudaConfig {
    nvcc: PathBuf,
    /// CUDA_ARCH as given or detected (sm_* / compute_*), for diagnostics.
    arch: String,
    /// The real architecture (always sm_*) the fatbin carries SASS for.
    real_arch: String,
    lb_threads: u32,
    lb_min_blocks: u32,
    batch_inv_window: u32,
}

/// A running nvcc compile of one module.
struct CudaJob {
    source: PathBuf,
    child: Child,
}

fn cuda_config() -> CudaConfig {
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=MTVK_CUDA_LB_THREADS");
    println!("cargo:rerun-if-env-changed=MTVK_CUDA_LB_MIN_BLOCKS");
    println!("cargo:rerun-if-env-changed=MTVK_CUDA_BATCH_INV_WINDOW");

    // Empirical tuning knobs (see src/kernels/ed25519.cu); overridable for
    // re-tuning on other GPUs.
    let lb_threads = tuning_var("MTVK_CUDA_LB_THREADS", 256);
    let lb_min_blocks = tuning_var("MTVK_CUDA_LB_MIN_BLOCKS", 1);
    let batch_inv_window = tuning_var("MTVK_CUDA_BATCH_INV_WINDOW", 256);
    // The kernel is compiled for exactly LB_THREADS threads per block; the
    // launch in src/gpu/cuda.rs must match.
    println!("cargo:rustc-env=MTVK_CUDA_BLOCK_SIZE={lb_threads}");
    // The Ed25519 host builds its affine step table for the compiled window.
    println!("cargo:rustc-env=MTVK_CUDA_BATCH_INV_WINDOW={batch_inv_window}");

    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "CUDA feature enabled but nvcc was not found. Install the CUDA Toolkit or set CUDA_NVCC/CUDA_HOME/CUDA_PATH."
        )
    });
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| default_cuda_arch(&nvcc));
    let real_arch = real_cuda_arch(&arch).unwrap_or_else(|| {
        panic!(
            "CUDA_ARCH={arch} is not a real GPU architecture. Use a value like sm_90 or compute_90."
        )
    });

    CudaConfig {
        nvcc,
        arch,
        real_arch,
        lb_threads,
        lb_min_blocks,
        batch_inv_window,
    }
}

/// Spawn (without waiting for) the nvcc compile of one module. The caller
/// collects the result with `finish_cuda_modules`.
fn compile_cuda_module(
    config: &CudaConfig,
    out_dir: &Path,
    source_name: &str,
    output_name: &str,
) -> CudaJob {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir missing"));
    let source = manifest_dir.join("src").join(source_name);
    // A fatbin (SASS for the detected arch + PTX for driver JIT on newer
    // GPUs). cuModuleLoadData accepts fatbin images.
    let output = out_dir.join(output_name);

    println!("cargo:rerun-if-changed={}", source.display());

    let arch_num = config
        .real_arch
        .strip_prefix("sm_")
        .expect("real_cuda_arch always returns sm_*");

    // MTVK_CUDA_DEFINES: extra whitespace-separated nvcc arguments
    // (e.g. "-DMESH_CRC_SLICE4=0") for kernel experiments; empty by default.
    println!("cargo:rerun-if-env-changed=MTVK_CUDA_DEFINES");
    let extra: Vec<String> = env::var("MTVK_CUDA_DEFINES")
        .map(|v| v.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    let child = Command::new(&config.nvcc)
        .arg("--fatbin")
        .arg("-O3")
        .arg("--std=c++17")
        .arg(format!("-DLB_THREADS={}", config.lb_threads))
        .arg(format!("-DLB_MIN_BLOCKS={}", config.lb_min_blocks))
        .arg(format!("-DBATCH_INV_WINDOW={}", config.batch_inv_window))
        .args(&extra)
        .arg(format!(
            "-gencode=arch=compute_{arch_num},code=[sm_{arch_num},compute_{arch_num}]"
        ))
        .arg("-Xptxas")
        .arg("-v")
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", config.nvcc.display()));

    CudaJob { source, child }
}

/// Wait for every nvcc compile, surface ptxas -v statistics in the cargo
/// build output and fail the build if any module did not compile.
fn finish_cuda_modules(config: &CudaConfig, jobs: Vec<CudaJob>) {
    // Drain each child's pipes on its own thread so a compile that is still
    // running can never stall on a full pipe while we wait for another one.
    let outputs = std::thread::scope(|scope| {
        let handles = jobs
            .into_iter()
            .map(|job| scope.spawn(move || (job.source, job.child.wait_with_output())))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("nvcc wait thread panicked"))
            .collect::<Vec<_>>()
    });

    let mut failed = None;
    for (source, result) in outputs {
        let cmd_output = result
            .unwrap_or_else(|error| panic!("failed to run {}: {error}", config.nvcc.display()));

        // Surface ptxas -v register/spill statistics in the cargo build output.
        for line in String::from_utf8_lossy(&cmd_output.stderr).lines() {
            if !line.trim().is_empty() {
                println!("cargo:warning=nvcc: {line}");
            }
        }

        if !cmd_output.status.success() && failed.is_none() {
            failed = Some(source);
        }
    }

    if let Some(source) = failed {
        panic!(
            "nvcc failed while compiling {} for CUDA_ARCH={} ({})",
            source.display(),
            config.arch,
            config.real_arch
        );
    }
}

fn tuning_var(name: &str, default: u32) -> u32 {
    match env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {value:?}")),
        Err(_) => default,
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("CUDA_NVCC").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }

    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(home) = env::var_os(var).map(PathBuf::from) {
            let candidate = home.join("bin").join(nvcc_exe());
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|path| path.join(nvcc_exe()))
        .find(|candidate| candidate.is_file())
}

/// Detect the compute capability of the GPU actually installed in this
/// machine via nvidia-smi (e.g. "9.0" -> sm_90). Without nvidia-smi it falls
/// back to the OLDEST toolkit-supported arch (then sm_75): the fatbin embeds
/// compute_XX PTX, which the driver JIT-compiles forward onto any newer GPU,
/// so the oldest arch is the portable choice (set CUDA_ARCH for best code).
fn default_cuda_arch(nvcc: &Path) -> String {
    if let Some(arch) = installed_gpu_arch() {
        return arch;
    }
    toolkit_fallback_arch(nvcc)
}

fn installed_gpu_arch() -> Option<String> {
    // Query only GPU 0: it is the first line of the unfiltered listing (the
    // one used here) and the device src/gpu/cuda.rs opens. Without
    // persistence mode nvidia-smi initialises every GPU it reports on, which
    // costs up to seconds per device on multi-GPU hosts.
    let output = Command::new("nvidia-smi")
        .arg("--id=0")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let cap = stdout.lines().next()?.trim();
    let (major, minor) = cap.split_once('.')?;
    let major: u32 = major.parse().ok()?;
    let minor: u32 = minor.parse().ok()?;
    Some(format!("sm_{major}{minor}"))
}

fn toolkit_fallback_arch(nvcc: &Path) -> String {
    let fallback = "sm_75";
    let output = Command::new(nvcc).arg("--list-gpu-code").output();
    let Ok(output) = output else {
        return fallback.to_string();
    };
    if !output.status.success() {
        return fallback.to_string();
    }

    let supported = String::from_utf8_lossy(&output.stdout);
    ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
        .into_iter()
        .find(|arch| supported.lines().any(|line| line.trim() == *arch))
        .unwrap_or(fallback)
        .to_string()
}

fn real_cuda_arch(arch: &str) -> Option<String> {
    arch.strip_prefix("compute_")
        .map(|suffix| format!("sm_{suffix}"))
        .or_else(|| arch.strip_prefix("sm_").map(|_| arch.to_string()))
}

fn nvcc_exe() -> &'static str {
    if cfg!(windows) {
        "nvcc.exe"
    } else {
        "nvcc"
    }
}

// ---------------------------------------------------------------------------
// Prime-field arithmetic on BigUint. Every value is kept fully reduced in
// [0, p), so additions and subtractions need a single conditional correction
// and products a single `%`. Build scripts are compiled without optimization
// by default, so the point arithmetic below is organised to keep the number
// of bignum operations small: projective coordinates everywhere and a single
// batched inversion per table row instead of one (or two) per point.
// ---------------------------------------------------------------------------

struct Field {
    p: BigUint,
}

impl Field {
    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let sum = a + b;
        if sum >= self.p {
            sum - &self.p
        } else {
            sum
        }
    }

    fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a - b
        } else {
            a + &self.p - b
        }
    }

    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a * b) % &self.p
    }

    fn sq(&self, a: &BigUint) -> BigUint {
        (a * a) % &self.p
    }

    fn inv(&self, a: &BigUint) -> BigUint {
        a.modinv(&self.p).expect("inverse of zero field element")
    }

    /// Inverses of every element of `values` using one field inversion
    /// (Montgomery's trick). No element may be zero.
    fn batch_inv(&self, values: &[BigUint]) -> Vec<BigUint> {
        // prefix[i] = values[0] * ... * values[i-1]
        let mut prefix = Vec::with_capacity(values.len());
        let mut acc = BigUint::one();
        for value in values {
            prefix.push(acc.clone());
            acc = self.mul(&acc, value);
        }
        let mut inv = self.inv(&acc);
        let mut out = vec![BigUint::zero(); values.len()];
        for i in (0..values.len()).rev() {
            out[i] = self.mul(&inv, &prefix[i]);
            inv = self.mul(&inv, &values[i]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Ed25519: -x^2 + y^2 = 1 + d x^2 y^2 in extended twisted Edwards coordinates
// (X : Y : Z : T) with x = X/Z, y = Y/Z, x*y = T/Z.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Extended {
    x: BigUint,
    y: BigUint,
    z: BigUint,
    t: BigUint,
}

struct Edwards {
    f: Field,
    d: BigUint,
    d2: BigUint,
}

impl Edwards {
    fn new() -> Edwards {
        let f = Field { p: modulus() };
        // d = -121665 / 121666
        let numerator = f.sub(&BigUint::zero(), &BigUint::from(121665u32));
        let d = f.mul(&numerator, &f.inv(&BigUint::from(121666u32)));
        let d2 = f.add(&d, &d);
        Edwards { f, d, d2 }
    }

    /// -x^2 + y^2 == 1 + d x^2 y^2 for an affine point.
    fn on_curve(&self, x: &BigUint, y: &BigUint) -> bool {
        let f = &self.f;
        let xx = f.sq(x);
        let yy = f.sq(y);
        let lhs = f.sub(&yy, &xx);
        let rhs = f.add(&BigUint::one(), &f.mul(&self.d, &f.mul(&xx, &yy)));
        lhs == rhs
    }

    /// Unified addition for a = -1 ("add-2008-hwcd"); complete on this curve,
    /// so it also serves as doubling.
    fn add(&self, p: &Extended, q: &Extended) -> Extended {
        let f = &self.f;
        let a = f.mul(&p.x, &q.x);
        let b = f.mul(&p.y, &q.y);
        let c = f.mul(&f.mul(&p.t, &self.d), &q.t);
        let dd = f.mul(&p.z, &q.z);
        let e = f.sub(
            &f.sub(&f.mul(&f.add(&p.x, &p.y), &f.add(&q.x, &q.y)), &a),
            &b,
        );
        let ff = f.sub(&dd, &c);
        let g = f.add(&dd, &c);
        let h = f.add(&b, &a);
        Extended {
            x: f.mul(&e, &ff),
            y: f.mul(&g, &h),
            z: f.mul(&ff, &g),
            t: f.mul(&e, &h),
        }
    }
}

fn generate_table() -> Vec<u32> {
    let ed = Edwards::new();
    let f = &ed.f;
    let mut words = Vec::with_capacity(WORDS);
    let mut point = basepoint(&ed);
    assert!(ed.on_curve(&point.x, &point.y), "basepoint is not on the curve");

    for _ in 0..DIGITS {
        // multiples[k] = (k + 1) * point, all in extended coordinates.
        let mut multiples: Vec<Extended> = Vec::with_capacity(MULTIPLES);
        multiples.push(point.clone());
        for k in 1..MULTIPLES {
            let next = ed.add(&multiples[k - 1], &point);
            multiples.push(next);
        }

        let zs: Vec<BigUint> = multiples.iter().map(|m| m.z.clone()).collect();
        let zinvs = f.batch_inv(&zs);
        for (multiple, zinv) in multiples.iter().zip(&zinvs) {
            let x = f.mul(&multiple.x, zinv);
            let y = f.mul(&multiple.y, zinv);
            // A single bad Z would corrupt the whole batch-inverted row, so
            // guard every affine point before it is written out.
            assert!(ed.on_curve(&x, &y), "table entry is not on the curve");
            // Affine Niels form: (y + x, y - x, 2 d x y).
            write_field(&mut words, &f.add(&y, &x), &f.p);
            write_field(&mut words, &f.sub(&y, &x), &f.p);
            write_field(&mut words, &f.mul(&f.mul(&ed.d2, &x), &y), &f.p);
        }

        for _ in 0..RADIX_BITS {
            point = ed.add(&point, &point);
        }
    }

    assert_eq!(words.len(), WORDS);
    words
}

fn basepoint(ed: &Edwards) -> Extended {
    let f = &ed.f;
    let x = BigUint::from_str_radix(
        "15112221349535400772501151409588531511454012693041857206046113283949847762202",
        10,
    )
    .expect("valid basepoint x");
    let y = f.mul(&BigUint::from(4u32), &f.inv(&BigUint::from(5u32)));
    let t = f.mul(&x, &y);
    Extended {
        x,
        y,
        z: BigUint::one(),
        t,
    }
}

/// Decompose an already-reduced field element (0 <= value < 2^255-19) into
/// the ten non-negative ref10 radix-2^25.5 limbs (26/25-bit alternating).
/// These are exactly the limb values ref10 holds for a reduced element; the
/// kernel loads them into signed 32-bit limbs.
fn write_field(words: &mut Vec<u32>, value: &BigUint, p: &BigUint) {
    assert!(value < p, "field element must be reduced");
    let digits = to_u64_limbs::<4>(value);
    assert!(digits[3] >> 63 == 0, "value exceeded 255 bits");
    let mut pos = 0usize;
    for bits in LIMB_BITS {
        words.push(extract_bits(&digits, pos, bits));
        pos += bits;
    }
}

/// Little-endian u64 limbs of `value`, zero-padded to `N` words.
fn to_u64_limbs<const N: usize>(value: &BigUint) -> [u64; N] {
    let digits = value.to_u64_digits();
    assert!(digits.len() <= N, "value does not fit in {N} u64 limbs");
    let mut out = [0u64; N];
    out[..digits.len()].copy_from_slice(&digits);
    out
}

/// Bits [pos, pos + bits) of a little-endian u64 limb array (bits <= 32).
fn extract_bits(digits: &[u64], pos: usize, bits: usize) -> u32 {
    let word = pos / 64;
    let offset = pos % 64;
    let mut value = digits[word] >> offset;
    if offset + bits > 64 {
        value |= digits[word + 1] << (64 - offset);
    }
    (value & ((1u64 << bits) - 1)) as u32
}

fn modulus() -> BigUint {
    (BigUint::one() << 255usize) - BigUint::from(19u32)
}
