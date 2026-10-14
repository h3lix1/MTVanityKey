// Differential harness for src/kernels/ed25519.cu (the MeshCore and
// Meshtastic kernels) compiled as host C++ (see shim.h). Driven by the
// `cuda_kernel_host_emulation_*` tests in src/gpu/cuda_host_tests.rs, which
// write a vector file of reference results and run this binary on it; every
// mismatch is printed and the exit status is non-zero.
//
// Built-in checks (always run, no vectors needed):
//   * fe_mul / fe_sq (mad.wide + int32-carry chain) vs the plain ref10
//     int64 code in ed25519_reference.h, limb-for-limb, on random limbs up
//     to the ref10 input bounds (both signs), the bound corners, canonical
//     values and chained walk steps.
//   * matches_prefix_masked vs the byte/nibble loop for every prefix length
//     0..16 on random and single-nibble near-miss inputs.
//   * fe_tobytes_words8 vs fe_tobytes32 and crc32_words8 vs a bitwise CRC.
//
// Vector file lines (hex words are 8-digit; Fe limbs are ten 8-digit
// two's-complement words, limb 0 first):
//   F a b mul(32 LE bytes) sq(32 LE bytes)   field ops vs BigUint
//   T words                                  the basepoint table
//   P words                                  the affine step table for the
//                                            compiled BATCH_INV_WINDOW
//   W threads iters nibbles pw0..pw7         start a MeshCore scenario
//   B words                                  one thread's base scalar (8)
//   X thread iter                            an expected hit
//   D words                                  expected debug words (8)
//   U words                                  the mesh (d*x*y) step table
//   M threads iters mask n t0..t63           start a Meshtastic scenario
//   K words                                  its CRC tables + bitmap (1152)
//   E                                        run the scenario

#include "shim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

static inline uint32_t atomicCAS(uint32_t* p, uint32_t cmp, uint32_t val) {
    uint32_t old = *p;
    if (old == cmp) *p = val;
    return old;
}

#include "ed25519.cu"
#include "ed25519_reference.h"

static int failures = 0;

static void fail(const char* what, const std::string& ctx) {
    failures++;
    if (failures <= 40) std::fprintf(stderr, "FAIL %s: %s\n", what, ctx.c_str());
}

static std::vector<uint32_t> parse_words(const std::string& hex) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i + 8 <= hex.size(); i += 8) {
        out.push_back((uint32_t)std::strtoul(hex.substr(i, 8).c_str(), nullptr, 16));
    }
    return out;
}

static Fe parse_fe(const std::string& hex) {
    std::vector<uint32_t> w = parse_words(hex);
    Fe f;
    for (int i = 0; i < 10; i++) f.v[i] = (int32_t)(i < (int)w.size() ? w[i] : 0u);
    return f;
}

static std::vector<uint8_t> parse_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i + 2 <= hex.size(); i += 2) {
        out.push_back((uint8_t)std::strtoul(hex.substr(i, 2).c_str(), nullptr, 16));
    }
    return out;
}

static std::string fe_hex(const Fe& f) {
    char buf[96];
    for (int i = 0; i < 10; i++) std::snprintf(buf + 8 * i, 9, "%08x", (uint32_t)f.v[i]);
    return std::string(buf, 80);
}

static std::string bytes_hex(const uint8_t* b, size_t n) {
    std::string s;
    char buf[4];
    for (size_t i = 0; i < n; i++) {
        std::snprintf(buf, sizeof buf, "%02x", b[i]);
        s += buf;
    }
    return s;
}

static bool fe_same(const Fe& a, const Fe& b) {
    return std::memcmp(a.v, b.v, sizeof(a.v)) == 0;
}

// Deterministic PRNG for the built-in checks (splitmix64).
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd64() {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ref10 fe_mul input bound: 1.65*2^26 for even limbs, 1.65*2^25 for odd.
static int32_t limb_bound(int i) {
    return (i & 1) ? (int32_t)(1.65 * 33554432.0) : (int32_t)(1.65 * 67108864.0);
}

// mode 0: uniform in [-bound, bound]; 1: each limb at +-bound or 0;
// 2: canonical reduced (even < 2^26, odd < 2^25, non-negative);
// 3: fe_mul-output-like (|even| <= 1.01*2^25, |odd| <= 1.01*2^24), as the
//    walk feeds back; 4: sparse (most limbs zero).
static Fe random_fe(int mode) {
    Fe f;
    for (int i = 0; i < 10; i++) {
        int32_t b = limb_bound(i);
        int64_t v;
        switch (mode) {
            case 1: {
                uint64_t r = rnd64() % 3;
                v = r == 0 ? b : (r == 1 ? -b : 0);
                break;
            }
            case 2:
                v = (int64_t)(rnd64() & ((i & 1) ? 0x1FFFFFFu : 0x3FFFFFFu));
                break;
            case 3: {
                int64_t ob = (i & 1) ? (int64_t)(1.01 * 16777216.0) : (int64_t)(1.01 * 33554432.0);
                v = (int64_t)(rnd64() % (uint64_t)(2 * ob + 1)) - ob;
                break;
            }
            case 4:
                v = (rnd64() % 4 == 0) ? (int64_t)(rnd64() % (uint64_t)(2 * b + 1)) - b : 0;
                break;
            default:
                v = (int64_t)(rnd64() % (uint64_t)(2 * b + 1)) - b;
        }
        f.v[i] = (int32_t)v;
    }
    return f;
}

static void check_field_pair(const Fe& a, const Fe& b, const char* ctx) {
    Fe m_new = fe_mul(a, b);
    Fe m_ref = ref_fe_mul(a, b);
    if (!fe_same(m_new, m_ref)) {
        fail("fe_mul vs ref", std::string(ctx) + " a=" + fe_hex(a) + " b=" + fe_hex(b)
            + " new=" + fe_hex(m_new) + " ref=" + fe_hex(m_ref));
    }
    Fe s_new = fe_sq(a);
    Fe s_ref = ref_fe_sq(a);
    if (!fe_same(s_new, s_ref)) {
        fail("fe_sq vs ref", std::string(ctx) + " a=" + fe_hex(a) + " new=" + fe_hex(s_new)
            + " ref=" + fe_hex(s_ref));
    }
}

static int builtin_field_checks(int rounds) {
    int checked = 0;
    // Bound corners: every limb at +bound / -bound / 0, all 3^2 sign
    // patterns for a few limb pairs plus the all-same patterns.
    Fe corners[5];
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < 10; i++) {
            int32_t b = limb_bound(i);
            corners[k].v[i] = k == 0 ? b : k == 1 ? -b : k == 2 ? 0 : k == 3 ? ((i & 1) ? b : -b) : ((i & 1) ? -b : b);
        }
    }
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            check_field_pair(corners[x], corners[y], "corner");
            checked++;
        }
    }
    for (int n = 0; n < rounds; n++) {
        int ma = n % 5, mb = (n / 5) % 5;
        check_field_pair(random_fe(ma), random_fe(mb), "random");
        checked++;
    }
    // Chained: a walk-like recurrence run through both implementations in
    // lock step, so every intermediate magnitude the kernel produces is
    // covered (sums and differences of two outputs, and a doubled square as
    // in 2Z^2 -+ K, feed the next multiply).
    Fe x_new = random_fe(2), y_new = random_fe(2);
    Fe x_ref = x_new, y_ref = y_new;
    for (int n = 0; n < 4000; n++) {
        Fe c_new = fe_add(x_new, y_new), d_new = fe_sub(x_new, y_new);
        Fe c_ref = fe_add(x_ref, y_ref), d_ref = fe_sub(x_ref, y_ref);
        Fe z2_new = fe_sq(y_new), z2_ref = ref_fe_sq(y_ref);
        Fe k_new = fe_mul(x_new, y_new), k_ref = ref_fe_mul(x_ref, y_ref);
        Fe den_new = fe_sub(fe_add(z2_new, z2_new), k_new);
        Fe den_ref = fe_sub(fe_add(z2_ref, z2_ref), k_ref);
        Fe nx_new = fe_mul(c_new, den_new);
        Fe nx_ref = ref_fe_mul(c_ref, den_ref);
        Fe ny_new = fe_mul(d_new, fe_add(fe_add(z2_new, z2_new), k_new));
        Fe ny_ref = ref_fe_mul(d_ref, fe_add(fe_add(z2_ref, z2_ref), k_ref));
        if (!fe_same(nx_new, nx_ref) || !fe_same(ny_new, ny_ref)) {
            fail("chained walk step", "n=" + std::to_string(n));
            break;
        }
        x_new = nx_new; y_new = ny_new; x_ref = nx_ref; y_ref = ny_ref;
        checked++;
    }
    return checked;
}

static int builtin_prefix_checks(int rounds) {
    int checked = 0;
    for (uint32_t nib = 0; nib <= 16; nib++) {
        for (int n = 0; n < rounds; n++) {
            uint64_t prefix = rnd64();
            // Like Prefix::packed_u64: bytes beyond the prefix are zero and an
            // odd trailing nibble leaves the low half of its byte zero.
            uint32_t full_bytes = nib >> 1;
            uint64_t keep = full_bytes >= 8 ? ~0ULL : ((1ULL << (8 * full_bytes)) - 1);
            if ((nib & 1) && full_bytes < 8) keep |= 0xF0ULL << (8 * full_bytes);
            prefix &= keep;
            SearchParams params;
            std::memset(&params, 0, sizeof params);
            params.prefix_nibbles = nib;
            params.prefix_bytes = (uint32_t)prefix;
            params.prefix_bytes_hi = (uint32_t)(prefix >> 32);
            uint64_t mask = prefix_mask8(nib);

            std::vector<uint64_t> ys;
            uint64_t r = rnd64();
            ys.push_back(r);
            // Exact match of the prefix, with random bits beyond it.
            uint64_t exact = (r & ~keep) | prefix;
            ys.push_back(exact);
            // One nibble flipped at every nibble position 0..15 (nibble k =
            // bits 4k..4k+3 of the little-endian u64; the prefix's nibble
            // order is high-then-low within each byte).
            for (int k = 0; k < 16; k++) {
                uint64_t flip = 1ULL << (4 * k + (rnd64() % 4));
                ys.push_back(exact ^ flip);
            }
            // First-byte rejection cases.
            ys.push_back((exact & ~0xFFULL));
            ys.push_back(exact | 0xFFULL);
            for (uint64_t y : ys) {
                bool a = ref_matches_prefix(y, &params);
                bool b = matches_prefix_masked(y, prefix, mask);
                if (a != b) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "nib=%u y=%016llx prefix=%016llx ref=%d new=%d",
                        nib, (unsigned long long)y, (unsigned long long)prefix, (int)a, (int)b);
                    fail("matches_prefix", buf);
                }
                checked++;
            }
        }
    }
    return checked;
}

// Meshtastic helpers: fe_tobytes_words8 must be fe_tobytes32 repacked, and
// crc32_words8 (slicing-by-4 over the host-built tables) must equal a
// bitwise CRC-32 (ErriezCRC32's loop) of the 32 bytes.
static uint32_t ref_crc32(const uint8_t* b, size_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= b[i];
        for (int j = 0; j < 8; j++) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

static std::vector<uint32_t> build_crc_tables() {
    std::vector<uint32_t> t(1024);
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[i] = c;
    }
    for (int k = 1; k < 4; k++) {
        for (int i = 0; i < 256; i++) {
            uint32_t prev = t[(k - 1) * 256 + i];
            t[k * 256 + i] = (prev >> 8) ^ t[prev & 0xFFu];
        }
    }
    return t;
}

static int builtin_mesh_checks(int rounds) {
    int checked = 0;
    std::vector<uint32_t> tab = build_crc_tables();
    for (int n = 0; n < rounds; n++) {
        Fe f = random_fe(n % 5);
        uint8_t bytes[32];
        fe_tobytes32(bytes, f);
        uint32_t words[8];
        fe_tobytes_words8(words, f);
        uint8_t repacked[32];
        for (int i = 0; i < 8; i++) {
            repacked[4 * i] = (uint8_t)words[i];
            repacked[4 * i + 1] = (uint8_t)(words[i] >> 8);
            repacked[4 * i + 2] = (uint8_t)(words[i] >> 16);
            repacked[4 * i + 3] = (uint8_t)(words[i] >> 24);
        }
        if (std::memcmp(bytes, repacked, 32) != 0) {
            fail("fe_tobytes_words8 vs fe_tobytes32", "fe=" + fe_hex(f) + " bytes=" + bytes_hex(bytes, 32)
                + " words=" + bytes_hex(repacked, 32));
        }
        uint32_t crc = crc32_words8(words, tab.data());
        uint32_t ref = ref_crc32(bytes, 32);
        if (crc != ref) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "kernel=%08x ref=%08x", crc, ref);
            fail("crc32_words8 vs bitwise", std::string(buf) + " bytes=" + bytes_hex(bytes, 32));
        }
        checked++;
    }
    // The IEEE check value through the word path.
    {
        uint8_t msg[32] = {'1','2','3','4','5','6','7','8','9'};
        if (ref_crc32(msg, 9) != 0xCBF43926u) fail("ref crc32 check value", "");
    }
    return checked;
}

// ---------------------------------------------------------------------------
// Vector-driven scenarios.
// ---------------------------------------------------------------------------

static std::vector<uint32_t> table_words;
static std::vector<uint32_t> step_words;
static std::vector<uint32_t> mesh_step_words;
static int scenarios_run = 0;
static int field_vectors = 0;

struct Scenario {
    bool active = false;
    bool mesh = false;
    uint32_t threads = 0, iters = 0, nibbles = 0;
    uint32_t prefix_words[8] = {0};
    MeshParams mesh_params;
    std::vector<uint32_t> mesh_tables;
    std::vector<uint32_t> bases;
    std::vector<std::pair<uint32_t, uint32_t>> hits; // (thread, iter)
    std::vector<uint32_t> debug;
};

static void run_scenario(Scenario& sc, int line_no) {
    scenarios_run++;
    std::string ctx = "scenario@" + std::to_string(line_no);
    if (table_words.empty()) {
        fail("scenario without table", ctx);
        return;
    }
    // The step table must be the one for the compiled window (W/2 + 1
    // entries of 30 words), or the walk reads past it.
    if (step_words.size() != (size_t)(WALK_HALF + 1) * 30) {
        fail("scenario step table size != (BATCH_INV_WINDOW/2 + 1) * 30", ctx);
        return;
    }
    if (sc.bases.size() != (size_t)sc.threads * 8) {
        fail("scenario base count", ctx);
        return;
    }
    std::sort(sc.hits.begin(), sc.hits.end());
    uint32_t debug_out[8] = {0};
    blockDim.x = 1;
    threadIdx.x = 0;

    if (sc.mesh) {
        if (mesh_step_words.size() != (size_t)(WALK_HALF + 1) * 30) {
            fail("mesh scenario step table size != (BATCH_INV_WINDOW/2 + 1) * 30", ctx);
            return;
        }
        if (sc.mesh_tables.size() != MESH_TABLE_WORDS) {
            fail("mesh scenario table words", ctx);
            return;
        }
        sc.mesh_params.count = sc.threads;
        sc.mesh_params.iters = sc.iters;
        uint32_t found_count = 0;
        uint32_t found_index[MAX_HITS_PER_LAUNCH], found_iter[MAX_HITS_PER_LAUNCH];
        std::memset(found_index, 0xFF, sizeof found_index);
        std::memset(found_iter, 0xFF, sizeof found_iter);
        for (uint32_t gid = 0; gid <= sc.threads; gid++) {
            blockIdx.x = gid;
            mesh_kernel(sc.bases.data(), &sc.mesh_params, sc.mesh_tables.data(), &found_count,
                found_index, found_iter, debug_out, table_words.data(), mesh_step_words.data());
        }
        std::vector<std::pair<uint32_t, uint32_t>> got;
        uint32_t n = found_count < MAX_HITS_PER_LAUNCH ? found_count : MAX_HITS_PER_LAUNCH;
        for (uint32_t i = 0; i < n; i++) got.push_back({found_index[i], found_iter[i]});
        std::sort(got.begin(), got.end());
        if (found_count != sc.hits.size() || got != sc.hits) {
            std::string s = ctx + " expected {";
            for (auto& h : sc.hits) s += "(" + std::to_string(h.first) + "," + std::to_string(h.second) + ")";
            s += "} got {";
            for (auto& h : got) s += "(" + std::to_string(h.first) + "," + std::to_string(h.second) + ")";
            s += "} count=" + std::to_string(found_count);
            fail("mesh: hit set", s);
        }
        if (!sc.debug.empty()) {
            if (sc.debug.size() != 8 || std::memcmp(sc.debug.data(), debug_out, 32) != 0) {
                fail("mesh debug u", ctx);
            }
        }
        return;
    }

    {
        SearchParams params;
        std::memset(&params, 0, sizeof params);
        params.count = sc.threads;
        params.prefix_nibbles = sc.nibbles;
        params.iters = sc.iters;
        for (int i = 0; i < 8; i++) params.prefix_full[i] = sc.prefix_words[i];
        params.prefix_bytes = sc.prefix_words[0];
        params.prefix_bytes_hi = sc.prefix_words[1];
        uint32_t found = 0, found_index = 0xFFFFFFFFu, found_iter = 0xFFFFFFFFu;
        // Emulate the grid: one thread per block so gid == blockIdx.x; the
        // extra out-of-range thread must return without touching anything.
        for (uint32_t gid = 0; gid <= sc.threads; gid++) {
            blockIdx.x = gid;
            search_kernel(sc.bases.data(), &params, &found, &found_index, &found_iter,
                debug_out, table_words.data(), step_words.data());
        }
        if (sc.hits.empty()) {
            if (found != 0) {
                fail("search: unexpected hit", ctx + " index=" + std::to_string(found_index)
                    + " iter=" + std::to_string(found_iter));
            }
        } else {
            if (found != 1) {
                fail("search: missed planted key", ctx + " expected "
                    + std::to_string(sc.hits.size()) + " candidate(s)");
            } else {
                std::pair<uint32_t, uint32_t> got(found_index, found_iter);
                if (!std::binary_search(sc.hits.begin(), sc.hits.end(), got)) {
                    fail("search: hit not in expected set", ctx + " index="
                        + std::to_string(found_index) + " iter=" + std::to_string(found_iter));
                }
            }
        }
    }

    if (!sc.debug.empty()) {
        if (sc.debug.size() != 8 || std::memcmp(sc.debug.data(), debug_out, 32) != 0) {
            fail("debug point", ctx);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s vectors.txt\n", argv[0]);
        return 2;
    }
    int field_checked = builtin_field_checks(20000);
    int prefix_checked = builtin_prefix_checks(200);
    int mesh_checked = builtin_mesh_checks(20000);

    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::string line;
    int line_no = 0;
    Scenario sc;
    while (std::getline(in, line)) {
        line_no++;
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "F") {
            std::string a, b, mul, sq;
            ss >> a >> b >> mul >> sq;
            Fe fa = parse_fe(a), fb = parse_fe(b);
            std::vector<uint8_t> em = parse_bytes(mul), es = parse_bytes(sq);
            uint8_t out[32];
            fe_tobytes32(out, fe_mul(fa, fb));
            if (em.size() != 32 || std::memcmp(out, em.data(), 32) != 0) {
                fail("F mul", "line " + std::to_string(line_no) + " got " + bytes_hex(out, 32));
            }
            fe_tobytes32(out, fe_sq(fa));
            if (es.size() != 32 || std::memcmp(out, es.data(), 32) != 0) {
                fail("F sq", "line " + std::to_string(line_no) + " got " + bytes_hex(out, 32));
            }
            check_field_pair(fa, fb, "vector");
            field_vectors++;
        } else if (tag == "T") {
            std::string w;
            ss >> w;
            table_words = parse_words(w);
        } else if (tag == "P") {
            std::string w;
            ss >> w;
            step_words = parse_words(w);
        } else if (tag == "U") {
            std::string w;
            ss >> w;
            mesh_step_words = parse_words(w);
        } else if (tag == "M") {
            sc = Scenario();
            sc.active = true;
            sc.mesh = true;
            std::memset(&sc.mesh_params, 0, sizeof sc.mesh_params);
            std::string mask;
            ss >> sc.threads >> sc.iters >> mask >> sc.mesh_params.target_count;
            sc.mesh_params.mask = (uint32_t)std::strtoul(mask.c_str(), nullptr, 16);
            for (int i = 0; i < 64; i++) {
                std::string w;
                ss >> w;
                sc.mesh_params.targets[i] = (uint32_t)std::strtoul(w.c_str(), nullptr, 16);
            }
        } else if (tag == "K") {
            std::string w;
            ss >> w;
            sc.mesh_tables = parse_words(w);
        } else if (tag == "W") {
            sc = Scenario();
            sc.active = true;
            ss >> sc.threads >> sc.iters >> sc.nibbles;
            for (int i = 0; i < 8; i++) {
                std::string w;
                ss >> w;
                sc.prefix_words[i] = (uint32_t)std::strtoul(w.c_str(), nullptr, 16);
            }
        } else if (tag == "B") {
            std::string w;
            ss >> w;
            std::vector<uint32_t> words = parse_words(w);
            if (words.size() != 8) fail("B words", "line " + std::to_string(line_no));
            sc.bases.insert(sc.bases.end(), words.begin(), words.end());
        } else if (tag == "X") {
            uint32_t t, i;
            ss >> t >> i;
            sc.hits.push_back({t, i});
        } else if (tag == "D") {
            std::string w;
            ss >> w;
            sc.debug = parse_words(w);
        } else if (tag == "E") {
            if (!sc.active) fail("E without scenario", "line " + std::to_string(line_no));
            else run_scenario(sc, line_no);
            sc = Scenario();
        } else {
            fail("unknown tag", "line " + std::to_string(line_no) + ": " + tag);
        }
    }
    std::printf(
        "builtin field checks %d, builtin prefix checks %d, builtin mesh checks %d, field vectors %d, scenarios %d failures %d\n",
        field_checked, prefix_checked, mesh_checked, field_vectors, scenarios_run, failures);
    return failures == 0 ? 0 : 1;
}
