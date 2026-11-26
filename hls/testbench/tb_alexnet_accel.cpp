// tb_alexnet_accel.cpp -- C simulation / C-RTL co-simulation testbench
//
// Part 1 (always): self-checking unit tests.  Random small layers are run
//   through alexnet_accel() and compared bit-exactly against an independent,
//   deliberately naive reference written directly from the math (unpacked
//   int arrays, direct 7-deep loops).  Covers K=3/K=5, pad 0/1/2, pooling,
//   ReLU on/off, saturation, negative values, FC requant and raw output and
//   the error-code paths.  These layers also fit the m_axi depth=8192 used
//   for co-simulation.
// Part 2 (csim only, when env ALEXNET_GOLDEN=<dir> is set): full AlexNet.
//   Loads build/deploy/golden/ produced by model/quantize_model.py, runs the
//   8 layers exactly as the PYNQ driver does and compares every layer output
//   with the bit-exact Python model.  (Too slow for RTL co-simulation.)
//
// Exit code 0 = PASS (required by Vitis HLS csim/cosim).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "alexnet_accel.h"

// ----------------------------------------------------------------------------
static uint32_t rng_state = 12345u;
static uint32_t rnd() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }
static int rnd_range(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// Every buffer handed to the DUT is at least COSIM_DEPTH words long, matching
// the m_axi depth=8192 pragma so RTL co-simulation never reads past an array.
static const int COSIM_DEPTH = 8192;
template <class T> static void pad_depth(std::vector<T> &v) { if ((int)v.size() < COSIM_DEPTH) v.resize(COSIM_DEPTH, 0); }

static int8_t lane(const word_t &w, int l) { return (int8_t)(unsigned)w.range(8 * l + 7, 8 * l); }
static void set_lane(word_t &w, int l, int8_t v) { w.range(8 * l + 7, 8 * l) = (uint8_t)v; }

static int8_t ref_requant(int64_t acc, int M, int sh, bool relu) {
    if (sh < 1) sh = 1;
    int64_t r = (acc * (int64_t)M + ((int64_t)1 << (sh - 1))) >> sh;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    if (relu && r < 0) r = 0;
    return (int8_t)r;
}

static word_t make_param(int32_t bias, int M, int sh) {
    word_t p = 0;
    p.range(31, 0) = (uint32_t)bias;
    p.range(47, 32) = (unsigned)M;
    p.range(53, 48) = (unsigned)sh;
    return p;
}

// ----------------------------------------------------------------------------
struct ConvCase { int cin, h, w, cout, k, pad, pool, relu, wmag, xmag; };

static int test_conv(const ConvCase &tc, int id) {
    const int Ho = tc.h + 2 * tc.pad - tc.k + 1, Wo = tc.w + 2 * tc.pad - tc.k + 1;
    const int Hp = tc.pool ? (Ho - 3) / 2 + 1 : Ho, Wp = tc.pool ? (Wo - 3) / 2 + 1 : Wo;
    std::vector<int8_t> x(tc.cin * tc.h * tc.w), wt(tc.cout * tc.cin * tc.k * tc.k);
    std::vector<int32_t> bias(tc.cout);
    std::vector<int> M(tc.cout), sh(tc.cout);
    for (auto &v : x) v = (int8_t)rnd_range(-tc.xmag, tc.xmag);
    for (auto &v : wt) v = (int8_t)rnd_range(-tc.wmag, tc.wmag);
    for (int o = 0; o < tc.cout; o++) {
        bias[o] = rnd_range(-20000, 20000);
        M[o] = rnd_range(16384, 32767);
        sh[o] = rnd_range(14, 24);
    }
    // ---- naive reference ----
    std::vector<int8_t> y(tc.cout * Ho * Wo), yp(tc.cout * Hp * Wp);
    for (int o = 0; o < tc.cout; o++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                int64_t acc = bias[o];
                for (int c = 0; c < tc.cin; c++)
                    for (int ky = 0; ky < tc.k; ky++)
                        for (int kx = 0; kx < tc.k; kx++) {
                            int iy = oy + ky - tc.pad, ix = ox + kx - tc.pad;
                            if (iy < 0 || iy >= tc.h || ix < 0 || ix >= tc.w) continue;
                            acc += (int64_t)x[(c * tc.h + iy) * tc.w + ix] *
                                   wt[((o * tc.cin + c) * tc.k + ky) * tc.k + kx];
                        }
                y[(o * Ho + oy) * Wo + ox] = ref_requant(acc, M[o], sh[o], tc.relu);
            }
    for (int o = 0; o < tc.cout; o++)
        for (int py = 0; py < Hp; py++)
            for (int px = 0; px < Wp; px++) {
                if (!tc.pool) { yp[(o * Hp + py) * Wp + px] = y[(o * Ho + py) * Wo + px]; continue; }
                int8_t m = -128;
                for (int dy = 0; dy < 3; dy++)
                    for (int dx = 0; dx < 3; dx++) {
                        int8_t v = y[(o * Ho + 2 * py + dy) * Wo + 2 * px + dx];
                        if (v > m) m = v;
                    }
                yp[(o * Hp + py) * Wp + px] = m;
            }
    // ---- pack for the kernel ----
    const int cig = tc.cin / 8;
    std::vector<word_t> in_w(cig * tc.h * tc.w, 0), out_w((tc.cout / 8) * Hp * Wp + 8, 0);
    for (int c = 0; c < tc.cin; c++)
        for (int i = 0; i < tc.h * tc.w; i++)
            set_lane(in_w[(c / 8) * tc.h * tc.w + i], c % 8, x[c * tc.h * tc.w + i]);
    std::vector<word_t> w_w, p_w;
    for (int t = 0; t < tc.cout / 16; t++)
        for (int to = 0; to < 16; to++)
            for (int g = 0; g < cig; g++)
                for (int ky = 0; ky < tc.k; ky++)
                    for (int kx = 0; kx < tc.k; kx++) {
                        word_t wd = 0;
                        int o = t * 16 + to;
                        for (int l = 0; l < 8; l++)
                            set_lane(wd, l, wt[((o * tc.cin + g * 8 + l) * tc.k + ky) * tc.k + kx]);
                        w_w.push_back(wd);
                    }
    for (int o = 0; o < tc.cout; o++) p_w.push_back(make_param(bias[o], M[o], sh[o]));
    // sentinel after output to detect overruns
    const word_t SENT = word_t("0xDEADBEEFCAFEF00D", 16);
    pad_depth(in_w); pad_depth(w_w); pad_depth(p_w); pad_depth(out_w);
    for (auto &v : out_w) v = SENT;

    int rc = alexnet_accel(in_w.data(), out_w.data(), w_w.data(), p_w.data(), OP_CONV,
                           tc.cin, tc.h, tc.w, tc.cout, tc.k, tc.pad, Ho, Wo,
                           tc.pool, tc.relu, 0);
    int errors = (rc != 0);
    for (int o = 0; o < tc.cout; o++)
        for (int i = 0; i < Hp * Wp; i++) {
            int8_t got = lane(out_w[(o / 8) * Hp * Wp + i], o % 8);
            int8_t exp = yp[o * Hp * Wp + i];
            if (got != exp && errors++ < 5)
                printf("  conv%d mismatch o=%d i=%d got %d exp %d\n", id, o, i, got, exp);
        }
    for (int i = (tc.cout / 8) * Hp * Wp; i < (int)out_w.size(); i++)
        if (out_w[i] != SENT) { errors++; printf("  conv%d overrun at word %d\n", id, i); break; }
    printf("[conv %d] cin=%d %dx%d cout=%d k=%d pad=%d pool=%d relu=%d -> %s (%d errors)\n",
           id, tc.cin, tc.h, tc.w, tc.cout, tc.k, tc.pad, tc.pool, tc.relu,
           errors ? "FAIL" : "PASS", errors);
    return errors;
}

static int test_fc(int cin, int cout, int relu, int raw, int id) {
    std::vector<int8_t> x(cin), wt(cout * cin);
    std::vector<int32_t> bias(cout);
    std::vector<int> M(cout), sh(cout);
    for (auto &v : x) v = (int8_t)rnd_range(-128, 127);
    for (auto &v : wt) v = (int8_t)rnd_range(-127, 127);
    for (int o = 0; o < cout; o++) { bias[o] = rnd_range(-100000, 100000); M[o] = rnd_range(16384, 32767); sh[o] = rnd_range(16, 26); }
    std::vector<word_t> in_w(cin / 8, 0), w_w(cout * cin / 8, 0), p_w(cout), out_w(cout + 4, 0);
    for (int i = 0; i < cin; i++) set_lane(in_w[i / 8], i % 8, x[i]);
    for (int o = 0; o < cout; o++) {
        for (int i = 0; i < cin; i++) set_lane(w_w[o * (cin / 8) + i / 8], i % 8, wt[o * cin + i]);
        p_w[o] = make_param(bias[o], M[o], sh[o]);
    }
    pad_depth(in_w); pad_depth(w_w); pad_depth(p_w); pad_depth(out_w);
    int rc = alexnet_accel(in_w.data(), out_w.data(), w_w.data(), p_w.data(), OP_FC,
                           cin, 1, 1, cout, 1, 0, 1, 1, 0, relu, raw);
    int errors = (rc != 0);
    for (int o = 0; o < cout; o++) {
        int64_t acc = bias[o];
        for (int i = 0; i < cin; i++) acc += (int64_t)x[i] * wt[o * cin + i];
        if (raw) {
            int32_t got = (int32_t)(uint32_t)out_w[o / 2].range(32 * (o & 1) + 31, 32 * (o & 1));
            if (got != (int32_t)acc && errors++ < 5) printf("  fc%d raw o=%d got %d exp %lld\n", id, o, got, (long long)acc);
        } else {
            int8_t got = lane(out_w[o / 8], o % 8), exp = ref_requant(acc, M[o], sh[o], relu);
            if (got != exp && errors++ < 5) printf("  fc%d o=%d got %d exp %d\n", id, o, got, exp);
        }
    }
    printf("[fc   %d] cin=%d cout=%d relu=%d raw=%d -> %s (%d errors)\n",
           id, cin, cout, relu, raw, errors ? "FAIL" : "PASS", errors);
    return errors;
}

static int test_errors() {
    std::vector<word_t> a(COSIM_DEPTH, 0), b(COSIM_DEPTH, 0);
    int e = 0;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), 7, 8, 4, 4, 16, 3, 1, 4, 4, 0, 1, 0) != ERR_BAD_OP;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), OP_CONV, 12, 4, 4, 16, 3, 1, 4, 4, 0, 1, 0) != ERR_CHANNELS;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), OP_CONV, 8, 4, 4, 16, 3, 1, 5, 4, 0, 1, 0) != ERR_KERNEL;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), OP_CONV, 8, 4, 4, 16, 7, 1, 0, 0, 0, 1, 0) != ERR_KERNEL;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), OP_CONV, 512, 13, 13, 16, 3, 1, 13, 13, 0, 1, 0) != ERR_CAPACITY;
    e += alexnet_accel(a.data(), b.data(), a.data(), a.data(), OP_FC, 64, 1, 1, 12, 1, 0, 1, 1, 0, 1, 0) != ERR_CHANNELS;
    bool untouched = true;
    for (auto &v : b) untouched &= (v == 0);
    e += !untouched;
    printf("[errors ] status-code checks -> %s\n", e ? "FAIL" : "PASS");
    return e;
}

// ----------------------------------------------------------------------------
static bool read_bin(const std::string &path, std::vector<word_t> &v) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { printf("cannot open %s\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f) / 8; fseek(f, 0, SEEK_SET);
    std::vector<uint64_t> raw(n);
    size_t got = fread(raw.data(), 8, n, f); fclose(f);
    if ((long)got != n) return false;
    v.resize(n);
    for (long i = 0; i < n; i++) v[i] = raw[i];
    return true;
}

static int full_network(const char *dir) {
    std::string d(dir);
    FILE *lf = fopen((d + "/layers.txt").c_str(), "r");
    if (!lf) { printf("no layers.txt in %s\n", dir); return 1; }
    std::vector<word_t> act;
    if (!read_bin(d + "/input.bin", act)) return 1;
    act.resize(IN_BUF_WORDS, 0);
    int errors = 0;
    for (int i = 0; i < 8; i++) {
        int op, cin, hin, win, cout, k, pad, hout, wout, pool, relu, raw, nw, now;
        if (fscanf(lf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d", &op, &cin, &hin, &win, &cout,
                   &k, &pad, &hout, &wout, &pool, &relu, &raw, &nw, &now) != 14) { fclose(lf); return 1; }
        std::vector<word_t> w, p, g;
        char nb[64];
        snprintf(nb, sizeof nb, "/w%d.bin", i); if (!read_bin(d + nb, w)) return 1;
        snprintf(nb, sizeof nb, "/p%d.bin", i); if (!read_bin(d + nb, p)) return 1;
        snprintf(nb, sizeof nb, "/golden%d.bin", i); if (!read_bin(d + nb, g)) return 1;
        std::vector<word_t> out(IN_BUF_WORDS, 0);
        int rc = alexnet_accel(act.data(), out.data(), w.data(), p.data(), op, cin, hin, win, cout,
                               k, pad, hout, wout, pool, relu, raw);
        int bad = 0;
        for (int j = 0; j < now; j++) bad += (out[j] != g[j]);
        printf("[full L%d] op=%d cin=%d cout=%d rc=%d words=%d mismatched_words=%d -> %s\n",
               i, op, cin, cout, rc, now, bad, (rc || bad) ? "FAIL" : "PASS");
        errors += bad + (rc != 0);
        act = out;
    }
    fclose(lf);
    // top-5 of raw logits (unscaled; per-class scales are applied on the PS)
    printf("fc8 raw accumulators [0..4]: ");
    for (int j = 0; j < 5; j++) printf("%d ", (int32_t)(uint32_t)act[j / 2].range(32 * (j & 1) + 31, 32 * (j & 1)));
    printf("\n");
    return errors;
}

int main() {
    int errors = 0;
    const ConvCase cases[] = {
        // cin  h  w cout k pad pool relu wmag xmag
        {16,  9,  9, 32, 3, 1, 1, 1, 127, 127},   // conv3/4/5-like, pooled
        { 8, 11, 11, 16, 5, 2, 1, 0,  60, 127},   // conv2-like (K5), no ReLU
        {16,  7,  7, 16, 3, 0, 0, 1, 127, 127},   // conv1'-like (pad 0), no pool
        { 8,  6,  5, 32, 3, 1, 0, 0, 127, 127},   // non-square, saturation-heavy
        {24,  5,  5, 16, 5, 2, 1, 1,   3,   5},   // small magnitudes, 3 groups
    };
    int id = 0;
    for (const auto &c : cases) errors += test_conv(c, id++);
    errors += test_fc(64, 24, 1, 0, 0);
    errors += test_fc(128, 10, 0, 1, 1);
    errors += test_fc(32, 8, 0, 0, 2);
    errors += test_errors();
#ifndef __SYNTHESIS__
    const char *g = getenv("ALEXNET_GOLDEN");
    if (g && *g) errors += full_network(g);
#endif
    printf("\n==== %s (%d errors) ====\n", errors ? "TEST FAILED" : "ALL TESTS PASSED", errors);
    return errors ? 1 : 0;
}
