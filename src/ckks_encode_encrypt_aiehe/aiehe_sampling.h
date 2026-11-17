#pragma once
#include "blake2xb_prng.h"
#include <cstdint>
#include <utility>
#include <vector>

// Error (centered binomial, SEAL's default) and ternary sampling with SEAL's
// CSPRNG (Blake2xb), so sampling draws the same distributions and the same kind
// of randomness as SEAL. The seed is fixed for reproducible experiments;
// production must seed from the OS.
class AieheSampler {
public:
  explicit AieheSampler(uint32_t seed)
      : e0_rng(derive_seed(seed, 0x101u)), v_rng(derive_seed(seed, 0x202u)), e1_rng(derive_seed(seed, 0x303u)) {}
  void sample_v(std::vector<int32_t> &v) {
    // Uniform in {-1, 0, 1} as SEAL: two random bits, rejecting 3.
    uint32_t bits = 0;
    int left = 0;
    for (auto &x : v) {
      uint32_t t;
      do {
        if (left == 0) {
          bits = v_rng();
          left = 16;
        }
        t = bits & 3;
        bits >>= 2;
        --left;
      } while (t == 3);
      x = int32_t(t) - 1;
    }
  }
  void sample_e0(std::vector<int32_t> &e0) { sample_cbd(e0_rng, e0); }
  void sample_e1(std::vector<int32_t> &e1) { sample_cbd(e1_rng, e1); }
private:
  // N is even, as in the original.
  void sample_cbd(Blake2xbPRNG &rng, std::vector<int32_t> &e) {
    for (size_t i = 0; i < e.size(); i += 2) {
      const auto [z0, z1] = cbd_pair(rng);
      e[i] = z0;
      e[i + 1] = z1;
    }
  }
  // Centered binomial as SEAL's sample_poly_cbd: popcount of 21 random bits
  // minus popcount of 21 more (variance 10.5, |e| <= 21). SEAL reads 6 bytes per
  // value; here two values take 96 bits, 42 of each 48.
  static int32_t cbd(uint64_t x) {
    constexpr uint64_t m21 = (uint64_t{1} << 21) - 1;
    return int32_t(__builtin_popcountll(x & m21)) - int32_t(__builtin_popcountll((x >> 21) & m21));
  }
  std::pair<int32_t, int32_t> cbd_pair(Blake2xbPRNG &rng) {
    const uint64_t w0 = rng(), w1 = rng(), w2 = rng();
    return {cbd(w0 | (w1 & 0xffff) << 32), cbd(w1 >> 16 | w2 << 16)};
  }
  // Expand the 32-bit experiment seed into a 64-byte key (splitmix64).
  static Blake2xbPRNG::seed_type derive_seed(uint32_t seed, uint32_t stream) {
    Blake2xbPRNG::seed_type key{};
    uint64_t x = (uint64_t(stream) << 32) | seed;
    for (auto &word : key) {
      uint64_t z = (x += 0x9e3779b97f4a7c15ull);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      word = z ^ (z >> 31);
    }
    return key;
  }
  Blake2xbPRNG e0_rng, v_rng, e1_rng;
};
