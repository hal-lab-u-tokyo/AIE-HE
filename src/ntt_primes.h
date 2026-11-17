#pragma once
// RNS primes shared by the AIE encrypt and the CPU baselines (SEAL, encode),
// so both sides compute with the same moduli: q = 1 mod 2N (a primitive 2N-th
// root for the negacyclic NTT), searched downward from 2^30 like SEAL's
// CoeffModulus::Create(N, {30, ...}). The AIE kernels need 2^29 < q < 2^30
// (w = 30): their Barrett and Shoup w' use the uint32 constant floor(2^61/q)
// and stay within one correction there (barrett() in ckks_encrypt/kernel_func.cc).
// NOTE: ckks_decrypt and ntt/ keep their own primes and the old int32 Barrett
// (floor(2^60/q), q < 2^29.5); port them before they use these primes.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Deterministic Miller-Rabin for n < 4,759,123,141 (bases 2, 7, 61).
inline bool is_prime_u32(uint64_t n)
{
  if (n < 2) return false;
  for (uint64_t p : {2, 3, 5, 7, 61})
    if (n % p == 0) return n == p;
  uint64_t d = n - 1;
  int r = 0;
  for (; d % 2 == 0; d /= 2) ++r;
  auto mulmod = [n](uint64_t a, uint64_t b) { return a * b % n; };  // n < 2^32
  for (uint64_t a : {2, 7, 61}) {
    uint64_t x = 1, base = a, e = d;
    for (; e; e >>= 1, base = mulmod(base, base))
      if (e & 1) x = mulmod(x, base);
    if (x == 1 || x == n - 1) continue;
    bool composite = true;
    for (int i = 1; i < r && composite; ++i) {
      x = mulmod(x, x);
      if (x == n - 1) composite = false;
    }
    if (composite) return false;
  }
  return true;
}

// The first `count` primes q = 1 mod 2^(n_log+1) below 2^30, descending (SEAL's order).
inline std::vector<uint64_t> ntt_primes(int count, int n_log)
{
  const uint64_t step = uint64_t{1} << (n_log + 1);
  std::vector<uint64_t> primes;
  for (uint64_t q = (uint64_t{1} << 30) - step + 1; int(primes.size()) < count; q -= step) {
    if (q < (uint64_t{1} << 29))
      throw std::runtime_error("not enough NTT primes in (2^29, 2^30) for logN=" + std::to_string(n_log) +
                               " (need " + std::to_string(count) + ", found " + std::to_string(primes.size()) + ")");
    if (is_prime_u32(q)) primes.push_back(q);
  }
  return primes;
}
