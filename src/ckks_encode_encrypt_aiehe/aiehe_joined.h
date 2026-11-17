#pragma once
// AIE-HE with its Encode connected to its Encrypt (README.md).
// The published hosts (../ckks_encrypt_planB, planA) encrypt an int32 test
// message, and the published Encode (../ckks_encode) decomposes the coefficients
// into residues that nothing reads, while the Encrypt's P1 input preparation
// reduces its input per modulus anyway. Connected, Encode stops at the rounded
// coefficients and the preparation reduces them; everything else is as published.
#include <cstdint>
#include <vector>

namespace aiehe_joined {

// x mod q for |x| < 2^62, q < 2^31, by Barrett (as SEAL's decomposition in the
// published Encode, not a 64-bit division).
inline uint32_t reduce(int64_t x, uint32_t q, uint64_t ratio, uint64_t bias)
{
  const uint64_t u = uint64_t(x + int64_t(bias));  // bias: a multiple of q above 2^62
  uint64_t r = u - uint64_t((unsigned __int128)u * ratio >> 64) * q;  // in [0, 3q)
  r -= r >= q ? q : 0;
  r -= r >= q ? q : 0;
  return uint32_t(r);
}
inline uint64_t ratio_of(uint32_t q) { return UINT64_MAX / q; }
inline uint64_t bias_of(uint32_t q) { return ((uint64_t(1) << 62) / q + 1) * q; }

// The published preparation, (x + q) * psi^i % q into AIE order, for x = m + e0 in int64.
inline void prepare(const std::vector<int64_t> &in, int *buf, int size, const int *idx, int q,
                    const std::vector<int32_t> &psi_pow)
{
  const uint64_t ratio = ratio_of(q), bias = bias_of(q);
  for (int i = 0; i < size; i++) buf[idx[i]] = int32_t(int64_t(reduce(in[i], q, ratio, bias)) * psi_pow[i] % q);
}

}  // namespace aiehe_joined

// Overloads of the hosts' own preparations (int32 input) for the connected input.
inline void prepare_for_Phase1(std::vector<int64_t> &in, int *buf, int size, int *idx, int q, std::vector<int32_t> &psi_pow)
{
  aiehe_joined::prepare(in, buf, size, idx, q, psi_pow);
}
inline void prepare_for_TuboiNTT(std::vector<int64_t> &in, int *buf, int size, int *idx, int q, std::vector<int32_t> &psi_pow)
{
  aiehe_joined::prepare(in, buf, size, idx, q, psi_pow);
}

#ifndef AIEHE_JOINED_NO_SEAL
#include "../ntt_primes.h"
#include "ckks_encoder.h"
#include "cxxopts.hpp"
#include <cmath>
#include <random>

namespace aiehe_joined {

inline void add_options(cxxopts::Options &options)
{
  options.add_options()("scale_bits", "CKKS scale 2^scale_bits of the connected Encode", cxxopts::value<int>()->default_value("28"));
}

inline seal::EncryptionParameters encode_parms(int n_log, int L)
{
  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(size_t(1) << n_log);
  const auto primes = ntt_primes(L + 1, n_log);  // Encode does not reduce
  parms.set_coeff_modulus(std::vector<seal::Modulus>(primes.begin(), primes.end()));
  return parms;
}

// N/2 random slots in [-1, 1], e0 (filled by the host from its sampler), and the Encrypt's input m + e0.
struct Message {
  seal::SEALContext context;
  ckks_encode::CKKSEncoderNoNTT encoder;
  double scale;
  std::vector<double> slots;
  std::vector<int32_t> e0;  // zero until the host samples it (aiehe_sampling.h) before each encode()
  std::vector<uint64_t> abs;
  std::vector<uint8_t> negative;
  std::vector<int64_t> md;

  Message(int n_log, int L, int scale_bits)
      : context(encode_parms(n_log, L), true, seal::sec_level_type::none), encoder(context),
        scale(std::ldexp(1.0, scale_bits)), slots(size_t(1) << (n_log - 1)), e0(size_t(1) << n_log),
        md(size_t(1) << n_log)
  {
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> slot(-1.0, 1.0);
    for (double &x : slots) x = slot(gen);
  }
  // Encode up to the rounded coefficients, then m + e0 (timed by the host).
  void encode()
  {
    if (encoder.encode_coefficients(slots, scale, abs, negative) > 61) throw std::invalid_argument("scale too large");
    for (size_t i = 0; i < md.size(); i++) md[i] = (negative[i] ? -int64_t(abs[i]) : int64_t(abs[i])) + e0[i];
  }
  // (m + e0) mod q for the reference, independently of reduce().
  int32_t md_mod(int i, int32_t q) const { return int32_t((md[i] % q + q) % q); }
};

}  // namespace aiehe_joined
#endif
