#pragma once

#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/memorymanager.h"
#include "seal/util/croots.h"
#include "seal/util/dwthandler.h"
#include "seal/util/pointer.h"
#include "../ntt_primes.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ckks_encode {

struct EncodedPlaintext {
  std::size_t coeff_count = 0;
  std::size_t coeff_modulus_size = 0;
  double scale = 1.0;
  std::vector<std::uint64_t> coeffs;

  const std::uint64_t *data(std::size_t modulus_index = 0) const;
  std::uint64_t *data(std::size_t modulus_index = 0);
};

class CKKSEncoderNoNTT {
public:
  explicit CKKSEncoderNoNTT(const seal::SEALContext &context);

  std::size_t slot_count() const noexcept;

  // SEAL's reduction loop order (coefficient-major, strided writes), only to
  // measure what the loop order costs (encode_loop_bench).
  bool seal_loop_order = false;

  template <typename T>
  EncodedPlaintext encode(const std::vector<T> &values, double scale) const
  {
    EncodedPlaintext destination;
    encode(values.data(), values.size(), context_.first_parms_id(), scale, destination, seal::MemoryManager::GetPool());
    return destination;
  }

  template <typename T>
  void encode(const std::vector<T> &values, double scale, EncodedPlaintext &destination) const
  {
    encode(values.data(), values.size(), context_.first_parms_id(), scale, destination, seal::MemoryManager::GetPool());
  }

  template <typename T>
  EncodedPlaintext encode(const T *values,
                          std::size_t values_size,
                          seal::parms_id_type parms_id,
                          double scale,
                          seal::MemoryPoolHandle pool) const;

  template <typename T>
  void encode(const T *values,
              std::size_t values_size,
              seal::parms_id_type parms_id,
              double scale,
              EncodedPlaintext &destination,
              seal::MemoryPoolHandle pool) const;

  // The modulus-independent part of encode (first_parms_id): the plaintext's
  // coefficients m_i = round(scale * IFFT(values)) in natural order as |m_i| and
  // sign, for a caller that reduces them per modulus itself (encode does both).
  // Needs |m_i| < 2^64. Returns the bit count of the largest |m_i| plus one (as
  // SEAL's encode).
  int encode_coefficients(const std::vector<double> &values, double scale, std::vector<std::uint64_t> &abs,
                           std::vector<std::uint8_t> &negative) const;

private:
  // conj_values = scale * IFFT(values) (real parts are the coefficients) and
  // the bit count of the largest coefficient, after encode's argument checks.
  template <typename T>
  int scaled_ifft(const T *values, std::size_t values_size, const seal::SEALContext::ContextData &context_data,
                  double scale, std::complex<double> *conj_values) const;

  using ComplexArith = seal::util::Arithmetic<std::complex<double>, std::complex<double>, double>;
  using FFTHandler = seal::util::DWTHandler<std::complex<double>, std::complex<double>, double>;

  seal::MemoryPoolHandle pool_ = seal::MemoryManager::GetPool();
  seal::SEALContext context_;
  std::size_t slots_ = 0;
  std::shared_ptr<seal::util::ComplexRoots> complex_roots_;
  seal::util::Pointer<std::complex<double>> inv_root_powers_;
  seal::util::Pointer<std::size_t> matrix_reps_index_map_;
  ComplexArith complex_arith_;
  FFTHandler fft_handler_;
};

// Benchmark moduli for L data primes counted as 30-bit ones (rns_L of the AIE
// encrypt), plus SEAL's special prime: the AIE's L primes (ntt_primes), or with
// prime_bits 60 L/2 of SEAL's 60-bit primes for the same log Q (L even; the
// AIE int32 kernels cannot use them, SEAL's 64-bit words can).
inline std::vector<seal::Modulus> bench_moduli(int L, int log_n, int prime_bits)
{
  if (prime_bits == 30) {
    const auto primes = ntt_primes(L + 1, log_n);
    return std::vector<seal::Modulus>(primes.begin(), primes.end());
  }
  return seal::CoeffModulus::Create(std::size_t(1) << log_n, std::vector<int>(L / (prime_bits / 30) + 1, prime_bits));
}

} // namespace ckks_encode
