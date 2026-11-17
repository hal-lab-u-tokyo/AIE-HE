#include "ckks_encoder.h"

#include "seal/util/common.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>

namespace ckks_encode {
namespace {

template <typename T>
constexpr bool is_ckks_value_v =
    std::is_same_v<std::remove_cv_t<T>, double> || std::is_same_v<std::remove_cv_t<T>, std::complex<double>>;

} // namespace

const std::uint64_t *EncodedPlaintext::data(std::size_t modulus_index) const
{
  return coeffs.data() + modulus_index * coeff_count;
}

std::uint64_t *EncodedPlaintext::data(std::size_t modulus_index)
{
  return coeffs.data() + modulus_index * coeff_count;
}

CKKSEncoderNoNTT::CKKSEncoderNoNTT(const seal::SEALContext &context) : context_(context)
{
  if (!context_.parameters_set()) {
    throw std::invalid_argument("encryption parameters are not set correctly");
  }

  auto &context_data = *context_.first_context_data();
  if (context_data.parms().scheme() != seal::scheme_type::ckks) {
    throw std::invalid_argument("unsupported scheme");
  }

  const std::size_t coeff_count = context_data.parms().poly_modulus_degree();
  slots_ = coeff_count >> 1;
  const int logn = seal::util::get_power_of_two(coeff_count);

  matrix_reps_index_map_ = seal::util::allocate<std::size_t>(coeff_count, pool_);

  std::uint64_t gen = 3;
  std::uint64_t pos = 1;
  const std::uint64_t m = static_cast<std::uint64_t>(coeff_count) << 1;
  for (std::size_t i = 0; i < slots_; i++) {
    const std::uint64_t index1 = (pos - 1) >> 1;
    const std::uint64_t index2 = (m - pos - 1) >> 1;

    matrix_reps_index_map_[i] = static_cast<std::size_t>(seal::util::reverse_bits(index1, logn));
    matrix_reps_index_map_[slots_ | i] = static_cast<std::size_t>(seal::util::reverse_bits(index2, logn));

    pos *= gen;
    pos &= (m - 1);
  }

  inv_root_powers_ = seal::util::allocate<std::complex<double>>(coeff_count, pool_);
  if (m >= 8) {
    complex_roots_ = std::make_shared<seal::util::ComplexRoots>(static_cast<std::size_t>(m), pool_);
    for (std::size_t i = 1; i < coeff_count; i++) {
      inv_root_powers_[i] = std::conj(complex_roots_->get_root(seal::util::reverse_bits(i - 1, logn) + 1));
    }
  } else if (m == 4) {
    inv_root_powers_[1] = {0, -1};
  }

  complex_arith_ = ComplexArith();
  fft_handler_ = FFTHandler(complex_arith_);
}

std::size_t CKKSEncoderNoNTT::slot_count() const noexcept
{
  return slots_;
}

template <typename T>
int CKKSEncoderNoNTT::scaled_ifft(const T *values, std::size_t values_size,
                                  const seal::SEALContext::ContextData &context_data, double scale,
                                  std::complex<double> *conj_values) const
{
  if (scale <= 0 || (static_cast<int>(std::log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count())) {
    throw std::invalid_argument("scale out of bounds");
  }

  const std::size_t n = seal::util::mul_safe(slots_, std::size_t(2));
  for (std::size_t i = 0; i < values_size; i++) {
    conj_values[matrix_reps_index_map_[i]] = values[i];
    conj_values[matrix_reps_index_map_[i + slots_]] = std::conj(values[i]);
  }

  double fix = scale / static_cast<double>(n);
  fft_handler_.transform_from_rev(conj_values, seal::util::get_power_of_two(n), inv_root_powers_.get(), &fix);

  double max_coeff = 0;
  for (std::size_t i = 0; i < n; i++) {
    max_coeff = std::max(max_coeff, std::fabs(conj_values[i].real()));
  }

  const int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;
  if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
    throw std::invalid_argument("encoded values are too large");
  }
  return max_coeff_bit_count;
}

int CKKSEncoderNoNTT::encode_coefficients(const std::vector<double> &values, double scale,
                                           std::vector<std::uint64_t> &abs, std::vector<std::uint8_t> &negative) const
{
  if (values.size() > slots_) {
    throw std::invalid_argument("values_size is too large");
  }
  const std::size_t n = slots_ * 2;
  auto conj_values = seal::util::allocate<std::complex<double>>(n, pool_, 0);
  const int bits = scaled_ifft(values.data(), values.size(), *context_.first_context_data(), scale, conj_values.get());
  if (bits > 64) {
    throw std::invalid_argument("encode_coefficients needs coefficients below 2^64");
  }
  abs.resize(n);
  negative.resize(n);
  for (std::size_t i = 0; i < n; i++) {
    const double coeffd = std::round(conj_values[i].real());
    abs[i] = static_cast<std::uint64_t>(std::fabs(coeffd));
    negative[i] = std::signbit(coeffd) ? 1 : 0;
  }
  return bits;
}

template <typename T>
EncodedPlaintext CKKSEncoderNoNTT::encode(const T *values,
                                          std::size_t values_size,
                                          seal::parms_id_type parms_id,
                                          double scale,
                                          seal::MemoryPoolHandle pool) const
{
  EncodedPlaintext destination;
  encode(values, values_size, parms_id, scale, destination, std::move(pool));
  return destination;
}

template <typename T>
void CKKSEncoderNoNTT::encode(const T *values,
                              std::size_t values_size,
                              seal::parms_id_type parms_id,
                              double scale,
                              EncodedPlaintext &destination,
                              seal::MemoryPoolHandle pool) const
{
  static_assert(is_ckks_value_v<T>, "CKKS encode supports double or std::complex<double>");

  auto context_data_ptr = context_.get_context_data(parms_id);
  if (!context_data_ptr) {
    throw std::invalid_argument("parms_id is not valid for encryption parameters");
  }
  if (!values && values_size > 0) {
    throw std::invalid_argument("values cannot be null");
  }
  if (values_size > slots_) {
    throw std::invalid_argument("values_size is too large");
  }
  if (!pool) {
    throw std::invalid_argument("pool is uninitialized");
  }

  auto &context_data = *context_data_ptr;
  auto &parms = context_data.parms();
  auto &coeff_modulus = parms.coeff_modulus();
  const std::size_t coeff_modulus_size = coeff_modulus.size();
  const std::size_t coeff_count = parms.poly_modulus_degree();

  if (!seal::util::product_fits_in(coeff_modulus_size, coeff_count)) {
    throw std::logic_error("invalid parameters");
  }

  const std::size_t n = seal::util::mul_safe(slots_, std::size_t(2));
  auto conj_values = seal::util::allocate<std::complex<double>>(n, pool, 0);
  const int max_coeff_bit_count = scaled_ifft(values, values_size, context_data, scale, conj_values.get());

  destination.coeff_count = coeff_count;
  destination.coeff_modulus_size = coeff_modulus_size;
  destination.scale = scale;
  destination.coeffs.resize(seal::util::mul_safe(coeff_count, coeff_modulus_size));

  const double two_pow_64 = std::pow(2.0, 64);
  auto rounded_coeffs = seal::util::allocate<double>(n, pool);
  auto coeff_is_negative = seal::util::allocate<std::uint8_t>(n, pool);
  for (std::size_t i = 0; i < n; i++) {
    const double coeffd = std::round(conj_values[i].real());
    rounded_coeffs[i] = std::fabs(coeffd);
    coeff_is_negative[i] = std::signbit(coeffd) ? 1 : 0;
  }

  if (max_coeff_bit_count <= 64 && seal_loop_order) {
    for (std::size_t i = 0; i < n; i++) {
      const std::uint64_t coeffu = static_cast<std::uint64_t>(rounded_coeffs[i]);
      for (std::size_t j = 0; j < coeff_modulus_size; j++) {
        const std::uint64_t reduced = seal::util::barrett_reduce_64(coeffu, coeff_modulus[j]);
        destination.coeffs[i + (j * coeff_count)] =
            coeff_is_negative[i] ? seal::util::negate_uint_mod(reduced, coeff_modulus[j]) : reduced;
      }
    }
  } else if (max_coeff_bit_count <= 64) {
    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
      std::uint64_t *dst = destination.data(j);
      for (std::size_t i = 0; i < n; i++) {
        const std::uint64_t coeffu = static_cast<std::uint64_t>(rounded_coeffs[i]);
        const std::uint64_t reduced = seal::util::barrett_reduce_64(coeffu, coeff_modulus[j]);
        dst[i] = coeff_is_negative[i] ? seal::util::negate_uint_mod(reduced, coeff_modulus[j]) : reduced;
      }
    }
  } else if (max_coeff_bit_count <= 128) {
    auto coeffu_lo = seal::util::allocate<std::uint64_t>(n, pool);
    auto coeffu_hi = seal::util::allocate<std::uint64_t>(n, pool);
    for (std::size_t i = 0; i < n; i++) {
      const double coeffd = rounded_coeffs[i];
      coeffu_lo[i] = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
      coeffu_hi[i] = static_cast<std::uint64_t>(coeffd / two_pow_64);
    }

    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
      std::uint64_t *dst = destination.data(j);
      for (std::size_t i = 0; i < n; i++) {
        std::uint64_t coeffu[2]{coeffu_lo[i], coeffu_hi[i]};
        const std::uint64_t reduced = seal::util::barrett_reduce_128(coeffu, coeff_modulus[j]);
        dst[i] = coeff_is_negative[i] ? seal::util::negate_uint_mod(reduced, coeff_modulus[j]) : reduced;
      }
    }
  } else {
    auto coeffu(seal::util::allocate_uint(coeff_modulus_size, pool));
    for (std::size_t i = 0; i < n; i++) {
      double coeffd = std::round(conj_values[i].real());
      const bool is_negative = std::signbit(coeffd);
      coeffd = std::fabs(coeffd);

      seal::util::set_zero_uint(coeff_modulus_size, coeffu.get());
      auto coeffu_ptr = coeffu.get();
      while (coeffd >= 1) {
        *coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
        coeffd /= two_pow_64;
      }

      context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);
      for (std::size_t j = 0; j < coeff_modulus_size; j++) {
        destination.coeffs[i + (j * coeff_count)] =
            is_negative ? seal::util::negate_uint_mod(coeffu[j], coeff_modulus[j]) : coeffu[j];
      }
    }
  }

  // Deliberately stop before SEAL's ntt_negacyclic_harvey step. The AIE encrypt
  // path owns the coefficient-domain-to-NTT-domain transform.
}

template EncodedPlaintext CKKSEncoderNoNTT::encode<double>(
    const double *, std::size_t, seal::parms_id_type, double, seal::MemoryPoolHandle) const;
template EncodedPlaintext CKKSEncoderNoNTT::encode<std::complex<double>>(
    const std::complex<double> *, std::size_t, seal::parms_id_type, double, seal::MemoryPoolHandle) const;
template void CKKSEncoderNoNTT::encode<double>(
    const double *, std::size_t, seal::parms_id_type, double, EncodedPlaintext &, seal::MemoryPoolHandle) const;
template void CKKSEncoderNoNTT::encode<std::complex<double>>(
    const std::complex<double> *, std::size_t, seal::parms_id_type, double, EncodedPlaintext &, seal::MemoryPoolHandle) const;

} // namespace ckks_encode
