#pragma once
// Port of SEAL's Blake2xbPRNG (microsoft/SEAL 7a931d5, native/src/seal/randomgen.cpp),
// SEAL's default PRNG: a 64-byte seed keys BLAKE2Xb, a 64-bit counter is the input,
// and each refill produces a 4096-byte buffer that is consumed in order.
// The blake2/ sources are SEAL's copies of the BLAKE2 reference code.
#include "blake2/blake2.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

class Blake2xbPRNG {
public:
  using result_type = std::uint32_t;  // UniformRandomBitGenerator, like SEAL's generate()
  using seed_type = std::array<std::uint64_t, 8>;

  explicit Blake2xbPRNG(const seed_type &seed) : seed_(seed) {}
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

  result_type operator()() {
    if (pos_ == buffer_.size()) refill();
    result_type r;
    std::memcpy(&r, buffer_.data() + pos_, sizeof(r));
    pos_ += sizeof(r);
    return r;
  }

private:
  void refill() {
    if (blake2xb(buffer_.data(), buffer_.size(), &counter_, sizeof(counter_), seed_.data(),
                 sizeof(seed_)) != 0)
      throw std::runtime_error("blake2xb failed");
    ++counter_;
    pos_ = 0;
  }

  seed_type seed_;
  std::uint64_t counter_ = 0;
  std::array<unsigned char, 4096> buffer_{};
  std::size_t pos_ = 4096;  // empty: the first call refills
};
