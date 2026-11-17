// Self-check of aiehe_joined::reduce:
// c++ -std=c++17 -DAIEHE_JOINED_NO_SEAL aiehe_joined_check.cpp && ./a.out
#include "aiehe_joined.h"
#include <cassert>
#include <cstdio>
#include <random>

int main()
{
  std::mt19937_64 gen(7);
  for (uint32_t q : {3u, 65537u, 1073479681u, 2147483647u}) {
    const uint64_t ratio = aiehe_joined::ratio_of(q), bias = aiehe_joined::bias_of(q);
    const int64_t lim = (int64_t(1) << 62) - 1;
    std::vector<int64_t> xs{0, 1, -1, lim, -lim, int64_t(q), -int64_t(q), int64_t(q) - 1, 1 - int64_t(q)};
    for (int k = 0; k < 200000; k++) xs.push_back(int64_t(gen() >> 2) * (k % 2 ? 1 : -1));
    for (int64_t x : xs) assert(aiehe_joined::reduce(x, q, ratio, bias) == uint32_t((x % int64_t(q) + q) % q));
  }
  std::puts("aiehe_joined ok");
}
