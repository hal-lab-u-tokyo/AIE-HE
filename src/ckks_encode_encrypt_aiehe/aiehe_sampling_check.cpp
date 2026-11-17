// Self-check of AieheSampler:
// cc -c blake2/blake2b.c blake2/blake2xb.c && c++ -std=c++17 aiehe_sampling_check.cpp blake2b.o blake2xb.o && ./a.out
#include "aiehe_sampling.h"
#include <cassert>
#include <cstdio>

int main()
{
  const size_t n = 1 << 20;
  std::vector<int32_t> v(n), e0(n), e1(n), v2(n), e02(n), e12(n);
  AieheSampler a(1), b(1);
  a.sample_v(v);
  a.sample_e0(e0);
  a.sample_e1(e1);
  b.sample_v(v2);
  b.sample_e0(e02);
  b.sample_e1(e12);
  assert(v == v2 && e0 == e02 && e1 == e12);

  size_t cnt[3] = {0, 0, 0};
  for (int32_t x : v) {
    assert(x >= -1 && x <= 1);
    cnt[x + 1]++;
  }
  for (size_t c : cnt) assert(c > n * 0.33 && c < n * 0.34);

  for (const auto *e : {&e0, &e1}) {
    double s = 0, s2 = 0;
    for (int32_t x : *e) {
      assert(x >= -21 && x <= 21);
      s += x;
      s2 += double(x) * x;
    }
    const double mean = s / n, var = s2 / n - mean * mean;
    assert(mean > -0.05 && mean < 0.05 && var > 10.3 && var < 10.7);
    std::printf("mean %.4f var %.4f\n", mean, var);
  }
  std::puts("aiehe_sampling ok");
}
