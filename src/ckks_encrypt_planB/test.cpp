//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../nttlib.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"
// AIE_HE_TIMELINE: prints the timeline with AIE completion times.
#include "../ckks_encrypt/encrypt_trace.h"
// Encode connected to the Encrypt (../ckks_encode_encrypt_aiehe): m + e0 in int64.
#include "aiehe_joined.h"
#include "aiehe_sampling.h"
#define MD_INPUT md_joined.md

#define REDUCE_PENALTY 1

// ===================================
// Common Parameters for each NTT
// ===================================
const int32_t COL_NUM_LOG = 2;
const int32_t COL_NUM = 1 << COL_NUM_LOG;
const int32_t RAW_NUM_LOG = 2;
const int32_t RAW_NUM = 1 << RAW_NUM_LOG;
const int32_t CORE_NUM_LOG = COL_NUM_LOG + RAW_NUM_LOG;
const int32_t CORE_NUM = COL_NUM * RAW_NUM;

const int32_t MD_CHARACTER_ORDER = 0;
const int32_t  A_CHARACTER_ORDER = 1;
const int32_t  B_CHARACTER_ORDER = 2;
const int32_t  V_CHARACTER_ORDER = 3;
const int32_t E1_CHARACTER_ORDER = 4;
const int32_t    NTT_CONST_ORDER = 5;


// ===================================
// Parameters for Divided NTT
// ===================================
const int32_t FLAG_PHASE1 = 0;
const int32_t FLAG_PHASE2 = 1;

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
int verify(int MSize, std::vector<std::vector<int32_t>> MDashNTTRef, std::vector<std::vector<int32_t>> A, std::vector<std::vector<int32_t>> B, std::vector<std::vector<int32_t>> VNTTRef, std::vector<std::vector<int32_t>> E1NTTRef, std::vector<std::vector<int32_t>> C0, std::vector<std::vector<int32_t>> C1, const int32_t* modulus_list, int32_t L, int verbosity)
{
  int errors = 0;
  for (int l = 0; l < L; l++) {
    int32_t Q = modulus_list[l];
    for (uint32_t i = 0; i < MSize; i++)
    {
      int md = MDashNTTRef[l][i];
      int a = A[l][i];
      int b = B[l][i];
      int v = VNTTRef[l][i];
      int e1 = E1NTTRef[l][i];
      int32_t c0 = ((int64_t)v * b + md + Q) % Q; // c0 = v*pk_b + m + e0 mod Q
      int32_t c1 = ((int64_t)v * a + e1 + Q) % Q; // c1 = v*pk_a + e1 mod Q
      int32_t ref0 = c0;
      int32_t ref1 = c1;
      // Let's check if they are inf or nan, and if so just pass because
      // comparisions will then fail, even for matches
      if (std::isinf(ref0) || std::isinf(ref1) || std::isinf(C0[l][i]) || std::isinf(C1[l][i]))
        break;
      if (std::isnan(ref0) || std::isinf(ref1) || std::isnan(C0[l][i]) || std::isinf(C1[l][i]))
        break;
      if (ref0 != C0[l][i] || ref1 != C1[l][i])
      {
        if (errors < 100)
        {
          if (ref0 != C0[l][i])
          {
            std::cout << "Error in C0[" << l << "][" << i << "] " << C0[l][i] << " != " << ref0 << " = CO_ref" << std::endl;
          }
          if(ref1 != C1[l][i])
          {
            std::cout << "Error in C1[" << l << "][" << i << "] " << C1[l][i] << " != " << ref1 << " = C1_ref" << std::endl;
          }
          std::cout << "===========================" << std::endl;
        }
        errors++;
      }
      else
      {
        if (verbosity > 1 && i < 10)
        {
          std::cout << "Correct in C0[" << l << "][" << i << "] " << C0[l][i] << " == " << ref0 << std::endl;
          std::cout << "Correct in C1[" << l << "][" << i << "] " << C1[l][i] << " == " << ref1 << std::endl;
          std::cout << "===========================" << std::endl;
        }
      }
    }
  }
  return errors;
}

int32_t generate_root(int modulo_q, int N_LOG)
{
  std::random_device rd;
  std::mt19937 genA(rd());
  std::uniform_int_distribution<> dis_A(2,modulo_q-1);

  assert((modulo_q - 1) % (1 << N_LOG) == 0);
  int k = (modulo_q - 1) >> N_LOG;
  while (true)
  {
    int g = dis_A(genA);
    int root = power(g, k, modulo_q);
    if (power(root, 1 << (N_LOG - 1), modulo_q) == modulo_q-1)
    {
      return root;
    }
  }
}

int two_adicity(int32_t q)
{
  int v = 0;
  int64_t x = static_cast<int64_t>(q) - 1;
  while ((x % 2) == 0) {
    x /= 2;
    v++;
  }
  return v;
}

bool is_prime_int32(int32_t n)
{
  if (n < 2) return false;
  if (n % 2 == 0) return n == 2;
  for (int64_t d = 3; d * d <= n; d += 2) {
    if (n % d == 0) return false;
  }
  return true;
}

std::vector<int32_t> build_modulus_list(int requested_l, int n_log)
{
  static constexpr int32_t kDefaultModulusList[] = {
    537133057, 539754497, 540540929, 541327361, 549978113, 551288833, 552861697, 555220993, 561774593, 564658177,
    568066049, 569638913, 570163201, 570949633, 576454657, 576716801, 581959681, 582746113, 583794689, 584581121,
    590872577, 595591169, 597688321, 605028353, 605552641, 606339073, 607911937, 608698369, 611844097, 612892673,
    615776257, 619184129, 621281281, 626262017, 629932033, 632553473, 635437057, 637009921, 638058497, 639369217,
    639631361, 644087809, 645922817, 648019969, 649592833, 651952129, 655360001, 657719297, 660078593, 663224321,
  };

  const int required_two_adicity = n_log + 1; // negacyclic NTT needs a primitive 2N-th root.
  std::vector<int32_t> selected;
  selected.reserve(requested_l);

  auto append_if_valid = [&](int32_t q) {
    if (static_cast<int>(selected.size()) >= requested_l) return;
    if (two_adicity(q) < required_two_adicity) return;
    if (std::find(selected.begin(), selected.end(), q) != selected.end()) return;
    selected.push_back(q);
  };

  for (int32_t q : kDefaultModulusList) {
    append_if_valid(q);
  }

  const int64_t step = int64_t{1} << required_two_adicity;
  const int64_t max_modulus = int64_t{1} << 30; // Keep the existing 30-bit modular arithmetic assumption.
  for (int64_t k = ((int64_t{1} << 29) + step - 1) / step;
       static_cast<int>(selected.size()) < requested_l && k * step + 1 < max_modulus;
       ++k) {
    int64_t q64 = k * step + 1;
    if (is_prime_int32(static_cast<int32_t>(q64))) {
      append_if_valid(static_cast<int32_t>(q64));
    }
  }

  if (static_cast<int>(selected.size()) < requested_l) {
    std::ostringstream oss;
    oss << "Not enough 30-bit NTT primes for logN=" << n_log
        << ", rns_L=" << requested_l
        << " (found " << selected.size()
        << ", need q-1 divisible by 2^" << required_two_adicity << ")";
    throw std::runtime_error(oss.str());
  }

  return selected;
}

int32_t sampling_from_gaussian(auto u1,auto u2, float stddev) {
  return (int32_t)std::round(stddev*std::sqrt(-2*std::log(u1)) * cos(2*M_PI*u2));
}

inline int rotl(int x, int shift, int bit_width) {
  assert(shift >= 0 && shift < bit_width && x < (1 << bit_width));
  return ((x << shift) | (x >> (bit_width - shift))) & ((1 << bit_width) - 1);
}

inline int rotr(int x, int shift, int bit_width) {
  assert(shift >= 0 && shift < bit_width && x < (1 << bit_width));
  return ((x >> shift) | (x << (bit_width - shift))) & ((1 << bit_width) - 1);
}

void output_data_from_ntt_to_mac(int *buf_for_mac, int *buf_from_ntt, int character_order, int *old_index_conversion_list, int loops, int block_volume, int mac_tile_volume) {
  for (int s=0; s<loops; s+=1) {
    int base_idx = s * block_volume;
    for (int i=0; i<mac_tile_volume; i++)
    {
      int old_idx = old_index_conversion_list[s * mac_tile_volume + i];
      int new_idx = base_idx + character_order * mac_tile_volume + i;
      buf_for_mac[new_idx] = buf_from_ntt[old_idx];
    }
  }
}

// Initialize twiddle factors for all columns
// Output buffer consists of `COL_NUM` loops of `FACTOR_SIZE_PER_CORE` twiddle factors
// Each `FACTOR_SIZE_PER_CORE` buffer is:
// [w, w^2, w^4, w^8, ..., w^(2^(N_LOG-1)), modulus, barrett_w, barrett_u]
void initialize_twfactor_for_tuboi(int *buff, int32_t size, int32_t mod, int32_t barrett_w, int32_t barrett_u, int32_t root, int N_LOG, int FACTOR_SIZE_PER_CORE)
{
  assert(size == FACTOR_SIZE_PER_CORE * CORE_NUM);
  for (int core = 0; core < CORE_NUM; core++){
    int w_temp = root;
    buff[FACTOR_SIZE_PER_CORE * core] = w_temp;
    for (int i = 1; i < N_LOG; i++){
      w_temp = ((uint64_t)w_temp * w_temp) % mod;
      buff[FACTOR_SIZE_PER_CORE * core + i] = w_temp;
    }
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG] = mod;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 1] = barrett_w;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 2] = barrett_u;
  }
}

// Bitreverse and separate odd-even bits
// Excmple:
// input: [0, 1, 2, 3, 4, 5, 6, 7]
// bit-reverse: [0, 4, 2, 6, 1, 5, 3, 7]
// separate odd-even bits: [0, 2, 1, 3, 4, 6, 5, 7]
void rearrange_to_aie_order_inline_for_tuboi(int *data, int logN, int logN_per_core)
{
    int N = 1 << logN;
    int32_t single_size = 1 << logN_per_core;
    int32_t *temp_ls = new int32_t[N];
    for (int i = 0; i < N; i++)
    {
        temp_ls[i] = data[i];
    }
    for (int i = 0; i < N; ++i)
    {
        int reversed_index = bit_reverse(i, logN);
        int32_t odd_even_bit = (reversed_index >> (0)) & 1;
        int32_t idx_block = reversed_index / single_size;
        int32_t idx_in_block = reversed_index % single_size;
        int32_t new_index = ((idx_in_block - odd_even_bit) >> 1) + (odd_even_bit * (1 << (logN_per_core - 1))) + idx_block * single_size;
        data[new_index] = temp_ls[i];
    }

    delete[] temp_ls;
}

// Rearrange data from AIE's output order to normal order
void rearrange_from_aie_order_inline_for_tuboi(int *data, int32_t logN, int32_t logN_per_core)
{
  int N = 1 << logN;
  int N_per_core = 1 << logN_per_core;
  int *temp = new int[N];

  // ===================
  // Inter tile order
  // ===================
  // Change core order
  std::vector<int> aie_order = {0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15};
  for (int i = 0; i < CORE_NUM; i++){
    int aie_order_index = aie_order[i];
    int *temp_ptr = temp + i * N_per_core;
    int *aie_ptr = data + aie_order_index * N_per_core;
    for (int j = 0; j < N_per_core; j++){
      temp_ptr[j] = aie_ptr[j];
    }
  }


  // ===================
  // Inner tile order
  // ===================
  // Reverse order in each tile
  // CPU: [0, 1, 2, 3, 4, 5, 6, 7]
  // AIE: [0, 2, 4, 6, 1, 3, 5, 7]
  for (int i = 0; i < CORE_NUM; i++){
    for (int j = 0; j < (N_per_core / 2); j++){
      int *data_ptr = data + i * N_per_core;
      int *temp_ptr = temp + i * N_per_core;
      data_ptr[2 * j] = temp_ptr[j];  // data[0,2,4,6] ← temp[0,1,2,3]
      data_ptr[2 * j + 1] = temp_ptr[(N_per_core / 2) + j];  // data[1,3,5,7] ← temp[4,5,6,7]
    }
  }
  delete[] temp;
}

// Initialize twiddle factors for all columns
// Output buffer consists of `COL_NUM` loops of `FACTOR_SIZE_PER_CORE` metadatas
// Each `FACTOR_SIZE_PER_CORE` buffer is:
// [w, w^2, w^4, w^8, ..., w^(2^(N_LOG-1)), modulus, barrett_w, barrett_u]
void initialize_metadata_for_divided_ntt(int *buff, int32_t size, int32_t mod, int32_t barrett_w, int32_t barrett_u, int32_t root, int32_t logn_for_current_ntt, int32_t if_phase2, int N_LOG, int FACTOR_SIZE_PER_CORE)
{
  assert(size == FACTOR_SIZE_PER_CORE * CORE_NUM);
  for (int core = 0; core < CORE_NUM; core++){
    int w_temp = root;
    buff[FACTOR_SIZE_PER_CORE * core] = w_temp;
    for (int i = 1; i < N_LOG; i++){
      w_temp = ((uint64_t)w_temp * w_temp) % mod;
      buff[FACTOR_SIZE_PER_CORE * core + i] = w_temp;
    }
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG] = mod;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 1] = barrett_w;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 2] = barrett_u;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 3] = logn_for_current_ntt;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 4] = 1 << logn_for_current_ntt;
    buff[FACTOR_SIZE_PER_CORE * core + N_LOG + 5] = if_phase2;
  }
}

// Bitreverse and separate odd-even bits
// Example:
// input: [0, 1, 2, 3, 4, 5, 6, 7]
// bit-reverse: [0, 4, 2, 6, 1, 5, 3, 7]
// separate odd-even bits: [0, 2, 1, 3, 4, 6, 5, 7]
//
// Arguments:
// logN: log2 of total data size
// logN_block:
//   - for whole-core NTT: log2 of data size per core, i.e., LOGN - CORE_NUM_LOG
//   - for each-core NTT: log2 of N for current phase
void rearrange_to_aie_order_inline_for_divided_ntt(int *data, int logN, int logN_per_block)
{
    const int N = 1 << logN;
    const int N_per_block = 1 << logN_per_block;
    const int number_of_blocks = N / N_per_block;
    int32_t *temp_ls = new int32_t[N];
    for (int i = 0; i < N; i++)
    {
        temp_ls[i] = data[i];
    }

    for (int i = 0; i < N; ++i)
    {
      int reversed_index = bit_reverse(i, logN);
      int32_t odd_even_bit = (reversed_index >> (0)) & 1;
      int32_t idx_block = reversed_index / N_per_block;
      int32_t idx_in_block = reversed_index % N_per_block;
      int32_t new_index = ((idx_in_block - odd_even_bit) >> 1) + (odd_even_bit * (1 << (logN_per_block - 1))) + idx_block * N_per_block;
      data[new_index] = temp_ls[i];
    }
    delete[] temp_ls;
}

// Rearrange data from AIE's output order to normal order
// Arguments:
// logN: log2 of total data size
// logN_per_block: log2 of N for current phase
void rearrange_from_aie_order_inline_for_divided_ntt(int *data, int32_t logN, int32_t logN_per_block)
{
  int N = 1 << logN;
  int N_per_block = 1 << logN_per_block;
  int number_of_blocks = N / N_per_block;
  int *temp = new int[N];

  for (int i = 0; i < N; i++)
  {
    temp[i] = data[i];
  }

  // ===================
  // Inner block order
  // ===================
  // Reverse order in each block
  // CPU: [0, 1, 2, 3, 4, 5, 6, 7]
  // AIE: [0, 2, 4, 6, 1, 3, 5, 7]
  for (int i = 0; i < number_of_blocks; i++){
    int *data_ptr = data + i * N_per_block;
    int *temp_ptr = temp + i * N_per_block;
    for (int j = 0; j < (N_per_block / 2); j++){
      data_ptr[2 * j] = temp_ptr[j];
      data_ptr[2 * j + 1] = temp_ptr[(N_per_block / 2) + j];
    }
  }
  delete[] temp;
}

void rotl_array(int *dst, int *src, int32_t logN, int32_t rotation){
  int N = 1 << logN;
  int *tmp = new int[N];
  for (int i = 0; i < N; i++){
    tmp[i] = src[rotl(i, rotation, logN)];
  }
  for (int i = 0; i < N; i++){
    dst[i] = tmp[i];
  }
  delete[] tmp;
}

void rearrange_and_rotl(int *dst, int *src, int32_t logN, int32_t logN_p1, int32_t logN_p2, int32_t rotation)
{
  int N = 1 << logN;
  int N_p1 = 1 << logN_p1;
  int N_p2 = 1 << logN_p2;
  int *temp = new int[N];

  // TODO: can we merge rearrangement and rotl in one loop?

  // ===================
  // Inner block order
  // ===================
  // Reverse order in each block
  // CPU: [0, 1, 2, 3, 4, 5, 6, 7]
  // AIE: [0, 2, 4, 6, 1, 3, 5, 7]
  for (int i = 0; i < N_p2; i++)
  {
    int *src_i = src + i * N_p1;
    int *temp_i = temp + i * N_p1;
    for (int j = 0; j < (N_p1 / 2); j++)
    {
      temp_i[2 * j] = src_i[j];
      temp_i[2 * j + 1] = src_i[(N_p1 / 2) + j];
    }
  }

  // ===================
  // rotl
  // ===================
  for (int i = 0; i < N; i++)
  {
    src[i] = temp[rotl(i, rotation, logN)];
  }

  // ===================
  // Inter block order
  // ===================
  for (int i = 0; i < N_p1; i++)
  {
    int *temp_i = temp + i * N_p2;
    int *src_i = src + i * N_p2;
    for (int j = 0; j < (N_p2 / 2); j++)
    {
      temp_i[j] = src_i[2 * j];
      temp_i[(N_p2 / 2) + j] = src_i[2 * j + 1];
    }
  }

  for (int i = 0; i < N; i++){
    dst[i] = temp[i];
  }

  delete[] temp;
}

void prepare_for_TuboiNTT(std::vector<int32_t> &vec_in, int *buf_in, int size, int *new_index_conversion_list, int modulo_q, std::vector<int32_t> &root_2N_power_list) {
  for (int i=0; i<size; i++){
    buf_in[new_index_conversion_list[i]] = (int64_t(vec_in[i] + modulo_q) * root_2N_power_list[i]) % modulo_q;
  }
}

void prepare_for_Phase1(std::vector<int32_t> &vec_in, int *buf_in, int size, int *new_index_conversion_list, int modulo_q, std::vector<int32_t> &root_2N_power_list) {
  for (int i=0; i<size; i++){
    buf_in[new_index_conversion_list[i]] = (int64_t(vec_in[i] + modulo_q) * root_2N_power_list[i]) % modulo_q;
  }
}

void prepare_for_Phase2(int *buf_out, int *buf_in, int size, int *old_index_conversion_list){
  for (int i=0; i<size; i++){
    buf_in[i] = buf_out[old_index_conversion_list[i]];
  }
}

// =========================================================
// Time Logger
// =========================================================
// =========================================================

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("CKKS Encrypt Test");
  cxxopts::ParseResult vm;

  test_utils::add_default_options(options);
  // 追加オプション
  options.add_options()
    ("benchmark", "flag for benchmark", cxxopts::value<int>()->default_value("0"))
    ("n_log", "log2 of N (input size)", cxxopts::value<int>()->default_value("16"))
    ("rns_L", "RNS level L", cxxopts::value<int>()->default_value("5"))
    ("tile_vol_pow", "tile volume power", cxxopts::value<int>()->default_value("9"))
    ("use_tuboi", "whether to use Tsuboi's NTT", cxxopts::value<int>()->default_value("0"))
    ("use_divided", "whether to use divided NTT", cxxopts::value<int>()->default_value("0"))
    ;

  aiehe_joined::add_options(options);
  test_utils::parse_options(argc, argv, options, vm);
  const int verbosity = vm["verbosity"].as<int>();
  const int do_verify = vm["verify"].as<bool>();
  const int n_iterations = vm["iters"].as<int>();
  const int n_warmup_iterations = vm["warmup"].as<int>();
  const int trace_size = vm["trace_sz"].as<int>();
  const int BENCHMARK = vm["benchmark"].as<int>();
  const int use_tsuboi = vm["use_tuboi"].as<int>();
  const int use_divided = vm["use_divided"].as<int>();

  const int L = vm["rns_L"].as<int>();
  const int N_LOG = vm["n_log"].as<int>();
  const int tile_volume_pow = vm["tile_vol_pow"].as<int>();

  const int MAX_LogN_FOR_TUBOI = 16;

  const int phase_1_step = (N_LOG+1)/2;
  const int phase_2_step = N_LOG - phase_1_step;

  const int N = 1<<N_LOG;
  const int MAC_TILE_VOLUME = 1<<tile_volume_pow;
  const int NTT_TILE_VOLUME = 1<<tile_volume_pow;
  const int32_t N_LOG_PER_CORE = N_LOG - CORE_NUM_LOG;
  const int32_t FACTOR_SIZE_PER_CORE_FOR_TUBOI = N_LOG + 3;
  const int IN_FACTOR_TUBOI_VOLUME = FACTOR_SIZE_PER_CORE_FOR_TUBOI * CORE_NUM;
  const int32_t FACTOR_SIZE_PER_CORE_FOR_DIVIDED = N_LOG + 8;
  const int IN_FACTOR_DIVIDED_VOLUME = FACTOR_SIZE_PER_CORE_FOR_DIVIDED * CORE_NUM;

  const int32_t N_LOG_PHASE1 = (N_LOG + 1) / 2;
  const int32_t N_LOG_PHASE2 = N_LOG - N_LOG_PHASE1;
  const int32_t N_PHASE1 = 1 << N_LOG_PHASE1;
  const int32_t N_PHASE2 = 1 << N_LOG_PHASE2;
  const int32_t LOOP_PHASE1 = N_LOG_PHASE2 - CORE_NUM_LOG;
  const int32_t LOOP_PHASE2 = N_LOG_PHASE1 - CORE_NUM_LOG;
  assert(N_LOG_PHASE1 <= 12); // Each core can handle max 4096 points NTT
  assert(N_LOG_PHASE2 <= 12);

  bool USETuboiNTT;
  if (use_tsuboi == 1 && use_divided == 1){
    std::cerr << "Error: --use_tuboi and --use_divided cannot be both 1\n";
    return 1;
  }else if (use_tsuboi == 1){
    USETuboiNTT = true;
  }
  else if (use_divided == 1){
    USETuboiNTT = false;
  }else {
    USETuboiNTT = (N_LOG <= MAX_LogN_FOR_TUBOI);
  }

  const int LOOP_for_MAC_assign = N/MAC_TILE_VOLUME;

  const int TWINDLE_FACTOR_VOLUME = N/2;
  const int EncrParam_size_per_M = 5; // m,a,b,v,e1
  const int EncrParam_const = 3; // Q, w, mu
  const int EncrParam_kernel_volume = EncrParam_size_per_M * MAC_TILE_VOLUME + EncrParam_const;
  const int IN_EncrParam_VOLUME = EncrParam_kernel_volume * (N / MAC_TILE_VOLUME);
  const int OUT_MAC_VOLUME = 2 * N;

  const int BLOCK_VOLUME = EncrParam_kernel_volume;

  size_t TWINDLE_FACTOR_SIZE = TWINDLE_FACTOR_VOLUME * sizeof(int32_t);
  size_t NTT_INOUT_SIZE = (N + TWINDLE_FACTOR_VOLUME) * sizeof(int32_t);
  size_t STANDARD_SIZE = N * sizeof(int32_t);
  size_t IN_EncrParam_SIZE = IN_EncrParam_VOLUME * sizeof(int32_t);
  size_t OUT_MAC_SIZE = OUT_MAC_VOLUME * sizeof(int32_t);
  size_t NTT_OUT_SIZE_bit = N * sizeof(int32_t) + trace_size;

  // TODO Remove trace for now?
  size_t OUT_SIZE = OUT_MAC_SIZE + trace_size;

  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v_tuboi_ntt =
      test_utils::load_instr_binary("build_tuboi_ntt/insts.bin");
  std::vector<uint32_t> instr_v_divided_ntt =
      test_utils::load_instr_binary("build_divided_ntt/insts.bin");
  std::vector<uint32_t> instr_v_mac =
      test_utils::load_instr_binary("build_mac/insts.bin");
  if (verbosity >= 1){
    std::cout << "Sequence instr_tuboi_ntt count: " << instr_v_tuboi_ntt.size() << "\n";
    std::cout << "Sequence instr_divided_ntt count: " << instr_v_divided_ntt.size() << "\n";
    std::cout << "Sequence instr_mac count: " << instr_v_mac.size() << "\n";
  }

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  // if (verbosity >= 1){
  //   std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";}
  auto xclbin_tuboi_ntt = xrt::xclbin("build_tuboi_ntt/final.xclbin");
  auto xclbin_divided_ntt = xrt::xclbin("build_divided_ntt/final.xclbin");
  auto xclbin_mac = xrt::xclbin("build_mac/final.xclbin");

  // Load the kernel
  // if (verbosity >= 1)
  //   std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels_tuboi_ntt = xclbin_tuboi_ntt.get_kernels();
  auto xkernel_tuboi_ntt = *std::find_if(xkernels_tuboi_ntt.begin(), xkernels_tuboi_ntt.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k)
                               {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                 {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelTuboiNTTName = xkernel_tuboi_ntt.get_name();

  auto xkernels_divided_ntt = xclbin_divided_ntt.get_kernels();
  auto xkernel_divided_ntt = *std::find_if(xkernels_divided_ntt.begin(), xkernels_divided_ntt.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k)
                               {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                 {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelDividedNTTName = xkernel_divided_ntt.get_name();

  auto xkernels_mac = xclbin_mac.get_kernels();
  auto xkernel_mac = *std::find_if(xkernels_mac.begin(), xkernels_mac.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k)
                               {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                 {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelMACName = xkernel_mac.get_name();

  // Register xclbin
  // if (verbosity >= 1)
  //   std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
  //             << "\n";
  device.register_xclbin(xclbin_tuboi_ntt);
  device.register_xclbin(xclbin_divided_ntt);
  device.register_xclbin(xclbin_mac);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context_tuboi_ntt(device, xclbin_tuboi_ntt.get_uuid());
  xrt::hw_context context_divided_ntt(device, xclbin_divided_ntt.get_uuid());
  xrt::hw_context context_mac(device, xclbin_mac.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1){
    std::cout << "Getting handle to kernel_tuboi_ntt:" << kernelTuboiNTTName << "\n";
    std::cout << "Getting handle to kernel_divided_ntt:" << kernelDividedNTTName << "\n";
    std::cout << "Getting handle to kernel_mac:" << kernelMACName << "\n";
  }
  auto kernel_tuboi_ntt = xrt::kernel(context_tuboi_ntt, kernelTuboiNTTName);
  auto kernel_divided_ntt = xrt::kernel(context_divided_ntt, kernelDividedNTTName);
  auto kernel_mac = xrt::kernel(context_mac, kernelMACName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  auto bo_instr_tuboi_ntt = xrt::bo(device, instr_v_tuboi_ntt.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel_tuboi_ntt.group_id(1));
  auto bo_tuboiNTT_in_factor = xrt::bo(device, IN_FACTOR_TUBOI_VOLUME * sizeof(int),
                              XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(4));
  auto bo_tuboiNTT_Mdin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(3));
  auto bo_tuboiNTT_Mdout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(5));
  auto bo_tuboiNTT_Vin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(3));
  auto bo_tuboiNTT_Vout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(5));
  auto bo_tuboiNTT_E1in = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(3));
  auto bo_tuboiNTT_E1out = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_tuboi_ntt.group_id(5));

  auto bo_instr_divided_ntt = xrt::bo(device, instr_v_divided_ntt.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel_divided_ntt.group_id(1));

  auto bo_dividedNTTP1_in_factor = xrt::bo(device, IN_FACTOR_DIVIDED_VOLUME * sizeof(int),
                              XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(4));
  auto bo_dividedNTTP1_Mdin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP1_Mdout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));
  auto bo_dividedNTTP1_Vin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP1_Vout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));
  auto bo_dividedNTTP1_E1in = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP1_E1out = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));

  auto bo_dividedNTTP2_in_factor = xrt::bo(device, IN_FACTOR_DIVIDED_VOLUME * sizeof(int),
                              XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(4));
  auto bo_dividedNTTP2_Mdin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP2_Mdout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));
  auto bo_dividedNTTP2_Vin = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP2_Vout = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));
  auto bo_dividedNTTP2_E1in = xrt::bo(device, STANDARD_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(3));
  auto bo_dividedNTTP2_E1out = xrt::bo(device, NTT_OUT_SIZE_bit,
                         XRT_BO_FLAGS_HOST_ONLY, kernel_divided_ntt.group_id(5));

  auto bo_instr_mac = xrt::bo(device, instr_v_mac.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel_mac.group_id(1));
  std::vector<xrt::bo> bo_inEncrParam_list, bo_outC0C1_list;
  for (int l = 0; l < L; l++){
    bo_inEncrParam_list.emplace_back(device, IN_EncrParam_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel_mac.group_id(3));
    bo_outC0C1_list.emplace_back(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel_mac.group_id(4));
  }
  // auto bo_inEncrParam =
  //     xrt::bo(device, IN_EncrParam_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel_mac.group_id(3));
  // auto bo_outC0C1 =
  //     xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel_mac.group_id(4));

  std::vector<int32_t> modulus_list = build_modulus_list(L, N_LOG);
  std::vector<int32_t> barrett_w_list(L), barrett_u_list(L), root_N_list(L), root_2N_list(L);
  std::vector<std::vector<int32_t>> root_2N_power_table(L, std::vector<int32_t>(N));
  for (int l = 0; l < L; l++) {
    uint32_t Q = modulus_list[l];
    int32_t w = std::bit_width(Q);
    int32_t mu = (1LL << (2 * w)) / Q;
    int32_t root_2N = generate_root(Q, N_LOG + 1); // primitive 2N-th root of unity
    int32_t root = power(root_2N, 2, Q); // primitive N-th root of unity
    assert(power(root, 1 << N_LOG, Q) == 1);
    barrett_w_list[l] = w;
    barrett_u_list[l] = mu;
    root_N_list[l] = root;
    root_2N_list[l] = root_2N;
    int root_2N_power = 1;
    for (int i = 0; i < N; i++)
    {
      root_2N_power_table[l][i] = root_2N_power;
      root_2N_power = ((int64_t)root_2N_power * root_2N) % Q;
    }
  }

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstrTuboiNTT = bo_instr_tuboi_ntt.map<void *>();
  memcpy(bufInstrTuboiNTT, instr_v_tuboi_ntt.data(), instr_v_tuboi_ntt.size() * sizeof(int));

  void *bufInstrDividedNTT = bo_instr_divided_ntt.map<void *>();
  memcpy(bufInstrDividedNTT, instr_v_divided_ntt.data(), instr_v_divided_ntt.size() * sizeof(int));

  void *bufInstrMAC = bo_instr_mac.map<void *>();
  memcpy(bufInstrMAC, instr_v_mac.data(), instr_v_mac.size() * sizeof(int));

  int32_t *buf_tuboiNTT_in_factor = bo_tuboiNTT_in_factor.map<int32_t *>();
  int32_t *buf_tuboiNTT_Mdin = bo_tuboiNTT_Mdin.map<int32_t *>();
  int32_t *buf_tuboiNTT_Mdout = bo_tuboiNTT_Mdout.map<int32_t *>();
  int32_t *buf_tuboiNTT_Vin = bo_tuboiNTT_Vin.map<int32_t *>();
  int32_t *buf_tuboiNTT_Vout = bo_tuboiNTT_Vout.map<int32_t *>();
  int32_t *buf_tuboiNTT_E1in = bo_tuboiNTT_E1in.map<int32_t *>();
  int32_t *buf_tuboiNTT_E1out = bo_tuboiNTT_E1out.map<int32_t *>();

  int32_t *buf_dividedNTTP1_in_factor = bo_dividedNTTP1_in_factor.map<int32_t *>();
  int32_t *buf_dividedNTTP1_Mdin = bo_dividedNTTP1_Mdin.map<int32_t *>();
  int32_t *buf_dividedNTTP1_Mdout = bo_dividedNTTP1_Mdout.map<int32_t *>();
  int32_t *buf_dividedNTTP1_Vin = bo_dividedNTTP1_Vin.map<int32_t *>();
  int32_t *buf_dividedNTTP1_Vout = bo_dividedNTTP1_Vout.map<int32_t *>();
  int32_t *buf_dividedNTTP1_E1in = bo_dividedNTTP1_E1in.map<int32_t *>();
  int32_t *buf_dividedNTTP1_E1out = bo_dividedNTTP1_E1out.map<int32_t *>();

  int32_t *buf_dividedNTTP2_in_factor = bo_dividedNTTP2_in_factor.map<int32_t *>();
  int32_t *buf_dividedNTTP2_Mdin = bo_dividedNTTP2_Mdin.map<int32_t *>();
  int32_t *buf_dividedNTTP2_Mdout = bo_dividedNTTP2_Mdout.map<int32_t *>();
  int32_t *buf_dividedNTTP2_Vin = bo_dividedNTTP2_Vin.map<int32_t *>();
  int32_t *buf_dividedNTTP2_Vout = bo_dividedNTTP2_Vout.map<int32_t *>();
  int32_t *buf_dividedNTTP2_E1in = bo_dividedNTTP2_E1in.map<int32_t *>();
  int32_t *buf_dividedNTTP2_E1out = bo_dividedNTTP2_E1out.map<int32_t *>();


  std::vector<int32_t> v_unNTTed_Vec(N);
  std::vector<int32_t> e1_unNTTed_Vec(N);
  std::vector<std::vector<int32_t>> AVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> SkVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> EVec(L, std::vector<int32_t>(N));

  std::vector<std::vector<int32_t>> MdVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> BVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> VVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> E1Vec(L, std::vector<int32_t>(N));

  std::vector<std::vector<int32_t>> ANTTRefVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> SkNTTRefVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> ENTTRefVec(L, std::vector<int32_t>(N));

  std::vector<std::vector<int32_t>> MdNTTRefVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> BNTTRefVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> VNTTRefVec(L, std::vector<int32_t>(N));
  std::vector<std::vector<int32_t>> E1NTTRefVec(L, std::vector<int32_t>(N));

  std::vector<std::vector<int32_t>> TwindleFactorVec(L, std::vector<int32_t>(TWINDLE_FACTOR_VOLUME));

  std::random_device rd;
  std::mt19937 genA(rd()), genSk(rd()), genE(rd());
  // std::mt19937 genM(40), genSk(41), genA(42), genV(43), genE(44), genE0(45), genE1(46);
  std::uniform_real_distribution<> dis_E(1e-9, 1.0);
  float err_stddev = 8/std::sqrt(2*M_PI);
  aiehe_joined::Message md_joined(N_LOG, L, vm["scale_bits"].as<int>());
  {  // The references are built from the values the timed loop draws again with the same seed.
    AieheSampler ref_sampler(1);
    ref_sampler.sample_v(v_unNTTed_Vec);
    ref_sampler.sample_e0(md_joined.e0);
    ref_sampler.sample_e1(e1_unNTTed_Vec);
  }
  md_joined.encode();
  for (int l = 0; l < L; l++) {
    int32_t modulo_q = modulus_list[l];
    int w_root = root_N_list[l];
    std::uniform_int_distribution<> dis_Q(0, modulo_q-1), dis_sk(-1,1);
    int root_2N = root_2N_list[l];
    int root_2N_power = 1;
    for (int i = 0; i < N; i++)
    {
      AVec[l][i] = dis_Q(genA);
      ANTTRefVec[l][i] = ((int64_t)AVec[l][i] * root_2N_power) % modulo_q;
      SkVec[l][i] = dis_sk(genSk); //暫定的 HWT(h)らしい
      SkNTTRefVec[l][i] = ((int64_t)SkVec[l][i] * root_2N_power) % modulo_q;
      EVec[l][i] = sampling_from_gaussian(dis_E(genE), dis_E(genE), err_stddev);
      ENTTRefVec[l][i] = ((int64_t)EVec[l][i] * root_2N_power) % modulo_q;

      MdNTTRefVec[l][i] = int64_t(md_joined.md_mod(i, modulo_q)) * root_2N_power % modulo_q;
      VNTTRefVec[l][i] = ((int64_t)(v_unNTTed_Vec[i] + modulo_q) * root_2N_power) % modulo_q;
      E1NTTRefVec[l][i] = ((int64_t)(e1_unNTTed_Vec[i] + modulo_q) * root_2N_power) % modulo_q;

      root_2N_power = ((int64_t)root_2N_power * root_2N) % modulo_q;
    }
    vector_bit_reverse(ANTTRefVec[l].data(), N_LOG);
    vector_bit_reverse(SkNTTRefVec[l].data(), N_LOG);
    vector_bit_reverse(ENTTRefVec[l].data(), N_LOG);

    vector_bit_reverse(MdNTTRefVec[l].data(), N_LOG);
    vector_bit_reverse(VNTTRefVec[l].data(), N_LOG);
    vector_bit_reverse(E1NTTRefVec[l].data(), N_LOG);

    ntt_cpu(ANTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);
    ntt_cpu(SkNTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);
    ntt_cpu(ENTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);

    ntt_cpu(MdNTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);
    ntt_cpu(VNTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);
    ntt_cpu(E1NTTRefVec[l].data(), N_LOG, w_root, modulo_q, false, N_LOG);
    for (int i = 0; i < N; i++)
    {
      BNTTRefVec[l][i] = ((-(uint64_t)ANTTRefVec[l][i] * SkNTTRefVec[l][i] + ENTTRefVec[l][i]) % modulo_q + modulo_q) % modulo_q;
    }
  }




  for (int l = 0; l < L; l++){
    auto bo_inEncrParam = bo_inEncrParam_list[l];
    int32_t *bufinEncrParam = bo_inEncrParam.map<int32_t *>();
    for (int s=0; s<LOOP_for_MAC_assign; s+=1) {
      int BASE_IDX = s * BLOCK_VOLUME;
      for (int i=0; i<MAC_TILE_VOLUME; i++)
      {
        bufinEncrParam[BASE_IDX + A_CHARACTER_ORDER * MAC_TILE_VOLUME + i] =  ANTTRefVec[l][s * MAC_TILE_VOLUME + i];
        bufinEncrParam[BASE_IDX + B_CHARACTER_ORDER * MAC_TILE_VOLUME + i] =  BNTTRefVec[l][s * MAC_TILE_VOLUME + i];
      }
      bufinEncrParam[BASE_IDX + NTT_CONST_ORDER * MAC_TILE_VOLUME + 0] = modulus_list[l];
      bufinEncrParam[BASE_IDX + NTT_CONST_ORDER * MAC_TILE_VOLUME + 1] = barrett_w_list[l];
      bufinEncrParam[BASE_IDX + NTT_CONST_ORDER * MAC_TILE_VOLUME + 2] = barrett_u_list[l];
    }
    // Sync buffers to update input buffer values
    bo_inEncrParam.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // rearrenge index conversion array
  int32_t *OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI = new int32_t[N];
  for (int i = 0; i < N; i++) {
    OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI[i] = i;
  }
  rearrange_to_aie_order_inline_for_tuboi(OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI, N_LOG, N_LOG_PER_CORE);

  // rearrenge index conversion array
  int32_t *OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED = new int32_t[N];
  for (int i = 0; i < N; i++) {
    OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED[i] = i;
  }
  rearrange_to_aie_order_inline_for_divided_ntt(OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED, N_LOG, N_LOG_PHASE1);

  int32_t *OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED = new int32_t[N];
  for (int i = 0; i < N; i++) {
    OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED[i] = i;
  }
  rearrange_and_rotl(OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED, OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED, N_LOG, N_LOG_PHASE1, N_LOG_PHASE2, N_LOG_PHASE1);

  int32_t *OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI = new int32_t[N];
  for (int i = 0; i < N; i++) {
    OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI[i] = i;
  }
  rearrange_from_aie_order_inline_for_tuboi(OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI, N_LOG, N_LOG_PER_CORE);

  int32_t *OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED = new int32_t[N];
  for (int i = 0; i < N; i++) {
    OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED[i] = i;
  }
  rearrange_from_aie_order_inline_for_divided_ntt(OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, N_LOG, N_LOG_PHASE2);
  rotl_array(OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, N_LOG, N_LOG_PHASE2);


  int32_t *NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI = new int32_t[N];
  int32_t *NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED = new int32_t[N];

  for (int i = 0; i < N; i++) {
    NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI[OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI[i]] = i;
    NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED[OLD_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED[i]] = i;
  }


  bo_instr_tuboi_ntt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr_divided_ntt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr_mac.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_squared_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++)
  {
    // Run kernel
    if (verbosity >= 1)
    {
      std::cout << "Running Kernel.\n";
    }



    std::vector<std::vector<int32_t>> C0Vec(L, std::vector<int32_t>(N));
    std::vector<std::vector<int32_t>> C1Vec(L, std::vector<int32_t>(N));// Copy output results and verify they are correct
    std::vector<int32_t> C0C1Vec(OUT_MAC_VOLUME);

    unsigned int opcode = 3;

    AieheSampler sampler(1);

    TimeLogger logger;
    logger.use_logger = std::getenv("AIE_HE_TIMELINE") && iter == n_warmup_iterations;
    auto start = std::chrono::high_resolution_clock::now();
    int event_id_total = logger.start_event("cpu", "all", "total", -1);
    {
      // SEAL's generator and distributions for v, e0, e1, before Encode adds e0
      sampler.sample_v(v_unNTTed_Vec);
      sampler.sample_e0(md_joined.e0);
      sampler.sample_e1(e1_unNTTed_Vec);
    }

    {  // Encode, connected: its rounded coefficients (+ e0) are the Encrypt's input.
      const int event_id_encode = logger.start_event("cpu", "m", "encode", 0);
      md_joined.encode();
      logger.stop_event(event_id_encode);
    }


    for (int l=0; l<L; l++){
      auto bo_inEncrParam = bo_inEncrParam_list[l];
      int32_t *bufinEncrParam = bo_inEncrParam.map<int32_t *>();

      int modulo_q = modulus_list[l];
      int barrett_w = barrett_w_list[l];
      int barrett_u = barrett_u_list[l];

      int root_N = root_N_list[l];
      int root_2N = root_2N_list[l];

      auto &root_2N_power_list = root_2N_power_table[l];

      if (USETuboiNTT) {

        // REDUCE_PENALTY == 0のとき、もしくは1だが最初のループのとき、m'の準備は隠蔽できない
        if (!REDUCE_PENALTY || l==0) {

          prepare_for_TuboiNTT(MD_INPUT, buf_tuboiNTT_Mdin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI, modulo_q, root_2N_power_list);
          bo_tuboiNTT_Mdin.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        }


        initialize_twfactor_for_tuboi(buf_tuboiNTT_in_factor, IN_FACTOR_TUBOI_VOLUME, modulo_q, barrett_w, barrett_u, root_N, N_LOG, FACTOR_SIZE_PER_CORE_FOR_TUBOI);
        bo_tuboiNTT_in_factor.sync(XCL_BO_SYNC_BO_TO_DEVICE);


        // NTT of m'
        int event_id_md_ntt = logger.start_event("npu", "md", "ntt", l);
        auto run_tuboi_ntt_md = kernel_tuboi_ntt(opcode, bo_instr_tuboi_ntt, instr_v_tuboi_ntt.size(), bo_tuboiNTT_Mdin, bo_tuboiNTT_in_factor, bo_tuboiNTT_Mdout);
        TimeLogger::AieWatch watch_run_tuboi_ntt_md;
        logger.watch(run_tuboi_ntt_md, watch_run_tuboi_ntt_md);

        // 前回のE1の出力を隠蔽
        if (REDUCE_PENALTY && l>0) {
          // auto inner_start = std::chrono::high_resolution_clock::now();

          // the computation time for outputting e1, except for the last one, can be concealed.
          auto bo_inEncrParam_previous = bo_inEncrParam_list[l-1];
          int32_t *bufinEncrParamPrevious = bo_inEncrParam_previous.map<int32_t *>();

          bo_tuboiNTT_E1out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

          output_data_from_ntt_to_mac(bufinEncrParamPrevious, buf_tuboiNTT_E1out, E1_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
          // Sync buffers to update input buffer values
          bo_inEncrParam_previous.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        }

        prepare_for_TuboiNTT(v_unNTTed_Vec, buf_tuboiNTT_Vin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI, modulo_q, root_2N_power_list);
        bo_tuboiNTT_Vin.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        logger.wait_run(run_tuboi_ntt_md, watch_run_tuboi_ntt_md, event_id_md_ntt);
        logger.stop_event(event_id_md_ntt);


        // NTT of v
        int event_id_v_ntt = logger.start_event("npu", "v", "ntt", l);
        auto run_tuboi_ntt_v = kernel_tuboi_ntt(opcode, bo_instr_tuboi_ntt, instr_v_tuboi_ntt.size(), bo_tuboiNTT_Vin, bo_tuboiNTT_in_factor, bo_tuboiNTT_Vout);
        TimeLogger::AieWatch watch_run_tuboi_ntt_v;
        logger.watch(run_tuboi_ntt_v, watch_run_tuboi_ntt_v);

        bo_tuboiNTT_Mdout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        output_data_from_ntt_to_mac(bufinEncrParam, buf_tuboiNTT_Mdout, MD_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
        // Sync buffers to update input buffer values
        bo_inEncrParam.sync(XCL_BO_SYNC_BO_TO_DEVICE);


        prepare_for_TuboiNTT(e1_unNTTed_Vec, buf_tuboiNTT_E1in, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI, modulo_q, root_2N_power_list);
        bo_tuboiNTT_E1in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        logger.wait_run(run_tuboi_ntt_v, watch_run_tuboi_ntt_v, event_id_v_ntt);
        logger.stop_event(event_id_v_ntt);

        // NTT of e1
        int event_id_e1_ntt = logger.start_event("npu", "e1", "ntt", l);
        auto run_tuboi_ntt_e1 = kernel_tuboi_ntt(opcode, bo_instr_tuboi_ntt, instr_v_tuboi_ntt.size(), bo_tuboiNTT_E1in, bo_tuboiNTT_in_factor, bo_tuboiNTT_E1out);
        TimeLogger::AieWatch watch_run_tuboi_ntt_e1;
        logger.watch(run_tuboi_ntt_e1, watch_run_tuboi_ntt_e1);

        bo_tuboiNTT_Vout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        output_data_from_ntt_to_mac(bufinEncrParam, buf_tuboiNTT_Vout, V_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
        // Sync buffers to update input buffer values
        bo_inEncrParam.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // REDUCE_PENALTY == 1かつ次のループがあるとき、m'の準備を隠蔽
        if (REDUCE_PENALTY && l+1<L) {

          prepare_for_TuboiNTT(MD_INPUT, buf_tuboiNTT_Mdin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_TUBOI, modulus_list[l+1], root_2N_power_table[l+1]);
          bo_tuboiNTT_Mdin.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        }


        logger.wait_run(run_tuboi_ntt_e1, watch_run_tuboi_ntt_e1, event_id_e1_ntt);
        logger.stop_event(event_id_e1_ntt);

        // REDUCE_PENALTY == 0のとき、もしくは1だが最後のループのとき、e1の出力は隠蔽できない
        if (!REDUCE_PENALTY || l+1==L) {

          bo_tuboiNTT_E1out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          output_data_from_ntt_to_mac(bufinEncrParam, buf_tuboiNTT_E1out, E1_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_TUBOI, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
          // Sync buffers to update input buffer values
          bo_inEncrParam.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        }

      }else{

        // REDUCE_PENALTY == 0のとき、もしくは1だが最初のループのとき、m'の準備は隠蔽できない
        if (!REDUCE_PENALTY || l==0) {
          int event_id_m_prepare_phase1 = logger.start_event("cpu","md","prepare_p1",l);
          prepare_for_Phase1(MD_INPUT, buf_dividedNTTP1_Mdin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED, modulo_q, root_2N_power_list);
          bo_dividedNTTP1_Mdin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          logger.stop_event(event_id_m_prepare_phase1);
          int event_id_init_factor_phase1 = logger.start_event("cpu","tw_factor","prepare_p1",l);
          initialize_metadata_for_divided_ntt(buf_dividedNTTP1_in_factor, IN_FACTOR_DIVIDED_VOLUME, modulo_q, barrett_w, barrett_u, root_N, N_LOG_PHASE1, FLAG_PHASE1, N_LOG, FACTOR_SIZE_PER_CORE_FOR_DIVIDED);
          bo_dividedNTTP1_in_factor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          logger.stop_event(event_id_init_factor_phase1);
        }

        int event_id_m_ntt_phase1 = logger.start_event("npu","md","ntt_p1",l);
        // m' phase 1 NTT
        auto run_divided_ntt_p1_md = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP1_Mdin, bo_dividedNTTP1_in_factor, bo_dividedNTTP1_Mdout);
        TimeLogger::AieWatch watch_run_divided_ntt_p1_md;
        logger.watch(run_divided_ntt_p1_md, watch_run_divided_ntt_p1_md);

        int event_id_v_prepare_phase1 = logger.start_event("cpu","v","prepare_p1",l);
        prepare_for_Phase1(v_unNTTed_Vec, buf_dividedNTTP1_Vin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED, modulo_q, root_2N_power_list);
        bo_dividedNTTP1_Vin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_v_prepare_phase1);

        // REDUCE_PENALTY == 1かつ最初のループでないとき、前回のe1の出力を隠蔽
        if (REDUCE_PENALTY && l>0) {
          int event_id_e1_output_phase1 = logger.start_event("cpu","e1","output",l-1);
          // the computation time for outputting e1, except for the last one, can be concealed.
          auto bo_inEncrParam_previous = bo_inEncrParam_list[l-1];
          int32_t *bufinEncrParamPrevious = bo_inEncrParam_previous.map<int32_t *>();

          bo_dividedNTTP2_E1out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          output_data_from_ntt_to_mac(bufinEncrParamPrevious, buf_dividedNTTP2_E1out, E1_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);

          // Sync buffers to update input buffer values
          bo_inEncrParam_previous.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          logger.stop_event(event_id_e1_output_phase1);
        }

        logger.wait_run(run_divided_ntt_p1_md, watch_run_divided_ntt_p1_md, event_id_m_ntt_phase1);
        logger.stop_event(event_id_m_ntt_phase1);


        int event_id_v_ntt_phase1 = logger.start_event("npu","v","ntt_p1",l);
        // v phase 1 NTT
        auto run_divided_ntt_p1_v = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP1_Vin, bo_dividedNTTP1_in_factor, bo_dividedNTTP1_Vout);
        TimeLogger::AieWatch watch_run_divided_ntt_p1_v;
        logger.watch(run_divided_ntt_p1_v, watch_run_divided_ntt_p1_v);

        int event_id_e1_prepare_phase1 = logger.start_event("cpu","e1","prepare_p1",l);
        prepare_for_Phase1(e1_unNTTed_Vec, buf_dividedNTTP1_E1in, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED, modulo_q, root_2N_power_list);
        bo_dividedNTTP1_E1in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_e1_prepare_phase1);

        logger.wait_run(run_divided_ntt_p1_v, watch_run_divided_ntt_p1_v, event_id_v_ntt_phase1);
        logger.stop_event(event_id_v_ntt_phase1);


        int event_id_e1_ntt_phase1 = logger.start_event("npu","e1","ntt_p1",l);
        // e1 phase 1 NTT
        auto run_divided_ntt_p1_e1 = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP1_E1in, bo_dividedNTTP1_in_factor, bo_dividedNTTP1_E1out);
        TimeLogger::AieWatch watch_run_divided_ntt_p1_e1;
        logger.watch(run_divided_ntt_p1_e1, watch_run_divided_ntt_p1_e1);

        int event_id_m_prepare_phase2 = logger.start_event("cpu", "md","prepare_p2",l);
        bo_dividedNTTP1_Mdout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        prepare_for_Phase2(buf_dividedNTTP1_Mdout, buf_dividedNTTP2_Mdin, N, OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED);
        bo_dividedNTTP2_Mdin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_m_prepare_phase2);
        int event_id_init_factor_phase2 = logger.start_event("cpu","tw_factor","prepare_p2",l);
        initialize_metadata_for_divided_ntt(buf_dividedNTTP2_in_factor, IN_FACTOR_DIVIDED_VOLUME, modulo_q, barrett_w, barrett_u, root_N, N_LOG_PHASE2, FLAG_PHASE2, N_LOG, FACTOR_SIZE_PER_CORE_FOR_DIVIDED);
        bo_dividedNTTP2_in_factor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_init_factor_phase2);

        logger.wait_run(run_divided_ntt_p1_e1, watch_run_divided_ntt_p1_e1, event_id_e1_ntt_phase1);
        logger.stop_event(event_id_e1_ntt_phase1);


        int event_id_m_ntt_phase2 = logger.start_event("npu","md","ntt_p2",l);
        // m` phase 2 NTT
        auto run_divided_ntt_p2_md = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP2_Mdin, bo_dividedNTTP2_in_factor, bo_dividedNTTP2_Mdout);
        TimeLogger::AieWatch watch_run_divided_ntt_p2_md;
        logger.watch(run_divided_ntt_p2_md, watch_run_divided_ntt_p2_md);

        int event_id_v_prepare_phase2 = logger.start_event("cpu","v","prepare_p2",l);
        bo_dividedNTTP1_Vout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        prepare_for_Phase2(buf_dividedNTTP1_Vout, buf_dividedNTTP2_Vin, N, OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED);
        bo_dividedNTTP2_Vin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_v_prepare_phase2);

        logger.wait_run(run_divided_ntt_p2_md, watch_run_divided_ntt_p2_md, event_id_m_ntt_phase2);
        logger.stop_event(event_id_m_ntt_phase2);

        int event_id_v_ntt_phase2 = logger.start_event("npu","v","ntt_p2",l);
        // v phase 2 NTT
        auto run_divided_ntt_p2_v = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP2_Vin, bo_dividedNTTP2_in_factor, bo_dividedNTTP2_Vout);
        TimeLogger::AieWatch watch_run_divided_ntt_p2_v;
        logger.watch(run_divided_ntt_p2_v, watch_run_divided_ntt_p2_v);

        int event_id_e1_prepare_phase2 = logger.start_event("cpu","e1","prepare_p2",l);
        bo_dividedNTTP1_E1out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        prepare_for_Phase2(buf_dividedNTTP1_E1out, buf_dividedNTTP2_E1in, N, OLD_IDX_CONVERT_REARRENGE_AND_ROTL_FOR_DIVIDED);
        bo_dividedNTTP2_E1in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        logger.stop_event(event_id_e1_prepare_phase2);

        int event_id_m_output = logger.start_event("cpu","md","output",l);
        bo_dividedNTTP2_Mdout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        output_data_from_ntt_to_mac(bufinEncrParam, buf_dividedNTTP2_Mdout, MD_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
        logger.stop_event(event_id_m_output);

        logger.wait_run(run_divided_ntt_p2_v, watch_run_divided_ntt_p2_v, event_id_v_ntt_phase2);
        logger.stop_event(event_id_v_ntt_phase2);

        int event_id_e1_ntt_phase2 = logger.start_event("npu","e1","ntt_p2",l);
        // e1 phase 2 NTT
        auto run_divided_ntt_p2_e1 = kernel_divided_ntt(opcode, bo_instr_divided_ntt, instr_v_divided_ntt.size(), bo_dividedNTTP2_E1in, bo_dividedNTTP2_in_factor, bo_dividedNTTP2_E1out);
        TimeLogger::AieWatch watch_run_divided_ntt_p2_e1;
        logger.watch(run_divided_ntt_p2_e1, watch_run_divided_ntt_p2_e1);

        int event_id_v_output = logger.start_event("cpu","v","output",l);
        bo_dividedNTTP2_Vout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        output_data_from_ntt_to_mac(bufinEncrParam, buf_dividedNTTP2_Vout, V_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
        logger.stop_event(event_id_v_output);

        // REDUCE_PENALTY == 1かつ次のループがあるとき、m'の準備を隠蔽
        if (REDUCE_PENALTY && l+1<L) {
          int event_id_m_prepare_next_phase1 = logger.start_event("cpu","md","prepare_p1",l+1);
          // the computation time for preparing m', except for the last one, can be hidden.
          prepare_for_Phase1(MD_INPUT, buf_dividedNTTP1_Mdin, N, NEW_IDX_CONVERT_TO_AIE_ORDER_FOR_DIVIDED, modulus_list[l+1], root_2N_power_table[l+1]);
          bo_dividedNTTP1_Mdin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          logger.stop_event(event_id_m_prepare_next_phase1);
          int event_id_init_factor_next_phase1 = logger.start_event("cpu","tw_factor","prepare_p1",l+1);
          initialize_metadata_for_divided_ntt(buf_dividedNTTP1_in_factor, IN_FACTOR_DIVIDED_VOLUME, modulus_list[l+1], barrett_w_list[l+1], barrett_u_list[l+1], root_N_list[l+1], N_LOG_PHASE1, FLAG_PHASE1, N_LOG, FACTOR_SIZE_PER_CORE_FOR_DIVIDED);
          bo_dividedNTTP1_in_factor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
          logger.stop_event(event_id_init_factor_next_phase1);
        }

        logger.wait_run(run_divided_ntt_p2_e1, watch_run_divided_ntt_p2_e1, event_id_e1_ntt_phase2);
        logger.stop_event(event_id_e1_ntt_phase2);

        // REDUCE_PENALTY == 0のとき、もしくは1だが最後のループのとき、e1の出力は隠蔽できない
        if (!REDUCE_PENALTY || l+1==L) {
          int event_id_e1_output = logger.start_event("cpu","e1","output",l);
          // the computation time for the final e1 output cannot be concealed.
          bo_dividedNTTP2_E1out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          output_data_from_ntt_to_mac(bufinEncrParam, buf_dividedNTTP2_E1out, E1_CHARACTER_ORDER, OLD_IDX_CONVERT_FROM_AIE_ORDER_FOR_DIVIDED, LOOP_for_MAC_assign, BLOCK_VOLUME, MAC_TILE_VOLUME);
          logger.stop_event(event_id_e1_output);
        }
      }


#if REDUCE_PENALTY
    }
    for (int l=0; l<L; l++){
      auto bo_inEncrParam = bo_inEncrParam_list[l];
      int32_t *bufinEncrParam = bo_inEncrParam.map<int32_t *>();
#endif

      auto bo_outC0C1 = bo_outC0C1_list[l];

      bo_inEncrParam.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      int event_id_mac = logger.start_event("npu", "c0c1", "mac", l);
      auto run_mac = kernel_mac(opcode, bo_instr_mac, instr_v_mac.size(), bo_inEncrParam, bo_outC0C1);
      TimeLogger::AieWatch watch_run_mac;
      logger.watch(run_mac, watch_run_mac);

      // 前回のC0,C1の出力を隠蔽
      if (REDUCE_PENALTY && l>0) {
        auto bo_outC0C1_previous = bo_outC0C1_list[l-1];
        bo_outC0C1_previous.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::int32_t *bufOutC0C1Previous = bo_outC0C1_previous.map<std::int32_t *>();
        for (int s=0; s<LOOP_for_MAC_assign; s+=1) {
          int BASE_IDX = s * 2 * MAC_TILE_VOLUME;
          for (int i=0; i<MAC_TILE_VOLUME; i++)
          {
            C0Vec[l-1][s * MAC_TILE_VOLUME + i] = bufOutC0C1Previous[BASE_IDX + 0 * MAC_TILE_VOLUME + i];
            C1Vec[l-1][s * MAC_TILE_VOLUME + i] = bufOutC0C1Previous[BASE_IDX + 1 * MAC_TILE_VOLUME + i];
          }
        }
      }

      logger.wait_run(run_mac, watch_run_mac, event_id_mac);
      logger.stop_event(event_id_mac);
      bo_outC0C1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

      std::int32_t *bufOutC0C1 = bo_outC0C1.map<std::int32_t *>();

      // memcpy(C0C1Vec.data(), bufOutC0C1, (C0C1Vec.size() * sizeof(int32_t)));

      if (!REDUCE_PENALTY || l+1==L) {
        // c0c1 = [c0[0][0:MAC_TILE_VOLUME-1], c1[0][0:MAC_TILE_VOLUME-1], c0[1][0:MAC_TILE_VOLUME-1], c1[1][0:MAC_TILE_VOLUME-1], ..., c0[0][MAC_TILE_VOLUME:2*MAC_TILE_VOLUME-1], c1[0][MAC_TILE_VOLUME:2*MAC_TILE_VOLUME-1], ... ]
        for (int s=0; s<LOOP_for_MAC_assign; s+=1) {
          int BASE_IDX = s * 2 * MAC_TILE_VOLUME;
          for (int i=0; i<MAC_TILE_VOLUME; i++)
          {
            C0Vec[l][s * MAC_TILE_VOLUME + i] = bufOutC0C1[BASE_IDX + 0 * MAC_TILE_VOLUME + i];
            C1Vec[l][s * MAC_TILE_VOLUME + i] = bufOutC0C1[BASE_IDX + 1 * MAC_TILE_VOLUME + i];
          }
        }
      }
    }
    logger.stop_event(event_id_total);
    auto stop = std::chrono::high_resolution_clock::now();

    logger.print_summary();

    // // Write trace values if trace_size > 0
    // if (trace_size > 0)
    // {
    //   test_utils::write_out_trace(((char *)bufOutC0C1) + OUT_MAC_SIZE, trace_size,
    //                               vm["trace_file"].as<std::string>());
    // }

    if (iter < n_warmup_iterations)
    {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    if (do_verify && n_warmup_iterations == iter && !BENCHMARK)
    {
      if (verbosity >= 1)
      {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      errors = verify(N, MdNTTRefVec, ANTTRefVec, BNTTRefVec, VNTTRefVec, E1NTTRefVec, C0Vec, C1Vec, modulus_list.data(), L, verbosity);
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1)
      {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    }
    else
    {
      std::cout << "WARNING: results not verified." << std::endl;
    }

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_squared_total += npu_time * npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  float avg = npu_time_total / n_iterations;
  float stddev =
      std::sqrt((npu_time_squared_total / n_iterations) - (avg * avg));

  if (BENCHMARK){
    std::string discriminator = (USETuboiNTT ? "tuboiNTT" : "dividedNTT");
    std::string filename_base = "benchmark_output";
    std::string penalty_type = REDUCE_PENALTY ? "reduce_penalty" : "normal_penalty";
    std::string filename = filename_base + "_" + penalty_type + "_" + discriminator;
    std::string txtfilename = filename + "_2604.txt";
    std::string csvfilename = filename +  "_2604.csv";
    std::ofstream txtfile(txtfilename, std::ios::app);
    std::ofstream csvfile(csvfilename, std::ios::app);
    txtfile << avg << "\t" << stddev << std::endl;
    csvfile << N << "," << L << "," << avg << "," << stddev << std::endl;
    txtfile.close();
    csvfile.close();
  }
  std::cout << "===========================" << std::endl;

  std::cout << "Iters: " << n_iterations << std::endl;

  std::cout << std::endl
            << "Avg NPU time: " << avg << " ± " << stddev << " us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors)
  {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  else
  {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
