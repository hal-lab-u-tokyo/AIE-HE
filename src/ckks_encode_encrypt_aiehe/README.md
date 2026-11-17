# Encode connected to Encrypt

This directory builds the hosts of `../ckks_encrypt_planA` and `../ckks_encrypt_planB` with their `Encode` connected to their `Encrypt`. Both hosts are built only through the `CMakeLists.txt` here.

- `Encode` runs up to the rounded coefficients (int64) and passes `m + e0` to the input preparation of `Encrypt`. The preparation computes `(x + q)·ψ^i mod q` and reduces `x` with the same Barrett reduction as the decomposition in SEAL (`aiehe_joined.h`). `Encode` is inside the timed interval (the timeline event `cpu,m,encode`).
- `v`, `e0` and `e1` are sampled inside the timed interval with the generator (BLAKE2Xb) and the distributions (uniform ternary, centered binomial) of SEAL (`aiehe_sampling.h`, `blake2xb_prng.h`, `blake2/`). The sampled values are the ones `Encrypt` uses. The seed is fixed, so every run draws the same values; a deployment must seed from the OS. The message slots and the key material are test fixtures generated with `std::mt19937`.
- The reference for verification is built from the connected `m + e0` (`Message::md_mod`, which uses `%` independently of the Barrett reduction). `--scale_bits` selects the scale (default 28).
- `aiehe_sampling_check.cpp` is a self-check of the sampler (the compile line is in its first comment).
- `aiehe_joined_check.cpp` is a self-check of the Barrett reduction (`c++ -std=c++17 -DAIEHE_JOINED_NO_SEAL aiehe_joined_check.cpp && ./a.out`).

Build and run (build the xclbins in the directory of the configuration first, then run there). `scripts/measure_aiehe.sh` does all of this.

```bash
cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release   # planA: -DAIEHE_SRC=../ckks_encrypt_planA/test.cpp
cmake --build _build -j 8
cd ../ckks_encrypt_planB && ../ckks_encode_encrypt_aiehe/_build/kernel -x build_mac/final.xclbin -i build_mac/insts.bin \
  -k MLIR_AIE --n_log 17 --rns_L 50 --tile_vol_pow 9 --scale_bits 58 --iters 1 --warmup 0
```
