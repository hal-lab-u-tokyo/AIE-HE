# AIE-HE

Implementation of AIE-HE, which accelerates the client-side processing of the CKKS homomorphic encryption scheme on the AMD AI Engine (AIE), the NPU of AMD Ryzen AI processors.

> Haruto Katakura, Ai Nozaki, Motoharu Tsuboi, Takuya Kojima, Hiroshi Nakamura, and Hideki Takase. AIE-HE: Accelerating Client-Side Processing of Homomorphic Encryption on AMD AI Engine. In Proceedings of the 55th International Conference on Parallel Processing (ICPP '26). https://doi.org/10.1145/3832810.3832849

This repository contains the `Encode`+`Encrypt` path: the AIE kernels for the NTT and the multiply-accumulate (MAC), the host program that drives them, and a script that measures the execution time.

## Configurations

| Name | Directory | NTT | MAC |
|---|---|---|---|
| `aiehe_d` | `src/ckks_encrypt_planA` | decomposed into two phases | fused with the second NTT phase in one kernel |
| `aiehe_u` | `src/ckks_encrypt_planB` | undecomposed for log N <= 16, decomposed for log N = 17 | separate kernel |

Both hosts are built through `src/ckks_encode_encrypt_aiehe`, which connects `Encode` to `Encrypt`: the rounded coefficients produced by `Encode` are the input of `Encrypt`, and the measured time covers both. See `src/ckks_encode_encrypt_aiehe/README.md`.

## Requirements

- An AMD Ryzen AI processor with the NPU, and XRT
- [mlir-aie](https://github.com/Xilinx/mlir-aie) (the Python package `aie`, Peano, `aiecc`, and its `test_utils` and `cxxopts`); tested with v1.1.3
- gcc 13 and CMake 3.30 or later
- [Microsoft SEAL](https://github.com/microsoft/SEAL) at `src/ckks_encrypt_SEAL/thirdparty/SEAL`:

```bash
git clone https://github.com/microsoft/SEAL src/ckks_encrypt_SEAL/thirdparty/SEAL
git -C src/ckks_encrypt_SEAL/thirdparty/SEAL checkout 79ad62fb50fb59880c7b717275c82fe497634639
```

## Measurement

```bash
bash scripts/measure_aiehe.sh [logNs] [Ls]    # defaults: "13 14 15 16 17" and "5 10 15 20 25 30 35 40 45 50"
bash scripts/measure_aiehe.sh "17" "50"       # one point
```

`L` is the number of 30-bit primes. For each log N the script builds the AIE designs and, before timing, checks the ciphertext against a CPU reference for the smallest and the largest `L`. Each time is the mean of 100 runs after 10 warm-up runs. The results are written to `results/aiehe-*/` (`aie.csv`, `summary.csv`).

## Limitations

This is a research prototype for performance evaluation and does not provide real security: the random seed is fixed, and the keys and the messages are generated as test fixtures.

## License

MIT; see `LICENSE`. Files derived from other projects keep their own notices; see `THIRD_PARTY_NOTICES.md`.
