# Third-party notices

## mlir-aie

The AIE designs (`src/ckks_encrypt_plan*/aie2_*.py`), the host programs (`src/ckks_encrypt_plan*/test.cpp`), the Makefiles, `src/ckks_encrypt/encrypt_trace.h`, `src/common.cmake` and `src/makefile-common` are based on the programming examples of [mlir-aie](https://github.com/Xilinx/mlir-aie), and `src/aie_kernel_utils.h` is taken from it. They are licensed under the Apache License v2.0 with LLVM Exceptions (https://llvm.org/LICENSE.txt); the files that carry the mlir-aie header keep it.

## Microsoft SEAL

`src/ckks_encode/ckks_encoder.{h,cpp}` are derived from the CKKS encoder of [Microsoft SEAL](https://github.com/microsoft/SEAL). `src/ckks_encode_encrypt_aiehe/blake2xb_prng.h` is a port of SEAL's `Blake2xbPRNG`, and `src/ckks_encode_encrypt_aiehe/blake2/` holds SEAL's copies of the BLAKE2 reference code. SEAL is distributed under the following license.

```
MIT License

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

## BLAKE2

The files in `src/ckks_encode_encrypt_aiehe/blake2/` come from the BLAKE2 reference source code package (https://blake2.net), which may be used under the terms of CC0 1.0, the OpenSSL Licence or the Apache Public License 2.0, as stated in each file.
