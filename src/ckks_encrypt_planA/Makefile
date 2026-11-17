##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2

targetname = kernel
kernelfile = kernel_func

devicename ?= npu
n_col_cores = 4
n_row_cores = 4
logN ?= 16
rns_L ?= 10
tile_vol_pow ?= 9
benchmark ?= 0

trace_size = 65536

aie_py_src_divided_ntt=aie2_divided_ntt.py
aie_py_src_mac=aie2_mac.py
aie_py_src_fused=aie2_divided_ntt_mac.py

trace_dir = bench_datas

trace_json_file_name = ${trace_dir}/trace_${targetname}_${trace_size}

colshift ?= $(if $(filter npu,$(devicename)),1,0)

all: build_mac/final.xclbin build_mac/insts.bin build_divided_ntt/final.xclbin build_divided_ntt/insts.bin


build_divided_ntt/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p build_divided_ntt
	cd build_divided_ntt && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}

build_divided_ntt/${kernelfile}.o: ${kernelfile}.cc
	cd build_divided_ntt && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I${aie2_runtime_dir} -c ../${kernelfile}.cc -o ${@F}

build_divided_ntt/kernels.a: build_divided_ntt/${kernelfile}.o build_divided_ntt/lut_based_ops.o
	ar rvs $@ $+

build_divided_ntt/aie.mlir: ${srcdir}/${aie_py_src_divided_ntt}
	mkdir -p build_divided_ntt
	python3 $< -n ${logN} > $@

build_divided_ntt/final.xclbin: build_divided_ntt/aie.mlir build_divided_ntt/kernels.a
	cd build_divided_ntt && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)



build_mac/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p build_mac
	cd build_mac && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}

build_mac/${kernelfile}.o: ${kernelfile}.cc
	cd build_mac && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I${aie2_runtime_dir} -c ../${kernelfile}.cc -o ${@F}

build_mac/kernels.a: build_mac/${kernelfile}.o build_mac/lut_based_ops.o
	ar rvs $@ $+

build_mac/aie.mlir: ${srcdir}/${aie_py_src_mac}
	mkdir -p build_mac
	python3 $< -n ${logN} -tv ${tile_vol_pow} > $@

build_mac/final.xclbin: build_mac/aie.mlir build_mac/kernels.a
	cd build_mac && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)


build_fused/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p build_fused
	cd build_fused && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}

build_fused/${kernelfile}.o: ${kernelfile}.cc
	cd build_fused && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I${aie2_runtime_dir} -c ../${kernelfile}.cc -o ${@F}

build_fused/kernels.a: build_fused/${kernelfile}.o build_fused/lut_based_ops.o
	ar rvs $@ $+

build_fused/aie.mlir: ${srcdir}/${aie_py_src_fused}
	mkdir -p build_fused
	python3 $< -n ${logN} > $@

build_fused/final.xclbin: build_fused/aie.mlir build_fused/kernels.a
	cd build_fused && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

clean:
	rm -rf build_mac build_divided_ntt build_fused _build ${targetname}.exe
