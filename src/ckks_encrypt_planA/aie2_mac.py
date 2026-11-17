# vector_exp/vector_exp_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys
from ml_dtypes import bfloat16

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.helpers.dialects.ext.scf import (
    _for as range_,
)  # scf (structured control flow) dialect
from aie.helpers.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent


# AI Engine structural design function
def encrypt(tile_vol_pow, logN):

    Q_list_length = 50


    N = 1<<logN
    datatype = np.int32
    tile_volume = 1<<tile_vol_pow # Tile sizes

    assert N % tile_volume == 0

    n_row_cores_log = 2
    n_col_cores_log = 2

    n_row_cores = 1<<n_row_cores_log
    n_col_cores = 1<<n_col_cores_log

    n_cores = n_row_cores * n_col_cores

    EncrParam_size_per_M = 5 # MDash, A, B, V, E1
    EncrParam_const = 3 # Q, w, mu
    C_size_per_M = 2 #c0,c1

    N_EncrParams_per_CT = EncrParam_size_per_M * tile_volume + EncrParam_const # (M, Sk, A, V, E, E0, E1) + (Q, w, mu)
    N_C_per_CT = C_size_per_M * tile_volume

    N_EncrParams_per_MTbuff = N_EncrParams_per_CT * n_row_cores
    N_C_per_MTbuff = N_C_per_CT * n_row_cores

    loop_in_CT = N // (tile_volume * n_cores)
    N_EncrParams_per_MT = N_EncrParams_per_MTbuff * loop_in_CT
    N_C_per_MT = N_C_per_MTbuff * loop_in_CT

    M_buffer_depth = 2
    EncrParam_buffer_depth = 2
    C_buffer_depth = 2

    datatype = np.int32

    # Device declaration - aie2 device NPU (aka Ryzen AI)
    @device(AIEDevice.npu1)
    def device_body():

        comTile_ty = np.ndarray[(tile_volume,), np.dtype[datatype]]

        # Type used in the tile memory
        EncrParam_ty = np.ndarray[(N_EncrParams_per_CT,), np.dtype[datatype]]
        C_ty = np.ndarray[(N_C_per_CT,), np.dtype[datatype]]

        # AIE Core Function declarations

        EncrParam_MT_ty = np.ndarray[(N_EncrParams_per_MTbuff,), np.dtype[datatype]]
        C_MT_ty = np.ndarray[(N_C_per_MTbuff,), np.dtype[datatype]]

        encrypt_mac = external_func("encrypt_mac", inputs=[EncrParam_ty, C_ty, np.int32]) # EncrParam, C0C1, tile_volume, scale, Q, w, mu)

        # Tile declarations
        ShimTiles = [tile(c, 0) for c in range(n_col_cores)]

        MemTiles = [tile(c, 1) for c in range(n_col_cores)]
        CompTiles = [[tile(c, 2 + r) for r in range(n_row_cores)] for c in range(n_col_cores)]

        # Input MSkAVEE0E1
        inEncrParam_fifos = [[] for _ in range(n_col_cores)]
        inEncrParams = [None] * n_col_cores
        for c in range(n_col_cores):
            inEncrParams[c] = object_fifo(f"inEncrParam_{c}", ShimTiles[c], MemTiles[c], EncrParam_buffer_depth, EncrParam_MT_ty)
            for r in range(n_row_cores):
                inEncrParam_fifos[c].append(
                    object_fifo(f"memEncrParam_{c}{r}", MemTiles[c], CompTiles[c][r], EncrParam_buffer_depth, EncrParam_ty)
                )
            if n_row_cores > 1:
                of_offsets = [
                    np.prod(np_ndarray_type_get_shape(EncrParam_ty)) * i
                    for i in range(n_row_cores)
                ]
            else:
                of_offsets = []
            object_fifo_link(inEncrParams[c], inEncrParam_fifos[c], [], of_offsets)

        # Output C0C1
        outC0C1_fifos = [[] for _ in range(n_col_cores)]
        outC0C1s = [None] * n_col_cores
        for c in range(n_col_cores):
            for r in range(n_row_cores):
                outC0C1_fifos[c].append(
                    object_fifo(f"memC0C1_{c}{r}", CompTiles[c][r], MemTiles[c], C_buffer_depth, C_ty)
                )
            outC0C1s[c] = object_fifo(f"outC0C1_{c}", MemTiles[c], ShimTiles[c], C_buffer_depth, C_MT_ty)
            if n_row_cores > 1:
                of_offsets = [
                    np.prod(np_ndarray_type_get_shape(C_ty)) * i
                    for i in range(n_row_cores)
                ]
            else:
                of_offsets = []
            object_fifo_link(outC0C1_fifos[c], outC0C1s[c], of_offsets, [])

        # Compute tile bodies
        for c in range(n_col_cores):
            for r in range(n_row_cores):
                # Compute tile [c][r]
                @core(CompTiles[c][r], "kernels.a")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range_(loop_in_CT):
                            elem_in_encr_param = inEncrParam_fifos[c][r].acquire(ObjectFifoPort.Consume, 1)
                            elem_out_c0c1 = outC0C1_fifos[c][r].acquire(ObjectFifoPort.Produce, 1)

                            encrypt_mac(elem_in_encr_param, elem_out_c0c1, tile_volume)

                            outC0C1_fifos[c][r].release(ObjectFifoPort.Produce, 1)
                            inEncrParam_fifos[c][r].release(ObjectFifoPort.Consume, 1)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(N,), np.dtype[datatype]] # for M
        tensor_EncrParam_ty = np.ndarray[(N_EncrParams_per_MT * n_col_cores,), np.dtype[datatype]] # for EncrParam
        tensor_C_ty = np.ndarray[(N_C_per_MT * n_col_cores,), np.dtype[datatype]] # for C

        @runtime_sequence(tensor_EncrParam_ty, tensor_C_ty)
        def sequence(EncrParam, C0C1):
            # タスクリストは列ごとに管理
            in_task_EncrParams = []
            out_task_C0C1s = []

            for c in range(n_col_cores):
                # Shim DMA は列ごとに1回設定すればよい
                # inM_fifos[c][r] (Core用) ではなく inAs[c] (Shim用) を使う
                in_task_EncrParam = shim_dma_single_bd_task(
                    inEncrParams[c], EncrParam, sizes=[1, 1, 1, N_EncrParams_per_MT], offset=c * N_EncrParams_per_MT, issue_token=True
                )
                out_task_C0C1 = shim_dma_single_bd_task(
                    outC0C1s[c], C0C1, sizes=[1, 1, 1, N_C_per_MT], offset=c * N_C_per_MT, issue_token=True
                )

                dma_start_task(in_task_EncrParam, out_task_C0C1)
                in_task_EncrParams.append(in_task_EncrParam)
                out_task_C0C1s.append(out_task_C0C1)


            for c in range(n_col_cores):
                dma_await_task(in_task_EncrParams[c], out_task_C0C1s[c])


p = argparse.ArgumentParser()
p.add_argument(
    "-tv",
    "--tile_vol_pow",
    required=True,
    dest="tile_vol_pow",
    help="Power of tile_volume",
)
p.add_argument("-n", "--logN", required=True, dest="logN", help="Polynomial size")
opts = p.parse_args(sys.argv[1:])

tile_vol_pow = int(opts.tile_vol_pow)
logN = int(opts.logN)

with mlir_mod_ctx() as ctx:
    encrypt(tile_vol_pow, logN)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
