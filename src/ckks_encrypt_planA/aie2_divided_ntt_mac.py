# section-4/section-4a/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates


import argparse

import aie.utils.trace as trace_utils
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx
import aie.extras.dialects.ext.arith as arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.extras import types as T

# ===================================
# Change here for different configurations
# all_size_log = 17
# ===================================

DEBUG_ON = 1
DEBUG_OFF = 0
STORE_FACTOR = 1
NO_STORE_FACTOR = 0

def divided_ntt_internal(trace_size, all_size_log):

    # enableTrace = opts.trace_size > 0
    enableTrace = False
    trace_size = opts.trace_size

    col_num_log = 2
    raw_num_log = 2
    cores_num_log = col_num_log + raw_num_log
    col_num = 1<<col_num_log
    raw_num = 1<<raw_num_log
    cores_num = 1<<cores_num_log

    n_log = all_size_log
    n1_log = (n_log + 1) // 2
    n2_log = n_log // 2
    n = 1 << n_log
    n1 = 1 << n1_log
    n2 = 1 << n2_log

    loops = 1 << (n1_log - col_num_log - raw_num_log)
    
    factor_buff_size = 16 * 3
    factor_FIFO_size = n_log + 8  # +4 for barrett_w, barrett_u, and logn

    @device(AIEDevice.npu1)
    def device_body():
        all_ty = np.ndarray[(n,), np.dtype[np.int32]]
        all_ty_in = np.ndarray[(n*5,), np.dtype[np.int32]]
        all_ty_out = np.ndarray[(n*2,), np.dtype[np.int32]]
        cores_ty = np.ndarray[(n2,), np.dtype[np.int32]]
        cores_ty_out = np.ndarray[(n2*2,), np.dtype[np.int32]]
        factor_buff_ty = np.ndarray[(factor_buff_size,), np.dtype[np.int32]]
        mem_ty = np.ndarray[(n2*raw_num,), np.dtype[np.int32]]
        mem_ty_out = np.ndarray[(n2*raw_num*2,), np.dtype[np.int32]]
        factor_FIFO_ty = np.ndarray[(factor_FIFO_size,), np.dtype[np.int32]]
        factor_mem_FIFO_ty = np.ndarray[(factor_FIFO_size,), np.dtype[np.int32]]
        factor_all_ty = np.ndarray[(factor_FIFO_size*col_num,), np.dtype[np.int32]]


        # ===================
        # External Function
        # ===================
        multi_NTT_in_a_tile = external_func(
            "multi_NTT_in_a_tile",
            inputs = [cores_ty,cores_ty,factor_buff_ty,factor_FIFO_ty,np.int32,np.int32,np.int32,np.int32,np.int32,np.int32,np.int32,np.int32],
        )

        # void elemwise_mac(int32_t *dst, int32_t *add, int32_t *mul1, int32_t *mul2, int32_t mod, int32_t barrett_w, int32_t barrett_u, int32_t buff_size) 
        elemwise_mac = external_func(
            "elemwise_mac",
            inputs = [cores_ty, cores_ty, cores_ty, cores_ty, np.int32, np.int32, np.int32, np.int32],
        )

        # void elemwise_mac_with_dst_offset(int32_t *dst, int32_t *add, int32_t *mul1, int32_t *mul2, int32_t mod, int32_t barrett_w, int32_t barrett_u, int32_t buff_size, int32_t offset)
        elemwise_mac_with_dst_offset = external_func(
            "elemwise_mac_with_dst_offset",
            inputs = [cores_ty_out, cores_ty, cores_ty, cores_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
        )

        # void copy_buff(int32_t *dst, int32_t *src, int32_t buff_size) {
        copy_buff = external_func(
            "copy_buff",
            inputs = [cores_ty, cores_ty, np.int32],
        )

        # void copy_buff_to_large_dst(int32_t *dst, int32_t *src, int32_t buff_size, offset)
        copy_buff_to_large_dst = external_func(
            "copy_buff_to_large_dst",
            inputs = [cores_ty_out, cores_ty, np.int32, np.int32],
        )

        # ===================
        # Define Tiles
        # ===================
        CT_tile_ls = []
        Mem_tile_ls = []
        Shim_tile_ls = []
        for col in range(col_num):
            Shim_tile_ls.append(tile(col,0))
            Mem_tile_ls.append(tile(col,1))
            CT_tile_ls.append([])
            for raw in range(raw_num):
                CT_tile_ls[col].append(tile(col,raw+2))


        # ===================
        # Define FIFOs
        # ===================
        Mem_CT_FIFO_ls = []
        Mem_CT_factor_FIFO_ls = []
        CT_Mem_FIFO_ls = []
        for col in range(col_num):
            Mem_CT_FIFO_ls.append([])
            CT_Mem_FIFO_ls.append([])
            Mem_CT_factor_FIFO_ls.append(object_fifo(f"of_Mem_{col}_CT_factor_FIFO", Mem_tile_ls[col], [CT_tile_ls[col][raw] for raw in range(raw_num)] , 1, factor_FIFO_ty))

            for raw in range(raw_num):
                Mem_CT_FIFO_ls[col].append(object_fifo(f"of_Mem_CT_FIFO_{col}_{raw}", Mem_tile_ls[col], CT_tile_ls[col][raw], 1, cores_ty))
                CT_Mem_FIFO_ls[col].append(object_fifo(f"of_CT_Mem_FIFO_{col}_{raw}", CT_tile_ls[col][raw], Mem_tile_ls[col], 1, cores_ty_out))

        Shim_Mem_FIFO_ls = []
        Shim_Mem_factor_FIFO_ls = []
        Mem_Shim_FIFO_ls = []
        for col in range(col_num):
            Shim_Mem_FIFO_ls.append(object_fifo(f"of_Shim_Mem_FIFO_{col}",  Shim_tile_ls[col],Mem_tile_ls[col], 1, mem_ty))
            Shim_Mem_factor_FIFO_ls.append(object_fifo(f"of_Shim_Mem_factor_FIFO_{col}",  Shim_tile_ls[col],Mem_tile_ls[col], 1, factor_mem_FIFO_ty))
            Mem_Shim_FIFO_ls.append(object_fifo(f"of_Mem_Shim_{col}_FIFO", Mem_tile_ls[col], Shim_tile_ls[col], 1, mem_ty_out))


        # ===================
        # Link FIFOs
        # ===================
        for col in range(col_num):
            object_fifo_link(Shim_Mem_FIFO_ls[col], [Mem_CT_FIFO_ls[col][raw] for raw in range(raw_num)],[],[raw*n2 for raw in range(raw_num)])
            object_fifo_link( [CT_Mem_FIFO_ls[col][raw] for raw in range(raw_num)], Mem_Shim_FIFO_ls[col],[raw*n2*2 for raw in range(raw_num)],[])
            object_fifo_link(Shim_Mem_factor_FIFO_ls[col], Mem_CT_factor_FIFO_ls[col],[],[raw * factor_FIFO_size for raw in range(raw_num)])


        # ===================
        # Buffers
        # ===================
        buff0_ls = []
        buff1_ls = []
        buff2_ls = []
        for col in range(col_num):
            buff0_ls.append([])
            buff1_ls.append([])
            buff2_ls.append([])
            for raw in range(raw_num):
                buff0_ls[col].append( buffer(tile=CT_tile_ls[col][raw], datatype= cores_ty))
                buff1_ls[col].append( buffer(tile=CT_tile_ls[col][raw], datatype= cores_ty))
                buff2_ls[col].append( buffer(tile=CT_tile_ls[col][raw], datatype= cores_ty))


        factor_buff_ls = []
        for col in range(col_num):
            factor_buff_ls.append([])
            for raw in range(raw_num):
                factor_buff_ls[col].append( buffer(tile=CT_tile_ls[col][raw], datatype= factor_buff_ty))

        # ===================
        # Core Body
        # ===================
        for col in range(col_num):
            for raw in range(raw_num):
                @core(CT_tile_ls[col][raw], "kernel_func.o")
                def core_body():
                    for _ in range_(sys.maxsize):
                        core_index = col * raw_num + raw

                        factor_FIFO_buff = Mem_CT_factor_FIFO_ls[col].acquire(ObjectFifoPort.Consume, 1)
                        factor_buff = factor_buff_ls[col][raw]
                        buff_v = buff0_ls[col][raw]
                        buff1 = buff1_ls[col][raw]
                        buff2 = buff2_ls[col][raw]

                        # =====================================
                        # Prepare parameters
                        # =====================================
                        modulo_q = factor_FIFO_buff[all_size_log]
                        barret_w = factor_FIFO_buff[all_size_log + 1]
                        barret_u = factor_FIFO_buff[all_size_log + 2]
                        logn_for_current_ntt = factor_FIFO_buff[all_size_log + 3]
                        n_for_current_ntt = factor_FIFO_buff[all_size_log + 4]
                        if_phase2 = 1                        
                        for loop in range_(loops):


                            w_offset_pow = loop * cores_num + core_index
                            w_offset_pow_i32 = arith.index_cast(w_offset_pow, to=np_dtype_to_mlir_type(np.int32))
                            
                            # =====================================
                            # Copy input to local memory of ComputeTile
                            # =====================================
                            out_vec = CT_Mem_FIFO_ls[col][raw].acquire(ObjectFifoPort.Produce, 1)
                            
                            # =====================================
                            # Calculate NTTP2 + MAC
                            # v = NTTP2(v)
                            # m' = NTTP2(m')
                            # e1 = NTTP2(e1)
                            # c0 = v * pkb + m'
                            # c1 = v * pka + e
                            # =====================================
                            # Read v and NTTP2
                            in_vec_v = Mem_CT_FIFO_ls[col][raw].acquire(ObjectFifoPort.Consume, 1)
                            copy_buff(buff_v, in_vec_v, n2)
                            Mem_CT_FIFO_ls[col][raw].release(ObjectFifoPort.Consume, 1)
                            multi_NTT_in_a_tile(buff_v, buff_v, factor_buff, factor_FIFO_buff, n_log,
                                                n2_log, n2_log, modulo_q, barret_w, barret_u, w_offset_pow_i32, if_phase2)
                            
                            # =========
                            # c0 = v * pkb + m'
                            # =========

                            # Read m' and NTTP2
                            in_vec_m = Mem_CT_FIFO_ls[col][raw].acquire(ObjectFifoPort.Consume, 1)
                            copy_buff(buff2, in_vec_m, n2)
                            Mem_CT_FIFO_ls[col][raw].release(ObjectFifoPort.Consume, 1)
                            multi_NTT_in_a_tile(buff2, buff2, factor_buff, factor_FIFO_buff, n_log,
                                                n2_log, n2_log, modulo_q, barret_w, barret_u, w_offset_pow_i32, if_phase2)
                            
                            # Read pkb
                            in_vec_b = Mem_CT_FIFO_ls[col][raw].acquire(ObjectFifoPort.Consume, 1)

                            # c0 = v * pkb + m'
                            elemwise_mac_with_dst_offset(out_vec, buff2, buff_v, in_vec_b, modulo_q, barret_w, barret_u, n2, 0)
                            Mem_CT_FIFO_ls[col][raw].release(ObjectFifoPort.Consume, 1)
                            
                            # =========
                            # c1 = v * pka + e1
                            # =========
                            # Read e and NTTP2
                            in_vec_e = Mem_CT_FIFO_ls[col][raw].acquire(ObjectFifoPort.Consume, 1)
                            copy_buff(buff2, in_vec_e, n2)
                            Mem_CT_FIFO_ls[col][raw].release(ObjectFifoPort.Consume, 1)
                            multi_NTT_in_a_tile(buff2, buff2, factor_buff, factor_FIFO_buff, n_log,
                                                n2_log, n2_log, modulo_q, barret_w, barret_u, w_offset_pow_i32, if_phase2)
                            
                            # Read pka
                            in_vec_a = Mem_CT_FIFO_ls[col][raw].acquire(ObjectFifoPort.Consume, 1)

                            # c1 = v * pka + e
                            elemwise_mac_with_dst_offset(out_vec, buff2, buff_v, in_vec_a, modulo_q, barret_w, barret_u, n2, n2)
                            Mem_CT_FIFO_ls[col][raw].release(ObjectFifoPort.Consume, 1)

                            CT_Mem_FIFO_ls[col][raw].release(ObjectFifoPort.Produce, 1)
                        
                        Mem_CT_factor_FIFO_ls[col].release(ObjectFifoPort.Consume, 1)

        # ===================
        # Runtime Sequence
        # ===================
        @runtime_sequence(all_ty_in,factor_all_ty,all_ty_out)
        def sequence(in_put,in_put_factor,out_put):
            inner_max = 512
            if n2 < inner_max:
                inner_max = n2
            for col in range(col_num):
                npu_dma_memcpy_nd(metadata = Shim_Mem_FIFO_ls[col], bd_id=col, mem=in_put ,
                                   sizes=[loops, 5, n2*raw_num//inner_max, inner_max],
                                   offsets=[0,0,0,col*n2*raw_num],
                                   strides=[5*n2*cores_num,n2*cores_num,inner_max,1],
                                   issue_token=True)

                npu_dma_memcpy_nd(metadata = Shim_Mem_factor_FIFO_ls[col], bd_id= col + col_num, mem=in_put_factor ,
                                   sizes=[1, 1, 1, factor_FIFO_size], offsets = [0,0,0,factor_FIFO_size*col],
                                   issue_token=True)

                npu_dma_memcpy_nd(metadata = Mem_Shim_FIFO_ls[col], bd_id=2*col_num+col, mem=out_put,
                                   sizes=[1, loops, n2*raw_num*2//inner_max, inner_max],
                                   offsets=[0,0,0,col*n2*raw_num*2],
                                   strides=[0,n2*cores_num*2,inner_max,1],
                                   )

            # npu_sync(column=0, row=0, direction=0, channel=0)
            dma_wait(*Mem_Shim_FIFO_ls)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    p.add_argument("-n", "--logN", required=True, dest="logN", help="Polynomial size")
    opts = p.parse_args(sys.argv[1:])
    trace_size = int(opts.trace_size)
    all_size_log = int(opts.logN)
    with mlir_mod_ctx() as ctx:
        divided_ntt_internal(trace_size, all_size_log)
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)