..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

CUDA C++/PTX intrinsics
=======================

当没有 tile primitive 覆盖你需要的操作时，有两个 escape hatch 可以直接触达硬件：
**调用 backend intrinsic**（来自 ``tvm.backend.cuda`` 的 ``T.cuda.*`` /
``T.ptx.*`` namespaces），或者 **inline raw CUDA** source。

Calling backend intrinsics
--------------------------

``T.cuda.*`` 和 ``T.ptx.*`` 直接暴露 CUDA backend 的 device intrinsics，包括
synchronization、mbarriers、reductions，以及 PTX data-movement / MMA families：

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier (__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
    T.ptx.mbarrier.try_wait(bar.data, phase)

一个完整可运行的例子：通过 ``T.tvm_warp_shuffle_xor`` 做 warp all-reduce：

.. code-block:: python

    @T.prim_func
    def warp_reduce(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32", align=16)
        T.device_entry()
        cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

这个 shuffle 会直接降低为 ``__shfl_xor_sync`` ：

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 下还有其他 families： ``cp_async`` (LDGSTS)、
``cp_async.bulk.tensor`` (TMA)、 ``ldmatrix`` / ``stmatrix`` 、 ``tcgen05.*``
（Blackwell MMA）、 ``atomic_add`` 、 ``fence`` 等。完整 ``tvm.backend.cuda``
reference 见 backend API reference。

Synchronization semantics
-------------------------

GEMM 和 FlashAttention kernels 中反复出现四种 synchronization mechanisms。因为
它们控制 asynchronous engines 和 parallel thread groups，误用其中任何一种通常会
导致 silent corruption 或 deadlock。

**Mbarrier Phases.** Mbarriers 用一个 internal phase bit 跟踪 arrivals。
``T.ptx.mbarrier.try_wait(bar, phase)`` intrinsic 会阻塞，直到 barrier 的
internal phase 与 caller 提供的 ``phase`` argument *不同*。因此，跨 loop
iterations 重用 barrier 时，caller 必须在每次 wait 后翻转本地 phase tracker
（ ``phase ^= 1`` ）。如果忘记这样做，后续 waits 会立刻返回，使 engine 可能读取
half-written memory。:ref:`chap_gemm_basics` 逐步展示了完整 phase-tracking
table。

Election. ``T.ptx.elect_sync()`` 会选出 *一个 warp 内的 single active
lane*，不是 lane 0，也不是每个 CTA 一个 thread。要把 issuer 缩窄到恰好一个
thread，必须配合 warp-level guard。:ref:`chap_gemm_basics` 中发出
``Tx.gemm_async`` 和 ``tcgen05.commit`` 时使用的模式是先写
``if warp_id == 0:`` ，再写 ``if T.ptx.elect_sync():`` 。

Named Warpgroup Barriers. ``T.cuda.cta_sync()`` 映射到 ``__syncthreads()`` ，
并要求 *每个* CTA thread 都到达。一旦 warpgroups specialize 到不同 code paths，
把 ``cta_sync()`` 放进 warpgroup branch 就会让 kernel deadlock，因为其他
warpgroups 永远到不了这里。硬件提供 16 个 named barriers（ID 0 到 15）；
``T.cuda.warpgroup_sync(10)`` 只同步一个 warpgroup 的 threads。不同
warpgroups 使用不同 IDs（例如 ``warpgroup_sync(wg_id + 10)`` ），这样不会撞到同
一个 hardware barrier。见 :ref:`chap_gemm_advanced`。

**Fences.** Fences 约束 producer writes 必须排在 consumer（通常是 asynchronous
engine）读取之前：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - 排序内容
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - thread 写入的 shared memory，排在 async proxy（TMA store / MMA）读取之前
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier initialization，排在后续 arrivals 或 waits 使用该 barrier 之前
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - ``tcgen05`` writeback edge 上的保守 ordering fence（Steps 8 和 9 会加它；TMA-to-MMA path 不需要）

Inlining raw CUDA
-----------------

如果某个操作完全没有 intrinsic，可以通过
``T.cuda.func_call(name, *args, source_code=..., return_type=...)`` 从 source
string 注入一个 ``__device__`` function：

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

source 会原样 emitted，并把 call 接上：

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
