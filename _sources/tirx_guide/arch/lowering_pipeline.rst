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

TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 会把你写好的 TIRx module 送入
**tirx pipeline** 。这是一串有序的 TIR passes，会把你写的 high-level constructs
（tile primitives、 ``TileLayout``-typed buffers、execution-scope ids）变成拆开的
**host** + **device** functions，然后由 CUDA backend 渲染成 source。pipeline
定义在 ``python/tvm/tirx/compilation_pipeline.py`` （ ``tirx_pipeline`` ）；本页按顺序
走一遍这些 passes。

Where it sits
-------------

``tvm.compile`` 首先 bind target，运行 **tirx pipeline**（下面这些 module-level
passes），然后分别对 host 和 device functions 应用 **finalization** passes，最后把每个
device function 交给 CUDA code generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

The passes
----------

``tirx_pipeline`` module pass 会应用下面这个精确顺序（少数 pass 受
``PassContext`` config 控制）：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - 核心 lowering，见下面的 `Inside LowerTIRx`_
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis bindings，使每个 ``threadIdx`` / ``blockIdx`` axis 只声明一次
   * - 3
     - ``StmtSimplify``
     - statement-level arithmetic simplification（arith analyzer）
   * - 4
     - ``LowerTIRxOpaque``
     - 将剩余 opaque TIRx constructs 降低为 plain TIR
   * - 5
     - ``FlattenBuffer``
     - 将 multi-dimensional ``BufferLoad`` / ``BufferStore`` flatten 到 1-D
   * - 6
     - ``BF16ComputeLegalize``
     - 将 ``bfloat16`` compute 改写为 legal（f32-up-cast）形式
   * - 7
     - ``NarrowDataType(32)``
     - 在可证明安全的位置，把 index/loop ``PrimExpr`` dtypes 收窄到 32-bit
   * - 8
     - ``VectorizeLoop``
     - 将 ``T.vectorized`` loops 变成 vector ops（如果设置 ``tir.disable_vectorize`` 则跳过）
   * - 9
     - ``UnrollLoop``
     - unroll 标记为 ``T.unroll`` 的 loops（以及小的 constant loops）
   * - 10
     - ``StmtSimplify``
     - 在 vectorize/unroll 暴露 constants 后再次 simplify
   * - 11
     - ``CommonSubexprElim``
     - 将重复 subexpressions hoist 到 temporaries（如果设置 ``tir.disable_cse_tir`` 则跳过）
   * - 12
     - ``FP8ComputeLegalize``
     - 将 ``float8`` compute 改写为 legal form
   * - 13
     - ``VerifyMemory``
     - 检查 host-side code 没有直接 dereference device memory（safety gate）
   * - 14
     - ``AnnotateEntryFunc``
     - 将单个 PrimFunc 标记为 module entry point
   * - 15
     - ``SplitHostDevice``
     - 在 ``launch_thread`` boundary 将每个 kernel 拆成 **host** function 和 **device** function
   * - 16
     - ``MakePackedAPI``
     - 将 host function 改写为 packed-func ABI（TVM 调用的 launcher）
   * - 17
     - ``FP8StorageLegalize``
     - legalize ``float8`` storage（打包进支持的 container types）
   * - 18
     - ``BF16StorageLegalize``
     - legalize ``bfloat16`` storage

随后 **Finalization** 会按 function kind 运行：

- **host**： ``LowerTVMBuiltin`` （lower ``tvm_*`` builtins）、 ``LowerIntrin``
  （target-specific intrinsics）
- **device**： ``LowerWarpMemory`` （warp-scoped buffers → shuffles）、 ``StmtSimplify`` 、
  ``LowerIntrin``

Inside LowerTIRx
----------------

``LowerTIRx`` 本身也是一个小 sequence（ ``src/tirx/transform/lower_tirx.cc`` ）：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 将每个 ``TilePrimitiveCall`` （ ``copy`` 、 ``gemm`` 、
  ``reduction`` 等）替换为所选 backend dispatch 发出的 body，也就是
  variant-selection 和 codegen 的结果。
- **``LowerTIRxCleanup``** 运行 ``LayoutApplier`` ：它把每个 ``TileLayout``-typed
  buffer access 解析成具体 physical address arithmetic
  （ ``addr = data + elem_offset + layout.apply(coord)`` ），flatten buffers，并把
  execution-scope ids（ ``T.cta_id`` / ``T.thread_id`` / …）通过 ``launch_thread``
  降低为 ``blockIdx`` / ``threadIdx`` 。

因此 ``LowerTIRx`` 之后，module 已经是 plain TIR：没有 tile primitives，没有
``TileLayout`` indirection，scope ids 已解析为 thread axes。

A worked example
----------------

看一个单行 scale kernel：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**经过 ``LowerTIRx`` 后**，scope ids 已经是真正的 thread axes，layout 也已经被
applied（ ``A_1`` / ``B_1`` 是 flattened 1-D views）：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**经过 ``SplitHostDevice`` + ``MakePackedAPI`` 后**，一个 function 变成两个：host
launcher 和 device kernel：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

随后 CUDA backend 会把 ``scale_kernel`` 渲染成 ``__global__`` function
（ ``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f`` ）。

Reproduce it yourself
---------------------

你可以手动运行 pipeline 的任意 prefix 来检查某个 stage。这些文档中的 IR snippets
就是这样生成的：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

或者编译完整 module 并读取 generated CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
