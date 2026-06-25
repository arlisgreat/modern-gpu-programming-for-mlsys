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

Buffers and memory
==================

Parameter buffers 使用 ``T.match_buffer`` 绑定；scratch buffers 在 body 中用下面两类
declaration APIs 创建。用 ``A[i, j]`` 索引 buffer，用 ``A[m0:m0+BM, 0:BK]`` 切片
（得到 ``BufferRegion`` ），并通过 ``A.ptr_to([i, j])`` 或 raw data pointer
``A.data`` 取 pointer。

Declaring buffers
-----------------

两个基础 API 用来创建 buffer：

- ``T.alloc_buffer(shape, dtype, scope=..., ...)`` — 分配新的 storage（发出
  ``AllocBuffer`` node）并返回 ``Buffer`` 。 ``T.alloc_shared`` / ``T.alloc_local``
  只是带 ``scope="shared"`` / ``scope="local"`` 的 ``alloc_buffer`` 。
- ``T.decl_buffer(shape, dtype, data=..., ...)`` — 在已有 pointer ``data`` 上
  声明一个 view（不分配）；用它来 alias 或 reinterpret storage，比如 pool 的
  sub-region，或 tensor-memory address。当 ``data=None`` 时，它会像
  ``alloc_buffer`` 一样分配。

buffer 的 ``data`` pointer 是 immutable ``Var`` （ ``alloc_buffer`` 定义它；
``decl_buffer`` 接收它）。如果要用 pointer *expression* 支撑一个 buffer，需要先
bind 它，见 :doc:`data_types`。

两者共享同一个 descriptor；最重要的参数如下：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - 含义
   * - ``dtype``
     - element type，例如 ``"float32"`` 、 ``"float16"`` 、 ``"float4_e2m1fn"`` 等
   * - ``shape``
     - logical shape（一组 extents）
   * - ``layout``
     - physical mapping（:ref:`TileLayout <chap_tirx_layout_api>`）； ``"default"`` = dense row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset`` （或 ``byte_offset`` ）把 *view* 放在 ``data`` 内的某个 offset； ``allocated_addr`` 携带预分配 address（tensor memory）
   * - ``align``
     - data pointer 的 alignment，以 bytes 计

``scope`` argument 选择 memory space：

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - Shorthand
     - Memory
   * - ``"global"``
     - (default)
     - device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - static shared memory（ ``__shared__`` ）
   * - ``"shared.dyn"``
     - (pool)
     - dynamic shared memory（pooled，见下文）
   * - ``"local"``
     - ``T.alloc_local``
     - per-thread registers
   * - ``"tmem"``
     - (TMEM pool)
     - Blackwell tensor memory（见下文）

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = T.alloc_shared((BM, BK), "float16")                 # new shared tile
    acc = T.alloc_local((4,), "float32")                     # register accumulator
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As

**基于 ptr 的 buffer 只是 pointer 上的一层 metadata。** 对任何 non-tmem buffer，
declaration 都是 pointer 加 layout，索引会解析成 address::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

（ ``layout.apply`` 返回 per-axis mapping；其中 ``"m"`` component 是 element
offset。）因此，*相同* logical access 会完全根据 buffer metadata 编译成不同的
address arithmetic。对 4×8 region 写 ``B[i, j] = A[i, j] + 1`` ，并用四种方式声明
``B`` ：

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

每种声明都会让 ``B[i, j]`` 在 generated CUDA 中降低为不同 index（ ``A[i, j]``
load 仍是 ``i*8 + j`` ，只有 ``B`` 的 metadata 变了）：

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Shared memory
-------------

Shared memory 有两种形式：static（compile time 固定）和 dynamic（launch
时确定大小）；此外还有一个 pool helper 用来管理 dynamic case。

Static
~~~~~~

最简单的 shared buffer 是 **static** buffer，也就是 ``T.alloc_shared`` （即
``scope="shared"`` ），大小在 compile time 确定。把 data stage 到其中，执行
``cta_sync`` 让整个 block 都看到写入，然后再读回：

.. code-block:: python

    @T.prim_func
    def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        B = T.match_buffer(B_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128])
        sm = T.alloc_shared((128,), "float32")   # static shared memory
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2.0)

它会降低成普通 ``__shared__`` array（generated CUDA，省略 boilerplate）：

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory（ ``scope="shared.dyn"`` ）按 launch 确定大小（由
``sharedMemBytes`` launch parameter 给出），不是 compile time 固定。一个 kernel
只能有 **一个** dynamic-shared allocation，也就是 *arena*。因此你只分配一次 arena，
然后用 ``T.decl_buffer`` 把每个 buffer 声明为它内部的 view： ``data=`` arena
pointer，再配合 ``elem_offset`` ：

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # the one arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个 views 共享同一个 ``extern __shared__`` arena（generated CUDA，省略
boilerplate；这里把 arena 命名为 ``smem`` 以便阅读）：

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

（两个独立的 ``alloc_buffer(scope="shared.dyn")`` 调用是错误的，*只允许一个
dynamic shared memory allocation*。）因此 static shared memory 在 compile time
定大小（ ``__shared__ T x[N];`` ）；dynamic shared memory 则是这个 launch-sized
arena，再在其中按 offsets 声明 views。

.. note::

   **TVM 如何标注 dynamic-shared size。** arena 的大小在 compile time 已知（这里
   ``128`` 个 floats = ``512`` bytes）。lowering 期间，TVM 会把
   ``"tirx.use_dyn_shared_memory"`` tag 追加到 device kernel 的
   ``tirx.kernel_launch_params`` ，host launcher 计算总 bytes，并把它作为最后一个
   launch argument 传入：

   .. code-block:: python

       # device kernel attribute:
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host-side launch call  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   运行时，这个 ``512`` 会成为 ``cuLaunchKernelEx`` call 中的
   ``config.sharedMemBytes`` 。你不需要手动设置它；它由 ``shared.dyn`` allocation
   的大小推导出来。

Pool sugar
~~~~~~~~~~

``T.SMEMPool`` 会自动处理 arena bookkeeping：它 bump-allocate offsets，因此你不用手动
``decl`` views。除了 ``alloc`` / ``commit`` ，它还提供 per-buffer ``align=`` 、
``alloc_mma`` helper（为你构造 MMA-compatible swizzle layout），以及用于回退 cursor
并复用空间的 ``move_base_to`` ：

.. code-block:: python

    pool = T.SMEMPool()                          # bump allocator over shared.dyn
    As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # MMA-compatible, swizzle inferred
    pool.commit()                                 # finalize the pool's size
    # pool.move_base_to(offset) rewinds the cursor to reuse space

TMEM pool（见下面的 `Tensor memory`_）构建在 ``SMEMPool`` 之上。

Registers
---------

Per-thread scratch 放在 registers 中。使用 ``T.alloc_local(shape, dtype)`` 分配它
（也就是 ``scope="local"`` ）：它对每个 thread 私有，并降低成保存在 registers 中的
local array。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # per-thread register array
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... compute on r[0..3] ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread, register-resident
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   ``alignas(64)`` 是 *default* buffer alignment：buffer 的 ``data_alignment``
   默认是 ``runtime::kAllocAlignment`` （64 bytes），CUDA codegen 会把它盖到每个
   allocation 上，包括这里这种 per-thread ``local`` arrays，虽然这里它没有意义。
   对这些 register-resident arrays，它 **没有 performance impact**：带静态可解析
   indices 的 thread-local array 会被 nvcc/ptxas promote 到 registers（scalar
   replacement of aggregates, SROA），因此它不会进入 addressable local memory，
   alignment 是 no-op。（如果一个 dynamically-indexed array spill 到 local memory，
   它确实会带上 over-alignment，但这不是常见情况。）register locals 的这种
   over-alignment 是已知 rough edge，后续计划修复（对 ``local`` scope 使用 dtype 的
   natural alignment）。

Scalar
~~~~~~

scalar 本质上只是 **一个元素** 的 register array；严格来说不需要单独概念。你可以分配
size-1 ``local`` buffer 并用 ``[0]`` 索引：

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 1-element register array
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

但到处写 ``phase[0]`` 很别扭，所以 **scalar** 正是这件事的 sugar：一个 one-element
register buffer，可以 **按名字** 读写：

.. code-block:: python

    phase: T.int32 = 0                 # mutable scalar (sugar for the above)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # explicit form; assign by name (s = ..., not s[0])
    acc: T.float32 = 0.0               # a type-annotated assignment also makes one

这两者不只是相似，而是会 parse 成 **structurally identical TIRx** 。sugar 完全在
parser 中解析： ``phase: T.int32`` *就是* 那个 one-element ``local`` buffer，而
``phase`` / ``phase += 1`` *就是* ``phase[0]`` / ``phase[0] += 1`` 。对两个
kernels 做 ``tvm.ir.assert_structural_equal`` 会通过，printer 甚至会把显式
``alloc_local`` + ``[0]`` form **打印回** scalar form。因此 parsing 之后二者没有任何
区别。两者都会降低成同一个 ``alignas(64) int phase_ptr[1];`` ；scalar 只是让你不用写
``[0]`` 。（ ``T.local_scalar`` / ``T.shared_scalar`` / ``T.alloc_scalar`` 用来显式选择
scope。）

.. note::

   **为什么不用 Var？** TIRx ``Var`` 是 *immutable*，即单个 static
   binding（也就是下面 ``T.let`` 产生的东西）。scalar 需要 *mutable*，因为 loops 和
   accumulators 中会反复给它赋值，所以它必须由一个可重复 store 的 one-element buffer
   支撑，而不是 ``Var`` 。

``let``
~~~~~~~

``T.let`` binding 是 **immutable** 的：它是单个 ``LetStmt`` （named value，不是
buffer）。它适合 derived constants：

.. code-block:: python

    n: T.let = M * K               # immutable binding (LetStmt)
    half: T.let[T.int32] = N // 2  # ... with an explicit type

它会降低成 **plain scalar C variable**，不是 buffer（没有 array，也没有 ``[0]`` ）。
例如 ``half: T.let = m * 2`` （ ``m`` 是 runtime value）：

.. code-block:: c++

    int half = m * 2;     // the `let` -> a const-like local

因为这个 value 是 immutable，simplifier 可以自由 propagate 和 CSE 它，所以在 use
sites 你经常会看到 ``m * 2`` 被直接替换进去（或通过 common-subexpression temporary
共享），而不是引用 ``half`` 。

.. note::

   **为什么需要 immutable binding？** 因为 value 不会改变，arithmetic analyzer 会把
   var bind 到它（simplify ``LetStmt`` 时执行 ``analyzer.Bind(var, value)`` ），所以
   关于这个 value 证明出的 facts，例如 constant bounds、modular set
   （divisibility / alignment）、ranges，都会 **贯穿每次使用传播** 。这会帮助 index
   simplification、bounds-check elimination，以及 alignment/vectorization decisions。
   *mutable* scalar 是 memory load（ ``buf[0]`` ）：analyzer 不能假设它保持不变，所以
   这些性质都无法传递。 ``let`` 还是 pure value：没有 allocation，并且可以自由
   inline / substitute / CSE；scalar 则是带 load/store semantics 的 one-element
   buffer。

Tensor memory
-------------

Blackwell *tensor memory* 不是普通 scratch scope：必须用 warp-uniform
``T.ptx.tcgen05.alloc`` / ``tcgen05.dealloc`` intrinsics 显式 reserve 和 free。
每个 tensor 都是其中的一个 view，用
``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)``
声明。 ``allocated_addr`` （column offset）是必需的，tensor-core dispatch 会 assert
它；因此 ``T.alloc_buffer(scope="tmem")`` （它 **不会** 设置 ``allocated_addr`` ）不能用。
不同于 shared memory，tensor memory 不能直接寻址；只能通过 ``tcgen05`` ``mma`` /
``ld`` / ``st`` / ``cp`` 读写。

手写时，一个 warp 把 allocation 发到 shared slot，你把每个 tensor ``decl`` 为某个
column offset 上的 view，并在结尾由一个 warp free 它：

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # slot for the allocated base
    if warp_id == alloc_warp:                         # tcgen05.alloc is warp-uniform
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # view at column 0
    # ... use acc as a gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)

你需要自己管理 column offsets 和 ``tmem_layout`` （datapath D/F layout）。下面的 pool
正是 emit 这套 sequence。

Pool
~~~~

``T.TMEMPool`` 封装了这些内容：warp-uniform alloc/dealloc、column
bump-allocation，以及 datapath layout：

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = the kernel's smem pool
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # allocated_addr set for you
    tmem_pool.commit()                               # emits tcgen05.alloc (one warp)
    # ... use acc ...
    tmem_pool.dealloc()                              # emits tcgen05.dealloc (one warp)

完整示例见 Part III 的 GEMM kernels。

Buffer APIs
-----------

``Buffer`` 是 pointer 上的 metadata（见上面的 *Declaring buffers*），因此大多数
methods 都是 *compile-time* reshapes/reinterprets：它们改变 index arithmetic，或把
pointer 交给你，本身不 emit runtime op。常见方法如下：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - 含义
   * - ``B.data``
     - raw data pointer（一个 ``Var`` ）；打印为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 指向某个 element 的 typed pointer（ ``address_of`` ）；打印为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - vectorized load / store；打印为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 用新的 shape/layout reinterpret 同一块 storage（无 copy）
   * - ``B.local(*shape, layout=…)``
     - calling thread 对 ``local`` buffer 拥有的 private register slice
   * - ``B.permute(*dims)``
     - axes 被 permute 后的 view（transposed layout）
   * - ``B.access_ptr(mask, …)``
     - masked access pointer（ ``tvm_access_ptr`` builtin），用于把 region 传给 intrinsic

Pointers — ``ptr_to`` / ``data``. ``ptr_to`` 用来把 element address 交给
intrinsic 或 inline function； ``data`` 是 base pointer：

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

**Vectorized access — ``vload`` / ``vstore``.** 把多个 elements 作为一次 wide
transfer 移动（另见 :doc:`data_types`）：

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret — ``view`` / ``permute``.** 二者都是 pure metadata；data
pointer 不变，只有 index arithmetic 不同。 ``A.view(64, 4)`` 把 256-element
buffer 看成 ``64×4`` ； ``A.permute(1, 0)`` 转置 axes：

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Registers — ``local``.** 把带 thread-axis 的 ``local`` layout 分解为 calling
thread 的 flat register bundle（tile primitives 中大量使用）：

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # this lane's 8 registers

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             // the lane's private registers
