(chap_gemm_async)=
# 将 GEMM 与 TMA 进行 pipeline 处理

:::{admonition} 概述
:class: overview

- 当两个阶段可以同时运行时，基本的 GEMM 却会浪费时间轮流执行（copy 一个 tile，compute，copy 下一个 tile）。
- 第 4 步切换到 TMA async load，第 5 步 double-buffer SMEM 并 prefetch（PIPE_DEPTH=2）；完整 load/compute overlap 在步骤 7 中随 warp specialization 到达，步骤 6 使 kernel 与 tile scheduler 保持一致。
- 目标是在 Tensor Cores 计算当前块的同时 load 下一个块。
:::

Tensor Cores 是芯片上最昂贵的单元，上一章中正确平铺的 GEMM 使它们在大部分时钟中处于空闲状态。kernel 轮流执行：threads 将一个 tile copy 到共享内存中，Tensor Cores 对其计算，threads copy 下一个 tile，然后 Tensor Cores 等待。即使 load 下一个 tile 并在当前 tile 上进行计算使用完全独立的硬件并且可以同时运行，每个阶段都会在前一个阶段停止。缩小这一差距并不需要新的数据路径；tile、layouts 和数学都已经正确了。必须改变的是工作“何时”发生以及“由谁”安排。本章将 tile 数据路径保持原样，并直接处理空闲状态。

我们通过三个渐进的步骤到达目的地，这有助于在开始之前了解目的地。在步骤 4 中，我们将批量 GMEM <-> SMEM 传输交给 TMA，以便专用 copy hardware 移动 tile 而不是 threads。在第 5 步中，我们添加了一个两级软件 pipeline，为下一个 K 块提供 landing stage，同时当前块仍在相乘。在第 6 步中，我们将 launch 重塑为由 tile scheduler 驱动的 persistent kernel，它分摊每个 tile setup，并让我们选择一个保持操作数热的 tile 顺序。在整个过程中，SMEM、TMEM 和寄存器 layouts 与我们在上一章中留下的内容完全相同。唯一真正的新想法是硬件单元之间的异步 handoff：让一个引擎先于另一个引擎运行，而不是让它们步调一致。

(chap_tma_async)=
## 步骤 4：TMA 异步加载

我们的第一步是让 copy 本身脱离关键路径。想一想 CTA 在步骤 1-3 中所做的事情：它的每一个 thread 都会计算地址并发出 load 指令，只是为了将块传送到 SMEM。这是本可用于 pipeline 的指令带宽，却没有用于数学计算。步骤 4 将同步 `Tx.copy` 替换为 TMA，其中单个 thread 发出一个 command，TMA engine 自行执行整个 tile transfer。从这里开始，示例以完整的 M=N=K=4096 大小运行，而不是步骤 1-3 的小大小，并且它们的端到端时序出现在 {ref}`chap_gemm_advanced` 末尾的“端到端结果”表中。

> **此步骤更改的内容：调度**
> - 范围：不变，1 个 warpgroup。
> - layout：不变，相同的 SMEM/TMEM/寄存器 tile。
> - 调度：GMEM → SMEM load 从同步 `Tx.copy` 移动到 TMA engine。

### TMA issue pattern

步骤 4 的一个更改是用 TMA load 替换同步 tile copy，因此有必要仔细查看该 load 的发出方式。对源代码的编辑只有几行，但这些行背后的执行模型是不同的。同步 `Tx.copy` 是 CTA threads 使用自己的指令自行完成的工作；TMA copy 是 thread 发出的 command，之后 TMA hardware 执行所有移动。两者并排看是值得的。

**之前（步骤 3）**：所有 128 个 threads 都参与 copy，然后 `cta_sync` 使共享内存写入可见：
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**之后（步骤 4）**：一个 thread 发出 TMA load，并且 mbarrier 跟踪硬件传输何时完成：
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

请注意，load 是在 `tid == 0` 上 gated，而不是在 `elect_sync()` 上 gated，而且这种区别比看起来更重要。`elect.sync` *为每个 warp* 选择一个 active lane，而 warpgroup 有四个 warps，因此 `elect_sync()` 实际上会让四个 threads 进入 load 协议。问题在于协议向 mbarrier 宣告预期的字节数，并且必须恰好宣告一次；四个公告会破坏计数，并且等待永远不会正确释放。通过 warpgroup 范围的 id 精确挑选一个 thread 是避免这种情况的干净方法。

诚实地了解加速的来源很重要。步骤 4 仍然在每个 TMA load 之后等待，因此我们尚未将 load 与 compute 重叠；这就是第 5 步的工作。这里的收益纯粹来自于数据移动路径的改变：

- `Tx.copy` 使用 CTA threads 来计算地址并发出 load/store 指令。
- TMA 使用一个 command 来启动硬件块传输。地址生成、coalescing 和 swizzling 由 TMA descriptor 描述，并由 TMA engine 执行。

因此，尽管第 4 步在每次 load 时仍然会阻塞，但它最终仍会更快。TMA 吸收了 bulk transfer，从而使 CTA threads 免于花费指令带宽来 shuffle，仅此节省就足以取得进展。

### TMA load 和 store 同步

我们已经看到了 TMA copy 是如何 issue 的；故事的另一半是知道它什么时候结束。切换到 TMA 会同时改变两件事：谁开始 copy，以及代码如何知道它何时完成。第一个从代码中显而易见；第二个很容易被忽视，如果出错的话会给你一个 silent correctness bug 而不是崩溃。使用 `Tx.cta.copy`，CTA threads 一起进行 copy，后面的 `cta_sync()` 足以知道它已完成。对于 TMA，选定的一个 thread 发出 `Tx.copy_async(..., dispatch="tma")`，引擎按自己的时间表执行传输，并通过 mbarrier 发出完成信号。

这正是 `cta_sync()` 不再足够的原因。`cta_sync()` 仅等待 CTA 自己的 threads 并仅约束其共享内存写入；它对 in-flight TMA 传输一无所知，因此当 tile 仍在到达时它也可能返回。修复方法是使完成变得明确：对于 TMA load，选定的 thread 首先告诉 mbarrier 需要多少字节，然后 CTA 在任何 MMA 接触该 SMEM tile 之前等待*那个* mbarrier。下图描绘了端到端的握手过程。

![TMA 异步加载：同步流程](../img/tma_sync_flow.png)

上图隔离了 load 端握手：选定的一个 thread 启动 TMA，mbarrier 计算预期字节，MMA 在读取 SMEM 之前等待释放。其中“选定 thread”表示选定的 thread 启动 TMA，在我们的代码中是 `tid == 0` thread，而不是 `elect_sync()` lane。

将 load path 放在一起看：选定的 thread 发出两个 `copy_async` 调用，并在它们之后调用 `arrive.expect_tx(total_bytes)`，其中字节数正是 mbarrier 应保留的数据量。一旦引擎移动了那么多字节，匹配的 `mbarrier.try_wait(phase)` 就会释放，只有这样 SMEM 块才能安全地馈送到 MMA。

store 端通过相同的硬件传输，但以不同的方式等待，因此有必要清楚地区分这两个协议：load 使用 mbarriers 和字节计数跟踪完成，而 store 使用 commit group 和 wait group 跟踪完成。在 threads 将其 fp16 结果写入 `Dsmem` 并同步后，选定的一个 thread 启动 `Tx.copy_async(D[...], Dsmem, dispatch="tma")`，然后是 `cp_async.bulk.commit_group()`，然后是 `cp_async.bulk.wait_group(0)`，直到 store drain。这种等待不是可选的：在前一个 store 消失之前，`Dsmem` 不能重新用于下一个 tile。

**尝试使用您的代理**：跟踪一个 K tile 的步骤 4 load 和 store 同步。识别哪个 thread 启动每个 TMA command，哪个 mbarrier 或 commit group 跟踪完成，哪个等待保护 `Asmem` 和 `Bsmem` 的 MMA 读取，哪个等待保护 `Dsmem` 的重用。为什么这里的 TMA load 协议选择 `elect_sync()` 是错误的 thread？

### 完整的内核

完整的 kernel 将 TMA load 和 store 折叠进步骤 3 的结构中，而该结构的其余部分保持不变。导入与以前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包装在 `hgemm_v4(M, N, K)` 中，这是我们始终遵循的模式：包装器将形状相关常量和 layouts 保留在使用它们的 kernel 旁边。

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()
    
        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0
    
        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )
    
        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
    
            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)
    
            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
    
            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)
    
            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1
    
        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    
        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### TMA 内核中的配置

kernel 中的几乎所有内容都是从步骤 3 继承的。只有五个配置点实际上携带 TMA 语义，并且值得通过名称来了解每个配置点：

- **TMA 配置**：`{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 告诉 `Tx.copy_async` 使用 TMA 并通过 `tma_bar` 报告加载完成。

- **字节计数**：`(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两个 fp16 操作数块加载的字节数。 `arrive.expect_tx(...)` 将该计数提供给 mbarrier。

- **mbarrier 初始化**：`init(tma_bar.ptr_to([0]), 1)` 创建 TMA load 使用的完成 barrier。

- **`@T.inline`**：`tma_load(...)` 和 `mma(...)` 是辅助函数。它们在编译时扩展为 kernel 主体，并且可以使用周围 kernel 中的变量。

- **TMA store 同步**：epilogue 首先将 fp16 行写入 `Dsmem`。 `fence.proxy_async` 和 `warpgroup_sync` 使那些 thread 写入的 SMEM 值准备好用于 TMA store 路径。然后，store 使用 `commit_group()` 和 `wait_group(0)` 等待 SMEM 到 GMEM 的传输完成。

此时数据路径是正确的，但调度还不对。步骤 4 仍然在开始匹配的 MMA 之前完成每个 load，因此 load 和 MMA 实际上不会同时运行；我们分开的两个引擎仍然轮流运转。下一步将 TMA load 和 store path 保持原样，而是重新安排 schedule，以便在计算另一个 tile 时可以继续 load 一个 K tile。

(chap_software_pipeline)=
## 步骤 5：软件 pipeline (PIPE_DEPTH=2)

当两个引擎明显独立时，为什么步骤 4 不能将 load 与 compute 重叠？原因是 storage。只有一个 SMEM tile pair 时，下一个 load 无处可去：直到当前 MMA 完成读取该 tile pair 后才能开始，因为提前开始会覆盖仍在使用的数据。步骤 5 通过 double-buffer shared memory 消除 storage conflict。单 warpgroup 循环在启动下一个 TMA load 之前仍然等待每个 MMA，但它现在具有不同的 stage 来 prefetch 和重用。我们仍然处于完整的 M=N=K=4096 大小。

> **此步骤更改的内容：layout**
> - 范围：不变，1 个 warpgroup。
> - layout：单个 SMEM tile pair 成为 `PIPE_DEPTH` stage ring buffer。
> - 调度：不变，TMA load 和 `tcgen05` MMA；此步骤添加了 prefetch 和 stage 重用，而完整 load/compute overlap 在步骤 7 中实现。

### pipeline 演练

对于 `PIPE_DEPTH=2`，kernel 分配两个 SMEM stage，为 load path 和 MMA path 提供单独的 slot。

将下图视为两级缓冲区要启用的 pipeline 结构，而不是该单 warpgroup kernel 的精确执行跟踪。第 5 步构建 ring buffer 并 prefetch 后面的 stage，但主循环在发出下一个 TMA load 之前仍然等待当前的 MMA。当 warp specialization 为 TMA 和 MMA 提供单独的角色时，完整 load/compute overlap 在步骤 7 中到达。

![*pipelinePIPE_DEPTH=2，目标进度；此单个 warpgroup 步骤仅预取，在步骤 7* 中与 warp specialization 完全重叠到达](../img/pipe_depth2.png)

一旦启动，循环就会交替经过两个 stage。两个 TMA load 预先填充两个 stage；之后，循环等待当前 stage，在其上运行 MMA，等待 MMA 完成读取该 stage，然后将 `k + PIPE_DEPTH` 的 load 启动到刚刚变得可重用的 stage。这还不是并发的 TMA/MMA 调度，但它建立了第 7 步将在生产者和消费者角色之间划分的 ring buffer 结构。

具体来说，该代码与步骤 4 有四个地方不同：

1. `Asmem` 和 `Bsmem` 获得领先的 `PIPE_DEPTH` 维度，因此每个阶段都有自己的 SMEM 存储。
2. `tma_bar` 成为一个阵列，每级一个 mbarrier。
3. 在主 K 循环之前，kernel 预取前两个阶段。
4. K 循环使用 `stage = k % PIPE_DEPTH`：等待当前阶段，在其上运行 MMA，然后为 `k + PIPE_DEPTH` 重用该阶段。

### pipeline 力学

**1. 预取**：在主循环运行之前，我们加载第一个 `PIPE_DEPTH` 阶段，以便循环始终在第一次迭代时找到等待它的数据：
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. 主循环**：对于每个 K tile，我们等待其 stage 准备就绪，在其上运行 MMA，然后通过提前启动 tile `PIPE_DEPTH` 的 load，立即使现在空闲的 stage 恢复工作：
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. 阶段管理**：这是令人困惑的部分，但规则比乍看起来要简单。每个 barrier 的 phase 翻转规则直接取决于该 barrier 有多少个槽，这就是两个 barrier 以不同节奏翻转的原因。 MMA 累加器位于一个 TMEM 槽中，因此 `mma_bar` 是每次迭代都会重新访问的单个 barrier (`mma_bar.ptr_to([0])`)，并且每次迭代都会重新访问的 barrier 必须在每次迭代时翻转其 phase。 TMA barrier 讲述了一个不同的故事：它们形成一个 `PIPE_DEPTH` 元素阵列，每个阶段有一个 barrier，并且任何给定阶段的 barrier 每次穿过环时只会返回一次。因此，`phase_tma` 仅当阶段索引回滚到 0 时才翻转：
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**尝试使用您的代理**：使用 `PIPE_DEPTH=2` 和 `K_TILES=5`，要求它跟踪主循环。对于每个 `k`，列出 `stage`、传递给等待的 `phase_tma` 和 `phase_mma` 值，以及是否发出新的预取。 `phase_tma` 到底在哪里翻转，为什么最后两次迭代没有预取？

### 完整的内核

完整的 kernel 保留步骤 4 的 TMA load 和 store path，然后将其包装在刚刚描述的 staged buffer 和 stage 逻辑中。imports 不变：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包裹在 `hgemm_v5(M, N, K)` 中。 `PIPE_DEPTH=2` 常量设置 pipeline 级数（这里有两个，这正是双缓冲）：

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## 第 6 步：persistent kernel + tile scheduler

到目前为止，一切都优化了单个 tile 内的工作。第 6 步更改问题的规模并跨 tile 进行优化。

第 5 步为每个 128 x 128 输出 tile 启动一个 CTA。对于 4096 x 4096 输出，这意味着 1024 个独立的 CTAs，每个都支付自己的设置成本，然后在其 tile 完成后消失。

步骤 6 改为启动 CTAs 的固定池，然后要求每个 CTA 依次处理许多 tile。这给我们带来了两件事：设置工作分摊到多个 tile 上，tile 分配在 kernel 内移动，scheduler 可以在其中选择重用操作数的顺序。我们保持完整的 M=N=K=4096 大小。

> **此步骤更改的内容：范围**
> - 范围：持久 CTAs 的固定池，每个池通过 scheduler 循环多个输出块。
> - layout：未更改，每个 tile SMEM/TMEM/寄存器路径相同。
> - 调度：不变。

### 持续调度

persistent kernel 的定义思想是根据硬件而不是问题来确定 grid 的大小。它推出 `SM_COUNT` CTAs，大约每个 SM 一个，无论有多少个输出块，目的是保持每个 SM 持续被占用。我们故意说“大致”：不能保证精确的 1:1 驻留，因为它取决于占用情况以及硬件选择如何安排 CTAs。

在 B200 上，我们的目标是 `SM_COUNT=148`。这 148 个 CTAs 中的每一个都会循环 `ClusterPersistentScheduler2D` 交给它的 tile。

第一个回报是摊销。 TMEM 分配、barrier 初始化和 scheduler 状态现在每个 CTA 发生一次，并在 CTA 处理的大约 7 个 tile 中重复使用，而不是在一次性 CTAs 上重复 1024 次。

第二个回报来自 scheduler 选择的顺序。设置 `l2_group_size=8` 将附近的 tile 分组在一起，因此共享行带的 tile 重复使用相同的 A 行 tile，共享列带的 tile 重复使用相同的 B tile。连续运行这些块可以使操作数在 L2 中保持热状态，而不是从 HBM 重新获取它们。这正是步骤 3 中留下的重用。

```python
bx = T.cta_id([SM_COUNT])  # 1D grid, one CTA per SM

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # Group 8 nearby tiles together
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

循环遍历 tile 会带来一种很容易被忽略的正确性结果。每个 tile 都运行自己的新 K 循环，这意味着其 barrier 阶段必须从已知状态开始。在步骤 5 中，CTA 只处理了一个 tile，因此一次初始化 `phase_tma` 和 `phase_mma` 就完全没问题了。在步骤 6 中，这些初始化器必须移动到 `while tile_scheduler.valid()` 循环的“内部”，以便每个 tile 以与其自己的 TMA 和 MMA 工作相匹配的相状态开始，而不是继承前一个 tile 碰巧留下的任何内容：

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### 完整的内核

从结构上来说，kernel 只不过是包裹在 tile 级外循环中的 Step 5 pipeline。唯一的新依赖项是 scheduler 本身，我们将其与其他依赖项一起导入：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

grid 维度现在只是 `SM_COUNT` 而不是 `(M//BLK_M, N//BLK_N)`，并且 `ClusterPersistentScheduler2D` 接管向每个 CTA 分配其 tile 的工作：

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 在步骤 4 中，`arrive.expect_tx`使用`(BLK_M * BLK_K + BLK_N * BLK_K) * 2`字节。如果此字节数太小或太大，mbarrier 会等待什么？
2. 在步骤 5 中，为什么每个 SMEM 级需要自己的 TMA barrier，而不是为两个级共享一个 `tma_bar`？
3. 在步骤 6 中，带有 `BLK_M=BLK_N=128` 的 4096 x 4096 输出有多少个输出 tile？对于 `SM_COUNT=148`，每个持久 CTA 平均处理多少个 tile？
