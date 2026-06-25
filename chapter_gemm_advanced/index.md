(chap_gemm_advanced)=
# 通过 Warp 专业化和 cluster 扩展 GEMM

:::{admonition} 概述
:class: overview

- pipeline 的 GEMM 仍然有一个 warpgroup 依次执行加载、MMA 和写回，本章消除了瓶颈。
- 步骤 7 将 warps 专门化为角色，步骤 8 添加 2-CTA cluster，步骤 9 添加多个消费者。
- 每个步骤都消除了串行瓶颈，最终吞吐量接近 state-of-the-art。
:::

上一章中的 pipeline GEMM ({ref}`chap_gemm_async`) 速度很快，但它仍然要求一个 warpgroup 完成所有操作：发出负载，运行 MMA，然后将结果写回。即使有软件 pipeline，threads 的一个团队也会成为所有三个引擎相遇的地方。

症状很容易看出。 TMA 单元在 Tensor Cores 运行时保持安静，Tensor Cores 在结果消耗到内存时保持安静，每个引擎通过一组 threads 等待其他引擎。解决这个问题的方法就是停止让一个团队做所有事情。

我们通过扩大合作的三个步骤来实现这一想法。步骤 7 ({ref}`chap_warp_specialization`) 将 warps 专门化为生产者、消费者和写回角色。步骤 8 ({ref}`chap_cta_cluster`) 将两个 CTAs 连接成一个 cluster，该 cluster 在其共享内存中共享操作数。步骤 9 ({ref}`chap_multi_consumer`) 添加第二个 MMA 消费者，因此一个分阶段的 tile 提供两倍的数学运算。

这有助于将这三个步骤视为不同尺度的一种模式。步骤 7 将完整 pipeline 保留在一个 CTA 中：TMA 和 MMA 共享一个 warpgroup，而写回在另一个中运行。第 8 步扩大了 CTAs 之间的合作，生成跨越两者的 256×256 tile。第 9 步进一步推动计算密度：cluster 输出增长到 512×256，每个阶段 B tile 都被两个消费者重复使用，我们得到了教程中最密集的变体。

在这一切过程中，有一件事始终保持不变。 SMEM、TMEM 和寄存器 layouts 仍然遵循我们在前两章中构建的合约；改变的是“谁合作”，而不是数据的 layout 方式。步骤 8 是协作 scope 第一次扩展超过单个 CTA，因此其操作数块被分割到两个 CTAs' 共享内存中，并且一个 layout 沿着 `cbx` 跨越两个 CTAs cluster 轴。


(chap_warp_specialization)=
## 第 7 步：扭曲专业化+pipeline

单 warpgroup kernel 将性能留在桌面上的原因很简单：每个 thread 都走相同的路径，加载，然后计算，然后写入，因此在加载时，Tensor Cores 无所事事，而在计算时，TMA 引擎无所事事。修复为 *warp specialization*。我们没有要求 threads 的一个团队轮流完成每项工作，而是将每项工作交给专门的 warp，并让这些 warps 同时运行，通过软件 pipeline 缝合在一起。这是 GEMM 路径中最大的架构变化，本章的其余部分建立在它的基础上。这里的基准使用 M=N=K=4096。

> **此步骤更改的内容：范围**
> - 范围：一个 warpgroup 步行负载 → MMA → 按顺序写回变成由满/空 barrier 连接的三个并发角色（TMA 生产者、MMA 消费者、写回）。
> - layout：不变，与步骤 6 相同的 SMEM 级和 TMEM 累加器。
> - 调度：不变，TMA 负载，`tcgen05` MMA。

**主题。**

- 扭曲专业化：将不同的 warps/warpgroups 专用于不同的任务

- 高级 barrier 抽象：`TMABar`、`TCGen05Bar`、`MBarrier`

- `PipelineState` 用于自动阶段/阶段管理

- `warpgroup_sync` 用于每个 warpgroup 同步的 barrier ID

（多级 SMEM pipeline 和持久 `ClusterPersistentScheduler2D` 与步骤 5-6 一样重复使用；这里只有 scope 拆分是新的。）

### 从顺序到并发

在介绍角色和 barrier 之前，先隔离一下 warp specialization 消除的调度瓶颈。下图使用 Step-4 样式的顺序时间线作为步骤 4-6 中预专业化 kernels 的紧凑参考，然后将其放在步骤 7 warp-specialized 时间表之上，因此引擎利用率的差异一目了然。

![扭曲专业化时间线](../img/warp_specialization_timeline.png)

最上面的是预专业化单 warpgroup 模式：相同的非专业化 thread 组同时拥有负载路径和 MMA 路径，因此一个引擎可以轻松地闲置，而另一个引擎则处于活动状态。步骤 5 和 6 通过双缓冲和 persistent scheduling 改进了该基线，但它们尚未将加载和计算拆分为独立的生产者和消费者角色。从底层来看，专业化打破了这种轮流制。当 MMA 消费者忙于计算时，TMA 生产者预取下一个 tile，并且写回自行进行。生产者 warp 3 发出下一个负载，而消费者 warp 0 仍在处理当前的 MMA，因此两个引擎都不必等待对方。负载/MMA 切换使用两个 barrier：

- **`tma2mma`** (TMA → MMA)：表示加载的 SMEM 数据已准备好供 MMA 使用。
- **`mma2tma`** (MMA → TMA)：表示 MMA 已完成读取缓冲区，因此 TMA 可以将其重新用于下一次加载。

图中的一个细节乍一看可能像是一个错误：`mma2tma` 箭头向前跳了一个阶段。原因是环形缓冲区。对于 `PIPE_DEPTH=2`，有两个 SMEM 缓冲区，阶段 0 和阶段 1； TMA Load k=0 填充缓冲区 0，TMA Load k=1 填充缓冲区 1。当 MMA Compute k=0 完成读取缓冲区 0 时，它向 `mma2tma` 发出信号，表示缓冲区空闲，但实际想要缓冲区 0 回来的负载是 TMA Load k=2，不是 k=1（使用缓冲区 1）。这就是为什么 `mma2tma` 箭头从 MMA 计算 k=0 一直到达 TMA 负载 k=2。仅仅因为环有两个插槽，该版本就跃升了一个阶段。

### 扭曲角色

时间表显示了我们“为什么”分工；下一个问题是“谁”负责每个部分。专业化将三个作业（加载、计算、写回）分配给特定的 warps，以便它们可以同时运行。与`WG_NUMBER=2`相比，kernel 使用两个 warpgroups（角色表中缩写为 WG）：

| 演员 | 地点 | 职位 |
|-------|----------|-----|
| **TMA 制作人** | 变形组 1，warp 3 | 通过 TMA 连续加载 A 和 B 块 |
| **MMA 消费者** | 变形组 1，warp 0 | 数据准备好后立即运行 MMA |
| **回写** | 变形组 0（全部 warps） | 读取 TMEM 结果，写入 GMEM |

### 4 barrier

三个同时进行的演员需要四个 barrier，而这四个 barrier 整齐地排列成两个相反的方向。前向路径（TMA→MMA→写回）发出数据*准备就绪*的信号；它的信息是“您正在等待的 tile 在这里”。反向路径（回写 → MMA → TMA）向缓冲区*释放*发出信号：“您想要的插槽再次空闲。”一旦您知道了命名约定，名称就会自行读取：每个都是 `source2destination`，因此 `tma2mma` 只是 TMA 发出 MMA 信号的 barrier。

| barrier | 类型 | 方向 | 意义 |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | “SMEM 数据已准备好” |
| **MMA2TMA** | `TCGen05Bar` | MMA -> TMA | “SMEM 缓冲区可以重复使用” |
| **MMA2LD** | `TCGen05Bar` | MMA -> 回写 | “TMEM 结果已准备就绪” |
| **ld2mma** | `MBarrier` | 回写-> MMA | “TMEM 下一块免费” |

为什么每个 barrier 都有它的*类型*？该类型遵循生产者宣布其完成的方式。 **TMA 负载**使用 `TMABar`，这是一个具有字节计数功能的 mbarrier：一旦传输的字节到达，TMA 硬件本身就会到达 barrier，因此消费者知道数据已准备好，而无需任何 thread 进行轮询。 **TMA store**不能使用此功能（store 没有人需要通知），因此它们回退到 `cp_async.bulk.commit_group()` + `wait_group(0)`，其中发行的 thread 只是等待其自己的写入耗尽。 **MMA 操作** 使用 `TCGen05Bar`，其中 `tcgen05.commit()` 指令在 MMA 完成时发出 barrier 信号。

这里的一个小细节将在步骤 8 中得到回报。`arrive` 调用会传递 `cta_mask=0`，因为在单个 CTA kernel 中，没有其他 CTA 需要发信号。当步骤 8 形成 cluster 时，该参数将变为非零，并成为唤醒协作 CTAs 的机制。

### PipelineState

四个 barrier 告诉角色“何时”缓冲区准备就绪；当 pipeline 循环时，仍然需要跟踪每个角色所在的*哪个*缓冲区。该簿记是 `PipelineState` 管理的。环形缓冲区同时承载两部分簿记：我们当前位于哪个槽位，以及我们正在等待该槽位 barrier 的哪个“阶段”。在 pipeline 循环中手动跟踪两者正是会产生逐一错误的情况，并且这里的逐一错误会导致整个 kernel 陷入僵局。 `PipelineState` 的存在是为了将两者保持在一起，这样您就不必：

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # Producer starts ready (phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

首字母`phase`决定了角色的第一个`wait`是让其运行还是使其阻塞，而正确的答案在 pipeline 的两端是相反的，这就是让人绊倒的部分：
- `phase=1`（生产者）-> 第一个 `wait(phase=1)` 看到 barrier 仍处于阶段 0，并且由于 0 != 1 它**立即通过**。这正是我们想要的，因为缓冲区一开始是空的，生产者应该可以立即开始填充它们。

- `phase=0`（消费者）-> 第一个 `wait(phase=0)` 在阶段 0 看到 barrier，并且由于 0 == 0 它**阻塞**。这又是我们想要的，因为还没有数据，消费者在生产者到达之前没有什么可读取的。

给两端相同的起始阶段，你就会陷入僵局，或者更糟糕的是，无声的腐败，所以这个选择值得做出正确的选择。

### `warpgroup_sync` barrier ID

专业化引入了很容易陷入的同步危险。一旦每个 warpgroup 运行不同的代码路径，熟悉的 `cta_sync()` 将陷入死锁：它使用硬件 barrier #0 并坚持*每个* CTA thread 到达，但在 warpgroup 分支内只有其中一些 threads 存在。相反，我们需要的是一个范围仅限于单个 warpgroup 的 barrier。 GPU 为我们提供了 16 个命名 barrier（ID 0–15），因此 kernels 达到了 `warpgroup_sync(10)`，它仅同步一个 warpgroup 内的 threads。当多个 warpgroups 各自需要自行同步时（如多消费者步骤 9 中发生的情况），它们通过 `warpgroup_sync(wg_id + 10)` 获取不同的 ID，这样它们就不会在同一硬件 barrier 上发生冲突。

**实施。**

我们在这里使用 `PIPE_DEPTH=2`，这是仍然允许加载和计算重叠的最小深度。更深层次隐藏更多内存延迟，直至 SMEM 预算的限制；下面的“当第 7 步行为不当时”讨论详细说明了这种权衡。现在掌握了所有部分（角色、四个 barrier、`PipelineState` 和 warpgroup 范围同步），我们可以将完整的 kernel 组合在一起：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # all 128 Warpgroup 0 threads arrive
        pool.commit()

        # --- TMEM alloc + fence ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA Consumer ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # Wait for TMEM to be free from previous tile's writeback
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # Signal results ready for writeback
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # Wait for MMA results
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # Read TMEM -> registers (warpgroup scope)
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # Signal TMEM free (all 128 threads arrive)
                ld2mma.arrive(0, cta_id=0, pred=True)

                # Cast fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # Write to Dsmem + TMA store
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

要运行其中任何一个 kernels，请重复使用我们在步骤 1 中展示过的相同编译/运行/检查工具 ({ref}`chap_gemm_basics`)：将 `hgemm_v1` 替换为 `hgemm_v7`、`hgemm_v8` 或 `hgemm_v9`，然后选择问题大小，例如`M=N=K=4096`。请记住，cluster 步骤需要 `M` 和 `N` 是其 cluster tile 的倍数（步骤 8 为 `256×256`，步骤 9 为 `512×256`），因此微小的 `128×128` 大小根本不会生成 tile。每个新的 Python 会话编译一个步骤，在切换步骤之前重新启动 kernel，因为 kernels 重用内部名称并且编译器保留每个会话的状态。每个步骤的计时收集在下面的*端到端结果*中。

### epilogue（回写）详细信息

第 7 步可以提供一个令人愉快的简单 epilogue。仅使用 `BLK_N=128` 列，回写 warpgroup 将整个 TMEM 块一次性读取到寄存器中，然后发出一个 TMA store。第 8 步和第 9 步不会有这种奢侈，这正是它们引入我们稍后添加的分块的原因，但现在的顺序是：

1. 等待 MMA：`mma2ld.wait(phase)`。本教程中的步骤 8 和 9 在此添加 `fence.after_thread_sync()` 作为保守的额外内容； MMA-完成 mbarrier 已经涵盖了排序，并且大多数 kernels（包括 CUTLASS）省略了它，所以步骤 7 也是如此。
2. 读取 TMEM -> 寄存器（每个 thread、warpgroup scope 128 个 fp32，先通过 `Tx.copy_async(reg_wg, tmem[:, :BLK_N])`，然后是 `T.ptx.tcgen05.wait.ld()`）。
3. 信号 MMA：`ld2mma.arrive(0, cta_id=0, pred=True)`（128 个 threads 全部到达）； TMEM 现在可以免费使用下一个 tile。两个 `arrive` kwargs 在 cluster 步骤中重复出现：`cta_id` 命名*哪个 CTA 的*信号 barrier 副本（`0` = 这个 CTA，本地 barrier；在步骤 8 中，合作社通过以下方式到达目标 CTA-0）： `cta_mask` 代替），而 `pred` 是每个 thread 的谓词，用于控制此 thread 是否实际到达（此处为 `True`，因此每个写回 thread 都计入到达总数）。
4. 将 fp32 -> fp16 投射到寄存器中。
5. 写入寄存器 -> Dsmem，然后 `fence.proxy_async("shared::cta") + warpgroup_sync(10)` 进行刷新。
6. TMA store Dsmem -> GMEM 通过 `cp_async.bulk.commit_group() + wait_group(0)`。

步骤 8（使用`BLK_N=256`）和步骤 9（每个消费者使用`MMA_N=256`）无法保留此一次性表格，原因是注册压力。读取每个 thread 的 256 个 fp32 值意味着 256 × 4 = 1024 字节必须同时存在于每个 thread 的寄存器中，这有溢出到本地内存的风险，而且最重要的是，强制使用更大的 Dsmem 缓冲区。因此，这些步骤将写回分解为 `EPI_N` 列块 (`EPI_N=64`)：每次迭代仅保留 `EPI_N` fp32 寄存器处于活动状态，并发出相应较小的 TMA store，用更多的存储指令换取保持舒适的寄存器预算。

**实施说明。**

- **持久 kernel**：`bx = T.cta_id([SM_COUNT])` --- 每个 SM 一个 CTA，在 tile 上循环

- **L2 友好的调度**：`ClusterPersistentScheduler2D` 为缓存位置订购切片

- 这种模式——warp specialization 加软件 pipeline——在高性能 GEMM kernels 中很常见，包括 CUTLASS 风格的设计。

### 当第 7 步出现问题时

第 7 步是第一个 GEMM kernel，其中 TMA load、`tcgen05` MMA 和写回都同时进行。步骤 8 和 9 中会出现相同的故障模式：barrier 计数不匹配、角色防护在错误的位置、缺少 barrier 或在 TMA store 耗尽之前重复使用暂存缓冲区。这些情况的调试清单收集在 {ref}`chap_warp_spec_debug` 中。

**pipeline 深度调整。** Step 7 kernel 在最小值 `PIPE_DEPTH=2` 上运行。将其推至 4 或 6 可以让 TMA 生产者进一步领先于 MMA 消费者，并隐藏更多内存延迟，但它是通过花费更多的 SMEM 来实现的，而 SMEM 是有限的。 B200 每个 SM 提供 228 KB（请参阅 {ref}`chap_background` 中的*要记住的数字*）。对于 `BLK_M=BLK_N=128, BLK_K=64, fp16`，A 和 B 的每个 pipeline 阶段的成本为 `(128*64 + 128*64) * 2 = 32 KB`，并且 `Dsmem` 回写暂存缓冲区在顶部又增加了 32 KB。这使得 `PIPE_DEPTH=4` 约为 160 KB，`PIPE_DEPTH=6` 约为 224 KB，完全超出了预算。要更深入地了解这一点，您必须重新考虑写回暂存策略。

---

经纱专业化得到了一台 CTA 合作的 threads。下一步将扩大这种合作，跨越 CTA 本身的边界，让其中的两个在一个更大的 tile 上工作。


(chap_cta_cluster)=
## 步骤 8：2-CTAcluster

第 7 步使引擎重叠，但每个 CTA 仍然独立计算自己的 128×128 tile，重新加载邻居无法借用的操作数。第 8 步打破了这种隔离。两个 CTAs 加入到一个 cluster 中，并获得访问彼此共享内存的能力，因此单个协作 `tcgen05` MMA 会生成一个跨越两个 CTAs 的 256×256 tile，并且 B 的一个负载现在可以提供两倍于 MMA 的工作量。如前所述，M=N=K=4096。

> **此步骤更改的内容：范围 + layout + 调度**
> - 范围：协作的 scope 现在跨越一个 cluster 中的两个 CTAs，而不是一个。
> - layout：操作数块分为两个 CTAs' SMEM； CTA 0 拥有共享完成 barrier (`remote_view`)。
> - 调度：MMA 获得 `cta_group` / `cta_mask`，因此 `tcgen05` 作为 2-CTA 协作操作运行。

**主题。**

- CTA clusters：多个 CTAs 在更大的 tile 上协作

- 通过 `map_shared_rank` 跨 CTA SMEM 访问

- `cta_group=2` 用于在 256x256 cluster tile 上进行协作 MMA

- 与 `cta_mask` 跨 CTA barrier 信号


### 簇状 tile shape

整个优化依赖于单一硬件功能：通过 `cta_group=2`，MMA 可以读取由“两个”CTAs 上演的操作数块，而不仅仅是其所在的操作数块。每个 CTA 加载一个存储 B 的 128 行切片，转置后变成 128 个逻辑输出列，并且协作 MMA 将两个切片重新缝合在一起成为一个操作数。下图描绘了两个 CTAs' A 和 B 切片如何组合成单个 256×256 cluster 切片：

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：每个 CTA 拥有一个 A 行片和一个存储 B 行片，然后通过 cluster (DSMEM) 读取另一个 CTA 的存储 B 片。在 `B.T` 之后，两个存储的 B 切片覆盖整个输出列跨度，因此这对生成一个 256×256 输出 tile。*

**为什么 A 和 B 在 cluster 上分割**：要了解 256×256 tile 如何分区，请回想一下教程将 GEMM 存储为 `D = A @ B.T`，其中存储的 B 具有形状 `N x K`。当 cluster 中包含两个 CTAs 时，裂缝会干净地脱落：

- **A 垂直分割**：CTA-0 保存 A0（第 0-127 行），CTA-1 保存 A1（第 128-255 行）。堆叠：`[A0; A1]`（256 行）。
- **存储的 B 按行分割**：CTA-0 加载 B 行 0-127，CTA-1 加载 B 行 128-255。因为数学使用 `B.T`，所以这两个存储的行切片成为逻辑右侧操作数的两个 128 列切片。
- 对于 `cta_group=2`，MMA 硬件通过跨 CTA 共享内存访问从 **两个** CTAs' SMEM 读取 B，因此它看到完整的逻辑输出列范围。
- 结果：两个 CTAs 在一个 256x256 输出块上协作。每个 CTA 写入该 tile 的 128x256 行条带。

值得停下来看看为什么这是一次真正的胜利，而不仅仅是工作的重新洗牌。每个 CTA 仍然只加载 A 的 128×K 和 B 的 128×K，因此 cluster 作为一个整体阶段大约是单个 CTA 操作数的 2 倍，但它会生成一个 256×256 的 tile，其中包含大约 4 倍 128×128 块的输出 FLOP。因此，MMA 在每个分段操作数字节上执行的工作大约是两倍，因为每个 CTA 的 B 片都通过协作 MMA 重用于其他 CTA 的 A 片。换句话说，算术强度大约翻倍，而这正是仍然依赖内存的 kernel 所需的杠杆：端到端表中约 2.2 倍的加速来自于将相同的字节提供给更多的数学运算。

### 平铺地址计算

既然 cluster 是工作单元，那么 tile scheduler 也必须计入 cluster 块中。它返回的每个 `(m_idx, n_idx)` 命名了一个完整的 256×256 区域，而 cluster 内的两个 CTAs 则将该区域分割开来。将 cluster 坐标转换为每个实际加载的每个 CTA 切片，如下所示：

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTAs 都在*相同* 256×256 cluster tile 上工作，并且单坐标 `cbx`（CTA 在 cluster 中的位置，0 或 1）负责挑出这个 CTA 沿两个轴的贡献。`m_st` 选择此 CTA 拥有的输出行条带，`n_st` 选择它馈入协作 MMA 的 stored-B 切片，并且稍后 writeback 会发出 256 列输出范围的两个 128 列一半。另请注意，`num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 计数的是 cluster tiles，而不是单个 CTA tiles。

乍一看，`cbx` 出现在 `m_st` 和 `n_st` 中，就好像行偏移以某种方式泄漏到列中一样，但两种用法都是正确的，并且值得弄清楚原因。在写回路径上，`cbx` 仅属于 M 轴：每个 CTA 拥有不同的 128 行条带（`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`，因此 CTA-0 写入 `m_idx*256 .. +128` 行，CTA-1 写入接下来的 128 行），然而，两个 CTAs 都写入了 cluster tile 的“完整”256 个输出列。这正是 store 从 cluster 的 `n_idx`（`n_st_epi = n_idx * 256 + no * 128`，看不到 `cbx`）而不是从每个 CTA `n_st` 派生其列的原因。 `n_st` 携带 `cbx` 的原因是每个 CTA 将不同的存储 B 行切片加载到 MMA 中：在那里，`cbx` 是*加载*偏移量，而不是 CTA 的输出列偏移量。

### 第 7 步的代码更改

与第 7 步的差异有六处编辑，每处都编码我们刚刚描述的 cluster 合约的一个片段：

```python
# 1. Cluster launch
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA index within cluster (0 or 1)

# 2. Cooperative MMA (was cta_group=1)
Tx.gemm_async(..., cta_group=2)

# 3. Cross-CTA shared memory access
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. Cross-CTA barrier
tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. mma2tma / mma2ld arrives go from cta_mask=0 (single CTA, Step 7)
#    to cta_mask=3 (signal both CTAs in the cluster)
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. Cluster sync replaces cta_sync at the end
T.cuda.cluster_sync()
```


### cluster 范围的变更

这六个编辑都源于同一个转变：协作的 scope 现在是 cluster，而不是单个 CTA。以下几点说明了这种扩大在实践中意味着什么：每个 CTA 如何找到自己的位置，cluster 协调的 barrier，以及 CTA 实际发行合作 MMA。

- **cluster CTA ID**：`cbx` 告诉每个 CTA 它在 cluster 中的位置（0 或 1）。 CTA-0 处理 A 行 0-127，CTA-1 处理 A 行 128-255。

- **远程 barrier 视图**：在 cluster 中，每个 CTA 都有自己的 SMEM 和自己的 barrier，这提出了一个明显的问题：如果 CTA-1 需要等待 CTA-0 生成的内容，那么它实际上会触及谁的 barrier？答案是指定 CTA-0 的 barrier 作为单个协调点，并让 cluster 中的任何 CTA 到达它们。 `map_shared_rank(tma2mma.ptr_to([0]), 0)` 使用 TIRx 包装器 `tma2mma.remote_view(0)` 返回一个指向 CTA-0 barrier 的 cluster 范围的指针，从那时起，每个到达和等待目标 CTA-0 的副本。

- **仅来自 CTA-0 的 MMA dispatch**：很容易将 `cta_group=2` 理解为并行启动两个引擎，但事实并非如此。 CTA-0 恰好发出一个 `tcgen05.mma`，然后硬件驱动一个跨 CTAs 的*单个协作* MMA，从两个 SMs' SMEM 读取操作数并在两个 SMs 上写入累加器 SMs'TMEM。 CTA-1 根本不发出 MMA。 （每个 SM 只有一个`tcgen05`引擎，所以`cta_group=2`是一个跨 SM MMA，而不是两个并排运行的引擎。）这就是代码用`if cbx == 0:`保护 MMA 的原因。

- **组播到达**：`tcgen05.commit(..., cta_group=2, cta_mask=3)` 仅由 CTA-0 发出，但向 CTAs 的 barrier 发出信号。 `cta_mask=3`（二进制`11`）表示 CTA-0 和 CTA-1 都是目标。

- **ld2mma 初始化计数**：`init(128 * CTA_GROUP)` --- CTAs 的写回 warpgroups（各 128 个 threads）均到达。


**实施。**

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- Cross-CTA barrier view ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 chunks of 128 columns = 256 total
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**2 CTAs 有何变化。**

- `CTA_GROUP = 2`、`MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- CTAs 的写回工作组均已到达

- TMA 到达字节计数包括 CTAs: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须使用 `cta_group=2`

- 回写将 256 个输出列分成两个 128 列块 --- 一次读取所有 256 个 TMEM 列超出了寄存器容量。步骤 9 将块进一步缩小为 `EPI_N=64`

- `cluster_sync()` 最后替换 `cta_sync()`（确保所有 CTAs 在 TMEM 释放之前完成）

所有额外的算术强度都直接显示在挂钟上：第 8 步在 4096³ 时达到 **0. 104 ms**，大约是相同大小的 70 ms Step-1 算法的 676 倍（请参阅端到端表）。 kernel 现在倾向于计算密集型，这正是设置步骤 9 的原因，我们在其中添加第二个 MMA 消费者，以保持更多的 Tensor Core 工作在进行中。

如果第 8 步的结果比第 7 步“慢”，那么罪魁祸首几乎总是新的 cluster 合约之一，输入略有错误。首先值得检查三件事：TMA 到达字节计数是 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`；对于 256×256 cluster tile，scheduler 尺寸为 `num_m_tiles=M//256, num_n_tiles=N//256`；该写回会发出两个 TMA 存储，每个 128 列块一个，每个存储在 Dsmem 重用之前耗尽。

---

cluster 提高了*跨*CTAs 的重用。最后一步转向内部，通过为生产者提供第二个 MMA 消费者来保持供给，从而提高每个 CTA 内的计算密度。


(chap_multi_consumer)=
## 第 9 步：多消费者 Warp 专业化

到第 8 步，MMA 确实很忙，但是单个消费者 warp 只能如此快速地咀嚼暂存的 B tile，并且该 B tile 始终位于 SMEM 中，可供任何愿意阅读它的人使用。最终的优化利用了这一点：它添加了第二个 MMA 消费者，将*不同的* A 块与*相同的* B 块相乘。每个 CTA 的计算密度翻倍，cluster 输出从 256×256 增长到 512×256。如前所述，M=N=K=4096。

> **此步骤更改的内容：范围 + layout**
> - 范围：一个 MMA 消费者变成两个，由`warp_id`选择。
> - layout：一个阶段 B tile 由两个消费者重复使用； A 获得消费者轴。
> - 调度：不变。

**主题。**

- 多个 MMA warps（消费者）以实现更高的吞吐量

- 具有独立 barrier 槽的多个写回 warpgroups

- 本教程中最优化的 GEMM 变体使用的结构


### 多消费者结构

添加第二个消费者意味着 kernel 现在需要布置更多不同的角色：两个 MMA warps 而不是一个，以及匹配的第二个写回 warpgroup 以耗尽额外的累加器。通过 `NUM_CONSUMER=2` 和 `WG_NUMBER=3`，kernel 现在跨越三个 warpgroups（角色表中的缩写 WG）：

| 扭曲群 | 扭曲 | 角色 |
|-----------|------|------|
| **工作组 2** | warp 0 | MMA 消费者 0：`Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| **工作组 2** | warp 1 | MMA 消费者 1：`Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| **工作组 2** | warp 3 | TMA 生产者：每阶段加载 2x A 块 + 1x B 块 |
| **工作组 0** | 全部 | 消费者 0 写回：读取 TMEM `[0:256]` |
| **工作组 1** | 全部 | 消费者 1 的写回：读取 TMEM `[256:512]` |

整个安排取决于一种不对称性。每个消费者将自己的 A 块与“相同”的分阶段 B 块相乘，因此单个 B 负载现在可以提供 2 倍的 MMA 工作，并且 B 的每个有用 FLOP 的负载成本实际上减半。我们共享 B 而不是 A 的原因是两个消费者覆盖不同的 M 行条带：它们的 A 块是真正不同的数据，而 B 块对于两者来说是相同的。练习 3 要求您说服自己这是唯一有效的分享。

### 第 8 步的更改

具体来说，支持第二个消费者会在少数几个地方触及 kernel，并且每一项更改都可以追溯到一个事实：现在每个阶段有两个 A 块和两个 TMEM 系列进行馈送和排空，而 B 保持共享。下面的编辑设置了一个额外的 A 块，为每个消费者提供了自己的 barrier 槽，并调整了更高的 512×256 cluster tile 的 tile 寻址。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 每级 2 个 A 块，每个消费者一个

- TMA 加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`，TMA 现在到达字节 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`（额外的 A 块）

- MMA warp `warp_id` 选择哪个 A 块和 TMEM 范围

- `mma2tma.init(NUM_CONSUMER)` --- 每个阶段两个消费者都发出 TMA 信号

- `mma2ld`和`ld2mma`有`depth=NUM_CONSUMER`——每个消费者使用自己的 barrier 槽（`warp_id`用于 MMA 侧，`wg_id`用于写回侧）

- tile 地址：`m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M 方向具有额外的 `NUM_CONSUMER` 因子，因为每个 cluster tile 现在跨越 M 中的 `NUM_CONSUMER` 消费者。 tilescheduler 使用 `num_m_tiles = M // 256 // NUM_CONSUMER`（cluster tile 为 512x256）

- 回写使用分块的 `EPI_N`，因此每次迭代在寄存器中保留更少的 TMEM 回读值


**实施。**

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, one slot per consumer
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, one slot per consumer
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # each stage expects 2 arrivals
        mma2ld.init(1)              # each slot gets 1 arrival
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (512x256 cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA Consumer: warp_id selects A block and TMEM range ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**实施说明。**

- 在此第 9 步设计中，`mma2ld` 和 `ld2mma` 都是与 `depth=NUM_CONSUMER` 共享的单个对象，而不是单独的每个消费者对象。插槽 0 将 MMA warp 0 连接到 Warpgroup 0，插槽 1 将 MMA warp 1 连接到 Warpgroup 1； MMA 侧索引为 `warp_id`，写回侧为 `wg_id`。

## 端到端结果

下表报告了从初始基线到 warp-specialized cluster kernel 的测量里程碑，以及 cuBLAS 参考。 NVIDIA B200、M=N=K=4096、fp16 上的参考编号、锁定时钟、1000 次迭代定时基准：

| 步骤 | 技术 | 时间 | 加速比 |
|------|-----------|------|---------|
| 1 | 同步负载+MMA | 70 毫秒 | 1× |
| 2 | K-循环累加 | --- | 手柄 K 大于一块 tile |
| 3 | 空间平铺 | 53.6 毫秒 | ~1.3× |
| 4 | TMA 异步加载 | 0.49 毫秒 | ~142× |
| 5 | 软件 pipeline | --- | 重叠负载+计算 |
| 6 | 持久 kernel | --- | L2 缓存局部性 |
| 7 | 经纱专业化 | 0.23 毫秒 | ~309× |
| 8 | 2-CTA cluster | 0.104 毫秒 | ~676× |
| 9 | 多消费者 | 0.094 毫秒 | ~744× |
| --- | cuBLAS（参考） | 0.094 毫秒 | ~744× |

此表中的每次，包括 70 ms 步骤 1 基线，都是在相同的 M=N=K=4096 大小下测量的，这使得加速链具有端到端的可比性。有必要准确说明 70 毫秒的实际含义，因为它很容易被误读。它*不是*来自 {ref}`chap_gemm_basics` 的单块 Step-1 kernel，在 4096³ 运行； kernel 仅计算一个 128×128 tile，并且仅以小尺寸运行。相反，70 毫秒是一个简单的全尺寸基线，它采用相同的顺序、单块方法并将其扩展到完整的 4096³ 问题。 {ref}`chap_gemm_basics` 在小尺寸（128×128 和 256³）中介绍了步骤 1-3，以保持最初的演练简单；这里的步骤 1 和步骤 3 行是它们的全尺寸基准对应行。其余的破折号（步骤 2、5、6）标记了结构所示的步骤，但没有单独计时。

将这些数字作为在受控条件下运行的单个 B200 参考运行来读取，而不是作为排行榜条目。每个步骤中嵌入的 `{.python .input}` 基准单元都是烟雾基准：它们有利于发现趋势，而不是声称峰值性能。

四种技术几乎占据了所有收益：

1. **TMA 异步数据移动**：硬件复制引擎取代了软件复制（从步骤 1 → 步骤 4 约为 142 倍）。正确读取这个 142× 非常重要：它反映了从单个 128×128 平铺 kernel (grid 1×1) 一直到具有 K 循环、空间平铺等的完整平铺和并行 kernel CTAs，*连同* TMA；这并不是 TMA 的孤立贡献。隔离 TMA 意味着比较两个仅在复制机制上不同的全尺寸 kernels。
2. **软件 pipeline + Warp 专业化**：通过赋予每个角色自己的专用角色来重叠加载和计算（步骤 4 → 步骤 7 的约 2.2 倍）。
3. **CTA cluster**：2-SM 协作 MMA 改进了 CTAs 中的 B-tile 重用（本基准测试中步骤 7 → 步骤 8 的约 2.2 倍）。
4. **多消费者**：两个 MMA warps 可实现更高的计算密度（步骤 8 → 步骤 9 的约 10%）。

在测量的里程碑上绘制的，这四个相同的贡献追踪从同步平铺 kernel 到 cuBLAS 参考的下降。下图显示了所选的测量点：

![GEMM 优化之旅](../img/gemm_perf.png)

请注意，随着我们沿着清单往下走，收益会缩小，这是结构性原因，而不是努力减弱。早期的步骤是针对“内存”瓶颈（TMA 替换软件副本，clusters 提高算术强度），而这正是 70 毫秒的大部分实际花费的地方，因此这些步骤的回报最大。到第 8 步，kernel 已经在 cuBLAS 的约 10% 范围内（0.104 vs 0.094 ms），并且接近*计算限制*，这意味着几乎没有内存停顿可以隐藏；第 9 步的多消费者重叠恢复了大部分剩余的内容。大约 10% 的最终增益正是接近计算上限时所期望的：它是接近解决的问题的收益递减，而不是弱优化的标志。

我们在本章中构建的所有内容（TMA 加载、`tcgen05` MMA、TMEM 读回和 warp-specialized barrier）将直接延续到下一章。 FlashAttention 重用所有这些，然后通过在两个 MMA 阶段之间插入在线 softmax 步骤而不是简单地重复单个阶段来提高难度。


## 练习

1. 如果在步骤 7 中将 TMA 和 MMA `PipelineState` 的初始 `phase` 设置为 `0` 会发生什么？画出死锁场景。
2. 对于步骤 8 中的 `cta_group=2`，TMA 到达字节计数为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`。当每个 CTA 加载自己的数据时，为什么要乘以`CTA_GROUP`？
3. 在步骤 9 中，每个消费者处理不同的 M 行但相同的 B tile。为什么共享 B（而不是 A）是正确的选择？

**与您的代理一起尝试**：粘贴步骤 7 kernel 并要求其通过四个 barrier（`tma2mma`、`mma2tma`、`mma2ld`、`ld2mma`）追踪一个 K-tile。对于每一个，询问谁在等待，谁到达，哪个 tile 可以安全读取，以及哪个缓冲区随后可以重用。
