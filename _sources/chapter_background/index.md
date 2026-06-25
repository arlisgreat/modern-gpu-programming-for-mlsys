(chap_background)=
# GPU 执行模型

:::{admonition} 概述
:class: overview

- A kernel runs over a thread hierarchy（thread → warp → warpgroup → CTA → cluster → grid），并跨 distinct memory spaces（registers、SMEM、GMEM、TMEM）访问数据。
- 计算分为 CUDA cores 和 Tensor Cores；像 TMA 这样的专用引擎会移动为其提供数据的数据。
- kernel 是一个 pipeline，可通过这些内存空间暂存数据，并在独立计算和数据移动引擎之间进行手工操作；反复出现的目标是让这些引擎立即保持忙碌。
:::

要编写快速的 GPU 程序，了解硬件本身以及代码如何在该硬件上运行非常重要。本章概述了 GPU 执行模型：执行工作的 thread 层次结构、保存和移动数据的内存空间以及执行繁重工作的计算和数据移动引擎。我们首先逐一介绍这些部分，然后将它们放在 GEMM pipeline 中，以便清楚数据和执行如何流经硬件。本书后面的几乎所有优化都是在这些相同的部分之间安排工作的某种方式。

现代 GPUs 还包含许多专用硬件单元。为了让您初步了解，在我们放大每个部分之前，下面把 Blackwell 原图以及 Ampere、Hopper 的 SM Architecture 放在同一组交互式图里。横向滚动可以在三代之间切换，点击每个硬件单元查看说明。

```{raw} html
<div style="overflow-x:auto; padding-bottom:8px;">
<div style="display:flex; gap:16px; width:max-content;">
<figure style="margin:0; width:1320px; flex:0 0 1320px;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
<figcaption style="font-size:0.92em; color:var(--pst-color-text-muted, #6c757d); margin-top:6px;">
交互式：Blackwell SM Architecture，展示 SMEM、TMEM、Tensor Core 和 TMA Engine 的 data path。
</figcaption>
</figure>
<figure style="margin:0; width:1320px; flex:0 0 1320px;">
<iframe src="../demo/ampere_sm_architecture.html" title="Ampere SM architecture" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
<figcaption style="font-size:0.92em; color:var(--pst-color-text-muted, #6c757d); margin-top:6px;">
交互式：Ampere SM Architecture，展示 <code>cp.async</code>/LD/ST staging、<code>ldmatrix</code>、<code>mma.sync</code> 和 Register File accumulators。
</figcaption>
</figure>
<figure style="margin:0; width:1320px; flex:0 0 1320px;">
<iframe src="../demo/hopper_sm_architecture.html" title="Hopper SM architecture" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
<figcaption style="font-size:0.92em; color:var(--pst-color-text-muted, #6c757d); margin-top:6px;">
交互式：Hopper SM Architecture，展示 TMA load 到 SMEM、<code>wgmma</code> 通过 SMEM descriptor 读取 operands，以及 Register File accumulators。
</figcaption>
</figure>
</div>
</div>
```

## 执行层次结构

我们从执行这项工作的 threads 开始。 GPU 不会将其数千个 threads 作为一个扁平池呈现。相反，它将它们分组到一个嵌套的层次结构中，这样做是因为合作同时发生在几个不同的规模上。每个级别的存在都是为了使某一规模的合作变得廉价。下图为 Blackwell 上的层次结构；您可以单击每个级别以突出显示它。

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*互动：点击关卡：thread → warp → warpgroup → CTA → cluster → grid。*

- **thread**：执行的标量单元。每个 thread 都有自己的程序计数器和自己的寄存器，并由其 warp 内的 lane ID 进行标识。
- **Warp**：在 SIMT 中执行的 32 个 threads（*单指令，多个 threads*）。 warp 的 lane 一起发出相同的指令，但每个 lane 都保留自己的寄存器，并且可以自行屏蔽，这使得单个 warp 的 lane 遵循不同的分支。
- **warpgroup**：四个连续的 warps，或 128 个 threads。 Hopper 引入了 warpgroup 作为发行 warpgroup 级 MMA（`wgmma`）的单元，在 Blackwell 上它承担了第二个角色：它是 Tensor Memory 接入的配合单元，其中 128 个 threads 一起将 TMEM 块移入或移出寄存器。
- **CTA**（*Cooperative Thread Array*，CUDA 也称为 thread block）：硬件调度的基本单元。 CTA 在单个 SM 上运行，并在其中拥有私有共享内存分配。多个 CTAs 可以同时驻留在同一个 SM 上，并且当它们驻留在同一 SM 上时，它们会在它们之间分配 SM 的共享内存容量。
- **cluster**：一组合作的 CTAs，可能生活在不同的 SMs 上。 cluster 中的 CTAs 可以相互同步，并且可以读写彼此的共享内存，这种功能称为分布式共享内存。

这些级别值得详细讨论，因为与早期架构不同，Blackwell 的关键操作**并非全部由同一组 threads** 发出。 TMA 副本由单个 thread 发起，然后由硬件执行。 TMEM 到寄存器的加载是 warpgroup 分布式的：四个 warps 协作，每个移动自己的 TMEM 切片。 `tcgen05` MMA 由一个当选的 thread 提交，而 cluster MMA 一次跨越两个 CTAs。因此，每个操作都有自己的自然粒度，运行它的 threads 集合就是我们所说的操作的 **scope**，这是本书反复讨论的三个重复设计元素（scope、layout 和 dispatch）中的第一个。

## 内存空间

该层次结构中的 threads 的速度与数据到达它们的速度一样快，因此我们接下来转向数据所在的位置。没有哪一个存储器既大又快。物理学迫使容量和速度之间进行权衡。因此，GPU 提供多个存储器，而不是一个，每个存储器在不同的点进行权衡，而 kernel 的工作原理是通过它们移动数据。每个空间都有自己的容量、自己的延迟以及自己的访问规则。

| 记忆 | 所有权 | 角色 | 笔记 |
|--------|-----------|------|-------|
| **全球 (GMEM)** | 设备范围 | 持久张量存储 | 大 HBM，所有人共享 SMs |
| **共享 (SMEM)** | 每 CTA（一个 SM） | tile staging | 低延迟暂存器； B200 上高达 228 KB/SM |
| **Tensor Memory (TMEM)** | 每 CTA | MMA 累加器存储 | Blackwell 上的新功能；由 `tcgen05` 使用 |
| **Register File (RF)** | 每 thread | 标量和每 thread 切片片段 | 快速地;保存 epilogue/临时值 |

按顺序阅读，这些空格描述了一条路径。本书中几乎每个 kernel 的数据路径都是**GMEM→SMEM→（计算）→寄存器→SMEM→GMEM**，对于 Tensor Corekernels TMEM 位于该路径的中间，在数学运行时持有累加器。

在这四款产品中，**Tensor Memory (TMEM)** 是唯一一款在 Blackwell 之前的硬件上没有模拟功能的产品，其完整详细信息要等到 {ref}`chap_tensor_cores` 为止。不过，现在值得理解其动机。早期的 GPUs 将大型 MMA 累加器保留在寄存器中，它们在寄存器中争夺稀缺资源。 Blackwell 相反，将 `tcgen05` 累加器输出写入 TMEM，这是一个 CTA 范围的 2D 暂存器，有 128 个 lane，每个 CTA 最多 512 个 32 位列（该阵列物理上位于 SM）。然后，kernel 必须在 epilogue 之前将 TMEM 显式读回到寄存器中。这个额外的步骤并不是免费的，它的两个后果将在整本书中重复出现。首先，TMEM 读取是**显式且 warpgroup 分布式**，由 warpgroup 的四个 warps 协作执行。第二个是 TMEM 与寄存器不同，必须**显式分配和释放**。

### Distributed Shared Memory 跨 cluster

cluster 是层次结构中的一个级别，其成员可以跨越多个 SMs，并且该范围购买了其他级别所缺乏的内存功能。一个 CTA 在一个 SM 上运行，并在 SM 的共享内存中工作，但单个 CTA 的 SMEM 预算是有限的，大块通常需要更多的操作数存储或更多的重用，而单个块无法提供。 Hopper 的答案是**thread blockcluster**：一组比独立块合作更紧密的 CTAs，因为它们可以一起同步并读写彼此的共享内存，这种能力称为**分布式共享内存（DSMEM）**。 Blackwell 保留 clusters 并向其添加，具有动态调度（{ref}`chap_clc`）和 2-CTA 协作 MMA。

DSMEM 让 CTA 直接寻址并访问对等 CTA 的共享内存。 thread 可以命名对等方的 SMEM 中的位置，并将 tile 直接从其自己的 SMEM 批量复制到对等方的中，一旦字节到达，就会提高完成 barrier（{ref}`chap_async_barriers`）。 Part III 中的 2-CTA cluster GEMM 正是基于此机制构建的，使用它在 CTAs 对之间共享操作数块，而无需通过全局内存将它们路由回。

下图显示了 CTA cluster 可能实现的额外 DSMEM 跳；单击一块可以查看每个 CTA 拥有什么以及跨 CTA 读取发生在哪里。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：2-CTA cluster，其中每个 CTA 拥有一半的 A 和一半的 B，通过 cluster (DSMEM) 读取另一个的 B，并且该对生成 256×256 的输出 tile。*

## 计算：CUDA Cores 和 Tensor Cores

threads 及其移动的数据必须在算术单元上满足，并且 SM 提供两种不同类型的数学引擎，而不是一种。两者之间的分工决定了几乎每个 kernel 的编写方式，并且它们发挥着互补的作用。

- **CUDA cores** 是通用 SIMT ALU。它们运行处理索引算术、元素数学、归约和控制流的标量和向量指令，以及围绕繁重矩阵工作的粘合逻辑。
- **Tensor Cores** 是固定功能单元，以 *tile* 粒度执行密集矩阵乘法累加，在单个指令中计算 $D = AB + C$。

这种划分很重要的原因是，Tensor Cores 提供的算术吞吐量远高于 CUDA cores，大约为 FLOP/s 的 10 倍或更多，因此密集线性代数（GEMM、卷积和 Attention）只有在 Tensor Cores 上运行时才能达到峰值性能。因此，获得性能很大程度上取决于让这些 Tensor Cores 得到满足。从一代 GPU 到下一代的变化是 Tensor Cores 的“如何”编程以及其结果“在哪里”得以保存。 Hopper 推出异步 warpgroup MMA（`wgmma.mma_async`）； Blackwell 的第五代 Tensor Core，`tcgen05`，将其累加器放在 Tensor Memory 而不是寄存器中，我们将`chap_tensor_cores`奉献给它。

cluster 以两种方式扩展这些引擎，这两种方式在 GEMM 章节中反复出现。 **2-CTA 协作 MMA** 允许两个 CTAs 各自将其 SMEM 操作数贡献给单个更大的 Tensor Core MMA 区块。 **TMA 多播**允许数据移动引擎的一个负载将相同的 GMEM 块同时传送到多个 CTAs，从而消除了单独负载可能产生的冗余全局流量。两者都建立在前面介绍的分布式共享内存的基础上。

## GEMM 数据 pipeline

到目前为止，我们已经分别介绍了硬件单元。为了了解它们如何协同工作，我们可以使用典型的通用矩阵乘法（GEMM）pipeline 作为示例。下面的交互式演示显示了三级 GEMM 切片 pipeline 中涉及的单元；单击 `tma load` 等操作可突出显示其跨硬件单元所采用的数据路径。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：加载→MMA→Blackwell 上的 epiloguepipeline；单击一个操作以跟踪其跨硬件单元的数据路径。*

单个 GEMM 块流经三个阶段。

1. **加载。** TMA 副本 ({ref}`chap_tma`) 将 A 或 B 操作数块从 GMEM 流式传输到 SMEM。一份 thread 发出副本，预先记录预计到达的字节数。当字节到达时，TMA 引擎会报告其进度，并且只有在所有预期字节都已交付后，完成 barrier 才会翻转。
2. **计算。** `tcgen05` MMA ({ref}`chap_tensor_cores`) 从 SMEM 中读取操作数块，并将乘积累加到 TMEM 块中。一个当选的 thread 发出它，当数学完成时它发出一个 barrier 信号。
3. **epilogue。** warpgroup 将 TMEM 累加器读回到寄存器中，将结果转换为输出数据类型，并将其存储到 GMEM，通常通过暂存 SMEM 并发出 TMA store 来实现。

以这种方式编写的三个阶段看起来严格顺序，但慢速 kernel 和快速 kernel 之间的全部区别在于**重叠**。天真的 kernel 确实按顺序运行步骤（加载、等待、计算、等待、存储），因此每个引擎在等待前一个引擎时都处于空闲状态。相反，快速的 kernel 对它们进行 pipeline 处理：当 Tensor Core 在 tile `k` 上计算时，TMA 引擎已经在获取 tile `k+1`，而 epilogue 正忙于耗尽 tile `k-1`，因此所有三个引擎同时保持占用状态。让三个异步引擎安全地相互传递工作正是 barrier 和阶段模型 ({ref}`chap_async_barriers`) 的工作，Part III 的 GEMM 梯子构建在其之上。

## 接下来读什么

现在我们已经看到了总体情况，我们可以继续深入探讨主要机制的章节：

- {ref}`chap_tensor_cores` 详细解释了 `tcgen05` 计算和 Tensor Memory。
- {ref}`chap_tma` 涵盖基于 TMA 的异步数据移动。
- {ref}`chap_async_barriers` 介绍了协调这些引擎的 mbarrier 和 phase 模型。
