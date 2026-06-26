(chap_async_barriers)=
# 异步协调：mbarriers

:::{admonition} 概述
:class: overview

- TMA 和 Tensor Core 是异步的，因此发出工作与完成工作不同，消费者需要一个显式的完成信号。
- mbarrier 就是这样的信号：生产者到达，消费者等待，并且它跟踪到达计数和（对于 TMA）字节计数。
- 每个 barrier 都有一个每轮都会翻转的“阶段”；等待正确的阶段才能安全地保护消费者。
:::

TMA ({ref}`chap_tma`) 和 Tensor Core ({ref}`chap_tensor_cores`) 操作是异步的。当 kernel 发出 TMA load 或 `tcgen05` MMA 时，发出 thread 不会等待操作完成。指令仅提交给硬件引擎；实际的数据移动或矩阵运算与程序的其余部分并行继续。

这很有用，因为它允许内存移动和计算重叠。这也意味着程序顺序不足以证明数据已准备好。后面的指令可能会在前面的异步操作完成之前运行。如果当 MMA 开始读取共享内存块时，TMA 仍在写入共享内存块，则 MMA 读取不完整的数据。如果 epilogue 在 Tensor Core 完成写入累加器之前读取 TMEM，则会读取错误值。如果 kernel 等待错误的条件，它可能永远不会取得进展。

因此，kernel 在每次异步切换时都需要一个明确的完成信号。 `mbarrier` 就是该信号。生产者在其工作完成后到达 barrier，而消费者在使用生成的数据之前在 barrier 上等待。相同的机制用于 TMA 到 MMA 切换、MMA 到 epilogue 切换以及跨 pipeline 阶段的缓冲区重用。

barrier 不仅仅是一面一次性旗帜。它携带一个 phase 位，并且每次 barrier 完成一轮到达时该 phase 位都会改变。阶段允许在许多循环迭代中重用一个 barrier，而不会混淆一次迭代的完成与另一次迭代的完成。

## mbarrier

`mbarrier`是内存 barrier 的缩写，是存储在共享内存中的硬件同步对象。从概念上讲，它包含两个状态：到达计数器和 phase 位。计数器告诉 barrier 本轮仍有多少到达者失踪。 phase 位告诉 kernel barrier 当前处于哪一轮。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：`mbarrier` 状态视图，显示到达计数器、phase 位以及 `init`、`arrive` 和 `wait` 操作；单击一个字段以将其聚焦。*

barrier 从初始化开始。在 `init` 期间，kernel 设置此 barrier 预计到达的人数。 barrier 从阶段 0 开始，其计数器加载到预期到达计数。从那时起，barrier 正在等待资源的所有必需生产者或用户报告他们已完成。

到达减少了 barrier 仍在等待的工作量。 kernel 的不同部分可以以不同的方式到达 barrier，并且区别很重要。

对于 TMA load，通常的到达路径是 tx-count arrive。像 `mbarrier.arrive.expect_tx(bytes)` 这样的操作会做两件事。首先，它算作 issuing thread 到达 barrier。其次，它记录 TMA engine 预计传输的字节数。仅仅因为 issuing thread 已经到达，barrier 并不完整。它还等待 TMA engine 在传输完成时 drain 字节计数。仅在满足两个条件后才会发生 phase 翻转：正常到达计数已达到零，并且待处理的 tx 字节计数已达到零。

这就是为什么 `expect_tx` 不应被解读为“又一个普通 arrive”。它为 async copy 设置字节预算。硬件稍后会通过 complete-tx 更新来计入实际的 copy 完成情况。仅当 arrive 和字节传输都完成时，barrier 才完成。

对于 Tensor Core 工作，到达路径不同。 `tcgen05` MMA 不会仅仅因为发出了 MMA 就自动推进 barrier。 kernel 必须显式地将 barrier 到达附加到提交路径，例如使用 `tcgen05.commit.mbarrier::arrive` 操作。当该提交组完成时，Tensor Core 侧执行 barrier 到达。如果 kernel 忘记提交到达，则在 barrier 上等待的消费者将永远等待。

普通的 thread 也可以直接到达 barrier。当普通 thread 代码是生产者时，或者当一组 threads 宣布它已完成资源的使用时，使用此方法。例如，在消费者完成读取共享内存缓冲区后，它可以到达一个 barrier，告诉生产者该缓冲区可以自由重用。

等待是同一协议的消费者端。消费者等待直到 barrier 完成当前迭代预期的阶段。只有这样才能安全地读取数据或重用受该 barrier 保护的资源。

重要的一点是，异步硬件不仅运行在程序之前，而且运行在程序之前。它还通过 barrier 报告完成情况。 TMA 可以发出共享内存块已准备就绪的信号。 Tensor Core 工作可以发出信号，表明 TMEM 结果已准备就绪。普通 threads 可以发出缓冲区不再使用的信号。 barrier 使所有这些情况都具有相同的生产者-消费者形状：生产者到达，消费者等待。

## phase 跟踪

barrier 通常不分配用于单一用途。 pipeline 的 K 循环可能会执行相同的切换数百次，并且为每次迭代分配新的共享内存 barrier 是不切实际的。相反，kernel 保留一小部分固定的 barrier，并在循环前进时重复使用它们。

phase 位使重用变得安全。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：跨多个 pipeline 迭代的重用 barrier，显示每轮完成后的 phase 位翻转。*

每次 barrier 完成当前轮次的所有到达时，它都会翻转阶段：阶段 0 变为阶段 1，阶段 1 变为阶段 0，依此类推。等待操作检查消费者期望的阶段。该预期 phase 由 kernel 保存在寄存器中。阶段成功等待一轮后，kernel 在使用下一轮 barrier 之前切换其本地 phase 值。

这可以防止 kernel 将旧完成误认为是新完成。假设一个 barrier 用于一台 TMA load 并且已经完成。如果下一个循环迭代在没有跟踪阶段的情况下重用相同的 barrier，消费者可能会观察到先前的完成情况并错误地假设新的负载已准备好。 phase 位将这两轮分开。迭代 0 等待一个阶段，迭代 1 等待相反的阶段，迭代 2 再次等待第一个阶段，并且该模式继续。

在真实的 pipeline 中，簿记通常是按阶段进行的。 kernel 具有固定数量的共享内存级、匹配的固定数量的 barrier 以及寄存器中的一小组 phase 值。随着循环的进行，每个逻辑迭代都映射到一个物理阶段，并且阶段值告诉等待操作它正在等待该物理 barrier 的哪一轮。

这就是为什么后来的 GEMM 代码不需要每个 K 区块 ({ref}`chap_gemm_async`) 一个 barrier。每个可重复使用的阶段需要一个 barrier，以及阶段跟踪。阶段索引选择共享内存缓冲区和 barrier。阶段值区分该阶段的当前使用与前一阶段。

**尝试使用您的代理**：给它一个两阶段 pipeline 并要求它跟踪四次迭代。对于每次迭代，列出阶段索引、局部阶段值、barrier 何时翻转，以及如果在重用阶段之前未切换阶段会出现什么问题。

## 同步规则

一旦 barrier 和 phase 机制明确，Tensor Core kernel 中的同步模式就相当机械化了。每当一条路径生成数据或释放另一条路径将消耗的资源时，必须明确进行切换。

常见的情况有以下三种。

第一种情况是为异步引擎生成数据的 thread 代码。如果 threads 写入共享内存，并且后面的 TMA store 或 MMA 指令读取该共享内存，则 kernel 必须在引擎读取 thread 写入之前使它们可见。这需要适当的 thread 级同步或栅栏。确切的指令取决于切换的 scope，但原因始终相同：在生成的 threads 完成写入之前，引擎不得观察共享内存缓冲区。

第二种情况是 TMA，为 MMA 生成数据。 TMA load 异步填充共享内存块。 MMA 路径不能仅仅因为发出了 TMA 指令就推断该 tile 已准备好。 TMA 操作必须与 `mbarrier` 关联，并且 MMA 路径必须在读取 tile 之前等待该 barrier。

第三个案例是 MMA，为 epilogue 生成数据。 `tcgen05` MMA 将其结果异步写入 TMEM。在 Tensor Core 完成相关工作之前，epilogue 无法安全地读取累加器。因此，MMA 提交路径到达完成 barrier，epilogue 在读取 TMEM 之前等待该 barrier。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：通过 `mbarrier` 完成 TMA load 信令。 MMA 路径在读取共享内存块之前等待 barrier。 Tensor Core 到 epilogue 的切换遵循相同的形状，只是 Tensor Core 提交路径执行到达而不是 TMA。*

同样的想法也适用于资源重用。 barrier 不仅仅是数据就绪信号。它也可以是“资源免费”的信号。在旧 tile 的所有使用者都使用完共享内存阶段之前，无法覆盖该阶段。在前一个用户完成读取或写入之前，TMEM 区域无法重复使用。在这些情况下，到达意味着“我已完成此资源”，而等待意味着“现在可以安全地在下一阶段重用此资源”。

这是在 pipeline GEMM kernel 中读取同步的正确方法。等待和到达并不作为防御性编程而分散。每一个都标志着一个具体的所有权转移：一个 tile 准备就绪，一个累加器变得可读，或者一个缓冲区变得可重用。一旦确定了这些切换，控制流程就变得更容易遵循。
