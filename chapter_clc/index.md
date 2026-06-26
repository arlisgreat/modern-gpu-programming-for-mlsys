(chap_clc)=
# 高级：Cluster Launch Control

:::{admonition} 概述
:class: overview

- persistent kernel 保留一组固定的 CTAs 或 CTA clusters 驻留（通常调整大小，以便每个 SM 大约有一个活动工作所有者，但不依赖于保证的 1:1 映射），并让它们循环遍历多个输出 tile，而不是启动一个每块 CTA。
- Cluster Launch Control 是 Blackwell 硬件机制，可让常驻 cluster 在运行时请求另一个 tile。这是一条围绕两条 PTX 指令构建的硬件工作窃取路径：一条指令请求工作，另一条指令读回请求是否成功。
- 主要好处是更好的尾部行为。当 tile 成本不均时，或者当 tile 数量在可用的 SMs 上分布不均时，提前完成的 CTAs 可以拉动更多工作而不是闲置。
:::

持久性 GEMM 不会将 CUDA grid 视为固定的每输出块一次 CTA 启动。相反，它推出了一组较小的长寿命 CTAs 或 CTA clusters。每个人计算一个 tile，前进到另一个 tile，再次计算，并继续计算，直到输出空间完成。这是在 {ref}`chap_gemm_advanced` 中构建的执行模式。

一旦 kernel 持久化，主要的调度问题就变得简单：在 CTA 或 cluster 完成其当前 tile 之后，下一个 tile 来自哪里？

最简单的答案是静态公式。例如，kernel 可以根据 CTA id 计算 tile 坐标，然后前进 grid 步幅。这很容易实现，并且当所有 tile 具有大致相同的成本并且 tile 数量均匀分布在 GPU 上时效果很好。但时间表是在工作实际进行之前决定的。如果一些 tile 需要更长的时间，或者如果最后几个 tile 分配不均匀，则一些 SMs 会提前完成其份额，而其他人仍在完成尾部工作。

Cluster Launch Control 或 CLC 更改了该调度模型。持久性 cluster 可以向硬件 grid scheduler 请求另一个尚未启动的 cluster 的工作，而不是预先决定整个分配。如果请求成功，则当前的 cluster 接管该 cluster 坐标并计算相应的 tile。如果请求失败，则不再有工作可窃取，并且循环退出。

这与 thread block clusters 本身不是一回事。 thread block clusters（CTAs 一起启动，具有 cluster 级同步和分布式共享内存访问）是在 Hopper（{ref}`chap_background`）中引入的。 CLC 是 Blackwell 的补充，它使 cluster 坐标上的调度变得动态。 cluster 已经是启动单元；CLC 允许已运行的 cluster 取消挂起的启动并继承其坐标。

## 两条指令

Cluster Launch Control 通过两条 PTX 指令公开。第一条指令向 grid scheduler 发送异步请求。第二条指令读取响应。

请求指令为`clusterlaunchcontrol.try_cancel.async`。

`try_cancel` 请求 scheduler 取消挂起的 cluster 的启动，并将 cluster 的坐标返回给调用者。响应作为 16 字节记录写入共享内存。由于请求是异步的，因此指令不会等待响应到达。相反，通过 `mbarrier` 报告完成情况，使用与 TMA 相同的势垒和相模型。

这是一个重要的细节，因为这意味着 CLC 不会引入新的等待模型。 kernel 发出请求，将其与 barrier 相关联，然后在读取响应之前等待 barrier。响应到达是通过字节计数完成的 barrier 发出信号的，其通用风格与其他异步硬件操作相同（请参阅{ref}`chap_async_barriers`）。

一旦 barrier 触发，kernel 将使用查询指令。

第一个查询是 `clusterlaunchcontrol.query_cancel.is_canceled`。它返回一个谓词，告诉 kernel 取消是否成功。真正的谓词意味着 scheduler 发现了一个待处理的 cluster 启动，取消了它，并返回了它的坐标。错误的谓词意味着没有剩余的待完成工作。

仅当 `is_canceled` 为 true 时，kernel 才应读取坐标。它使用 `clusterlaunchcontrol.query_cancel.get_first_ctaid` 执行此操作，提取已取消的 cluster 的第一个 CTA id。 That CTA id is a coordinate vector, usually read as `(x, y, z)`, and the kernel decodes it into the output tile it should compute next.

该协议中没有数字哨兵 tile ID。 kernel 在谓词上分支。如果谓词为真，则坐标有效。如果谓词为假，则工作窃取循环完成。

在底层，这种形状直接遵循 CLC 正在做的事情。硬件没有从软件队列中分配抽象任务。它正在取消尚未 launch 的 cluster launch。因此，成功的响应包含真实的 cluster 坐标。响应失败仅仅意味着 launch queue 已耗尽。

## 工作窃取循环

有了这两条指令，持久 scheduler 就变成了一个短循环。

在循环中的任何一点，cluster 都有一个块负责计算。在启动该 tile 之前，它会发送一个 `try_cancel` 请求来获取下一个 tile。该请求异步运行。当 scheduler 处理该请求时，cluster 计算其当前 tile。

当前 tile 完成后，cluster 等待与 `try_cancel` 响应关联的 `mbarrier`。一旦响应准备好，它就会调用 `query_cancel.is_canceled`。如果谓词为 true，它将调用 `query_cancel.get_first_ctaid`，解码返回的坐标，并将其用作下一个 tile。如果谓词为假，则不再有任何工作，并且 cluster 退出。

在代码形式中，循环是：

1. 发出 `try_cancel` 以获得可能的下一个 tile；
2. 在请求进行时计算当前 tile；
3. 等待响应 barrier；
4. 查询是否取消成功；
5. 继续使用返回的坐标或退出。

请求的放置使得循环变得有用。 cluster 不会等到完成当前 tile 才请求更多工作。它首先询问，然后计算。这将 scheduler 请求与有用的工作重叠。当当前 tile 完成时，下一个 tile 的答案通常已经可用。

这与 persistent kernels 在其他地方使用异步 copy 和 Tensor Core barrier 的基本原因相同。 kernel 避免将长延迟操作直接放在关键路径上。 CLC 将相同的思想应用于 tile scheduling：尽早请求下一个工作单元，计算当前单元，然后在需要时消耗调度结果。

## 与持久 GEMM 的关系

{ref}`chap_gemm_advanced` 中的持久 GEMM 在主要演练中使用静态 scheduler。静态 scheduler 更容易解释，因为可以直接从循环状态计算下一个 tile。例如，诸如 `ClusterPersistentScheduler2D` 之类的 scheduler 可以在输出 tile 空间上使用 grid 步幅模式来分配 tile。

CLC 是该静态分配的动态替代。外循环保持不变：每个驻留 cluster 重复计算一个输出 tile，然后前进到另一个输出 tile。改变的是下一个 tile 的来源。使用静态 scheduler，下一个 tile 是通过公式计算的。对于 CLC，下一个 tile 是通过硬件工作窃取返回的。

这种差异在接近 epilogue 时最为重要。在静态计划中，剩余的工作可能不会均匀分配。一些 SMs 可能会用完分配的 tile，而其他一些则还剩下几个 tile。对于 CLC，提前完成的 cluster 请求另一个待处理的 cluster 坐标。只要 launch queue 中还有剩余工作，早期完成者就会不断拉出更多的 tile。

当 tile 成本不统一时，这也很重要。由于边界、掩蔽、稀疏性、分组调度或围绕主矩阵乘法的融合工作，某些 GEMM 块可能会采用不同的路径。静态调度假设在观察到任何这些成本之前 tile 分配足够好。 CLC 不需要该假设。仅当 cluster 可用后，它才会分配更多工作。

因此，在 TIRx 中，CLC 可以公开为动态 tile scheduler。编程模型不需要改变 tile 的计算。 tile 主体与静态 scheduler 使用的持久 GEMM 主体相同。 scheduler 从“根据公式计算下一个 tile 坐标”更改为“向硬件询问下一个可用的 cluster 坐标”。结果是相同的持久循环，但采用硬件驱动的工作分配，而不是固定的启动时间安排。
