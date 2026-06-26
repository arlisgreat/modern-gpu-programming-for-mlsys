(chap_tma)=
# 异步数据移动：TMA

:::{admonition} 概述
:class: overview

- TMA 是用于全局内存和共享内存之间异步 tile copy 的硬件引擎。一个 thread 发出 copy command，引擎移动字节。
- TMA copy 由 tensor map descriptor 描述。该 descriptor 告诉引擎全局张量形状、步幅、tile 坐标和共享内存 swizzle 模式。
- 在 load path 上，TMA 在写入共享内存时可以 swizzle tile，因此 tile 落在 Tensor Core 预期的 layout 中。
- TMA load 通过具有字节计数跟踪功能的 `mbarrier` 完成。TMA store 使用 commit group 和 wait group。
:::

Tensor Core 只有在有可供使用的数据时才有帮助。在 GEMM 或 Attention kernel 中，一旦 pipeline 已满 ({ref}`chap_performance`)，数学部分可能会受到 compute 限制，但只有下一个 operand tile 及时到达时，pipeline 才会保持满状态。

移动 tile 的较旧方法是让 threads 自行 copy。每个 thread 计算地址，从全局内存发出 load，并将值 store 到共享内存中。这是可行的，但它在地址算术和 copy bookkeeping 上花费 warp 指令，而不是计算。它还使 copy path 在应该馈送到 Tensor Core 的同一 warps 的指令流中可见。

Tensor Memory Accelerator 或 TMA 将此工作移至 hardware copy engine 中。一个 thread 发出 tile copy。然后，copy engine 在全局内存和共享内存之间异步移动矩形块。当引擎移动字节时，CTA 的其余部分可以继续执行其他工作。

TMA 还处理部分 layout 问题。Tensor Core 不仅仅需要共享内存中的正确值。它需要它们位于正确的共享内存 layout 中。在 load path 上，TMA 可以在写入 tile 时应用共享内存 swizzle。这使得 tile 可以直接落在后来的 MMA 所期望的 layout 中。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：TMA 将 tile 从全局内存 copy 到共享内存。切换 swizzle 模式并将鼠标悬停在源单元上以查看它在共享内存中的位置。*

## 一个 thread 发出请求，硬件移动 tile

TMA copy 由 issuing thread 发起。thread 不会循环遍历 tile 中的所有元素。它为硬件提供 copy descriptor，然后 TMA 引擎执行传输。

主要输入是 tensor map descriptor。该 descriptor 描述了全局张量以及如何从中读取 tile。它记录了张量形状、步幅、元素大小、tile shape 和 swizzle 模式等信息。issuing thread 还提供了 tile 应放置的共享内存地址。

发出指令后，copy 异步运行。issuing thread 可以继续。CTA 中的其他 threads 也可以继续。现在，传输由 TMA 引擎负责，而不是普通 load/store 指令的循环。

这为 kernel 提供了两种不同的方式来表达相同的逻辑操作“copy 这个 tile”。

一种路径是 thread copy。threads 协作从全局内存 load 并 store 到共享内存中。这使 kernel 能够直接控制每次访问，但它会消耗 thread 指令和寄存器来进行地址计算。

另一个路径是 TMA copy。一个 thread 发出传输，hardware copy engine 执行矩形 copy。这是大型常规 tile 的自然路径，尤其是 Tensor Core kernels 使用的操作数 tile。

这两条路径具有不同的同步规则和不同的性能行为。在它们之间进行选择是一个 dispatch 的决定。layout 告诉 kernel 它需要什么样的内存排列。scope 告诉它哪个 threads 或 CTAs 正在参与。dispatch 决定 copy 是通过普通 thread 代码实现还是通过 TMA 实现。

## swizzled layout

移动 tile 是不够的。该 tile 还必须放置在 layout 中的共享内存中，以便 Tensor Core 可以有效地读取。

这是使用 TMA swizzle 的地方。当 TMA 将 tile 写入共享内存时，它可以排列共享内存地址模式。全局内存块仍然是逻辑矩形，但共享内存中的目标 layout 可以 swizzle。

swizzle 模式是 TMA descriptor 的一部分。一旦设置了 descriptor，发出 thread 就不必手动应用 swizzle。引擎会在字节落入共享内存时应用它。

关键要求是一致性。TMA descriptor、共享内存块 layout 和后面的 MMA 指令必须都描述相同的 layout ({ref}`chap_data_layout`)。如果 TMA 使用一个 swizzle 写入该 tile，但 MMA 按另一个 layout 读取该 tile，硬件仍会执行操作，但对计算而言，字节排列可能不正确。

正是在这一点上，layout 表示法不再只是一种记账工具。DSL 使用的 layout 必须与 TMA descriptor 和 Tensor Core 指令使用的硬件 layout 相匹配。例如，如果 kernel 表示操作数块存储在 128 字节 swizzled layout 中，则 TMA descriptor 必须使用匹配的 swizzle 模式，并且 MMA dispatch 必须期望相同的共享内存安排。上面的演示可以让您在无 swizzle 和 128 字节 swizzle 之间切换；应用 swizzle 后，将鼠标悬停在源元素上即可查看其着陆位置。

理解 swizzle 的一个有用方式是：TMA 不会更改逻辑块。它改变的是逻辑元素在共享内存中的物理位置。后来的 MMA 仍然消耗相同的逻辑 A 或 B 块。swizzle 仅决定该 tile 如何跨 shared memory banks 排列。

## 用于 tiling 和 swizzling 的 3D TMA

普通的 TMA copy 会移动平面 2D tile，但 Tensor Core 想要的共享内存 layout 通常会 tile 到 swizzle atom（来自 {ref}`chap_data_layout` 的 8 x 128 字节 atom）。TMA 使用额外的 descriptor 维度来处理该问题。**3D TMA** 将共享内存 box 描述为 `(group, row, col)`，其中 group 维度跨越 atom 和一个 atom 内的内部两个地址。然后，单个 3D copy 会按 atom tile，并在每个 atom 内应用 swizzle，因此数据已经到达 MMA 期望的 layout 中，无需单独的 tiling 或 swizzling pass。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：3D TMA copy，寻址为（group、row、col），tile 到 swizzled shared memory 中。*

选择 swizzle *格式*与此 tiling 相关。更宽的 swizzle 将一列分散在更多的 bank 上，因此 128 字节 swizzle 是适合时的默认值，但 N 字节 atom 需要 tile 的连续尺寸来填充它。因此，由于形状限制而较小的 tile 不能使用 128 字节 swizzle，而必须降低到 64 字节或 32 字节：经验法则是选择 tile 可以填充的最大 swizzle ({ref}`chap_data_layout`)。下面的演示直接显示了约束：只有当 tile 被切成与 atom 匹配的 16 x 8 group 时，16 x 16 tile 上的 128 字节 swizzle 才会变得无冲突。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*交互式：16 x 16 tile 上的 128 字节 swizzle，一旦 tile 成 16 x 8 group 就无冲突。*

## 完成：load

copy 是异步的，因此仅仅发出 copy 还不够。消费者不能仅仅因为发出了 TMA 指令就读取共享内存块。仅当引擎完成字节写入后，该 tile 才可安全读取。

对于 TMA load，完成信号是 `mbarrier` ({ref}`chap_async_barriers`)。

通常的顺序是：

1. 为 pipeline 阶段初始化或重用 `mbarrier`；
2. 告诉 barrier TMA 传输预计写入多少字节；
3. issue TMA load；
4. 让 TMA 引擎在字节到达时更新 barrier；
5. 让消费者在读取共享内存块之前等待 barrier 阶段。

字节计数通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这有两个作用。它记录预期的传输大小，并且还执行 issuing thread 到达 barrier 的操作。仅仅因为这个调用发生，barrier 并不完整。它仍然等待 TMA engine 报告预期字节已到达。

随着传输的进行，引擎针对 barrier 执行完整的交易更新。仅当满足两个条件后，barrier 阶段才会翻转：满足到达计数，并且待处理字节计数达到零。

然后消费者在该 barrier 上等待。一旦预期阶段的等待完成，共享内存块就准备好了。此时 MMA 路径可以安全地读取它。

![TMA load 同步流程](../img/tma_sync_flow.png)

这与其他异步生产者-消费者切换使用的 barrier 模型相同。生产者是 TMA engine。消费者是 MMA 路径或读取共享内存块的任何其他代码。barrier 是它们之间的明确切换。

## 完成：store

TMA store 以相反方向移动数据，从共享内存到全局内存。它们也是异步的，但完成机制不同。

TMA load 通常在同一个 kernel 内供后续消费者读取。MMA 路径需要知道共享内存块何时准备就绪。这就是加载路径使用 `mbarrier` 的原因。

TMA store 通常将最终数据写入全局内存。通常没有立即的 in-kernel 消费者等待存储的结果。 kernel 需要知道的主要事情是何时可以安全地重用共享内存缓冲区或完成存储序列。

为此，TMA store 使用 commit group 和 wait group。kernel 发出一个或多个 store，提交该 group，然后等待该 group drain。等待完成后，从 kernel 的角度来看，该 group 中的 store 已完成，并且可以安全地重用 store 所使用的共享内存区域。

所以规则很简单：

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

这两种机制在不同的切换点具有相同的目的。load 需要使共享内存 tile 对后来的消费者可见。store 需要确保在 kernel 重用源存储或依赖已 drain 的 store 之前完成传出传输。

## 为什么 TMA 对于 pipeline 很重要

TMA 在作为 pipeline 一部分时最有用。kernel 可以为未来的 tile 发出 load，而 Tensor Core 计算当前的 tile。load 在后台运行。计算在前台运行。当未来的 tile 变成当前的 tile 时，barrier 将两者连接起来。

典型的 GEMM 循环重复使用此结构。共享内存的一级保存 MMA 当前消耗的 tile。TMA 正在填补另一个 stage。随着循环的推进，角色轮换。在 MMA 读取 stage 之前，它会等待该 stage 的 load barrier。在 TMA 覆盖 stage 之前，kernel 确保前一个消费者已完成该 stage。

这就是为什么 TMA 和 `mbarrier` 通常一起出现在 Blackwell-和 Hopper-风格的 kernels 中。TMA 为 kernel 提供异步 copy engine。barrier 为 kernel 提供了一种精确的方法来了解 copy 的字节何时准备就绪。
