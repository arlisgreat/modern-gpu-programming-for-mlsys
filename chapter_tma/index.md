(chap_tma)=
# 异步数据移动：TMA

:::{admonition} 概述
:class: overview

- TMA 是用于全局内存和共享内存之间异步切片复制的硬件引擎。一个 thread 发出副本，引擎移动字节。
- TMA 副本由张量映射描述符描述。该描述符告诉引擎全局张量形状、步幅、tile 坐标和共享内存 swizzle 模式。
- 在加载路径上，TMA 在写入共享内存时可以 swizzle tile，因此 tile 落在 Tensor Core 预期的 layout 中。
- TMA 通过具有字节计数跟踪功能的 `mbarrier` 完成加载。 TMA 存储使用提交组和等待组。
:::

Tensor Core 只有在有可供使用的数据时才有帮助。在 GEMM 或 Attention kernel 中，一旦 pipeline 已满 ({ref}`chap_performance`)，数学部分可能会受到 compute 限制，但只有下一个 operand tile 及时到达时，pipeline 才会保持满状态。

移动 tile 的较旧方法是让 threads 自行复制它。每个 thread 计算地址，从全局内存发出负载，并将值存储到共享内存中。这是可行的，但它在地址算术和复制簿记上花费 warp 指令，而不是计算。它还使复制路径在应该馈送到 Tensor Core 的同一 warps 的指令流中可见。

Tensor Memory Accelerator 或 TMA 将此工作移至硬件复制引擎中。一个 thread 发出平铺副本。然后，复制引擎在全局内存和共享内存之间异步移动矩形块。当引擎移动字节时，CTA 的其余部分可以继续执行其他工作。

TMA 还处理部分 layout 问题。 Tensor Core 不仅仅需要共享内存中的正确值。它需要它们位于正确的共享内存 layout 中。在加载路径上，TMA 可以在写入 tile 时应用共享内存 swizzle。这使得 tile 可以直接落在后来的 MMA 所期望的 layout 中。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：TMA 将 tile 从全局内存复制到共享内存。切换 swizzle 模式并将鼠标悬停在源单元上以查看它在共享内存中的位置。*

## 一个 thread 出现问题，硬件会移动 tile

TMA 副本以发行 thread 的副本开始。 thread 不会循环遍历 tile 中的所有元素。它为硬件提供副本的描述，然后 TMA 引擎执行传输。

主要输入是张量映射描述符。该描述符描述了全局张量以及如何从中读取 tile。它记录了张量形状、步幅、元素大小、平铺形状和 swizzle 模式等信息。发布的 thread 还提供了 tile 应放置的共享内存地址。

发出指令后，副本异步运行。发行 thread 可以继续。 CTA 中的其他 threads 也可以继续。现在，传输由 TMA 引擎负责，而不是普通加载和存储指令的循环。

这为 kernel 提供了两种不同的方式来表达相同的逻辑操作“复制此 tile”。

一种路径是 thread 副本。 thread 协作从全局内存加载并存储到共享内存中。这使 kernel 能够直接控制每次访问，但它会消耗 thread 指令和寄存器来进行地址计算。

另一个路径是 TMA 副本。一个 thread 发出传输，硬件复制引擎执行矩形复制。这是大型常规 tile 的自然路径，尤其是 Tensor Core kernels 使用的操作数 tile。

这两条路径具有不同的同步规则和不同的性能行为。在它们之间进行选择是一个 dispatch 的决定。 layout 告诉 kernel 它需要什么样的内存排列。 scope 告诉它哪个 threads 或 CTAs 正在参与。 dispatch 决定复制是通过普通 thread 代码实现还是通过 TMA 实现。

## 混合 layout

移动 tile 是不够的。该 tile 还必须放置在 layout 中的共享内存中，以便 Tensor Core 可以有效地读取。

这是使用 TMA 调配的地方。当 TMA 将 tile 写入共享内存时，它可以排列共享内存地址模式。全局内存块仍然是逻辑矩形，但共享内存中的目标 layout 可以混合。

swizzle 模式是 TMA 描述符的一部分。一旦设置了描述符，发出 thread 就不必手动应用 swizzle。引擎将其应用为字节落在共享内存中。

重要的要求是同意。 TMA 描述符、共享内存块 layout 和后面的 MMA 指令必须都描述相同的 layout ({ref}`chap_data_layout`)。如果 TMA 使用一个 swizzle 写入该 tile，但 MMA 读取该 tile 就好像它有另一个 TMA 一样，硬件仍将按照要求执行操作。对于计算而言，字节的排列可能不正确。

正是在这一点上，layout 表示法不再只是一种记账工具。 DSL 使用的 layout 必须与 TMA 描述符和 Tensor Core 指令使用的硬件 layout 相匹配。例如，如果 kernel 表示操作数块存储在 128 字节混合 layout 中，则 TMA 描述符必须使用匹配的 swizzle 模式，并且 MMA dispatch 必须期望相同的共享内存安排。上面的演示可以让您在无 swizzle 和 128 字节 swizzle 之间切换；应用 swizzle 后，将鼠标悬停在源元素上即可查看其着陆位置。

读取 swizzle 的一个有用方法是 TMA 不会更改逻辑块。它正在改变逻辑元素在共享内存中的物理位置。后来的 MMA 仍然消耗相同的逻辑 A 或 B 块。 swizzle 仅决定该 tile 如何跨共享 bank 排列。

## 用于平铺和混合的 3D TMA

普通的 TMA 副本会移动平面 2D tile，但 Tensor Core 想要的共享内存 layout 通常会“平铺”到 swizzle 原子（来自 {ref}`chap_data_layout` 的 8 x 128 字节原子）。 TMA 使用额外的描述符维度来处理该问题。 **3D TMA** 将共享内存盒描述为 `(group, row, col)`，其中组维度跨越原子和一个原子内的内部两个地址。然后，单个 3D 副本将逐个原子地平铺（平铺）并在每个原子内应用 swizzle，因此数据已经到达 MMA 期望的 layout 中，无需单独的平铺或混合 lane。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：3D TMA 副本，寻址为（组、行、列），平铺到混合共享内存中。*

选择 swizzle *格式*与此平铺相关。更宽的 swizzle 将一列分散在更多的 bank 上，因此 128 字节 swizzle 是它适合的默认值，但 N 字节原子需要 tile 的连续尺寸来填充它。因此，由于形状限制而较小的 tile 不能使用 128 字节 swizzle，而必须降低到 64 字节或 32 字节：经验法则是选择 tile 可以填充的最大 swizzle ({ref}`chap_data_layout`)。下面的演示直接显示了约束：只有当 tile 被分成与原子匹配的 16 x 8 组时，16 x 16 tile 上的 128 字节 swizzle 才会变得无冲突。

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
*交互式：16 x 16 tile 上的 128 字节 swizzle，一旦平铺为 16 x 8 组就无冲突。*

## 完成：负载

副本是异步的，因此发出副本是不够的。消费者不能仅仅因为发出了 TMA 指令就读取共享内存块。仅当引擎完成字节写入后，该 tile 才可安全读取。

对于 TMA 负载，完成信号是 `mbarrier` ({ref}`chap_async_barriers`)。

通常的顺序是：

1. 为 pipeline 阶段初始化或重用 `mbarrier`；
2. 告诉 barrier TMA 传输预计写入多少字节；
3. 签发 TMA load；
4. 让 TMA 引擎在字节到达时更新 barrier；
5. 让消费者在读取共享内存块之前等待 barrier 阶段。

字节计数通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这有两个作用。它记录预期的传输大小，并且还执行发出 thread 到达 barrier 的操作。仅仅因为这个调用的发生，barrier 并不完整。它仍然等待 TMA 引擎报告预期字节已到达。

随着传输的进行，引擎针对 barrier 执行完整的交易更新。仅当满足两个条件后，barrier 阶段才会翻转：满足到达计数，并且待处理字节计数达到零。

然后消费者在该 barrier 上等待。一旦预期阶段的等待完成，共享内存块就准备好了。此时 MMA 路径可以安全地读取它。

![TMA load 同步流程](../img/tma_sync_flow.png)

这与其他异步生产者-消费者切换使用的 barrier 模型相同。生产者是 TMAengine。消费者是 MMA 路径或读取共享内存块的任何其他代码。 barrier 是它们之间的明确切换。

## 完成：store

TMA 以相反的方向存储移动数据，从共享内存到全局内存。它们也是异步的，但完成机制不同。

TMA load 通常在同一个 kernel 内为消费者提供食物。 MMA 路径需要知道共享内存块何时准备就绪。这就是加载路径使用 `mbarrier` 的原因。

TMA store 通常将最终数据写入全局内存。通常没有立即的 in-kernel 消费者等待存储的结果。 kernel 需要知道的主要事情是何时可以安全地重用共享内存缓冲区或完成存储序列。

为此，TMA 存储使用提交组和等待组。 kernel 发出一个或多个存储，提交该组，然后等待该组耗尽。等待完成后，从 kernel 的角度来看，该组中的存储已完成，并且可以安全地重用存储所使用的共享内存区域。

所以规则很简单：

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

这两种机制在不同的切换点具有相同的目的。负载需要使共享内存 tile 对后来的消费者可见。存储需要确保在 kernel 重用源存储或依赖已耗尽的存储之前完成传出传输。

## 为什么 TMA 对于 pipeline 很重要

TMA 当它是 pipeline 的一部分时最有用。 kernel 可以为未来的 tile 发出负载，而 Tensor Core 计算当前的 tile。负载在后台运行。计算在前台运行。当未来的 tile 变成当前的 tile 时，barrier 将两者连接起来。

典型的 GEMM 循环重复使用此结构。共享内存的一级保存 MMA 当前消耗的 tile。 TMA 正在填补另一个阶段。随着循环的推进，角色轮换。在 MMA 读取阶段之前，它会等待该阶段的负载 barrier。在 TMA 覆盖阶段之前，kernel 确保前一个消费者已完成该阶段。

这就是为什么 TMA 和`mbarrier`通常一起出现在 Blackwell-和 Hopper-风格的 kernels 中。 TMA 为 kernel 提供异步复制引擎。该 barrier 为 kernel 提供了一种精确的方法来了解复制的字节何时准备就绪。
