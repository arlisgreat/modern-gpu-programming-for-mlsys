(chap_data_layout)=
# 数据 layout 及其表示法

:::{admonition} 概述
:class: overview

- *数据 layout* 将张量的逻辑索引映射到物理位置，并决定 coalescing、bank conflict 以及引擎是否可以读取 tile。
- 本书用一种表示法编写 layouts：`S[(shape) : (strides)]`，带有命名轴（`@laneid`、`@TLane`，...）和用于广播或复制数据的复制术语 `R[...]`。
- Swizzle 是地址的 XOR 重新映射，可消除 shared memory bank conflict。
:::

以不同物理排列写入内存的相同数字在同一 GPU 上运行的数量级可能相差一个数量级。

原因是张量的逻辑索引没有说明其字节实际所在的位置。硬件对这种 layout 高度敏感：它决定 32 个 lane 的负载是否合并为一个事务或分散为 32 个事务，它们的地址是否位于不同的 bank 中或发生冲突和序列化，甚至一个块是否与 Tensor Core 可以读取的字节排列相匹配。

机器学习程序通常通过张量的逻辑形状来描述张量。 **数据 layout** 添加了缺失的物理部分：它表示具有逻辑索引 `(i, j, …)` 的元素所在的位置，无论是在内存中、在寄存器中还是在某些其他硬件存储中。

本章介绍了现代 GPU 编程中出现的主要 layouts。为了使讨论易于处理，我们开发了一种紧凑的**符号**来描述机器学习系统遇到的情况。我们以 **swizzling** 结束，该机制可以同时高效地按行和按列访问 tile。

## 形状-步幅模型

在我们讨论 GPU 特定的 layouts 之前，值得从最简单的开始，因为本章中的其他所有内容都是建立在它之上的。从本质上讲，layout 只有两件事：**形状**和一组匹配的**步幅**。我们将这对写为 `S[(shape) : (strides)]`，为了找到逻辑索引所在的位置，我们采用该索引与步幅的点积。例如，row-major 4×4 矩阵如下所示：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这只不过是经典的形状/步幅模型，编写紧凑（比 CuTe 表示法的 row-major 版本更简洁），并且接下来的所有内容都是根据它构建的。

事实上，您几乎肯定已经使用过这个模型。任何编写过 PyTorch 或 NumPy 的人都会这样做，因为这些库中的张量*就是*精确的形状以及平面存储缓冲区上的跨距：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

一旦您以这种方式看到张量，就会清楚为什么如此多的“重塑”操作根本不接触数据。他们只是重写步幅并在同一存储上传回**视图**，最清楚的例子是转置或排列：

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True, same bytes
```

这里，`t.permute(1, 0)` 是“相同”内存上的 `S[(4, 3) : (1, 4)]`：转置纯粹是步幅的变化，没有移动任何字节。对于连续张量上的 `reshape` 或 `view` 来说，情况是相同的：新的形状和对旧存储的新跨越。 （NumPy 的行为相同；唯一的区别是它的 `.strides` 以字节而不是元素来计数。）

这正是 layouts 在 GPU 上的工作方式，本章的其余部分实际上是一个想法的一系列变体：tile 的映射（无论是到内存，还是通过我们稍后介绍的命名轴，到 lane 和寄存器）是固定缓冲区上的跨步规则，因此重新排列 tile 通常是 *layout* 的更改，而不是副本。不过，我们应该小心这种推理的界限。零拷贝的故事完全适用于单个线性地址空间的逻辑视图；在 GPU 上，仅当新视图与现有字节和所有权安排兼容时才适用。当您更改哪个 thread 或寄存器拥有一个元素，或者更改 thread 或寄存器，或者更改 SMEM swizzle 时，您通常需要真正的数据移动：加载、存储、洗牌、`ldmatrix`、转置。

## 平铺 layout

到目前为止，我们已经描述了整个张量的 layouts。然而，GPU kernels 很少同时对整个矩阵进行操作；它们在较小的 tile 上工作，这些 tile 由硬件的不同部分加载、转换和计算。好消息是，tile 不需要任何新东西。它仍然只是一个 layout，只是现在多写了几个维度。将 8×8 矩阵切成 2×4 块，我们得到一个 4-D layout，坐标为 `(tile_row, row_in_tile, tile_col, col_in_tile)` 并选择步幅，以便每个块保持连续：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

逻辑上的 `(i, j)` 首先变为 `(i//2, i%2, j//4, j%4)`，然后跨步运行。值得注意的是，该表示法在表达平铺时根本没有任何特殊的“平铺”概念：它与之前的形状-步幅模型相同，只是将索引分为外坐标和内坐标。

下面的交互式可视化显示了逻辑矩阵索引如何分解为 tile 坐标，然后映射到物理地址。

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：单击单元格即可查看其平铺索引和地址。*

## 命名轴

到目前为止，`S[...]` 中的每一步都已命名为线性内存中的偏移量，并且我们已将地址视为其中的位置。不过，在 GPU 上，数据可以驻留在多个位置：除了内存之外，tile 还可以分布在 warp lane、thread 寄存器或 TMEM lane 和列上。为了统一描述所有这些，我们用**命名轴**扩展符号。这个想法是让每个步幅系数带有一个轴标签，说明它移动的空间：`@m`用于普通内存，`@laneid`用于 warp lane，`@reg`用于寄存器，`@warpid`用于 warps，以及`@TLane` / `@TCol` 为 TMEM 坐标。有了标签，单个 layout 不仅可以描述数据在内存中的位置，还可以描述数据如何分布在运行其上的硬件资源上。

一旦内存标签变得明确，内存中的 row-major 8×16 块就很简单了

```text
S[(8, 16) : (16@m, 1@m)]
```

当 layout 描述的数据*分布在 threads* 上而不是布置在内存中时，标签就开始发挥作用。以 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` 为例：它不是指向线性内存，而是将行和列映射到 lane ID 和 per-lane register 上。这里，`laneid` 表示 warp 内的 warp lane index，即 `thread_index % warp_size`。这正是您将在 {ref}`chap_layout_generations` 中遇到的 Tensor Core register fragment。

下面的交互式可视化显示了 layout 如何在 warp lane 和每 lane 寄存器之间分配张量元素，而不是将它们放置在线性存储器中。

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：layout 优于 `@laneid` 和 `@reg`；单击一个单元格即可查看哪个 lane/寄存器保存它。*

## 分布式 layout

命名轴如此有用的原因在于，它们让我们能够在系统的多个级别上统一描述放置，包括*跨整个设备*的放置。我们刚刚将它们用于单个 GPU 内的 lane 和寄存器，但同样的想法向外延伸：诸如 `@gpuid_x` 和 `@gpuid_y` 之类的轴可以表示数据位于 GPU 网格中的位置，并且通过它们，符号捕获了分布式训练和推理中出现的分片模式。轴尚未捕获的一件事是“复制”，即复制到多个位置的数据，因此我们添加符号 `R[n : stride]`，其中 `R` 标记复制的维度。例如，`R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` 轴的复制。将两者放在一起，单个表达式可以在 2×2 GPU 网格上分割张量并沿一个轴复制它：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的演示展示了小型 GPU 网格上的组合分区和复制模式。单击任何单元格即可查看哪个设备拥有它，并观察 `@gpuid_x` 复制如何在配对设备上放置相同的副本；按钮在完全分片、分片 + 副本和分片 + 偏移 layouts 之间切换。

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：layout 分布在 2×2 GPU 网格上；单击一个单元格即可查看哪个 device(s) 持有它。*

### 内核内复制模式：TMEM 中的 scale factor

我们刚刚为 GPU 网格引入的复制维度 `R[...]` 不仅仅涉及多个设备。事实证明，相同的构造也描述了完全在单个 kernel 内部发生的事情：硬件“跨 lane 广播”的数据。 Blackwell 的 block-scaled MMA ({ref}`chap_layout_generations`) 就是一个很好的例子。它的 scale factor 位于 TMEM 中，其中 128 行 scale vector 仅存储在 **32 个 TMEM lane** 中，其中逻辑行 `r` 转到 TMEM lane `r % 32`，其中 `r // 32` 沿列运行。然后，这 32 个存储的 TMEM lane **沿 TMEM `TLane` 轴**复制，从 32 个到 128 个 TMEM lane，以便读取 warpgroup 中的四个 warps 中的每一个都拥有自己的 32-lane TMEM 窗口。这是一个 `warpx4` 广播，我们用复制维度来编写它。读取本身由 warps' threads 执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这以 32 个 TMEM lane 的跨度提供了四个副本：TMEM lane `l`、`l+32`、`l+64` 和 `l+96` 都保持相同的比例。和以前一样，复制维度没有携带新数据；它只是说“相同的值，位于四个 TMEM lane 位置”，就像 `@gpuid_x` 刚才在 GPU 网格上广播一行一样。

下面的交互式演示同时显示了这两个步骤：紧凑打包到 32 个 TMEM lane 中，然后 `warpx4` 广播到 128 个读取 lane。

```{raw} html
<iframe src="../demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：点击 scale factor `SFA[m, sf]`；它在 lane `m mod 32`、列 `(m // 32)·4 + sf` 处打包到 TMEM 中，然后通过 `TLane` 轴将 `warpx4` 广播到四个 lane 副本（`l`、`l+32`、`l+64`、`l+96`），每个 warp 的 32-lane 窗口一个。*

每列内的字节打包（`scale_vec` 1X/2X/4X 模式）和 `cta_group::2` 分割在 {ref}`chap_layout_generations` 中介绍。

已经了解 CuTe 的读者可以将本章中的符号视为它的 row-major 变体，并使用显式硬件命名轴和专用复制结构进行扩展。

## swizzled layout

本章中最后一个 layout 的存在是为了解决一个特定的硬件问题。 GPU 上的 shared memory 被组织成 banks，当不同 lane 落在 different banks 上时，访问运行速度最快。当多个 lane 到达 same bank 中的不同地址时，硬件别无选择，只能将它们序列化，并且我们付出了 **bank conflict** 的代价。

在张量程序中，这是很难避免的，因为内存不是以纯线性顺序访问的。使用矩阵时，我们通常需要读取同一 tile 的行切片和列切片，这会产生真正的紧张：对行访问有效的 layout 往往会在列访问中产生 bank conflict，而有利于列的访问会损害行。 **Swizzling** 是旨在打破这种紧张的技术。

swizzle 背后的想法是排列地址映射，通常通过对列索引与行进行异或，以便行和列访问最终分布在 bank 中。它提供的无冲突保证是特定的：它适用于匹配元素宽度、swizzle 模式和访问模式（引擎描述符期望的模式），而不适用于任意元素宽度或对齐方式。

下面的第一个交互式演示使这一点变得具体。单击列索引并观察每个元素落在哪个 bank 中：在左侧的普通 row-major tile 中，一列将所有八个元素汇集到一个 bank 中，因此读取序列化为八个周期；在右侧的 XOR swizzled layout 中，同一列分布在八个不同的 bank 中，并在一个周期中读取。

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：8×8 块，按普通 row-major 的列发生 bank conflict，XOR swizzle 后无冲突。*

这个 8×8 的小例子抓住了核心思想，但真正的 GPU 存储器的 bank 比玩具图片显示的要多得多。为了使 swizzling 全面进行，我们不会将整个 tile 视为一个整体对象。相反，我们将内存切成小段，并在每个段内应用 swizzle 模式。实践中最常见的情况是 `SWIZZLE_128B`，它围绕 128 字节段组织，因此相同的行/列重新映射技巧自然适合 32-bank memory system。

下面的交互式演示显示了一个具体硬件 swizzle、`SWIZZLE_128B`，因此在我们推广跨格式之前，逐段重复的模式是可见的。

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：128 字节段内的`SWIZZLE_128B`模式；逐步执行读取周期以查看 `physical_sector = logical_sector XOR row` 将每列分布在不同的 bank 中。*

同样的想法也适用于 128 字节的情况。为了简化可视化，我们现在将使用单一色块来引用一个段，而不是绘制单独的 bank。一般来说，硬件定义了一个小的重复**原子**，在其上应用排列，并且不同的 swizzle 模式选择不同的原子大小。 `SWIZZLE_128B` 使用 8×128 B 原子，`SWIZZLE_64B` 使用 8×64 B 原子，`SWIZZLE_32B` 使用 8×32 B 原子；然后，整个 tile 将由正在使用的原子进行平铺。

最终的交互式演示可让您在这些格式之间切换（包括 16 B 交错模式），选择一种数据类型，并将鼠标悬停在任何单元格上以直接检查一个原子内的元素排列，这是推理加载/存储指令所需的 swizzle 的正确详细程度。

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：选择 swizzle 格式（和数据类型）以查看其原子形状（8 × N B）；将鼠标悬停在单元格上即可查看其元素如何排列。*

您应该选择哪种模式？经验法则是选择 tile 可以填充的“最大”原子。 N 字节原子需要 tile 的连续维度至少为 N 字节及其倍数，因此 `SWIZZLE_128B` 仅当行跨越至少 128 字节或 64 个 `float16` 元素时才适用。当它适合时，它是默认选择，因为它的 8 × 128 B 原子覆盖了完整的 128 字节 bank 行，因此一次将一列分散在所有 32 个 bank 中，从而在 fp16 中一次提供对 8 行和 8 列的无冲突访问。但是，当问题的形状迫使连续尺寸变小时，tile 无法再填充 128 B 原子，并且您将下降到 `SWIZZLE_64B` 或 `SWIZZLE_32B`，这是该行仍然可以覆盖的最大原子。

您永远不会手动计算出这些排列后的地址，并且值得精确了解 swizzle 与 `S[...]` 符号的关系：它“不是”仿射映射的一部分。它是一个独立的非仿射层，位于其之上。 `S[...]` layout 将一个元素放置在线性存储器 (`@m`) 地址处，然后 swizzle 将写入 TIRx layout API 中的该地址排列为 `ComposeLayout(swizzle, tile)` （{ref}`chap_tirx_layout_api`）。您的工作只是在每个接触 tile 的操作中选择一种一致的模式，然后让组合的 layout 完成其余的工作。

同样组成的 layout 也是硬件所填充的，这就是混合和平铺结合在一起的地方。 TMA 描述符是多维的，因此单个三维框可以描述 tile 的原子平铺和每个原子内的 swizzle；然后，一个 TMA load 将 tile 逐个原子地铺开，而 swizzles 在写入共享内存 ({ref}`chap_tma`) 时将其铺开，无需单独的 swizzling pass。 *swizzle 每个 engine 的需求是特定于一代的，这是下一章的主题。
