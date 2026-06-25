(chap_tirx_layout_api)=
# TIRxlayoutAPI

:::{admonition} 概述
:class: overview

- TIRx layout API 将 layout 表示法从 `chap_data_layout` 转换为编译器对象。主要对象为`TileLayout`、`SwizzleLayout`、`ComposeLayout`。
- `TileLayout` 描述了指定硬件轴上的仿射放置。它是根据分片规格 `S[...]`、副本规格 `R[...]` 和可选偏移量构建的。
- layout 将一个逻辑坐标映射到一个或多个物理坐标。 `layout.apply()` 评估该映射。
- `SwizzleLayout` 描述了用于避免 bank 冲突的基于 XOR 的共享内存 swizzles。 `ComposeLayout` 将 swizzle 堆叠在 tile layout 的顶部。
- 现成的构造函数如 `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 覆盖了 kernels 中重复出现的硬件 layouts。
:::

{ref}`chap_data_layout` 介绍了本书中使用的符号：平铺形状、命名轴上的一组步幅以及用于复制而不是分区的值的可选复制术语。本章将这种表示法转化为编译器使用的 API。

目标是页面上的符号和 kernel 中的代码看起来几乎相同。当您编写 layout 时，例如：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你不只是写一个解释。您正在构造一个可以附加到缓冲区的 `TileLayout` 对象。之后，每个接触缓冲区的 tile 操作都可以从 layout 读取其位置。该 layout 写入一次，检查一次，并由编译器重用。

当从池中分配或声明缓冲区时，会附加 layout：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那时起，缓冲区就开始了它的物理放置。平铺操作不需要重复每个元素所在的位置。

layout 对象位于一个模块中：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

API 背后有一个中心思想。 layout 不必将逻辑索引映射到单个物理地址。它将逻辑索引映射到命名轴上的一组物理坐标。在通常情况下，该集合只有一个元素。当存在复制时，同一逻辑元素具有多个物理位置。

这就是为什么 layout 模型具有三部分：分片、副本和偏移量。分片放置元素。副本将其复制到其他坐标。偏移会改变整个位置。

## layout 示例

下面的示例显示了 API 的基本形式。

TMEM 中的累加器可以直接放置在 TMEM 轴上：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里逻辑行映射到`TLane`，逻辑列映射到`TCol`。在`chap_tmem`中，硬件坐标称为 Lane 和 Col。在 TIRx layout 表示法中，这些硬件轴被写为`TLane`和`TCol`。

block-scaled MMA 比例因子 layout 使用复制：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

该分片在 TMEM 中放置了一个 32 行组。副本以 32 lane 的跨度重复该组四次，因此 32 行组在整个 128 lane TMEM 空间中可见。

Tensor Core register fragment 可以分布在 lanes 和 warps 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一物理轴可以出现多次。在此示例中，两个不同的 iter 都对 `laneid` 做出贡献。没有显式轴的步幅使用默认内存轴 `m`。

在真实的 kernels 中，常见的硬件 layouts 通常来自构造函数：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些构造函数返回普通的 `TileLayout` 对象。它们是一种便利，而不是一个单独的机制。您可以检查返回的 layout，将其与其他 layouts 组合，或者当形状不寻常时手写底层`S[...]`和`R[...]`形式。

## 互动演示

在研究机制之前，先了解一些具体的东西是有帮助的。下面的演示允许您选择预设 layout，编辑逻辑形状和 `S` 或 `R` 术语，选择数据类型和 swizzle 模式，然后单击一个元素以查看哪个或哪些物理坐标拥有它。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ Open the demo full screen ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

该演示非常有用，因为大多数 API 只是演示所显示内容的精确版本。逻辑元素进入 layout。 layout 将其展平，将其拆分为其迭代，在命名轴上累积坐标，然后在需要时应用复制。

## TileLayout

`TileLayout` 是主要仿射 layout 对象。通常使用与文本中使用的相同的符号来编写：

```python
TileLayout(S[shape : strides])
```

`S` 术语是分片规范。您可以将其解读为：采用此形状的逻辑 tile，并使用这些步幅将其放置在命名轴上。

当一个值需要出现在多个位置时，分片规范将使用副本规范进行扩展：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

还可以添加可选的偏移量：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在表面之下，这些片段由 iters 表示。 iter 是一个三元组：

```text
(extent, stride, axis)
```

它描述了在一个指定轴上的跨步行走。范围表示 iter 有多少个位置。步幅表示每一步移动的距离。该轴指示正在更改哪个硬件坐标。

layout 具有三个部分。

### 碎片

分片或 `D` 是由 `S[...]` 构建的部分。它将逻辑索引划分为一个或多个迭代并生成基本物理坐标。

例如：

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个分片 iter。它们的范围是 `8`、`2`、`4` 和 `2`。他们的步幅再次将数据放置在 `laneid`、`warpid`、`laneid` 以及默认内存轴 `m` 上。

这概括了普通的形状和步幅规则。不同之处在于步幅附加到命名的硬件轴而不是单个平面地址。

### 复制品

副本或 `R` 描述同一逻辑元素的附加物理副本。副本迭代独立于逻辑索引。它们枚举硬件空间中的额外偏移量。

例如：

```python
R[2 : 4@warpid]
```

在 `warpid` 轴上创建由四个 warps 分隔的两个副本。

复制并不是为了方便而采取的伎俩。它描述了真实的硬件行为。一些数据跨 warps、lane 或内存区域广播。逻辑到物理的映射自然支持这一点，因为一个逻辑元素可以映射到一组物理坐标。

### 偏移量

偏移量或 `O` 是添加到每个结果的固定坐标。

例如：

```python
5@warpid
```

将 `warpid` 轴上的整个 layout 移动五倍。

偏移量用于将 tile 放置在选定的基坐标处、保留专用区域或描述在同一资源中的另一个 tile 之后开始的 tile。

### 将各个部分组合在一起

layout 按顺序应用这三个部分。

首先，分片计算基础坐标。然后副本扇形协调成零个或多个附加副本。最后，偏移量会移动每个坐标。

对于逻辑坐标 `x`，结果为：

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有副本，`R` 仅包含零偏移量，因此结果是单例集。如果存在副本，则结果包含每个副本位置的一个坐标。

在 TIRx 语法中，完整的 layout 可能如下所示：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右读取，分片放置逻辑块，副本在四个 warp ID 之外创建第二个副本，并且偏移量将整个放置移动到从 `warpid = 5` 开始。

如果迭代器已经构建为对象，则可以直接构建相同的 layout：

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数用户代码使用 `S[...]` 和 `R[...]` 表示法，因为它更接近数学形式。

## 命名轴

layout 中的轴不是匿名尺寸。每个轴命名一个真实的硬件坐标或编译器级放置坐标。

示例包括：

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

`bx`、`by` 和 `bz` 等网格轴在 CTAs 上工作。 `cbx`、`cby` 和 `cbz` 等 cluster 轴将工作置于 CTA cluster 内。 `tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 等螺纹轴描述了 CTA 或 warpgroup 内部的所有权。轴 `m` 是默认的线性记忆轴。 `P` 和 `F` 用于二维暂存器式放置。 `Bank` 命名共享内存组。 `TLane` 和 `TCol` 是 TMEM Lane 和 Col 坐标的 TIRx layout 名称。

轴名称是 layout 的一部分。这很重要，因为具有相同整数值的两个坐标可能意味着不同的硬件事物。 `1@tx` 与 `1@tid_in_wg` 不同。 `1@laneid` 与 `1@TLane` 不同。 layout 使这些含义保持明确。

## 正向映射

评估 layout 意味着获取逻辑坐标并计算其物理着陆位置。 API 方法是：

```python
layout.apply(*coord)
```

对于没有复制的 layout，结果是一个坐标字典。通过复制，结果是一组坐标字典。坐标字典将轴名称映射到整数位置，例如：

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

评估规则有四个步骤。

首先，按行优先顺序展平逻辑坐标。对于逻辑坐标：

```text
x = (x0, x1, ..., xr-1)
```

在逻辑形状内：

```text
(S0, S1, ..., Sr-1)
```

平坦指数为：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

其次，将该扁平索引拆分到分片范围内。如果分片范围是：

```text
(e0, e1, ..., en-1)
```

然后分割产生组件：

```text
c0, c1, ..., cn-1
```

在分片范围上使用相同的行主顺序。

第三，使用其步幅将每个组件累积到其轴上。如果分片 iter `k` 具有范围 `ek`、步长 `sk` 和轴 `ak`，则组件 `ck` 贡献：

```text
ck * sk @ ak
```

对同一轴的所有贡献都加在一起。然后添加偏移量。

第四，应用副本迭代器。每个副本迭代器都会贡献一个独立于逻辑坐标的附加偏移量。如果有多个副本迭代器，则 layout 枚举所有组合。

此规则的一个有用结果是 layout 不需要对输入形状进行硬编码。它需要的是逻辑 tile 的元素总数与分片范围的乘积相同。一旦成立，展平和分割就定义了映射。

## 案例研究：Tensor Core 寄存器块

考虑一个逻辑 `(8, 16)` 区块分布在两个 warps（各 32 个 lane）上。每个 lane 拥有一个小的寄存器片段。寄存器槽由默认内存轴 `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从 `(8, 16)` tile 中获取逻辑元素 `(i, j)`。

行主平坦索引为：

```text
flat = 16 * i + j
```

按分片范围 `(8, 2, 4, 2)` 进行拆分得出：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

分片贡献为：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

添加偏移量 `5@warpid` 后，变为：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

复制品术语：

```python
R[2 : 4@warpid]
```

将 `0` 或 `4` 添加到 `warpid`。所以完整的映射是：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, where r in {0, 1}
m      = j mod 2
```

分片将 tile 放置在 warps 5 和 6 上。副本然后将其复制到 warps 9 和 10。因此，相同的逻辑元素出现在两个 warp 位置中。

此示例说明了模型使用一组物理坐标的原因。复制并不自然地用从物理坐标到逻辑坐标的函数来表示。它很自然地用从一个逻辑坐标到几个物理坐标的函数来表示。

## 案例研究：Blackwell Tensor Memory

相同的 layout 模型适用于内存 layout。这些轴不必是 thread 轴。它们可以是记忆轴。

TMEM 通过硬件 Lane 和 Col 坐标进行寻址。在 TIRx layout 表示法中，这些轴被写为 `TLane` 和 `TCol`。

考虑这个 layout：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果逻辑 tile shape 是 `(2, 128, 112)`，则分割组件只是逻辑坐标本身。对于元素 `(a, l, c)`，映射为：

```text
TLane = l
TCol  = 112 * a + c
```

步幅为 `1@TLane` 的范围为 128 的 iter 填充了 128 TMEM Lane 行。跨度为 `112@TCol` 的范围为 2 的迭代器和跨度为 `1@TCol` 的范围为 112 的迭代器总共覆盖 224 列：

```text
TCol in [0, 224)
```

224 列跨度是有意为之的。 TMEM layouts 不必是 2 的幂。 block-scaled FP8 GEMM 可以选择 224 列累加器，因为完整的 256 列 tile 不会为两个累加器级加上比例因子留下足够的 TMEM 容量。 layout API 可以直接表达该形状。

## 比例因子 layout

上面的累加器 layout 是纯粹的放置。每个逻辑累加器元素映射到一个 TMEM 坐标。 block-scaled MMA 的比例因子不同，因为同一物理组可能需要在多个 warp 窗口中可见。这就是复制发挥作用的地方。

紧凑比例因子 layout 可写为：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

分片在 TMEM 中放置一个 32 行比例因子组：

```text
TLane = r
TCol  = s
```

逻辑比例坐标 `(r, s)`。

副本术语创建由 32 个 lane 分隔的四个副本：

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

因此，32 行组在 TMEM lane 0 到 31、32 到 63、64 到 95 以及 96 到 127 处可见。这是 `warpx4` 广播模式 ({ref}`chap_layout_generations`)。四个 warp 大小的 TMEM lane 窗口中的每一个都看到相同的比例因子组。

在完整的 block-scaled MMA layout 中，该原子与 M 行和 K 比例因子组上的外部迭代组合。多个比例因子也可以打包到一个 32 位 `TCol` 单元中，具体取决于比例因子 dtype。例如，fp8 比例因子可以将四个值打包到一个 32 位列单元中。然后，可选的零步重用和 pipeline 深度迭代器可以描述跨多个 MMA 和双缓冲的规模重用。

重要的是，相同的 `TileLayout` 模型描述了这两种情况。累加器单独放置在 TMEM 中。比例因子是同一 TMEM 地址空间中的复制放置。

## 现成的 layout

大多数 kernels 不会手写每个硬件 layout。 TIRx 为经常出现的 layouts 提供了构造函数。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回由 `tcgen05.mma` 写入的 TMEM 累加器 layout。 `datapath` 参数选择行放置模式。例如，`"D"` 对应于 `M = 128` 身份式 layout，而 `"F"` 对应于 `M = 64` 分散 layout。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回由 `tcgen05.ld` 或 `tcgen05.st` 原子移动的寄存器块 layout。指令形状的示例包括`.32x32b`、`.16x64b`、`.16x128b`和相关形式。在 DSL 级别，这是 warpgroup 分布式 tile。在降低过程中，它变成四个 warp 集体 `tcgen05.ld` 或 `tcgen05.st` 指令，每个 warp 一个，每个 warp 处理自己的 32 个 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回 warpgroup 本地寄存器块，通常在 `tid_in_wg` 上每个 thread 一行。

这些助手的作用是避免手动重写常见的硬件映射。他们不隐藏模型。每个助手返回一个由上述相同的 `S` 和 `R` 片段构建的普通 `TileLayout`。

## SwizzleLayout 和 ComposeLayout

`TileLayout` 是仿射的。它可以表达指定轴上的步幅、复制和偏移。这对于许多放置来说已经足够了，包括 thread 片段、TMEM 切片和紧凑比例因子 layouts。

共享内存 swizzles 需要其他东西。用于避免库冲突的 swizzle 不是仿射步幅模式。它是线性共享内存地址的基于异或的排列。

因此，TIRx 作为单独的 layout 对象保持混合：

```python
SwizzleLayout(...)
```

并将其与 tile layout 组合：

```python
ComposeLayout(swizzle, tile)
```

块 layout 首先产生一个线性内存地址。然后 swizzle 排列该地址。保持这两层分离比将 XOR 排列强制到仿射 layout 模型中更清晰。

## 为什么要调配

共享内存分为 32 个 bank，每个 bank 字保存 4 个字节。当一次访问的 lane 接触同一 bank 中的不同地址时，该访问会因 bank 冲突而串行化。

普通的行主 tile 可能会在结构上造成这种冲突。考虑具有行主 layout 的 `(8, 64)` float16 tile：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 具有线性元素地址：

```text
m = 64 * i + j
```

每行有 64 个 float16 值，即 128 个字节。这正是一个完整的共享 bank 行。如果 warp 向下读取具有固定 `j` 的列，则每一行步长都会前进一个完整的 128 字节行。 Bank 索引重复，因此列读取会跨行折叠到同一 Bank 上。

swizzle 通过使低地址位依赖于高行位来改变这一点。本来会反复登陆同一岸的纵队却分散在不同的岸上。

## Swizzle 变换

`SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S
```

输入是线性元素地址`m`。

`m` 的低 `M` 位保持不变。这保留了一小部分连续的元素。高位被下移为临时值：

```text
x = m >> M
```

然后将`x`的`[S, S + B)`位置处的位组异或为`x`的位组`[0, B)`。然后通过将未更改的低 `M` 位放回来形成混合地址。

等效地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

为了使 layout 格式良好，`S` 必须至少为 `B`。

变换的重点不是改变 tile 中的逻辑元素。它改变了这些元素在共享内存中的位置。 MMA 仍然读取相同的逻辑块。 swizzle 让实体银行模式变得更好。

## 选择 Swizzle 参数

在正常使用中，swizzle 参数是从 dtype 和共享内存 swizzle 模式中选择的。常见的模式有 32 字节、64 字节、128 字节 swizzles。

选择 `per_element` 参数，以便小向量大小的组保持连续。对于 float16，16 字节向量包含 8 个元素，因此：

```text
M = log2(8) = 3
```

对于 128 字节 swizzle，layout 使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这使 16 字节向量组保持完整，同时仍然对较大的共享内存地址模式进行排列，足以打破列组冲突。

大多数代码不应该手动导出这些参数。 dtype 和描述符模式通常决定它们。对于程序员来说重要的是 TIRx layout 中的 swizzle、TMA 描述符和 MMA 期望都匹配。

因此，混合共享内存分配如下所示：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

组成的 layout 被附加到共享内存缓冲区。

## 元素的岸和线

要查看 swizzle 是否有帮助，请将混合后的元素地址转换回共享内存组。

令 `addr` 为混合元素地址，并令 `b` 为元素大小（以字节为单位）。字节地址为：

```text
byte = addr * b
```

银行是：

```text
bank = floor(byte / 4) mod 32
```

128 字节存储行是：

```text
line = floor(byte / 128)
```

对于 float16、`b = 2`，因此银行公式变为：

```text
bank = floor(addr / 2) mod 32
```

这是下面的工作示例中使用的公式。

## 工作示例：`(8, 64)` float16 tile 上的 128B Swizzle

返回行主 float16 tile：

```text
m = 64 * i + j
```

用途：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

变换变为：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

自：

```text
m = 64 * i + j
```

我们可以写：

```text
q = floor(j / 8)
r = j mod 8
```

混合后的地址是：

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在查看 `j = 0` 列。然后是 `q = 0` 和 `r = 0`，所以：

```text
addr = 72 * i
```

对于 float16，银行是：

```text
bank = floor(addr / 2) mod 32
```

所以八行映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

该专栏现在涉及八家不同的银行。冲突消失了。

没有混合，同一列有地址：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每行都位于 bank 0 上，因此访问是串行的。 swizzle 仅更改物理 layout，但这足以将列访问转变为无冲突访问。

此保证取决于 swizzle 的设计方式。 dtype、swizzle 宽度和访问形状必须与 TMA 和 MMA 描述符模式匹配。 128 字节 float16 swizzle 是围绕相关 16 字节行块和 Tensor Core 访问模式设计的。这并不能保证任意共享内存访问不会发生冲突。本章顶部的演示使这一点变得可见：选择一种 dtype 和 swizzle 模式，并观察列折叠到没有 swizzle 的一个 bank 上，然后在应用匹配的 swizzle 后分散在 bank 视图中。

## 设计原理

layout API 遵循三种设计选择。

首先，它支持一般形状。硬件块并不总是 2 的幂。全局张量、共享内存阶段、TMEM 累加器和比例因子缓冲区通常具有来自容量限制或算法选择的形状。 layout 模型将这些形状视为正常形状。

其次，从逻辑坐标到物理坐标的映射。这个方向很重要，因为复制很常见。一个逻辑元素可能存在于多个物理位置。逻辑到物理映射将其直接表示为一组坐标。

第三，硬件轴明确。 layout 不使用匿名维度，并稍后依靠上下文来解释它们。 `tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的差异写入 layout 本身。

合法性和可行性检查并不是 layout 对象单独的工作。 layout 可以说出数据放置的位置。更高级别的 tile primitive 决定给定操作是否可以合法且有效地使用该放置。这种分离使 layout API 保持较小，同时仍为编译器提供足够的信息来进行 dispatch 实际硬件操作。
