(chap_tirx_primer)=
# TIRx 简介

:::{admonition} 概述
:class: overview

- TIRx 是一种 Python DSL，用于在 IR 级别编写 GPU kernels：您可以直接命名硬件，但通过结构化 IR。
- 每个 tile 操作均由三个设计元素控制：*scope*（其中 threads）、*layout*（tile 所在位置）和 *dispatch*（哪个硬件路径）。
- 一个可运行的单 MMA GEMM 显示所有三个；本书的其余部分是这些按比例设计的元素。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，例如 B200）。 TIRx 编译器作为 Apache TVM 轮的 `tvm.tirx` 模块提供；将其与 PyTorch 的 CUDA 版本一起安装：

```bash
pip install apache-tvm
```

使用 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认导入。相同的设置运行书中的每个可运行示例。
:::

第一部分解释了硬件是什么。为了让它计算任何东西，我们需要一种对其进行编程的方法。

我们可以编写原始 CUDA 或 PTX，许多快速 kernels 正是这样编写的。问题在于，很难看到真正决定 kernel 行为的决策：哪个​​ threads 运行某个操作、每个数据块位于何处以及哪个硬件路径执行它。这些选择隐藏在内在参数、地址算术和约定中。

TIRx (Tensor IR neXt) 是一种 Python DSL，它将这三个决策公开：**scope**（哪些 threads 运行操作）、**layout**（操作数 tile 所在的位置）和 **dispatch**（哪个硬件路径执行它）。它仍然直接命名硬件概念，包括 threads、shared memory、Tensor Memory、barriers 和 `tcgen05` MMA。不同之处在于，这些选择现在是结构化 IR，编译器可以降低、检查和调度。

我们不会抽象地介绍这些想法，而是从单个完整的 kernel 开始工作：最小的单个 MMA GEMM。我们先让它运行，然后逐行读回它，看看 scope、layout 和 dispatch 是如何塑造它的，以及 kernel 是如何编译的。 kernel 所依赖的张量 layout 模型是在 {ref}`chap_tirx_layout_api` 中自行开发的，完整的语言功能集在 {ref}`chap_language_reference` 中；这里我们重点关注 kernel 这一块以及三个设计元素。

## 第一个内核：单 MMA GEMM

我们承诺的 kernel 是最小的 GEMM，缩减到仍然使用 Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 x 128 输出 tile，其中 K = 64。整个计算从头到尾表示为一个 `Tx.gemm_async` tile 操作。（该 tile 操作不会映射到单个硬件指令：因为硬件 MMA K-atom 为 16，所以 K=64 tile 会降低为沿 K 步进的 `tcgen05.mma` 指令短序列。 DSL 的要点恰恰是我们编写 tile，而不是手写指令序列。）围绕该操作，kernel 执行通常的杂务：它分配 shared memory (SMEM) 和 Tensor Memory (TMEM)，将 A 和 B 从 global memory 复制到 shared memory，将 tile MMA 发送到 TMEM accumulator，通过 registers 读回该 accumulator，并存储结果。尽管 kernel 很小，但它是我们在 {ref}`chap_gemm_basics` 中攀爬的 GEMM 阶梯的第一步，它会带着完整的演练返回。

每个 TIRx kernel 都是从相同的进口产品开始的，因此值得一睹它们的风采：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们将 kernel 包装在一个小型构建器 `hgemm_v1(M, N, K)` 中，该构建器采用问题形状并返回 `PrimFunc`。对于我们选择的形状 `M=N=128, K=64`，启动恰好包含一个输出 tile，这使得第一个版本足够简单，可以一次性阅读：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
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
        phase_mma: T.int32 = 0
    
        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

在阅读 kernel 之前，让我们先确保它能正常工作。我们编译它并根据 torch 参考检查其输出。我们不必详细说明确切的架构：拱门（e.g.`sm_100a`）是从设备自动检测到的，因此目标 `"cuda"` 就足够了，而 `tir_pipeline="tirx"` 就是选择 TIRx 下降 pipeline 的原因。编译后，`ex.mod(...)` 直接采用 torch 张量，中间无需手动转换。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly, the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## 范围、layout、调度

现在 kernel 运行了，我们可以读回它并询问它的行实际上决定了什么。这样看来，整个 kernel 是三个设计元素的一组选择。其中的每个操作都回答相同的三个问题：*谁*运行它、*在哪里*存储数据以及*如何*执行，而这三个答案正是 scope、layout 和 dispatch。本节的其余部分将逐一介绍设计元素；下面的交互式演示可以让您看到每个设计元素控制哪些行。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：单击范围/layout/调度以突出显示每个设计元素控制的 kernel 的线条。*

使用演示时，请注意三个问题：

- **范围：谁运行操作？** `Tx.cta.copy(...)` 的范围是 CTA，因此所有 128 个 threads 都有助于 GMEM -> SMEM 副本。 `Tx.gemm_async(...)` 由当选的 thread 发布一次，因为每个降低的 `tcgen05.mma` 指令已经是合作的 MMA 发布。 `Tx.wg.copy_async(...)` 是 warpgroup 范围的，因此 warpgroup 的 128 threads 将 TMEM 回读逐行拆分。
- **layout：每个 tile 位于哪里？** A 和 B 使用 `tcgen05.mma` 期望的混合 SMEM layouts。累加器位于 `TLane`/`TCol` layout 下的 TMEM 中。寄存器回读视图将行映射到 `tid_in_wg`，因此每个 warpgroup thread 拥有一个行片段。
- **调度：哪个硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core 路径。复制操作也有 dispatch 选择：第一个 kernel 使用普通的 thread 副本，后面的 GEMM 步骤将这些副本交换为 TMA，而不更改周围的 scope 或 layout。

**与您的代理一起尝试**：从第一个 kernel 中选择三行：一份副本、一份 MMA 和一份 TMEM 回读。要求它用 scope、layout 和 dispatch 标记每一行，然后检查答案是否与代码中的守卫、缓冲区 layouts 和 `dispatch=` 参数匹配。

## 编译的工作原理

我们已经编译了上面的 kernel 来测试一下；现在我们更仔细地看看这一步的作用。配方很简短：将 `PrimFunc` 包裹在 `IRModule` 中，然后交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这将运行 TIRx 降低 pipeline 并返回您直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少在概要上，`tir_pipeline="tirx"` 的启动机制是值得了解的。 pipeline 的中央 lane `LowerTIRx` 根据其 scope / layout / dispatch 合约解析每个 tile 原语：这就是我们刚刚讨论的三个设计元素实际上兑现为指令的地方。之后，通常的主机/设备拆分和最终步骤会生成可启动模块。如果您愿意，还可以在 `with target:` 块内进行编译，这让 kernel 拾取周围的目标上下文。

此流程的一个很好的特性是您不会隐藏任何内容：可以在两个级别检查结果。您可以使用 `.show()` 或 `.script()` 读取 IR 本身，并且可以读取编译器最终直接从编译模块发出的 CUDA C。

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(ex.mod.imports[0].inspect_source())
```

这只是一个草图。有关完整的降低故事，涵盖所有 lane、如何解析 tile primitive dispatch 以及如何完成主机/设备拆分，请参阅 {ref}`chap_arch`。

## 下一步去哪里

一个 kernel 足以满足 scope、layout 和 dispatch 并查看它们的编译和运行。三个设计元素中的每一个以及 kernel 本身都将进入一个更进一步的章节：

- {ref}`chap_tirx_layout_api`：上面的操作数和累加器放置所基于的张量 layout 模型（`TileLayout`，命名轴，swizzle）。如果 layout 设计元素感觉像是三个元素中最神秘的，就从这里开始吧。
- {ref}`chap_language_reference`：完整的语言功能集，涵盖解析器实用程序、数据类型、缓冲区和内存、控制流以及 thread 同步，适合当您想要完整的词汇而不是浏览时。
- {ref}`chap_gemm_basics`：此 kernel 作为 GEMM 优化路径的步骤 1，通过 K 循环积累、空间平铺、TMA 和 warp specialization 构建。如果您想看到相同的三个设计元素扩展到真正的 kernel，那么这自然是下一站。
