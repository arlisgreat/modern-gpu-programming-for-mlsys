(chap_warp_spec_debug)=
# 调试 Warp 专用内核

GEMM {ref}`chap_gemm_advanced` 中的步骤 7-9 与 TMA load、`tcgen05` MMA 和 TMEM/SMEM 写回重叠。相同的调试方法适用于 FlashAttention 切换：识别角色，识别每个角色拥有的存储，然后根据该模型验证生成的 CUDA。

不要从重写 kernel 开始。首先确保运行有效，然后检查生成的 CUDA。排除环境和编译时问题后，这些 kernels 中的运行时故障通常会减少为中断的切换：未初始化的 barrier、错误的到达计数、隐藏在角色防护内的集合、过时的 barrier 阶段或在生产者使其写入可见之前重用的存储。

## 调试内核之前

首先排除运行时上下文：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernels 目标为 Blackwell (`sm_100a`)。如果 Python 导入过时的 TVM 签出，或者 GPU 不是 Blackwell 类，请在更改 kernel 之前修复该问题。然后运行 ​​kernel 的最小正确性检查，例如 `run_correctness()`，然后再查看性能。

## 调试工作流程

1. 以仍然失败的最小形状重现失败。如果失败是非法内存访问，请在下次运行之前重新启动 Python。
2. 如果编译失败，请在读取运行时同步代码之前检查已安装的 API、目标、`dispatch=`和缓冲区 scopes。
3. 保存 `inspect_source("cuda")` 输出。在再次阅读 Python 之前，搜索角色防护、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的 kernel 路径写入角色/存储/切换/生命周期表。
5. 根据该表检查生成的 CUDA：角色分支之前的 barrier inits、预期的 TMA 生产者、MMA issuer(s)、写回 group(s)，并且仅 warpgroup 分支内没有 CTA 范围的集合。
6. 将运行分类为死锁、崩溃、错误结果或正确但缓慢的运行，然后使用下面的匹配部分。
7. 一次更改一个切换：初始化计数、到达/等待阶段、角色防护、围栏、TMA store 耗尽、TMEM 分配/释放或切片 scheduler 提前。
8. 在测量性能之前重新运行正确性。

## 什么转移

对于任何异步 kernel，在更改代码之前制作一个小工作表：

| 项目 | 写下什么 |
|---|---|
| 角色 | 发出每个异步操作的确切 threads、warps、warpgroups 或 CTAs。 |
| 存储 | 每个步骤中每个 tile 的实时位置：GMEM、SMEM、TMEM 或寄存器。 |
| 切换 | 生产者、消费者、信号对象、到达计数、阶段以及使数据可见的栅栏或漏极。 |
| 终身 | 每个存储槽可以被重用、读回或释放的最早点。 |

然后对照工作表验证生成的 CUDA：

- 角色守卫与角色表相匹配。
- barrier 初始化出现在受保护的角色分支之前。
- 集体行动不会被 lane、warp 或 warpgroup 守卫意外缩小。
- 到达/等待阶段与切换表匹配。
- TMA store 耗尽、TMEM 释放和 SMEM 重用仅在生命周期表表明它们合法之后才会发生。

对 TMA->MMA->写回 GEMM pipeline 以及 FlashAttention 中的分数/softmax/值/校正切换使用相同的工作表。

## 如果编译失败

在调试运行时同步之前修复编译时失败：

| 症状 | 可能的区域 | 首先检查 |
|---|---|---|
| 未知 TIRx API 或属性错误 | 安装的轮子与教程代码不匹配 | 打印`tvm.__file__`和`tvm.__version__`；将 API 名称与 {ref}`chap_language_reference` 进行比较。 |
| 不支持 `dispatch=` | 选定的目标或基元不支持该路径 | 检查`dispatch`参数和目标能力；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| 缓冲区 scope 不匹配 | 通过错误的硬件路径使用缓冲区 | 检查工作表的存储行：TMEM 必须通过`tcgen05`访问，TMA 操作数必须使用兼容的 GMEM/SMEM layouts。 |
| 编译成功但生成的 CUDA 缺少预期路径 | 派遣并没有按照您预期的方式降低 | 在更改算法之前，检查生成的 CUDA 中的 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成的代码

对于任何已编译的 kernel，请保存 CUDA，以便您可以搜索和比较它：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成的代码将 TIRx 构造映射到 CUDA，如下所示：

| TIRx | 生成 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部防护罩 | `((int)threadIdx.x) < 1`（仅 CTA thread 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

在阅读完整的 kernel 之前扫描这些字符串：

| 生成 CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单个 CTA-thread 防护，常 barrier 初始化 |
| `mbarrier_init` | barrier 初始化存在并出现在角色分支之前 |
| `tcgen05` | 生成了 Tensor Core 路径 |
| `cp.async.bulk.tensor` | 副本降为 TMA |
| `cta_sync();` | CTA 全护栏；它不能位于 `wg_id` 分支内 |

## 步骤 7 参考骨架

正确编译的 Step 7 kernel 具有此顶级形状。为了便于阅读，下面的守卫都写上了角色名称；在生成的 CUDA 中，从上表中搜索对应的表达式。

```c
// (1) Barrier inits: top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc: WG0 warp 0, all lanes of the issuing warp
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup: issuing warp, no lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

在更改算法之前检查这些：

- barrier 初始化位于顶层，而不是在 `wg_id` 防护内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有一个 warp 守卫，但没有 lane 守卫；发行 warp 的所有泳道参与。
- TMA 和 MMA 循环均迭代 `K_TILES` 次。
- 阶段初始化为生产者=`1`，消费者=`0`。

## 症状图

从症状开始，但将其视为线索而不是最终诊断：

| 线索 | 可能的区域 | 首先检查 |
|---|---|---|
| 内核挂起，然后运行时报告未指定的启动失败 | 死锁 | barrier 初始化放置、到达计数、`cta_sync()` 放置和 `next_tile()` 参与 |
| 非法内存访问、XID 或以后不相关的 CUDA 调用也会失败 | 崩溃/中毒上下文 | 重新启动 Python，然后检查指针范围、存储寿命和集体参与 |
| 128 行或平铺大小的条纹中出现错误的行 | 同步竞赛或 tile 索引不匹配 | 生产者/消费者阶段、scheduler 提前以及 warpgroup 拥有每个行条带 |
| `NaN` 或明显无效的值 | 描述符、操作数设置或未初始化的累加 | SMEM/TMEM 描述符设置、swizzle/layout 和累加器初始化 |
| 有限但模式化的错误价值观 | 陈旧或部分可见的数据 | 缺少栅栏、缺少 TMA store 排水管或在生命周期表允许之前重复使用存储 |
| 输出正确，但没有预期的加速 | 调度或资源问题 | 生成 CUDA 路径、pipeline 深度、占用率和寄存器溢出 |

## 何时重新启动 Python

CUDA 错误并不总是会自行清除。在发生非法内存访问、XID 或“CUDA 上下文中毒”错误后，后续不相关的调用（例如 `torch.randn`）可能会继续失败。在测试下一个修复之前重新启动 Python 进程，否则您可能正在调试之前的崩溃而不是当前的代码。

## 僵局

按顺序检查这些：

- **到达计数与初始计数不匹配。**常见情况：`MBarrier.init(128)`，但 `arrive` 由 `if warp_id == 0: if lane_id == 0:` 保护，因此只有 1 个 thread 到达，并且等待永远不会返回。

  | barrier | init(count) | 谁来了 | 到达航班 |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA engine 通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar`（mma->tma，mma->ld） | 1 | MMA warp 通过 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | 所有 WG0 threads 通过 `arrive` | 128 |

- **barrier 初始化嵌套在 `wg_id` 保护内部。** `.init()` 降低到 `if threadIdx.x < 1:`，这意味着 CTA thread 0。 CTA thread 0 位于 WG0 中，因此`if wg_id == 1:` 阻止每个 thread 运行 init。 Inits 必须位于顶层； `grep mbarrier_init` 在`inspect_source()`中进行验证。

- **`cta_sync()` 在 warpgroup 分支内。** `cta_sync` 是 `__syncthreads()`，它需要所有 CTA threads。在`if wg_id == 0:`内部，WG1 永远不会到达它。将 `T.cuda.warpgroup_sync(10)` 用于单个 warpgroup barrier。

- **`tile_scheduler.next_tile()` 被某些消费者跳过 - warpgroup threads。** scheduler 跟踪每个 thread 状态； threads 跳过它可以永远循环。

- **TMA 和 MMA 在 K 区块计数上存在分歧。** 如果 MMA 执行 `K_TILES - 1` 而不是 `K_TILES`，则势垒 phase 漂移并且第二个外部区块死锁。

- **`PipelineState` 初始阶段错误。** 生产者从 `phase=1` 开始，因此第一个等待通过；消费者从 `phase=0` 开始，因此第一个等待会阻塞。如果两者从同一阶段开始，则第一次切换可能会立即陷入僵局。

## 崩溃和上下文中毒

常见原因：

- **`pool.commit()` 之后的 `pool.alloc`。** barrier 包装器在内部调用 `alloc`。正确顺序：`tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **带有 lane 防护装置的 `tcgen05.alloc` 或 `tcgen05.dealloc`。** 签发的 warp 必须参与所有 lane。 `if lane_id == 0:` 运行一个 thread，这是未定义的行为。
- **`tcgen05.dealloc` 之前缺少 `cta_sync()`。** TMEM 在写回仍在读取时被释放。
- **超出范围的 GMEM 或 SMEM 访问。** 缩小到一个 tile，检查 scheduler 的 `m_idx` / `n_idx`，并检查当前形状是否是 kernel 的 tile 或 cluster tile 的倍数。

## 错误的结果

在猜测之前按模式对错误输出进行分类。整行条纹通常指向生产者/消费者阶段、切片索引或角色所有权不匹配。 `NaN` 输出通常指向描述符设置、操作数设置或未初始化的累加。有限但模式化的错误值通常意味着消费者读取了旧的 tile、部分写入的 tile 或存储尚未耗尽的数据。

- **`tcgen05.commit` 在 `elect_sync` 之外。** 所有 32 个 threads 创建提交组； 31 个空组立即向 mbarrier 发出信号。 TMA 可以在 MMA 读取之前覆盖 SMEM。
- **在 TMA store 之前缺少 `fence.proxy_async("shared::cta")`。** TMA 引擎可能看不到来自 threads 的 SMEM 写入。
- **在 TMA store 之后缺少 `cp_async.bulk.commit_group()` 加上 `wait_group(0)`。** 下一个 tile 可以在 store 排水之前重复使用 Dsmem。
- **持久 kernel 在小尺寸（例如 1024x1024）下会间歇性失败。**较大尺寸可能会掩盖较长 K 循环的竞争。重新检查 tile 之间的阶段重置和 TMA 存储提交/等待。
- **`fence.after_thread_sync()` 通常不是修复方法。** MMA 完成 mbarrier 已经带有释放获取语义。第 8 步和第 9 步保守地将其添加到回写边缘，即 `mma2ld.wait` 之后和第一个 `tcgen05.ld` 之前；不要将其常规添加到 TMA-to-MMA 边缘。

## 正确但缓慢

如果输出正确但性能远低于预期，请使用相同的检查循环：

| 线索 | 可能的区域 | 首先检查 |
|---|---|---|
| 生成的 CUDA 没有`cp.async.bulk.tensor` | 副本没有降低到 TMA | 检查 `dispatch="tma"`、目标能力和操作数 layout |
| 生成的 CUDA 没有`tcgen05`路径 | MMA 未降低至 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、目标能力和操作数 layouts |
| TMA 和 MMA 不重叠 | pipeline 太浅或阶段序列化生产者/消费者 | 检查生成的 CUDA 中等待/到达/提前的顺序 |
| 小形状正确性好，但大形状速度差 | 记录溢出、占用或分段缓冲区压力 | 查看编译器资源报告；减少切片大小、块写回或降低 pipeline 深度 |

## 提交一个好问题

如果经过上述检查后故障仍然存在，请在 [Apache TVM GitHub 存储库](https://github.com/apache/tvm/issues) 上提交问题之前减少故障。包括：

- `tvm.__file__` / `tvm.__version__` 输出和 GPU 功能；
- 重现故障的最小形状；
- 失败是否是编译时、死锁、崩溃、错误结果或正确但缓慢；
- 最小的 kernel 或笔记本单元及其正确性检查；
- 保存的 `inspect_source("cuda")` 输出，或显示可疑守卫、barrier 或 dispatch 路径的最小摘录。
