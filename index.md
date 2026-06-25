# MLSys 的现代 GPU 编程

机器学习系统是现代人工智能工作负载的核心。在这些系统中，性能通常取决于少数 GPU kernels 的质量。 Attention kernels、LLM prefill 和 decode kernels、低精度 block-scaled GEMM、fused MoE 层和其他大型 fused kernels，都会直接影响训练和 serving 的端到端速度。

然而，为了使这些 kernels 更快，我们需要的不仅仅是一系列优化技巧。现代 GPUs 不再是相同旧设计的简单变体。最近的架构引入了更丰富的内存空间、新的访问模式和日益专业化的执行单元。为了对它们进行良好的编程，我们既需要清晰的硬件思维模型，又需要对高性能 kernels 的构建方式有实际的了解。本书就是关于两者的发展。

本书遵循一个简单的流程：首先了解 GPU 硬件，然后学习我们将使用的编程模型，最后一步步构建 state-of-the-art kernels。我们的主要目标是 Blackwell 一代，我们的主要运行示例是快速矩阵乘法（GEMM）和 FlashAttention。在此过程中，我们还将研究 GPU 优化背后的核心要素：数据 layout、异步数据移动和异步协调。

该材料源自卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列。为了使这些想法更容易学习和更容易运行，本书使用 **TIRx** Python DSL 一步步构建真正的 GPU kernel 示例。 TIRx 与硬件保持密切联系，这使我们能够推理低级控制，同时仍然通过可运行代码进行学习。

## 本书的结构

- **Part I，了解 GPU。** 这部分介绍了 GPU 的整体组织、快速编写 kernels 的一般秘诀，以及数据 layout、异步内存操作和协调等关键概念。它构建了本书其余部分所依赖的硬件直觉。
- **Part II、TIRx Overview。**这部分介绍了 TIRx 的关键元素，它们作为全书代码示例的基础。
- **Part III、GEMM：平铺至 SOTA。** 优化平铺 GEMM 的完整指南，通过 TMA pipeline、persistent scheduling、warp specialization 和 2-CTA 构建 clusters。
- **Part IV、FlashAttention 4。** 从 Part III 技术构建的完整 Attention kernel：两个 MMA，其间带有 softmax、online softmax rescaling、causal masking 和 GQA。
- **Reference。** TIRx 语言参考和编译器内部结构。

```{toctree}
:caption: Part I, 理解 GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: Part II, TIRx 概览
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "Part III, GEMM: Tiled to SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: Part IV, FlashAttention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

appendix/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
tirx_guide/language_reference/index
```
