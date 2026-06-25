# MLSys 的现代 GPU 编程

本书以渐进方式教授现代 GPU kernel 编程：**了解 GPU 硬件 → 学习编程 → 编写 state-of-the-art kernels。** 它把 Blackwell 类 GPU 作为真正的主题：它的内存层次结构、Tensor Memory、Tensor Core 和异步数据移动引擎、warpgroups 与 clusters。承载这些内容的是 **TIRx** (Tensor IR neXt)，一种用于在 IR 级别编写 GPU kernels 的 Python DSL。

📖 **在线阅读：<https://mlc.ai/modern-gpu-programming-for-mlsys/>**

## 里面有什么

- **Part I — 了解 GPU。** 执行和内存模型、性能模型（roofline、重叠）、深入研究数据 layout、内存和计算引擎（TMA、Tensor Memory、Tensor Cores)、异步协调和高级调度(CLC)。
- **Part II — 使用 TIRx 编程 GPU。** 通过一个可运行的单 MMA GEMM — scope、layout 和 layout 介绍 TIRx dispatch，以及编译如何工作 - 加上张量 layout 模型（`TileLayout`，命名轴，swizzle）。
- **Part III — GEMM：平铺至 SOTA。** 通过 TMA pipeline 构建的平铺 GEMM、persistent scheduling、warp specialization 和 2-CTA clusters。
- **Part IV — FlashAttention 4。** 从 Part III 技术构建的完整 Attention kernel：两个 MMA，其间带有 softmax、online softmax rescaling、causal masking 和 GQA。
- **Reference。** TIRx 语言参考和编译器内部结构。

## 在本地构建本书

这本书是一个 [Sphinx](https://www.sphinx-doc.org/) 站点（Markdown/MyST + reStructuredText）：

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

### 预览

```bash
python -m http.server -d _build/html 8000
```

打开<http://localhost:8000>。服务器在远程计算机上运行，​​因此转发端口 - `ssh -L 8000:localhost:8000 user@your-server` - 然后在本地打开 URL。 （VS Code 远程 SSH 会自动转发它。）

## 运行 kernels（需要 Blackwell GPU）

本书中的 kernels 以 Blackwell（`sm_100a`）为目标，因此运行它们需要 Blackwell GPU（例如 B200）、TIRx 编译器和 CUDA PyTorch 的构建。

**1. 安装 TIRx 编译器。** 它作为 Apache TVM 轮的 `tvm.tirx` 模块提供：

```bash
pip install apache-tvm
```

核实：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

**2. 使用与您的 GPU 匹配的 PyTorch** 安装 PyTorch**（用于示例输入和参考检查） — 请参阅 <https://pytorch.org>。

**3.  （可选）参考 kernels。** 完整的 GEMM 和 FlashAttention 4 kernels 位于配套的 `tirx-kernels` 包中（`pip install -e .` 来自结帐）；使用 e.g.、`python -m tirx_kernels.test --kernel fp16_bf16_gemm` 运行它们。

TIRx 通过 Python 源检查解析 kernel 源，因此示例应该位于文件或笔记本单元中，而不是位于 `python -c` 内。

## 部署

每次对 `main` 的推送都是由 GitHub Actions (`.github/workflows/build_deploy.yaml`) 自动构建并发布到 <https://mlc.ai/modern-gpu-programming-for-mlsys/>。
