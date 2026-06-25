..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Control flow
============

Control flow 包括 ``if`` 、loop family 和 ``while`` ，它们都会映射到直接对应的
CUDA。

if
--

Python ``if`` / ``else`` 会变成 CUDA ``if`` / ``else`` 。可以用 thread/lane
comparison 做 guard，也可以用 ``T.ptx.elect_sync()`` 选出一个 issuing thread：

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # one elected lane (e.g. to issue TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

如果只是 expression-level choice（不需要 branch），使用
``T.if_then_else(cond, a, b)`` 。它会降低成 ternary，因此不会引入 control-flow
divergence：

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

Uniform vs. divergent control flow
----------------------------------

``if tx < 128`` 这样的 per-thread guards 对普通工作没问题，但 **collective**
operations 必须被它们同步范围内的每个 thread *uniformly* 到达。

例如， ``T.cuda.cta_sync()`` 映射到 ``__syncthreads()`` ，它要求 thread block
里的所有 threads 都到达。它绝不能放在 thread-divergent 或 warpgroup-divergent
branch 中：如果把它放在 ``if wg_id == 0:`` 里，其他 warpgroups 永远不会到达，
kernel 会 deadlock。只有一个 warpgroup 需要同步时，使用 warpgroup-scoped
``T.cuda.warpgroup_sync(id)`` （见 :ref:`chap_gemm_advanced` 和
:doc:`threads_sync`）。

同样的谨慎也适用于 barrier setup。 ``mbarrier`` 的 ``.init()`` 会降低为
single-thread guard（ ``if (threadIdx.x < 1)`` ）。如果把它嵌在另一个 divergent
branch 里，barrier 可能保持 uninitialized，进而导致未定义的 launch failures。

loop
----

Loop 有四种形式；普通 Python ``range`` 会变成 ``T.serial`` ：

- ``T.serial(n)`` — sequential loop（ptxas 仍然可能 unroll 它）。
- ``T.unroll(n)`` — fully unrolled（展开成 straight-line statements）。
- ``T.vectorized(n)`` — vectorized loop。
- ``T.grid(*extents)`` — nested loop nest。

``break`` / ``continue`` 可以在 loops 内使用。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 则会展开成四条 straight-line statements，不留下 loop。

while
-----

``while`` loop 会一直运行到 condition 为 false。使用 mutable scalar counter（见
:doc:`buffers`）：

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它会降低为带 early-exit ``break`` 的 ``while (1)`` （counter 是一个 one-element
register buffer）：

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
