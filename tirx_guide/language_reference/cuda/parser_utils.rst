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

Parser utilities
================

有几个 helper 会在 **parse time** 起作用，也就是 TVMScript 被转换为 TIRx
的时候。它们让你可以 inline Python 计算出的值，抽出可复用片段，并打包
parser-side state。

``T.meta_var`` — inline 一个 Python value
------------------------------------------

``T.meta_var(x)`` 告诉 parser 把 ``x`` （在 **Python** 中算出的值）当作
compile-time *meta* value，并直接 inline 到 IR 中，而不是把它 parse 成 script
variable。这样可以避免一次性 local，也支撑 metaprogramming：对 meta value
写普通 Python ``for`` ，会在 parser 中 unroll。

.. code-block:: python

    n = T.meta_var(4)              # n is a Python int, inlined
    for j in range(n):            # unrolled at parse time
        acc[0] = acc[0] + A[tx, j]

``@T.inline`` — inline functions
--------------------------------

``@T.inline`` 定义一个函数，其 body 会在 parsing 期间 **inline 到每个 call
site**，生成代码中不会出现 call。它遵循 Python 的 lexical (LEGB) scoping 和
late binding，因此参数会 shadow 外层变量：

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` — parser-side state objects
---------------------------------------------

``@T.meta_class`` 标记一个普通 Python class；它的 **instances 是 parser meta
values** 。这些 instance 的 fields 可以保存 buffers 和 scalars，因此你可以把相关
allocations 和 state 组合到一个对象里，并在 kernel body 中使用。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
    # ... s.buf[i] ...

这适合把 kernel 的 pipeline state（barriers、accumulators、scratch views）分组，
不用把许多分散的 locals 一路传过 body。

``T.constexpr``
---------------

``T.constexpr`` 标记 compile-time kernel parameter，并由 ``@T.jit`` 的
``.specialize(...)`` bake in。细节见 :ref:`chap_tirx_primer`。
