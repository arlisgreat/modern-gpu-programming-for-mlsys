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

.. _chap_language_reference:

TIRx Language Reference
=======================

这里整理了编写 TIRx device kernels 所需的完整 language-feature 集合，并从
:ref:`chap_tirx_primer` walkthrough 中拆分出来：parser utilities、data types
and expressions、buffers and memory、control flow，以及 thread synchronization。
当你需要确认某个 feature 的精确写法或语义时，查这一组页面。

.. toctree::
   :maxdepth: 1

   cuda/parser_utils
   cuda/data_types
   cuda/buffers
   cuda/control_flow
   cuda/threads_sync
