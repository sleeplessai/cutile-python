# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import compile_tile
from cuda.tile._ir.ops import Loop, MakePartitionView
from cuda.tile.compilation import ArrayConstraint, KernelSignature, CallingConvention


# MakePartitionView must not be hoisted away from its consumer (version < V_13_3)
@pytest.mark.parametrize("version", [BytecodeVersion.V_13_1, BytecodeVersion.V_13_2])
def test_partition_view_grouped_with_consumer(version):
    def kernel(x):
        for i in range(10):
            n = ct.num_tiles(x, 0, shape=(1,))
            v = ct.load(x, i, shape=(1,))
            ct.store(x, i, v + n)

    x_constraint = ArrayConstraint(dtype=ct.float32, ndim=1,
                                   stride_lower_bound_incl=0, alias_groups=(),
                                   may_alias_internally=False)
    sig = KernelSignature([x_constraint], CallingConvention.cutile_python_v1())
    [root_block] = compile_tile(kernel, [sig], bytecode_version=version,
                                return_final_ir=True, return_cubin=False).final_ir

    root_pvs = [op for op in root_block if isinstance(op, MakePartitionView)]
    assert len(root_pvs) == 1, "Expected 1 MakePartitionView hoisted to root (for num_tiles)"

    loop = next(op for op in root_block if isinstance(op, Loop))
    body_pvs = [op for op in loop.body if isinstance(op, MakePartitionView)]
    assert len(body_pvs) == 2, "Expected 2 MakePartitionViews inside loop (for load and store)"
