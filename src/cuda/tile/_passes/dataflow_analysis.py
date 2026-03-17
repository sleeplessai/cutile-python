# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, TypeVar, Generic, Any

from cuda.tile._ir.ir import Var, Block
from cuda.tile._ir.ops import Assign, GetArrayListItem, \
    Loop, IfElse, Continue, Break, EndBranch, MakePartitionView, PointerOffset, \
    TileBroadcast, TileReshape, MakeTensorView, MakeListView, AssumeDivBy, TileReduce, TileScan
from cuda.tile.compilation._signature import ParameterConstraint, \
    ArrayConstraint, ListConstraint, ScalarConstraint


ALIAS_UNIVERSE = -1
ALIAS_EMPTY = 0
AliasSet = int


@dataclass(frozen=True)
class DataPredicate:
    alias_set: AliasSet

    def unify(self, other: "DataPredicate") -> "DataPredicate":
        return DataPredicate(alias_set=self.alias_set | other.alias_set)


@dataclass
class DataflowResult:
    predicates: Dict[str, DataPredicate]

    def __getitem__(self, var_name: str) -> DataPredicate:
        return self.predicates[var_name]


def dataflow_analysis(root_block: Block,
                      parameter_constraints: Sequence[tuple[tuple[Var, ...], ParameterConstraint]]
                      ) -> DataflowResult:
    state = _State(_Tracker(), _Tracker())
    alias_set_mapper = _AliasSetMapper()
    for flat_params, constraint in parameter_constraints:
        if isinstance(constraint, ArrayConstraint):
            base_ptr = flat_params[0]
            state.tracker.update(base_ptr,
                                 DataPredicate(alias_set=alias_set_mapper(constraint.alias_groups)))
            state.list_array_tracker.update(base_ptr, ALWAYS_TRUE_AGG_PREDICATE)
        elif isinstance(constraint, ListConstraint):
            assert isinstance(constraint.element, ArrayConstraint)
            base_ptr = flat_params[0]
            state.tracker.update(base_ptr,
                                 DataPredicate(alias_set=alias_set_mapper(constraint.alias_groups)))
            element_alias_set = alias_set_mapper(constraint.element.alias_groups)
            state.list_array_tracker.update(
                    base_ptr,
                    _AggregatePredicate({BASE_PTR: DataPredicate(alias_set=element_alias_set)}))
        elif isinstance(constraint, ScalarConstraint):
            [only_param] = flat_params
            state.set_always_true(only_param)
        else:
            assert False

    _analyze_aliases_in_block(root_block, state, None, None)

    while state.dirty:
        state.reset_dirty()
        _analyze_aliases_in_block(root_block, state, None, None)

    return DataflowResult(state.tracker.finalize())


class _AliasSetMapper:
    def __init__(self):
        self._bit_seq = map(lambda x: 1 << x, itertools.count())
        self._mapping: dict[str, int] = dict()

    def __call__(self, alias_groups: Sequence[str] | None) -> AliasSet:
        if alias_groups is None:
            return ALIAS_UNIVERSE
        elif len(alias_groups) == 0:
            return next(self._bit_seq)
        else:
            ret = 0
            for ap in alias_groups:
                bit = self._mapping.get(ap)
                if bit is None:
                    bit = next(self._bit_seq)
                    self._mapping[ap] = bit
                ret |= bit
            return ret


ALWAYS_TRUE_PREDICATE = DataPredicate(alias_set=ALIAS_UNIVERSE)


class _AggregatePredicate:
    def __init__(self, items: Dict[Any, DataPredicate]):
        self._items = items

    def __getitem__(self, key):
        return self._items.get(key, ALWAYS_TRUE_PREDICATE)

    def unify(self, other: "_AggregatePredicate") -> "_AggregatePredicate":
        all_keys = set(self._items.keys()) | other._items.keys()
        unified_items = {k: self[k].unify(other[k]) for k in all_keys}
        return _AggregatePredicate(unified_items)

    def __eq__(self, other: "_AggregatePredicate") -> bool:
        all_keys = set(self._items.keys()) | other._items.keys()
        return all(self[k] == other[k] for k in all_keys)


ALWAYS_TRUE_AGG_PREDICATE = _AggregatePredicate({})


BASE_PTR = "base_ptr"


P = TypeVar("P")


class _Tracker(Generic[P]):
    def __init__(self):
        self.dirty = False
        self._predicates: dict[str, P] = dict()

    def __getitem__(self, var: Var) -> P:
        return self._predicates[var.name]

    def update(self, var: Var, pred: P):
        old_pred = self._predicates.get(var.name)
        if old_pred is None:
            new_pred = pred
        else:
            new_pred = old_pred.unify(pred)
            if new_pred == old_pred:
                return
        self.dirty = True
        self._predicates[var.name] = new_pred

    def finalize(self) -> dict[str, P]:
        return self._predicates


@dataclass
class _State:
    tracker: _Tracker[DataPredicate]
    list_array_tracker: _Tracker[_AggregatePredicate]

    def propagate(self, src: Var, dst: Var):
        self.tracker.update(dst, self.tracker[src])
        self.list_array_tracker.update(dst, self.list_array_tracker[src])

    def set_always_true(self, var: Var):
        self.tracker.update(var, ALWAYS_TRUE_PREDICATE)
        self.list_array_tracker.update(var, ALWAYS_TRUE_AGG_PREDICATE)

    @property
    def dirty(self):
        return self.tracker.dirty or self.list_array_tracker.dirty

    def reset_dirty(self):
        self.tracker.dirty = False
        self.list_array_tracker.dirty = False


def _analyze_aliases_in_block(block: Block,
                              state: _State,
                              innermost_loop: Loop | None,
                              innermost_branch: IfElse | TileReduce | TileScan | None):
    for op in block.operations:
        if isinstance(op, Assign):
            state.propagate(op.value, op.result_var)
        elif isinstance(op, AssumeDivBy):
            state.propagate(op.x, op.result_var)
        elif isinstance(op, GetArrayListItem):
            # TODO: more granular array list get item alias analysis
            # Propagate to the base pointer of the array
            state.tracker.update(op.result_vars[0], state.list_array_tracker[op.x][BASE_PTR])
            state.list_array_tracker.update(op.result_vars[0], ALWAYS_TRUE_AGG_PREDICATE)
            for v in op.result_vars[1:]:
                state.set_always_true(v)
        elif isinstance(op, MakeTensorView):
            state.propagate(op.base_ptr, op.result_var)
        elif isinstance(op, MakePartitionView):
            state.propagate(op.array, op.result_var)
        elif isinstance(op, MakeListView):
            state.propagate(op.base_ptr, op.result_var)
        elif isinstance(op, PointerOffset):
            state.propagate(op.pointer, op.result_var)
        elif isinstance(op, TileBroadcast | TileReshape):
            # Needed for tiles of pointers produced by gather/scatter
            state.propagate(op.x, op.result_var)
        elif isinstance(op, Loop):
            if op.is_for_loop:
                state.set_always_true(op.induction_var)

            for init, body, result in zip(op.initial_values, op.body_vars, op.result_vars,
                                          strict=True):
                # Loop initial values flow into body values.
                state.propagate(init, body)

                # `For` loop initial values can flow into result values if
                # loop runs for 0 iteration.
                if op.is_for_loop:
                    state.propagate(init, result)

            _analyze_aliases_in_block(op.body, state, op, None)

        elif isinstance(op, Continue):
            for next, body, result in zip(op.values, innermost_loop.body_vars,
                                          innermost_loop.result_vars, strict=True):
                # Loop next values can flow into body values
                state.propagate(next, body)

                # `For` loop next values can flow into result values when
                # the iterator is exhausted.
                if innermost_loop.is_for_loop:
                    state.propagate(next, result)

        elif isinstance(op, Break):
            for output, result in zip(op.values, innermost_loop.result_vars, strict=True):
                state.propagate(output, result)

        elif isinstance(op, IfElse):
            _analyze_aliases_in_block(op.then_block, state, innermost_loop, op)

            _analyze_aliases_in_block(op.else_block, state, innermost_loop, op)

        elif isinstance(op, EndBranch):
            for output, result in zip(op.outputs, innermost_branch.result_vars, strict=True):
                state.propagate(output, result)

        elif isinstance(op, TileReduce | TileScan):
            for v in op.body.params:
                state.set_always_true(v)
            _analyze_aliases_in_block(op.body, state, None, op)

        else:
            assert len(op.nested_blocks) == 0
            for v in op.result_vars:
                state.set_always_true(v)
