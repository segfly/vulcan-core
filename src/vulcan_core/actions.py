# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import partial
from typing import cast

from vulcan_core.ast_utils import ASTProcessor
from vulcan_core.models import ActionCallable, ActionReturn, DeclaresFacts, Fact, FactHandler


@dataclass(frozen=True, slots=True)
class Action(FactHandler[ActionCallable, ActionReturn], DeclaresFacts):
    """
    Represents a deferred result calculation of a rule.
    """

    output_classes: tuple[type[Fact], ...] = field(default_factory=tuple)

    def __call__(self, *args: Fact) -> ActionReturn:
        return self._evaluate(*args)

    def _evaluate(self, *args: Fact) -> ActionReturn:
        return self.func(*args)


def _infer_output_classes(processor: ASTProcessor[ActionCallable]) -> tuple[type[Fact], ...]:
    """Infer the `Fact` subclasses produced by an action callable.

    Class names that appear as constructors in the lambda body are resolved first from
    `analysis.fact_classes` (populated via attribute-access tracking), then by looking
    up the name in the callable's `__globals__` and confirming it is a `Fact` subclass.
    This handles action lambdas that construct new facts without reading any fact
    attributes (e.g. ``lambda: User(name="x")``).

    Results are deduplicated while preserving first-seen order.

    Args:
        processor: A fully initialised processor for the callable being analysed.

    Returns:
        A tuple of `Fact` subclass types produced by the callable.
    """
    fact_classes = processor.analysis.fact_classes
    func_globals: dict[str, object] = getattr(processor.func, "__globals__", {})
    seen: dict[str, type[Fact]] = {}

    def _resolve(name: str) -> type[Fact] | None:
        if name in fact_classes:
            return fact_classes[name]
        val = func_globals.get(name)
        if isinstance(val, type) and issubclass(val, Fact):
            return val
        return None

    for node in ast.walk(processor.analysis.ast_body):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name):
            continue

        # partial(ClassName, ...) pattern — check before the generic constructor case
        if node.func.id == "partial" and node.args and isinstance(node.args[0], ast.Name):
            cls = _resolve(node.args[0].id)
            if cls is not None:
                seen.setdefault(node.args[0].id, cls)

        # Direct constructor call: ClassName(...)
        else:
            cls = _resolve(node.func.id)
            if cls is not None:
                seen.setdefault(node.func.id, cls)

    return tuple(seen.values())


def action(value: ActionCallable | ActionReturn) -> Action:
    if not isinstance(value, partial) and callable(value):
        processed = ASTProcessor[ActionCallable](value, action, ActionReturn)  # ty:ignore[invalid-argument-type] - needs to be reworked to avoid runtime checks on TypeAliasTypes
        return Action(processed.facts, processed.func, _infer_output_classes(processed), analysis=processed.analysis)
    else:
        # Determine output_classes from the static value type(s)
        if isinstance(value, tuple):
            classes = []
            for v in value:
                if isinstance(v, Fact):
                    classes.append(type(v))
                elif isinstance(v, partial):
                    classes.append(cast("type[Fact]", v.func))
            output_classes: tuple[type[Fact], ...] = tuple(classes)
        elif isinstance(value, partial):
            output_classes = (cast("type[Fact]", value.func),)
        else:
            # value is narrowed to Fact at this point
            output_classes = (type(value),)

        return Action((), lambda: value, output_classes)  # ty:ignore[invalid-argument-type] - We know value is an ActionReturn at this point
