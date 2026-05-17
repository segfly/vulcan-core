# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

import ast
from functools import partial

from vulcan_core.actions import action
from vulcan_core.ast_utils import AnalysisInfo
from vulcan_core.conditions import Condition, condition
from vulcan_core.models import Fact


# Minimal Fact subclasses used across all tests in this module
class User(Fact):
    name: str
    age: int


class Order(Fact):
    total: float


class TestAnalysisInfoOnCondition:
    def test_lambda_condition_fact_classes(self):
        cond = condition(lambda: User.age >= 18)

        assert cond.analysis is not None
        assert cond.analysis.fact_classes == {"User": User}

    def test_lambda_condition_ast_body_is_compare(self):
        cond = condition(lambda: User.age >= 18)

        assert cond.analysis is not None
        assert isinstance(cond.analysis.ast_body, ast.Compare)

    def test_decorated_function_condition_has_analysis(self):
        @condition
        def is_adult(user: User) -> bool:
            return user.age >= 18

        assert isinstance(is_adult, Condition)
        assert is_adult.analysis is not None
        assert isinstance(is_adult.analysis, AnalysisInfo)


class TestActionOutputClassesFromLambda:
    def test_constructor_call_lambda(self):
        act = action(lambda: User(name="x", age=0))

        assert act.output_classes == (User,)

    def test_partial_call_lambda(self):
        act = action(lambda: partial(User, name="x", age=0))

        assert act.output_classes == (User,)

    def test_tuple_return_yields_both_classes(self):
        act = action(lambda: (User(name="x", age=0), Order(total=1.0)))

        assert set(act.output_classes) == {User, Order}

    def test_lambda_analysis_stored(self):
        act = action(lambda: User(name="x", age=0))

        assert act.analysis is not None
        assert isinstance(act.analysis, AnalysisInfo)


class TestActionOutputClassesFromStaticValue:
    def test_static_fact_instance(self):
        act = action(User(name="x", age=0))

        assert act.output_classes == (User,)

    def test_static_partial(self):
        act = action(partial(User, name="x", age=0))

        assert act.output_classes == (User,)

    def test_static_tuple_of_mixed_types(self):
        act = action((User(name="x", age=0), partial(Order, total=2.0)))

        assert set(act.output_classes) == {User, Order}
