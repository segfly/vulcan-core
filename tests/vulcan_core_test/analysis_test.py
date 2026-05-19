# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

from unittest.mock import MagicMock

import pytest
import yaml

from vulcan_core.analysis import (
    AnalysisReport,
    ConditionEncoder,
    CycleIssue,
    CyclePath,
    Findings,
    UnreachableIssue,
    UnsatIssue,
    ValidationInternalError,
    ValidationResult,
)
from vulcan_core.conditions import AICondition, OnFactChanged, condition
from vulcan_core.models import Fact


class Score(Fact):
    value: int
    active: bool = True


@pytest.fixture
def empty_report() -> AnalysisReport:
    return AnalysisReport(unsat_issues=(), cycle_issues=(), unreachable_issues=())


@pytest.fixture
def unsat_issue() -> UnsatIssue:
    return UnsatIssue(rule_id="abcd1234", rule_name="my_rule", condition="Foo.x > 10 and Foo.x < 5")


@pytest.fixture
def cycle_path() -> CyclePath:
    return CyclePath(
        rule_id="abcd1234",
        rule_name="my_rule",
        fact_read="Foo.x",
        fact_written="Bar",
        action_type="then",
        condition="Foo.x > 0",
    )


@pytest.fixture
def unreachable_issue() -> UnreachableIssue:
    return UnreachableIssue(rule_id="abcd1234", rule_name="my_rule", missing_triggers=("Foo.x",))


class TestValidationResult:
    def test_valid_combined_with_other_raises(self, empty_report):
        with pytest.raises(ValidationInternalError):
            ValidationResult(findings=Findings.VALID | Findings.UNSAT, report=empty_report)


class TestAnalysisReportYaml:
    def test_empty_report_structure(self, empty_report):
        parsed = yaml.safe_load(empty_report.generate_yaml_report())

        assert parsed == {"analysis": {"unsat": [], "cycles": [], "unreachable": []}}

    def test_unsat_issue_yaml(self, unsat_issue):
        report = AnalysisReport(unsat_issues=(unsat_issue,), cycle_issues=(), unreachable_issues=())
        parsed = yaml.safe_load(report.generate_yaml_report())

        assert parsed["analysis"]["unsat"] == [
            {"rule_id": "abcd1234", "rule_name": "my_rule", "condition": "Foo.x > 10 and Foo.x < 5"}
        ]

    def test_cycle_issue_yaml(self, cycle_path):
        issue = CycleIssue(
            cycle_type=Findings.CONDITIONAL_CYCLE,
            path=(cycle_path,),
            classification_reason="Condition is satisfiable but not a tautology.",
        )
        report = AnalysisReport(unsat_issues=(), cycle_issues=(issue,), unreachable_issues=())
        parsed = yaml.safe_load(report.generate_yaml_report())

        assert parsed["analysis"]["cycles"] == [
            {
                "cycle_type": "CONDITIONAL_CYCLE",
                "classification_reason": "Condition is satisfiable but not a tautology.",
                "path": [
                    {
                        "rule_id": "abcd1234",
                        "rule_name": "my_rule",
                        "fact_read": "Foo.x",
                        "fact_written": "Bar",
                        "action_type": "then",
                        "condition": "Foo.x > 0",
                    }
                ],
            }
        ]

    def test_unreachable_issue_yaml(self, unreachable_issue):
        report = AnalysisReport(unsat_issues=(), cycle_issues=(), unreachable_issues=(unreachable_issue,))
        parsed = yaml.safe_load(report.generate_yaml_report())

        assert parsed["analysis"]["unreachable"] == [
            {"rule_id": "abcd1234", "rule_name": "my_rule", "missing_triggers": ["Foo.x"]}
        ]


class TestConditionEncoderSatisfiability:
    @pytest.fixture
    def encoder(self) -> ConditionEncoder:
        return ConditionEncoder()

    def test_satisfiable_int_comparison(self, encoder):
        cond = condition(lambda: Score.value >= 0)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True
        assert result.has_ai is False

    def test_unsat_contradictory_int_range(self, encoder):
        cond = condition(lambda: Score.value >= 100 and Score.value < 0)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is False
        assert result.has_ai is False

    def test_satisfiable_bool_field(self, encoder):
        cond = condition(lambda: Score.active)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True
        assert result.has_ai is False

    def test_satisfiable_inverted_bool(self, encoder):
        cond = ~condition(lambda: Score.active)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True
        assert result.has_ai is False

    def test_unsat_bool_self_contradiction(self, encoder):
        cond = condition(lambda: Score.active and not Score.active)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is False
        assert result.has_ai is False

    def test_on_fact_changed_is_satisfiable(self, encoder):
        cond = OnFactChanged(("Score.value",), func=lambda: True)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True

    def test_compound_and_unsat(self, encoder):
        cond = condition(lambda: Score.value > 0) & condition(lambda: Score.value < 0)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is False

    def test_compound_or_satisfiable(self, encoder):
        cond = condition(lambda: Score.value > 0) | condition(lambda: Score.value < 0)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True

    def test_compound_xor_satisfiable(self, encoder):
        cond = condition(lambda: Score.value > 0) ^ condition(lambda: Score.active)

        result = encoder.is_satisfiable(cond)

        assert result.satisfiable is True

    def test_ai_condition_has_ai_flag(self, encoder):
        ai_cond = _make_ai_condition()

        result = encoder.is_satisfiable(ai_cond)

        assert result.has_ai is True
        assert result.satisfiable is not False

    def test_compound_with_ai_has_ai_flag(self, encoder):
        cond = _make_ai_condition() & condition(lambda: Score.value > 0)

        result = encoder.is_satisfiable(cond)

        assert result.has_ai is True


class TestConditionEncoderTautology:
    @pytest.fixture
    def encoder(self) -> ConditionEncoder:
        return ConditionEncoder()

    def test_on_fact_changed_is_tautology(self, encoder):
        cond = OnFactChanged(("Score.value",), func=lambda: True)

        result = encoder.is_tautology(cond)

        assert result.is_tautology is True
        assert result.has_ai is False

    def test_disjunction_covering_all_integers_is_tautology(self, encoder):
        cond = condition(lambda: Score.value >= 0) | condition(lambda: Score.value < 0)

        result = encoder.is_tautology(cond)

        assert result.is_tautology is True

    def test_partial_range_is_not_tautology(self, encoder):
        cond = condition(lambda: Score.value > 5)

        result = encoder.is_tautology(cond)

        assert result.is_tautology is False


def _make_ai_condition() -> AICondition:
    """Construct a minimal `AICondition` using mocked LangChain dependencies."""
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = MagicMock()
    mock_chain = MagicMock()
    return AICondition(
        facts=("Score.value",),
        chain=mock_chain,
        model=mock_model,
        system_template="system",
        attachments_template="attachments",
        inquiry="{Score.value}",
    )
