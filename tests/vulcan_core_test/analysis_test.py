# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

import pytest
import yaml

from vulcan_core.analysis import (
    AnalysisReport,
    CycleIssue,
    CyclePath,
    Findings,
    UnreachableIssue,
    UnsatIssue,
    ValidationInternalError,
    ValidationResult,
)


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
