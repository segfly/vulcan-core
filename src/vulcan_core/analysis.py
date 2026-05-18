# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

"""Static analysis for Vulcan rule engine rulesets.

Provides satisfiability checking (via Z3), cycle detection (via Tarjan's SCC), and reachability analysis for rules
registered with a `RuleEngine`. The public entry point is `RulesetAnalyzer.validate`, which returns a `ValidationResult`
summarizing all findings.
"""

from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:  # pragma: no cover - not used at runtime
    from vulcan_core.engine import RuleEngine


class ValidationInternalError(Exception):
    """Raised when validation logic produces an internally inconsistent result."""


class Findings(Flag):
    """Flag enum representing the set of issues found during ruleset validation.

    Values may be combined with bitwise OR except that `VALID` cannot be
    combined with any other flag. Use `Findings.VALID` to indicate a clean
    ruleset.
    """

    VALID = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    UNCONDITIONAL_CYCLE = auto()
    CONDITIONAL_CYCLE = auto()
    INVERSE_CYCLE = auto()
    UNREACHABLE = auto()


# ---------------------------------------------------------------------------
# Issue dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnsatIssue:
    """A rule whose `when` condition can never be satisfied.

    Args:
        rule_id: The first 8 characters of the rule's UUID.
        rule_name: The human-readable name of the rule, or `None` if unnamed.
        condition: Human-readable lambda body text for the condition.
    """

    rule_id: str
    rule_name: str | None
    condition: str

    def to_dict(self) -> dict:
        """Return a dictionary representation suitable for YAML serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "condition": self.condition,
        }


@dataclass(frozen=True, slots=True)
class CyclePath:
    """One step in a detected dependency cycle.

    Args:
        rule_id: The first 8 characters of the rule's UUID.
        rule_name: The human-readable name of the rule, or `None` if unnamed.
        fact_read: The ``"ClassName.attr"`` string that routes to this rule.
        fact_written: The fact class name written by the action.
        action_type: Either ``"then"`` or ``"inverse"``.
        condition: Human-readable lambda body text for the rule condition.
    """

    rule_id: str
    rule_name: str | None
    fact_read: str
    fact_written: str
    action_type: str
    condition: str

    def to_dict(self) -> dict:
        """Return a dictionary representation suitable for YAML serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "fact_read": self.fact_read,
            "fact_written": self.fact_written,
            "action_type": self.action_type,
            "condition": self.condition,
        }


@dataclass(frozen=True, slots=True)
class CycleIssue:
    """A detected dependency cycle in the ruleset.

    Args:
        cycle_type: One of `UNCONDITIONAL_CYCLE`, `CONDITIONAL_CYCLE`, or
            `INVERSE_CYCLE`.
        path: The ordered sequence of steps forming the cycle.
        classification_reason: A plain-English explanation of why this
            classification was chosen.
    """

    cycle_type: Findings
    path: tuple[CyclePath, ...]
    classification_reason: str

    def to_dict(self) -> dict:
        """Return a dictionary representation suitable for YAML serialization."""
        return {
            "cycle_type": self.cycle_type.name,
            "classification_reason": self.classification_reason,
            "path": [step.to_dict() for step in self.path],
        }


@dataclass(frozen=True, slots=True)
class UnreachableIssue:
    """A rule that can never be triggered given the reachable set of fact types.

    Args:
        rule_id: The first 8 characters of the rule's UUID.
        rule_name: The human-readable name of the rule, or `None` if unnamed.
        missing_triggers: Fact attribute strings (``"ClassName.attr"``) that
            would trigger the rule but are never produced.
    """

    rule_id: str
    rule_name: str | None
    missing_triggers: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return a dictionary representation suitable for YAML serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "missing_triggers": list(self.missing_triggers),
        }


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Structured findings from a ruleset validation pass.

    Args:
        unsat_issues: Rules whose conditions are unsatisfiable.
        cycle_issues: Dependency cycles detected in the ruleset.
        unreachable_issues: Rules that can never be reached.
    """

    unsat_issues: tuple[UnsatIssue, ...]
    cycle_issues: tuple[CycleIssue, ...]
    unreachable_issues: tuple[UnreachableIssue, ...]

    def to_dict(self) -> dict:
        """Return a dictionary representation suitable for YAML serialization.

        Returns:
            A dictionary with a top-level ``"analysis"`` key containing
            ``"unsat"``, ``"cycles"``, and ``"unreachable"`` sub-keys.
        """
        return {
            "analysis": {
                "unsat": [issue.to_dict() for issue in self.unsat_issues],
                "cycles": [issue.to_dict() for issue in self.cycle_issues],
                "unreachable": [issue.to_dict() for issue in self.unreachable_issues],
            }
        }

    def generate_yaml_report(self) -> str:
        """Serialize the report to a YAML string.

        Returns:
            A YAML-formatted string representation of the analysis report.
        """
        from vulcan_core.reporting import _dump_yaml

        return _dump_yaml(self.to_dict())


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel, frozen=True):
    """The result of a static validation pass over a `RuleEngine` ruleset.

    Args:
        findings: A `Findings` flag value summarizing all detected issues.
        report: Structured per-issue detail for all findings.
    """

    findings: Findings
    report: AnalysisReport

    @model_validator(mode="after")
    def _validate_findings(self) -> Self:
        """Reject combinations of `VALID` with any other finding flag."""
        if Findings.VALID in self.findings and self.findings != Findings.VALID:
            msg = f"VALID cannot be combined with other findings; got: {self.findings!r}"
            raise ValidationInternalError(msg)
        return self

    def yaml_report(self) -> str:
        """Return the analysis report serialized as a YAML string."""
        return self.report.generate_yaml_report()


# ---------------------------------------------------------------------------
# Stub components (implemented in later tasks)
# ---------------------------------------------------------------------------


class ConditionEncoder:
    """Encodes Vulcan conditions as Z3 formulas for satisfiability checking.

    Not yet implemented.
    """

    def encode(self) -> None:
        """Encode the condition as a Z3 formula."""
        raise NotImplementedError


class DependencyGraph:
    """Bipartite directed graph of rules and fact class names for cycle detection.

    Not yet implemented.
    """

    def build(self) -> None:
        """Build the dependency graph from a registered ruleset."""
        raise NotImplementedError


class RulesetAnalyzer:
    """Performs static analysis of a `RuleEngine` ruleset.

    Orchestrates satisfiability checking, cycle detection, and reachability
    analysis, returning a consolidated `ValidationResult`.
    """

    def validate(self, _engine: "RuleEngine") -> ValidationResult:
        """Run all static analyses against the registered ruleset.

        Args:
            engine: The `RuleEngine` instance whose ruleset to analyze.

        Returns:
            A `ValidationResult` containing a `Findings` flag summary and a
            structured `AnalysisReport` with per-issue detail.
        """
        raise NotImplementedError
