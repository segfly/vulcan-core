# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

"""Static analysis for Vulcan rule engine rulesets.

Provides satisfiability checking (via Z3), cycle detection (via Tarjan's SCC), and reachability analysis for rules
registered with a `RuleEngine`. The public entry point is `RulesetAnalyzer.validate`, which returns a `ValidationResult`
summarizing all findings.
"""

import ast
import logging
from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, Self, get_type_hints

from pydantic import BaseModel, model_validator
from z3 import (
    And,
    ArithRef,
    Bool,
    BoolRef,
    BoolVal,
    ExprRef,
    Int,
    IntVal,
    Not,
    Or,
    Real,
    RealVal,
    Solver,
    String,
    StringVal,
    Xor,
    sat,
    unsat,
)

from vulcan_core.conditions import AICondition, CompoundCondition, Condition, Expression, OnFactChanged, Operator
from vulcan_core.models import Fact

if TYPE_CHECKING:  # pragma: no cover - not used at runtime
    from vulcan_core.engine import RuleEngine

logger = logging.getLogger(__name__)


class ValidationInternalError(Exception):
    """Raised when validation logic produces an internally inconsistent result."""


class Findings(Flag):
    """Flag enum representing the set of issues found during ruleset validation."""

    VALID = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    UNCONDITIONAL_CYCLE = auto()
    CONDITIONAL_CYCLE = auto()
    INVERSE_CYCLE = auto()
    UNREACHABLE = auto()


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
        fact_read: The `ClassName.attr` string that routes to this rule.
        fact_written: The fact class name written by the action.
        action_type: Either `then` or `inverse`.
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
        classification_reason: A plain-English explanation of why this classification was chosen.
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
        missing_triggers: Fact attribute strings (`ClassName.attr`) that would trigger the rule but are never produced.
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
            A dictionary with a top-level `analysis` key containing
            `unsat`, `cycles`, and `unreachable` sub-keys.
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


@dataclass(frozen=True, slots=True)
class EncodedExpression:
    """A Z3 boolean formula derived from a Vulcan `Expression`.

    Produced by `ConditionEncoder.encode` and its private helpers. Carries the formula itself alongside a flag that
    indicates whether any `AICondition` sub-expression was encountered during encoding, which signals that the result is
    only partially statically analysable.

    Args:
        formula: The Z3 boolean formula representing the expression.
        has_ai: `True` if any `AICondition` was encountered during encoding.
    """

    formula: BoolRef
    has_ai: bool


@dataclass(frozen=True, slots=True)
class SatisfiabilityResult:
    """Outcome of a Z3 satisfiability check on a Vulcan `Expression`.

    Produced by `ConditionEncoder.is_satisfiable`.

    Args:
        satisfiable: `True` if the formula is satisfiable, `False` if unsatisfiable, or `None` if the solver returns  
          unknown (e.g., due to an `AICondition`).
        has_ai: `True` if any `AICondition` was encountered during encoding.
    """

    satisfiable: bool | None
    has_ai: bool


@dataclass(frozen=True, slots=True)
class TautologyResult:
    """Outcome of a Z3 tautology check on a Vulcan `Expression`.

    Produced by `ConditionEncoder.is_tautology`.

    Args:
        is_tautology: `True` if the formula is `True` for all possible variable assignments.
        has_ai: `True` if any `AICondition` was encountered during encoding.
    """

    is_tautology: bool
    has_ai: bool


@dataclass(frozen=True, slots=True)
class ConditionEncoder:
    """Encode Vulcan `Expression` instances as Z3 formulas for satisfiability checking.

    Stateless encoder that dispatches on expression type and recursively converts AST nodes captured at
    condition-construction time into Z3 boolean expressions.
    """

    def encode(self, expr: Expression) -> EncodedExpression:
        """Encode an `Expression` as a Z3 formula.

        Args:
            expr: The expression to encode.

        Returns:
            An `EncodedExpression` containing the Z3 formula and an `has_ai` flag that is `True` if any `AICondition`
            was encountered during encoding.

        Raises:
            ValidationInternalError: If a `Condition` has no captured analysis metadata.
        """
        # OnFactChanged always fires; represent as a tautology
        if isinstance(expr, OnFactChanged):
            return EncodedExpression(formula=BoolVal(True), has_ai=False)  # noqa: FBT003

        # AICondition is opaque to static analysis; represent as a free variable
        if isinstance(expr, AICondition):
            return EncodedExpression(formula=Bool(f"ai_{id(expr)}"), has_ai=True)

        # CompoundCondition: recursively encode both sides and combine
        if isinstance(expr, CompoundCondition):
            return self._encode_compound(expr)

        # Condition: walk the captured AST body
        if isinstance(expr, Condition):
            return self._encode_condition(expr)

        # Unknown Expression subtype: treat as opaque free variable
        logger.debug("Encoding unknown Expression subtype %s as opaque bool", type(expr).__name__)
        return EncodedExpression(formula=Bool(f"opaque_{id(expr)}"), has_ai=False)

    def is_satisfiable(self, expr: Expression) -> SatisfiabilityResult:
        """Check whether the encoded expression is satisfiable.

        Args:
            expr: The expression to check.

        Returns:
            A `SatisfiabilityResult` whose `satisfiable` field is `True` if the formula is satisfiable, `False` if
            unsatisfiable, or `None` if the solver returns unknown.
        """
        encoded = self.encode(expr)
        solver = Solver()
        solver.add(encoded.formula)
        result = solver.check()
        if result == sat:
            return SatisfiabilityResult(satisfiable=True, has_ai=encoded.has_ai)
        elif result == unsat:
            return SatisfiabilityResult(satisfiable=False, has_ai=encoded.has_ai)
        else:
            return SatisfiabilityResult(satisfiable=None, has_ai=encoded.has_ai)

    def is_tautology(self, expr: Expression) -> TautologyResult:
        """Check whether the encoded expression is a tautology.

        A tautology is a formula that is `True` for all possible variable assignments, such as `x > 5 or x <= 5`. This
        is verified by asserting `Not(formula)` and confirming the result is UNSAT.

        Args:
            expr: The expression to check.

        Returns:
            A `TautologyResult` whose `is_tautology` field is `True` if the formula holds for all variable assignments.
        """
        encoded = self.encode(expr)
        solver = Solver()
        solver.add(Not(encoded.formula))
        return TautologyResult(is_tautology=solver.check() == unsat, has_ai=encoded.has_ai)

    def _encode_condition(self, expr: Condition) -> EncodedExpression:
        """Encode a `Condition` by walking its captured AST body.

        Args:
            expr: The condition to encode.

        Returns:
            An `EncodedExpression` with `has_ai` always `False` for non-AI conditions.

        Raises:
            ValidationInternalError: If the condition has no captured analysis metadata.
        """
        if expr.analysis is None:
            msg = f"Condition source is unavailable for analysis: {expr!r}"
            raise ValidationInternalError(msg)

        # Build formula from the captured AST body
        z3_vars: dict[str, ExprRef] = {}
        formula = self._encode_ast(expr.analysis.ast_body, expr.analysis.fact_classes, z3_vars)

        return EncodedExpression(formula=Not(formula) if expr.inverted else formula, has_ai=False)

    def _encode_compound(self, expr: CompoundCondition) -> EncodedExpression:
        """Encode a `CompoundCondition` by recursively encoding its sub-expressions.

        Args:
            expr: The compound condition to encode.

        Returns:
            An `EncodedExpression` where `has_ai` is the OR of both sub-expression `has_ai` flags.
        """
        left = self.encode(expr.left)
        right = self.encode(expr.right)
        has_ai = left.has_ai or right.has_ai

        # Combine sub-formulas with the declared logical operator
        if expr.operator == Operator.AND:
            formula: BoolRef = And(left.formula, right.formula)
        elif expr.operator == Operator.OR:
            formula = Or(left.formula, right.formula)
        else:
            formula = Xor(left.formula, right.formula)

        return EncodedExpression(formula=Not(formula) if expr.inverted else formula, has_ai=has_ai)

    def _encode_ast(self, node: ast.expr, fact_classes: dict[str, type[Fact]], z3_vars: dict[str, ExprRef]) -> BoolRef:
        """Recursively encode an AST expression node as a Z3 boolean formula.

        Args:
            node: The AST expression node to encode.
            fact_classes: Map from class name to `Fact` class for typed variable creation.
            z3_vars: Mutable cache of already-created Z3 variables keyed by `ClassName.attr`.

        Returns:
            A Z3 boolean expression representing the AST node.
        """
        # Boolean operators: And / Or
        if isinstance(node, ast.BoolOp):
            operands = [self._encode_ast(v, fact_classes, z3_vars) for v in node.values]
            return And(*operands) if isinstance(node.op, ast.And) else Or(*operands)

        # Logical negation
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return Not(self._encode_ast(node.operand, fact_classes, z3_vars))

        # Comparison expressions (e.g., Foo.bar > 5)
        if isinstance(node, ast.Compare):
            return self._encode_compare(node, fact_classes, z3_vars)

        # Standalone attribute access treated as a boolean variable (e.g., Foo.flag)
        if isinstance(node, ast.Attribute):
            class_name = node.value.id if isinstance(node.value, ast.Name) else ""
            key = f"{class_name}.{node.attr}"
            var = self._get_or_create_z3_var(key, class_name, node.attr, fact_classes, z3_vars)
            if isinstance(var, BoolRef):
                return var
            # Attribute has a non-bool type but is used in boolean position; use an opaque Bool
            logger.debug("Attribute %s used in boolean position but resolved to non-Bool type; using opaque Bool", key)
            return Bool(f"bool_{key}")

        # Boolean constants True / False
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return BoolVal(node.value)

        # Unrecognized node: treat as an opaque free boolean variable
        logger.debug("Encoding unrecognized AST node %s as opaque bool", type(node).__name__)
        return Bool(f"opaque_{ast.dump(node)}")

    def _encode_compare(
        self, node: ast.Compare, fact_classes: dict[str, type[Fact]], z3_vars: dict[str, ExprRef]
    ) -> BoolRef:
        """Encode an AST comparison node as a Z3 boolean expression.

        Chained comparisons (e.g., `1 < x < 10`) are encoded as a conjunction of pairwise comparisons.

        Args:
            node: The comparison AST node.
            fact_classes: Map from class name to `Fact` class.
            z3_vars: Mutable cache of Z3 variables.

        Returns:
            A Z3 boolean expression representing the comparison.
        """
        # Build a list of pairwise Z3 comparisons
        comparisons: list[BoolRef] = []
        left = self._encode_operand(node.left, fact_classes, z3_vars)
        for op, right_node in zip(node.ops, node.comparators, strict=False):
            right = self._encode_operand(right_node, fact_classes, z3_vars)

            if isinstance(op, ast.Eq):
                comparisons.append(left == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(left != right)
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                if isinstance(left, ArithRef) and isinstance(right, ArithRef):
                    if isinstance(op, ast.Lt):
                        comparisons.append(left < right)
                    elif isinstance(op, ast.LtE):
                        comparisons.append(left <= right)
                    elif isinstance(op, ast.Gt):
                        comparisons.append(left > right)
                    else:
                        comparisons.append(left >= right)
                else:
                    logger.debug(
                        "Ordering comparison on non-arithmetic operands (%s, %s); using opaque Bool",
                        type(left).__name__,
                        type(right).__name__,
                    )
                    comparisons.append(Bool(f"opaque_{ast.dump(node)}"))
            else:
                comparisons.append(Bool(f"opaque_{ast.dump(node)}"))

            left = right

        return And(*comparisons) if len(comparisons) > 1 else comparisons[0]

    def _encode_operand(
        self, node: ast.expr, fact_classes: dict[str, type[Fact]], z3_vars: dict[str, ExprRef]
    ) -> ExprRef:
        """Encode a comparison operand as a typed Z3 expression.

        Args:
            node: The operand AST node (attribute reference or literal constant).
            fact_classes: Map from class name to `Fact` class.
            z3_vars: Mutable cache of Z3 variables.

        Returns:
            A typed Z3 expression matching the operand's value domain.
        """
        # Fact attribute reference (e.g., Foo.bar)
        if isinstance(node, ast.Attribute):
            class_name = node.value.id if isinstance(node.value, ast.Name) else ""
            key = f"{class_name}.{node.attr}"
            return self._get_or_create_z3_var(key, class_name, node.attr, fact_classes, z3_vars)

        # Literal constant: create a typed Z3 value
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return BoolVal(value)
            if isinstance(value, int):
                return IntVal(value)
            if isinstance(value, float):
                return RealVal(value)
            if isinstance(value, str):
                return StringVal(value)

        # Fall back to the general AST encoder for other node types
        return self._encode_ast(node, fact_classes, z3_vars)

    def _get_or_create_z3_var(
        self,
        key: str,
        class_name: str,
        attr_name: str,
        fact_classes: dict[str, type[Fact]],
        z3_vars: dict[str, ExprRef],
    ) -> ExprRef:
        """Return a cached Z3 variable or create and cache a new typed one.

        The variable type is inferred from the `Fact` subclass field's type hint. Unrecognized types fall back to
        `z3.Bool` with a warning.

        Args:
            key: Cache key in `ClassName.attr` format.
            class_name: The `Fact` subclass name.
            attr_name: The attribute name on the `Fact` subclass.
            fact_classes: Map from class name to `Fact` class.
            z3_vars: Mutable cache of Z3 variables.

        Returns:
            A typed Z3 variable for the given fact attribute.
        """
        if key in z3_vars:
            return z3_vars[key]

        # Resolve the attribute's type hint from the Fact subclass
        fact_class = fact_classes.get(class_name)
        type_hint = get_type_hints(fact_class).get(attr_name) if fact_class else None

        # Create a Z3 variable with the appropriate sort
        if type_hint is int:
            var: ExprRef = Int(key)
        elif type_hint is float:
            var = Real(key)
        elif type_hint is str:
            var = String(key)
        else:
            if type_hint is not None and type_hint is not bool:
                logger.warning("Unrecognized type %s for %s.%s; treating as Bool", type_hint, class_name, attr_name)
            var = Bool(key)

        z3_vars[key] = var
        return var


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
