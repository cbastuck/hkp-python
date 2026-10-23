"""The expression evaluator's reads of what is not there — the same cases are
pinned for the browser and hkp-node (expressionEval.test.ts, expression-eval.test.ts)
and for hkp-rt (expression.test.cpp)."""
from __future__ import annotations

from typing import Any

from hkp.services.expression import Expression, compile_expression


def run(source: str, params: Any = None) -> Any:
    return Expression.parse(source).evaluate(params)


def reads(source: str, params: Any = None) -> list[str]:
    missing: list[str] = []
    Expression.parse(source).evaluate(params, on_missing=missing.append)
    return missing


SCOPE = {"a": {"b": 1}, "list": [1], "none": None}


def test_a_member_of_nothing_is_nothing():
    assert run("params.x", SCOPE) is None
    assert run("params.x.y.z", SCOPE) is None
    assert run("params.none.y", SCOPE) is None
    assert run("params.x.y", None) is None
    assert run("params.x.y || 'fallback'", SCOPE) == "fallback"
    assert run("params.a.b", SCOPE) == 1


def test_reports_the_reads_that_found_nothing_once_per_chain():
    assert reads("params.a.b", SCOPE) == []
    assert reads("params.titel", SCOPE) == ["params.titel"]
    assert reads("params.x.y.z", SCOPE) == ["params.x"]
    assert reads("params.a.c", SCOPE) == ["params.a.c"]
    assert reads("params.list[3]", SCOPE) == ["params.list[3]"]
    assert reads("params['x y']", SCOPE) == ['params["x y"]']
    assert reads("params.id || params.link", SCOPE) == ["params.id", "params.link"]
    # Present, if empty, is not missing.
    assert reads("params.none", SCOPE) == []
    # No input at all: the first read of it is what is missing.
    assert reads("params.title", None) == ["params.title"]


def test_a_predicate_inside_a_term_reports_nothing_of_its_own():
    assert reads("find(params.list, 'item.nope')", SCOPE) == []


def test_a_compiled_term_passes_on_missing_through_and_names_a_parse_error():
    missing: list[str] = []
    assert compile_expression("params.titel")({}, missing.append) is None
    assert missing == ["params.titel"]

    broken = compile_expression("nope(")
    assert broken.parse_error.startswith("invalid expression 'nope('")
    assert compile_expression(5)({}) == 5
