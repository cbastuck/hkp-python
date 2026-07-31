"""Expression evaluation for services that let a board author write small
dynamic terms — a Map template's ``key=`` rows, for example.

An expression is a single JavaScript-style expression evaluated with the
incoming data bound to ``params`` and the helper functions below in scope. The
browser, Node and C++ runtimes evaluate the same sources, so the dialect and the
helper set are kept aligned: a template authored in the shared Map UI behaves
the same whichever runtime hosts the service. Helpers that need a browser (DOM,
vault, AudioContext) have no counterpart here.

Supported: literals (number, string, true/false/null), array literals,
identifiers, member and index access, calls into the builtin table, unary
``!``/``-``/``+``, ``* / %``, ``+ -``, ``< <= > >=``, ``== != === !==``,
``&& ||`` and the ternary operator. There are no assignments, no statements and
no lambdas — ``find``/``filter`` take their predicate as an expression string
with the element bound to ``item`` (and its position to ``index``).

Python spellings are accepted alongside the JavaScript ones (``and``/``or``/
``not``, ``True``/``False``/``None``) and the builtin table carries the Python
conversions the previous evaluator exposed, so flat templates written against
it keep working.

    expression = Expression.parse("round(params.value * 2)")
    result = expression.evaluate(input)   # raises EvaluationError on failure
"""

from __future__ import annotations

import json as json_module
import math
import random
import re
import time
import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import quote


class ParseError(Exception):
    """Raised when an expression source cannot be parsed."""


class EvaluationError(Exception):
    """Raised when a parsed expression cannot be evaluated."""


# ── Value helpers ──────────────────────────────────────────────────────────────


def is_truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return True  # objects and arrays are truthy, empty or not


def to_number(value: Any) -> float:
    """JS-style numeric coercion. Returns NaN for anything that is not numeric."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return math.nan
    return math.nan


def to_text(value: Any) -> str:
    """Renders a value the way JSON.stringify would, so 42 does not become 42.0."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return json_module.dumps(value, separators=(",", ":"))


def _number(value: float) -> Any:
    """An integral result is stored as an int; NaN has no JSON form and is null."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            return int(value)
    return value


# ── Date helpers (moment-style token subset shared with the other runtimes) ────

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_WEEKDAYS = [
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
]

# Longest token first, so "MMMM" is matched before "MM".
_TOKEN_PATTERN = re.compile(
    r"\[([^\]]*)\]|YYYY|YY|MMMM|MMM|MM|M|DD|D|dddd|ddd|HH|H|hh|h|mm|m|ss|s|A|a"
)


def _format_date(value: datetime, fmt: str) -> str:
    weekday = (value.weekday() + 1) % 7  # datetime weeks start on Monday
    hours12 = value.hour % 12 or 12

    def render(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            return match.group(1)
        token = match.group(0)
        return {
            "YYYY": f"{value.year}",
            "YY": f"{value.year % 100:02d}",
            "MMMM": _MONTHS[value.month - 1],
            "MMM": _MONTHS[value.month - 1][:3],
            "MM": f"{value.month:02d}",
            "M": f"{value.month}",
            "DD": f"{value.day:02d}",
            "D": f"{value.day}",
            "dddd": _WEEKDAYS[weekday],
            "ddd": _WEEKDAYS[weekday][:3],
            "HH": f"{value.hour:02d}",
            "H": f"{value.hour}",
            "hh": f"{hours12:02d}",
            "h": f"{hours12}",
            "mm": f"{value.minute:02d}",
            "m": f"{value.minute}",
            "ss": f"{value.second:02d}",
            "s": f"{value.second}",
            "A": "AM" if value.hour < 12 else "PM",
            "a": "am" if value.hour < 12 else "pm",
        }[token]

    return _TOKEN_PATTERN.sub(render, fmt)


def _parse_date(text: str, fmt: str) -> datetime | None:
    """Reads a date written in the token subset above, or None when it does not fit."""
    pattern = ""
    captured: list[str] = []
    last = 0

    for match in _TOKEN_PATTERN.finditer(fmt):
        pattern += re.escape(fmt[last : match.start()])
        last = match.end()

        if match.group(1) is not None:
            pattern += re.escape(match.group(1))
            continue

        token = match.group(0)
        captured.append(token)
        if token == "YYYY":
            pattern += r"(\d{4})"
        elif token in ("MMMM", "MMM", "dddd", "ddd"):
            pattern += r"([A-Za-z]+)"
        elif token in ("A", "a"):
            pattern += r"([AaPp][Mm])"
        else:
            pattern += r"(\d{1,2})"
    pattern += re.escape(fmt[last:])

    if not captured:
        return None

    match = re.fullmatch(pattern, str(text).strip())
    if not match:
        return None

    parts = {"year": 1970, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0}
    pm = False
    hours12 = False

    for token, raw in zip(captured, match.groups()):
        if token == "YYYY":
            parts["year"] = int(raw)
        elif token == "YY":
            parts["year"] = 2000 + int(raw)
        elif token in ("MMMM", "MMM"):
            name = raw.lower()
            for index, month in enumerate(_MONTHS):
                if month.lower().startswith(name):
                    parts["month"] = index + 1
                    break
        elif token in ("MM", "M"):
            parts["month"] = int(raw)
        elif token in ("DD", "D"):
            parts["day"] = int(raw)
        elif token in ("HH", "H"):
            parts["hour"] = int(raw)
        elif token in ("hh", "h"):
            parts["hour"] = int(raw)
            hours12 = True
        elif token in ("mm", "m"):
            parts["minute"] = int(raw)
        elif token in ("ss", "s"):
            parts["second"] = int(raw)
        elif token in ("A", "a"):
            pm = raw.lower() == "pm"

    if hours12:
        parts["hour"] = (parts["hour"] % 12) + (12 if pm else 0)

    try:
        return datetime(**parts)  # type: ignore[arg-type]
    except ValueError:
        return None


def _to_epoch_millis(value: Any) -> float | None:
    """Milliseconds for anything the other runtimes accept: a number or ISO-8601."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp() * 1000
    except ValueError:
        return None


def _now_millis() -> int:
    return int(time.time() * 1000)


def _uuid_v7() -> str:
    millis = _now_millis()
    data = bytearray(random.getrandbits(8) for _ in range(16))
    for i in range(6):
        data[i] = (millis >> (8 * (5 - i))) & 0xFF
    data[6] = (data[6] & 0x0F) | 0x70  # version 7
    data[8] = (data[8] & 0x3F) | 0x80  # variant
    return str(uuid_module.UUID(bytes=bytes(data)))


# ── AST ────────────────────────────────────────────────────────────────────────


@dataclass
class _Node:
    kind: str
    value: Any = None
    name: str = ""
    computed: bool = False
    left: "_Node | None" = None
    right: "_Node | None" = None
    third: "_Node | None" = None
    items: list["_Node"] = field(default_factory=list)


# Lowest precedence first. The Python spellings are aliases for the JS operators.
_BINARY_LEVELS: list[list[str]] = [
    ["||", "or"],
    ["&&", "and"],
    ["===", "!==", "==", "!="],
    ["<=", ">=", "<", ">"],
    ["+", "-"],
    ["*", "/", "%"],
]

_WORD_OPERATORS = {"or", "and", "not"}

_IDENTIFIER_START = re.compile(r"[A-Za-z_$]")
_IDENTIFIER_PART = re.compile(r"[A-Za-z0-9_$]")


class _Parser:
    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0

    def parse(self) -> _Node:
        self._skip_space()
        node = self._parse_expression()
        self._skip_space()
        if self._pos != len(self._source):
            raise ParseError(f"unexpected {self._source[self._pos:]!r}")
        return node

    # ── Grammar ────────────────────────────────────────────────────────────────

    def _parse_expression(self) -> _Node:
        return self._parse_conditional()

    def _parse_conditional(self) -> _Node:
        test = self._parse_binary(0)
        self._skip_space()
        if self._peek() != "?":
            return test
        self._pos += 1
        consequent = self._parse_expression()
        self._skip_space()
        self._expect(":")
        alternate = self._parse_expression()
        return _Node("conditional", left=test, right=consequent, third=alternate)

    def _parse_binary(self, level: int) -> _Node:
        if level >= len(_BINARY_LEVELS):
            return self._parse_unary()

        left = self._parse_binary(level + 1)
        while True:
            self._skip_space()
            matched = self._match_operator(_BINARY_LEVELS[level])
            if not matched:
                return left
            right = self._parse_binary(level + 1)
            kind = "logical" if matched in ("&&", "||", "and", "or") else "binary"
            left = _Node(kind, name=matched, left=left, right=right)

    def _match_operator(self, operators: list[str]) -> str | None:
        for operator in operators:
            if not self._source.startswith(operator, self._pos):
                continue
            end = self._pos + len(operator)
            if operator in _WORD_OPERATORS:
                # "and"/"or" are operators, "android" is an identifier.
                if end < len(self._source) and _IDENTIFIER_PART.match(self._source[end]):
                    continue
            self._pos = end
            return operator
        return None

    def _parse_unary(self) -> _Node:
        self._skip_space()
        char = self._peek()
        if char in ("!", "-", "+") and not self._source.startswith("!=", self._pos):
            self._pos += 1
            return _Node("unary", name=char, left=self._parse_unary())
        if self._match_operator(["not"]):
            return _Node("unary", name="!", left=self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self) -> _Node:
        node = self._parse_primary()
        while True:
            self._skip_space()
            char = self._peek()
            if char == ".":
                self._pos += 1
                node = _Node("member", left=node, name=self._parse_identifier())
            elif char == "[":
                self._pos += 1
                index = self._parse_expression()
                self._skip_space()
                self._expect("]")
                node = _Node("member", left=node, right=index, computed=True)
            elif char == "(":
                self._pos += 1
                node = _Node("call", left=node, items=self._parse_arguments(")"))
            else:
                return node

    def _parse_arguments(self, closing: str) -> list[_Node]:
        args: list[_Node] = []
        self._skip_space()
        if self._peek() == closing:
            self._pos += 1
            return args

        while True:
            args.append(self._parse_expression())
            self._skip_space()
            if self._peek() == ",":
                self._pos += 1
                continue
            self._expect(closing)
            return args

    def _parse_primary(self) -> _Node:
        self._skip_space()
        char = self._peek()

        if char == "":
            raise ParseError("unexpected end of expression")

        if char == "(":
            self._pos += 1
            node = self._parse_expression()
            self._skip_space()
            self._expect(")")
            return node

        if char == "[":
            self._pos += 1
            return _Node("array", items=self._parse_arguments("]"))

        if char in ("'", '"'):
            return _Node("literal", value=self._parse_string(char))

        if char.isdigit() or (char == "." and self._peek(1).isdigit()):
            return _Node("literal", value=self._parse_number())

        name = self._parse_identifier()
        if name in ("true", "True"):
            return _Node("literal", value=True)
        if name in ("false", "False"):
            return _Node("literal", value=False)
        if name in ("null", "undefined", "None"):
            return _Node("literal", value=None)
        return _Node("identifier", name=name)

    def _parse_string(self, quote_char: str) -> str:
        self._pos += 1  # opening quote
        text = ""
        escapes = {"n": "\n", "t": "\t", "r": "\r"}
        while self._pos < len(self._source) and self._source[self._pos] != quote_char:
            char = self._source[self._pos]
            if char == "\\" and self._pos + 1 < len(self._source):
                self._pos += 1
                char = escapes.get(self._source[self._pos], self._source[self._pos])
            text += char
            self._pos += 1
        if self._pos >= len(self._source):
            raise ParseError("unterminated string literal")
        self._pos += 1  # closing quote
        return text

    def _parse_number(self) -> Any:
        start = self._pos
        while self._pos < len(self._source):
            char = self._source[self._pos]
            previous = self._source[self._pos - 1] if self._pos > start else ""
            if char.isdigit() or char in (".", "e", "E"):
                self._pos += 1
            elif char in ("+", "-") and previous in ("e", "E"):
                self._pos += 1
            else:
                break

        text = self._source[start : self._pos]
        try:
            return _number(float(text))
        except ValueError as error:
            raise ParseError(f"invalid number {text!r}") from error

    def _parse_identifier(self) -> str:
        self._skip_space()
        start = self._pos
        if self._pos < len(self._source) and _IDENTIFIER_START.match(self._source[self._pos]):
            self._pos += 1
            while self._pos < len(self._source) and _IDENTIFIER_PART.match(
                self._source[self._pos]
            ):
                self._pos += 1
        if self._pos == start:
            raise ParseError(f"expected an identifier at position {start}")
        return self._source[start : self._pos]

    # ── Scanning ───────────────────────────────────────────────────────────────

    def _skip_space(self) -> None:
        while self._pos < len(self._source) and self._source[self._pos].isspace():
            self._pos += 1

    def _expect(self, char: str) -> None:
        if self._peek() != char:
            raise ParseError(f"expected {char!r}")
        self._pos += 1

    def _peek(self, offset: int = 0) -> str:
        index = self._pos + offset
        return self._source[index] if index < len(self._source) else ""


def _callee_name(node: _Node | None) -> str | None:
    """Flattens an identifier/member chain into a dotted name, so ``uuid.v4`` can
    be looked up in the builtin table. Returns None for anything else."""
    if node is None:
        return None
    if node.kind == "identifier":
        return node.name
    if node.kind == "member" and not node.computed:
        prefix = _callee_name(node.left)
        return None if prefix is None else f"{prefix}.{node.name}"
    return None


def _member_of(obj: Any, key: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        if key == "length":
            return len(obj)
        index = to_number(key)
        if math.isnan(index) or index < 0 or index >= len(obj):
            return None
        return obj[int(index)]

    if isinstance(obj, str):
        if key == "length":
            return len(obj)
        index = to_number(key)
        if math.isnan(index) or index < 0 or index >= len(obj):
            return None
        return obj[int(index)]

    if isinstance(obj, dict):
        return obj.get(key if isinstance(key, str) else to_text(key))

    return None


def _loose_equals(left: Any, right: Any) -> bool:
    number_types = (int, float)
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, number_types) and isinstance(right, number_types):
        return float(left) == float(right)
    if isinstance(left, number_types) and isinstance(right, str):
        other = to_number(right)
        return not math.isnan(other) and float(left) == other
    if isinstance(left, str) and isinstance(right, number_types):
        other = to_number(left)
        return not math.isnan(other) and other == float(right)
    return left == right


def _apply_binary(operator: str, left: Any, right: Any) -> Any:
    if operator == "+":
        if isinstance(left, str) or isinstance(right, str):
            return to_text(left) + to_text(right)
        return _number(to_number(left) + to_number(right))
    if operator == "-":
        return _number(to_number(left) - to_number(right))
    if operator == "*":
        return _number(to_number(left) * to_number(right))
    if operator == "/":
        divisor = to_number(right)
        if divisor == 0:
            return None  # JS yields Infinity/NaN, neither of which is JSON
        return _number(to_number(left) / divisor)
    if operator == "%":
        divisor = to_number(right)
        if divisor == 0:
            return None
        return _number(math.fmod(to_number(left), divisor))

    if operator == "==":
        return _loose_equals(left, right)
    if operator == "!=":
        return not _loose_equals(left, right)
    if operator == "===":
        return type(left) is type(right) and left == right
    if operator == "!==":
        return not (type(left) is type(right) and left == right)

    if isinstance(left, str) and isinstance(right, str):
        values: tuple[Any, Any] = (left, right)
    else:
        values = (to_number(left), to_number(right))
        if any(isinstance(value, float) and math.isnan(value) for value in values):
            return False

    if operator == "<":
        return values[0] < values[1]
    if operator == "<=":
        return values[0] <= values[1]
    if operator == ">":
        return values[0] > values[1]
    if operator == ">=":
        return values[0] >= values[1]

    raise EvaluationError(f"unknown operator {operator!r}")


def _evaluate_node(node: _Node | None, scope: dict[str, Any]) -> Any:
    if node is None:
        return None

    if node.kind == "literal":
        return node.value

    if node.kind == "identifier":
        if node.name in scope:
            return scope[node.name]
        if node.name in _builtins():
            raise EvaluationError(f"{node.name!r} must be called")
        return None

    if node.kind == "member":
        obj = _evaluate_node(node.left, scope)
        key = _evaluate_node(node.right, scope) if node.computed else node.name
        return _member_of(obj, key)

    if node.kind == "array":
        return [_evaluate_node(item, scope) for item in node.items]

    if node.kind == "call":
        name = _callee_name(node.left)
        if name is None:
            raise EvaluationError("only builtin functions can be called")
        builtin = _builtins().get(name)
        if builtin is None:
            raise EvaluationError(f"unknown function {name!r}")
        return builtin([_evaluate_node(item, scope) for item in node.items])

    if node.kind == "unary":
        operand = _evaluate_node(node.left, scope)
        if node.name == "!":
            return not is_truthy(operand)
        if node.name == "-":
            return _number(-to_number(operand))
        return _number(to_number(operand))

    if node.kind == "logical":
        left = _evaluate_node(node.left, scope)
        if node.name in ("&&", "and"):
            return _evaluate_node(node.right, scope) if is_truthy(left) else left
        return left if is_truthy(left) else _evaluate_node(node.right, scope)

    if node.kind == "binary":
        return _apply_binary(
            node.name,
            _evaluate_node(node.left, scope),
            _evaluate_node(node.right, scope),
        )

    if node.kind == "conditional":
        return (
            _evaluate_node(node.right, scope)
            if is_truthy(_evaluate_node(node.left, scope))
            else _evaluate_node(node.third, scope)
        )

    raise EvaluationError(f"unknown node {node.kind!r}")


class Expression:
    """A parsed expression. Parsing happens once — when a service is configured —
    so evaluating per input costs only the tree walk."""

    def __init__(self, source: str, root: _Node) -> None:
        self.source = source
        self._root = root

    @staticmethod
    def parse(source: str) -> "Expression":
        return Expression(source, _Parser(source).parse())

    def evaluate(self, params: Any = None, scope: dict[str, Any] | None = None) -> Any:
        return _evaluate_node(self._root, scope if scope is not None else {"params": params})


# ── Builtins ───────────────────────────────────────────────────────────────────


def _arg(args: list[Any], index: int, default: Any = None) -> Any:
    return args[index] if index < len(args) else default


def _matches(predicate: Expression, item: Any, index: int) -> bool:
    return is_truthy(predicate.evaluate(scope={"item": item, "index": index}))


def _find(args: list[Any]) -> Any:
    array, source = _arg(args, 0), _arg(args, 1)
    if not isinstance(array, list) or not isinstance(source, str):
        return None
    predicate = Expression.parse(source)
    for index, item in enumerate(array):
        if _matches(predicate, item, index):
            return item
    return None


def _filter(args: list[Any]) -> Any:
    array, source = _arg(args, 0), _arg(args, 1)
    if not isinstance(array, list) or not isinstance(source, str):
        return []
    predicate = Expression.parse(source)
    return [item for index, item in enumerate(array) if _matches(predicate, item, index)]


def _slice(args: list[Any]) -> Any:
    array = _arg(args, 0)
    if not isinstance(array, list):
        return []
    offset = int(to_number(_arg(args, 1, 0)) or 0)
    raw_step = to_number(_arg(args, 2, 1))
    step = 1 if math.isnan(raw_step) or raw_step < 1 else int(raw_step)
    raw_end = to_number(_arg(args, 3, len(array)))
    end = len(array) if math.isnan(raw_end) else int(raw_end)
    return array[max(0, offset) : end : step]


def _parse_json(args: list[Any]) -> Any:
    value = _arg(args, 0)
    if not isinstance(value, str):
        return "<parse undefined>"
    try:
        return json_module.loads(value)
    except ValueError:
        return None


def _reformat_date(args: list[Any]) -> Any:
    text = to_text(_arg(args, 0))
    parsed = _parse_date(text, to_text(_arg(args, 1)))
    if parsed is None:
        millis = _to_epoch_millis(_arg(args, 0))
        if millis is None:
            return text
        parsed = datetime.fromtimestamp(millis / 1000)
    return _format_date(parsed, to_text(_arg(args, 2)))


def _numeric_list(args: list[Any]) -> list[float] | None:
    array = _arg(args, 0)
    return [to_number(entry) for entry in array] if isinstance(array, list) else None


def _sum(args: list[Any]) -> Any:
    values = _numeric_list(args)
    return _arg(args, 0) if values is None else _number(math.fsum(values))


def _avg(args: list[Any]) -> Any:
    values = _numeric_list(args)
    if values is None:
        return _arg(args, 0)
    return None if not values else _number(math.fsum(values) / len(values))


def _flat_sum(args: list[Any]) -> Any:
    array = _arg(args, 0)
    if not isinstance(array, list):
        return 0
    total = 0.0
    for entry in array:
        if isinstance(entry, list):
            total += math.fsum(to_number(inner) for inner in entry)
        else:
            total += to_number(entry)
    return _number(total)


def _at(args: list[Any]) -> Any:
    array = _arg(args, 0)
    if not isinstance(array, list) or not array:
        return None
    return array[int(abs(round(to_number(_arg(args, 1, 0))))) % len(array)]


_BUILTINS: dict[str, Callable[[list[Any]], Any]] | None = None


def _builtins() -> dict[str, Callable[[list[Any]], Any]]:
    global _BUILTINS
    if _BUILTINS is not None:
        return _BUILTINS

    _BUILTINS = {
        "print": lambda args: print(to_text(_arg(args, 0))),
        "log": lambda args: print(to_text(_arg(args, 0))),
        "round": lambda args: _number(float(round(to_number(_arg(args, 0))))),
        "sin": lambda args: math.sin(to_number(_arg(args, 0))),
        "min": lambda args: _number(min((to_number(a) for a in args), default=math.nan)),
        "max": lambda args: _number(max((to_number(a) for a in args), default=math.nan)),
        "rand": lambda args: random.random(),
        "number": lambda args: _number(to_number(_arg(args, 0))),
        "string": lambda args: to_text(_arg(args, 0)),
        "stringify": lambda args: json_module.dumps(_arg(args, 0), separators=(",", ":")),
        "parse": _parse_json,
        "concat": lambda args: "".join(to_text(a) for a in args),
        "encodeURI": lambda args: quote(to_text(_arg(args, 0)), safe="-_.!~*'();/?:@&=+$,#"),
        "slug": lambda args: re.sub(r"[^a-z0-9_-]", "", to_text(_arg(args, 0)).lower()),
        "now": lambda args: _now_millis(),
        "range": lambda args: list(range(max(0, int(round(to_number(_arg(args, 0, 0))))))),
        "sum": _sum,
        "avg": _avg,
        "flatSum": _flat_sum,
        "at": _at,
        "slice": _slice,
        "find": _find,
        "filter": _filter,
        "isFuture": lambda args: (lambda ms: ms is not None and ms >= _now_millis())(
            _to_epoch_millis(_arg(args, 0))
        ),
        "isPast": lambda args: (lambda ms: ms is not None and ms < _now_millis())(
            _to_epoch_millis(_arg(args, 0))
        ),
        "formatNow": lambda args: _format_date(datetime.now(), to_text(_arg(args, 0))),
        "reformatDate": _reformat_date,
        "uuid.v4": lambda args: str(uuid_module.uuid4()),
        "uuid.v7": lambda args: _uuid_v7(),
        # Python conversions the previous evaluator exposed, kept so flat
        # templates written against it keep working.
        "len": lambda args: len(_arg(args, 0)) if hasattr(_arg(args, 0), "__len__") else None,
        "str": lambda args: to_text(_arg(args, 0)),
        "int": lambda args: int(to_number(_arg(args, 0))),
        "float": lambda args: to_number(_arg(args, 0)),
        "bool": lambda args: is_truthy(_arg(args, 0)),
        "abs": lambda args: _number(abs(to_number(_arg(args, 0)))),
        "list": lambda args: list(_arg(args, 0)) if isinstance(_arg(args, 0), (list, tuple)) else [],
    }
    return _BUILTINS


def compile_expression(source: Any) -> Callable[[Any], Any]:
    """Compiles an expression source into a callable. Non-string sources are
    constants and are returned as-is; a source that does not parse raises when
    called, so the caller decides how a broken term is reported."""
    if not isinstance(source, str):
        return lambda params: source

    try:
        expression = Expression.parse(source)
    except ParseError as error:
        message = str(error)

        def broken(params: Any) -> Any:
            raise EvaluationError(f"invalid expression {source!r}: {message}")

        return broken

    return expression.evaluate
