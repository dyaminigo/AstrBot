import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from astrbot.core.utils.error_redaction import redact_sensitive_text

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[_\-.])(?:api[_-]?key|access[_-]?token|auth(?:orization)?|credential|"
    r"password|private[_-]?key|refresh[_-]?token|secret|session[_-]?id)(?:$|[_\-.])",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
_OPAQUE_PATTERN = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_ACTION_IDENTIFIER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}$")
_STRUCTURED_SCALAR_PATTERN = re.compile(
    r"^(?P<key>[^\s:;,]{2,32})\s*:\s*(?P<value>[^\n]+)$"
)
_CODE_MARKERS = re.compile(
    r"(?:^|\s)(?:class|def|function|import|from|return|SELECT|INSERT|UPDATE|DELETE)\s"
    r"|[{};]\s*$|(?:&&|\|\||=>)",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ToolCallArgumentSummary:
    """Human-readable, safely redacted parts of a tool call notification."""

    actions: tuple[str, ...] = ()
    intention: str | None = None
    details: str | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: int
    order: int
    text: str
    source_id: int
    depth: int = 0


def _readable_key(key: object) -> str:
    text = str(key).strip().replace("_", " ").replace("-", " ")
    return " ".join(text.split()) or "parameter"


def _is_sensitive(key: object, schema: object) -> bool:
    if _SENSITIVE_KEY_PATTERN.search(str(key)):
        return True
    if not isinstance(schema, Mapping):
        return False
    if schema.get("writeOnly") is True or schema.get("format") == "password":
        return True
    description = schema.get("description")
    return isinstance(description, str) and bool(
        re.search(r"\b(?:credential|password|secret|token)\b", description, re.I)
    )


def _schema_for_property(schema: object, key: object) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    property_schema = properties.get(key)
    return property_schema if isinstance(property_schema, Mapping) else {}


def _looks_opaque(text: str) -> bool:
    compact = text.strip()
    if len(compact) < 32 or not _OPAQUE_PATTERN.fullmatch(compact):
        return False
    unique_ratio = len(set(compact)) / len(compact)
    entropy = -sum(
        (compact.count(char) / len(compact))
        * math.log2(compact.count(char) / len(compact))
        for char in set(compact)
    )
    return unique_ratio > 0.3 and entropy > 4.0


def _string_kind(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if _looks_opaque(stripped):
        return "opaque"
    if "\n" in stripped or "\r" in stripped or _CODE_MARKERS.search(stripped):
        return "code" if _CODE_MARKERS.search(stripped) else "multiline"
    if len(stripped) > 500:
        return "long"
    if _ACTION_IDENTIFIER_PATTERN.fullmatch(stripped):
        return "action"
    words = _WORD_PATTERN.findall(stripped)
    if len(words) >= 4 and len("".join(words)) >= len(stripped) * 0.45:
        return "natural"
    return "short"


def _format_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        text = redact_sensitive_text(value.strip())
        structured = _STRUCTURED_SCALAR_PATTERN.fullmatch(text)
        if structured:
            return (
                f"{_readable_key(structured.group('key'))}: "
                f"{structured.group('value').strip()}"
            )
        return text
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return redact_sensitive_text(str(value))
    except Exception:
        return f"<{type(value).__name__}>"


def format_tool_call_arguments(
    arguments: object,
    schema: Mapping[str, Any] | None = None,
) -> ToolCallArgumentSummary:
    """Build a safe semantic summary of arbitrary tool arguments.

    The ranking uses value shape, tree position, and JSON Schema metadata. It
    deliberately does not depend on tool names, MCP servers, or known parameter
    names.

    Args:
        arguments: Tool-call arguments from the model.
        schema: Optional JSON Schema for the arguments.

    Returns:
        Structured notification lines for nested actions, intention, and details.
    """
    if arguments is None:
        return ToolCallArgumentSummary()
    if not isinstance(arguments, Mapping):
        if isinstance(arguments, (str, int, float, bool)):
            return ToolCallArgumentSummary(
                details=f"value: {_format_scalar(arguments)}"
            )
        return ToolCallArgumentSummary(
            details=f"parameters: {type(arguments).__name__}"
        )
    if not arguments:
        return ToolCallArgumentSummary()

    actions: list[str] = []
    intentions: list[_Candidate] = []
    details: list[_Candidate] = []
    fallbacks: list[_Candidate] = []
    seen: set[int] = set()
    order = 0
    stack: list[tuple[Mapping[object, object], Mapping[str, Any], int, bool]] = [
        (arguments, schema or {}, 0, False)
    ]

    while stack:
        current, current_schema, depth, sequence_item = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        try:
            entries = list(current.items())
        except Exception:
            continue

        required = set()
        if isinstance(current_schema.get("required"), list):
            required = {str(item) for item in current_schema["required"]}

        for key, value in entries:
            order += 1
            property_schema = _schema_for_property(current_schema, key)
            required_bonus = 12 if str(key) in required else 0
            default_penalty = -16 if property_schema.get("default") == value else 0
            metadata_bonus = 0
            description = property_schema.get("description")
            if (
                isinstance(description, str)
                and len(_WORD_PATTERN.findall(description)) >= 3
            ):
                metadata_bonus = 4
            label = _readable_key(key)
            source_id = id(value)

            if _is_sensitive(key, property_schema):
                fallbacks.append(
                    _Candidate(-20, order, f"{label}: [REDACTED]", source_id, depth)
                )
                continue

            if value is None or isinstance(value, (bool, int, float)):
                score = required_bonus + default_penalty + (18 if depth > 0 else 8)
                details.append(
                    _Candidate(
                        score,
                        order,
                        f"{label}: {_format_scalar(value)}",
                        source_id,
                        depth,
                    )
                )
                continue

            if isinstance(value, str):
                stripped = value.strip()
                kind = _string_kind(stripped)
                if kind == "natural":
                    intentions.append(
                        _Candidate(
                            80 + required_bonus + metadata_bonus - depth * 2,
                            order,
                            redact_sensitive_text(stripped),
                            source_id,
                            depth,
                        )
                    )
                elif kind == "action":
                    if sequence_item and stripped not in actions:
                        actions.append(stripped)
                    else:
                        fallbacks.append(
                            _Candidate(
                                -10,
                                order,
                                f"{label}: {stripped}",
                                source_id,
                                depth,
                            )
                        )
                elif kind == "code":
                    line_count = len(value.splitlines()) or 1
                    details.append(
                        _Candidate(
                            24 + required_bonus,
                            order,
                            f"{label}: code, {line_count} line{'s' if line_count != 1 else ''}",
                            source_id,
                            depth,
                        )
                    )
                elif kind == "multiline":
                    line_count = len(value.splitlines())
                    details.append(
                        _Candidate(
                            22 + required_bonus,
                            order,
                            f"{label}: text, {line_count} lines",
                            source_id,
                            depth,
                        )
                    )
                elif kind in {"opaque", "long"}:
                    fallbacks.append(
                        _Candidate(
                            -12,
                            order,
                            f"{label}: text, {len(value)} characters",
                            source_id,
                            depth,
                        )
                    )
                elif kind == "short":
                    formatted = _format_scalar(value)
                    text = (
                        formatted
                        if _STRUCTURED_SCALAR_PATTERN.fullmatch(stripped)
                        else f"{label}: {formatted}"
                    )
                    details.append(
                        _Candidate(
                            34
                            + required_bonus
                            + metadata_bonus
                            + default_penalty
                            + (8 if depth > 0 else 0),
                            order,
                            text,
                            source_id,
                            depth,
                        )
                    )
                continue

            if isinstance(value, Mapping):
                count = len(value)
                fallbacks.append(
                    _Candidate(
                        2,
                        order,
                        f"{label}: {count} field{'s' if count != 1 else ''}",
                        source_id,
                        depth,
                    )
                )
                if depth < 3:
                    stack.append((value, property_schema, depth + 1, sequence_item))
                continue

            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                count = len(value)
                fallbacks.append(
                    _Candidate(
                        1,
                        order,
                        f"{label}: {count} item{'s' if count != 1 else ''}",
                        source_id,
                        depth,
                    )
                )
                if depth < 3:
                    item_schema = property_schema.get("items", {})
                    if not isinstance(item_schema, Mapping):
                        item_schema = {}
                    for item in reversed(value[:8]):
                        if isinstance(item, Mapping):
                            stack.append((item, item_schema, depth + 1, True))
                continue

            fallbacks.append(
                _Candidate(
                    0,
                    order,
                    f"{label}: {type(value).__name__}",
                    source_id,
                    depth,
                )
            )

    intention = None
    intention_source_id = None
    if intentions:
        best_intention = max(intentions, key=lambda item: (item.score, -item.order))
        intention = best_intention.text
        intention_source_id = best_intention.source_id

    ranked_details = sorted(details, key=lambda item: (-item.score, item.order))
    has_salient_nested_detail = any(
        candidate.depth > 0 and candidate.score >= 34 for candidate in ranked_details
    )
    has_semantic_context = bool(actions or intention) and any(
        candidate.score >= 34 for candidate in ranked_details
    )
    selected: list[str] = []
    for candidate in ranked_details:
        if candidate.source_id == intention_source_id or candidate.text in selected:
            continue
        if has_salient_nested_detail and candidate.depth == 0 and candidate.score < 20:
            continue
        if has_semantic_context and candidate.score < 20:
            continue
        selected.append(candidate.text)
        if len(selected) == 4:
            break

    if not selected and not actions and intention is None:
        for candidate in sorted(fallbacks, key=lambda item: (-item.score, item.order)):
            if candidate.text in selected:
                continue
            selected.append(candidate.text)
            if len(selected) == 3:
                break

    return ToolCallArgumentSummary(
        actions=tuple(actions),
        intention=intention,
        details=" · ".join(selected) if selected else None,
    )
