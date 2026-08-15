from copy import deepcopy

import pytest

from astrbot.core.utils.tool_call_formatter import (
    ToolCallArgumentSummary,
    format_tool_call_arguments,
)


@pytest.mark.parametrize("arguments", [None, {}])
def test_empty_arguments_have_no_summary(arguments):
    assert format_tool_call_arguments(arguments) == ToolCallArgumentSummary()


def test_scalar_fields_are_readable_and_not_mutated():
    arguments = {
        "location_name": "Prague",
        "days": 5,
        "include_hourly": True,
        "threshold": None,
    }
    original = deepcopy(arguments)

    summary = format_tool_call_arguments(arguments)

    assert summary.intention is None
    assert summary.details == (
        "location name: Prague · days: 5 · include hourly: yes · threshold: null"
    )
    assert arguments == original


def test_natural_language_value_becomes_primary_summary():
    summary = format_tool_call_arguments(
        {
            "opaque_reference": "gain_21",
            "user_intent": "Merge the two pending changes using the squash method",
            "dry_run": False,
        }
    )

    assert summary.intention == "Merge the two pending changes using the squash method"
    assert summary.details == "opaque reference: gain_21"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("script", "def main():\n    return 1\n", "script: code, 2 lines"),
        ("notes", "first line\nsecond line\nthird line", "notes: text, 3 lines"),
        ("blob", "A" * 600, "blob: text, 600 characters"),
    ],
)
def test_large_or_multiline_strings_are_summarized(key, value, expected):
    assert format_tool_call_arguments({key: value}).details == expected


def test_nested_objects_and_arrays_are_compact():
    summary = format_tool_call_arguments(
        {
            "filters": {"state": "open", "labels": ["bug", "urgent"]},
            "operations": [{"target": "one"}, {"target": "two"}],
        }
    )

    assert summary.intention is None
    assert summary.details is not None
    assert "state: open" in summary.details
    assert "target: one" in summary.details
    assert "target: two" in summary.details


def test_large_array_is_not_expanded():
    summary = format_tool_call_arguments({"items": list(range(100))})

    assert summary.intention is None
    assert summary.details == "items: 100 items"


def test_secrets_are_redacted_by_key_schema_and_value_pattern():
    schema = {
        "type": "object",
        "properties": {
            "login": {"type": "string", "format": "password"},
            "message": {"type": "string"},
        },
    }

    summary = format_tool_call_arguments(
        {
            "api_key": "do-not-show",
            "login": "also-hidden",
            "message": "Use bearer abcdefghijklmnopqrstuvwxyz for this request",
        },
        schema,
    )

    assert summary.intention == "Use bearer [REDACTED] for this request"
    assert summary.details is None or "do-not-show" not in summary.details


def test_schema_required_and_default_affect_ranking():
    schema = {
        "type": "object",
        "required": ["target"],
        "properties": {
            "target": {"type": "string", "description": "Object being changed"},
            "mode": {"type": "string", "default": "normal"},
        },
    }

    summary = format_tool_call_arguments(
        {"mode": "normal", "target": "project-alpha"}, schema
    )

    assert summary.intention is None
    assert summary.details == "target: project-alpha · mode: normal"


def test_equally_informative_candidates_use_stable_argument_order():
    summary = format_tool_call_arguments(
        {
            "first": "Review the pending notification settings for this account",
            "second": "Update the selected notification settings for this account",
        }
    )

    assert summary.intention == (
        "Review the pending notification settings for this account"
    )
    assert summary.details is None


def test_deep_and_recursive_values_do_not_recurse_forever():
    recursive: dict[str, object] = {"level": 1}
    recursive["child"] = recursive
    deep: dict[str, object] = {"leaf": "visible"}
    for index in range(20):
        deep = {f"level_{index}": deep}

    summary = format_tool_call_arguments({"recursive": recursive, "deep": deep})

    assert summary.intention is None
    assert summary.details is not None
    assert "level: 1" in summary.details


def test_non_mapping_and_nonstandard_values_degrade_gracefully():
    class OddValue:
        pass

    assert format_tool_call_arguments([1, 2]).details == "parameters: list"
    summary = format_tool_call_arguments({"unknown_payload": OddValue()})
    assert summary.intention is None
    assert summary.details == "unknown payload: OddValue"


def test_nested_operation_identifiers_are_preserved_without_known_field_names():
    summary = format_tool_call_arguments(
        {
            "batch": [
                {
                    "kind": "ALPHA_FETCH_RECORDS",
                    "payload": {"owner": "sample-user", "project": "sample-repo"},
                },
                {
                    "kind": "BETA_UPDATE_ENTRY",
                    "payload": {"entry": "Planning meeting", "day": "2026-08-18"},
                },
            ],
            "transport": {"opaque": "random-session-value"},
            "background": False,
        }
    )

    assert summary.actions == ("ALPHA_FETCH_RECORDS", "BETA_UPDATE_ENTRY")
    assert summary.details is not None
    assert "owner: sample-user" in summary.details
    assert "project: sample-repo" in summary.details
    assert "batch:" not in summary.details


def test_discovery_like_array_prefers_intention_and_structured_context_to_counts():
    summary = format_tool_call_arguments(
        {
            "requests": [
                {
                    "description": "Get the latest workflow runs in a repository or fork",
                    "context": "repo:sample-repo",
                },
                {
                    "description": "Find recent automation runs for the selected project",
                    "context": "owner:sample-user",
                },
            ],
            "runtime": {"generate": True},
        }
    )

    assert summary.actions == ()
    assert summary.intention == ("Get the latest workflow runs in a repository or fork")
    assert summary.details is not None
    assert "repo: sample-repo" in summary.details
    assert "owner: sample-user" in summary.details
    assert "requests:" not in summary.details
    assert "generate:" not in summary.details


def test_execution_like_payload_omits_low_value_root_booleans_when_context_exists():
    summary = format_tool_call_arguments(
        {
            "operations": [
                {
                    "operation": "OMEGA_LIST_RUNS",
                    "arguments": {
                        "account": "sample-user",
                        "repository": "sample-repo",
                    },
                }
            ],
            "explanation": "Fetch the latest workflow runs for the selected repository",
            "synchronous": False,
            "phase": "FETCHING_WORKFLOW_RUNS",
        }
    )

    assert summary.actions == ("OMEGA_LIST_RUNS",)
    assert summary.intention == (
        "Fetch the latest workflow runs for the selected repository"
    )
    assert summary.details == "account: sample-user · repository: sample-repo"
