"""Tests for loom.profile — shared profile loader."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from loom.profile import (
    flatten_profile,
    format_profile_context,
    get_colleague_names,
    get_user_emails,
    get_user_name,
    load_profile,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures: sample profiles in both formats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V2_PROFILE = {
    "identity": {
        "name": "Alex Chen",
        "preferred_name": "Al",
        "emails": ["user@example.com", "alex@example.com"],
        "timezone": "America/Los_Angeles",
        "location": {"city": "Seattle", "neighborhood": "Magnolia"},
    },
    "professional": {
        "company": "Alteris",
        "role": "CTO and Co-founder",
        "context": "PhD in EE/ML, ex-Amazon robotics.",
        "key_colleagues": ["Sam P.", "Jordan L.", "Robin S."],
    },
    "family_and_relationships": {
        "immediate_family": [
            "Dana Kim (Wife)",
            "Maya Chen (Daughter, age 5.5)",
            "Leo (Son, age 2)",
        ],
    },
    "work_patterns": {
        "deep_work_windows": ["Pre-noon at home office", "21:00-23:30: Push work"],
        "non_negotiable_blocks": [
            "07:25: School drop-off",
            "15:00: Maya pickup",
            "18:00: Dinner",
        ],
    },
}

V1_PROFILE = {
    "name": "Alex Chen",
    "emails": ["user@example.com"],
    "phones": ["+1-555-0100"],
    "timezone": "America/Los_Angeles",
    "role": "CTO",
    "context": "Early-stage startup founder.",
    "family": {
        "spouse": "Dana Kim",
        "children": [
            {"name": "Maya", "nicknames": ["May"]},
        ],
        "care_providers": [
            {"name": "Flor", "role": "babysitter"},
        ],
    },
    "key_colleagues": ["Sam P.", "Jordan L."],
    "sensitivity_mode": "cloud_all",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: format_profile_context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_format_empty():
    assert format_profile_context(None) == ""
    assert format_profile_context({}) == ""


def test_format_v2_has_all_sections():
    result = format_profile_context(V2_PROFILE)
    assert result.startswith("USER PROFILE:")
    assert "Alex Chen (Al)" in result
    assert "CTO and Co-founder at Alteris" in result
    assert "Sam P." in result
    assert "Dana Kim" in result
    assert "America/Los_Angeles" in result
    assert "Pre-noon" in result
    assert "07:25" in result


def test_format_v1_has_key_sections():
    result = format_profile_context(V1_PROFILE)
    assert result.startswith("USER PROFILE:")
    assert "Alex Chen" in result
    assert "CTO" in result
    assert "Sam P." in result
    assert "Dana Kim" in result


def test_format_context_is_compact():
    """Profile context should be <400 tokens (~1600 chars)."""
    result = format_profile_context(V2_PROFILE)
    assert len(result) < 1600


def test_format_truncates_long_context():
    profile = {
        "professional": {
            "role": "Engineer",
            "context": "x" * 500,
        },
    }
    result = format_profile_context(profile)
    assert "..." in result
    assert len(result) < 600


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: get_user_emails
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_emails_v2():
    assert get_user_emails(V2_PROFILE) == ["user@example.com", "alex@example.com"]


def test_emails_v1():
    assert get_user_emails(V1_PROFILE) == ["user@example.com"]


def test_emails_empty():
    assert get_user_emails({}) == []
    assert get_user_emails(None) == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: get_user_name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_name_v2():
    assert get_user_name(V2_PROFILE) == "Alex Chen"


def test_name_v1():
    assert get_user_name(V1_PROFILE) == "Alex Chen"


def test_name_empty():
    assert get_user_name({}) == ""
    assert get_user_name(None) == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: get_colleague_names
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_colleagues_v2():
    names = get_colleague_names(V2_PROFILE)
    assert "Sam P." in names
    assert "Jordan L." in names
    assert len(names) == 3


def test_colleagues_v1():
    names = get_colleague_names(V1_PROFILE)
    assert "Sam P." in names
    assert len(names) == 2


def test_colleagues_empty():
    assert get_colleague_names({}) == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: flatten_profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_flatten_v2():
    flat = flatten_profile(V2_PROFILE)
    assert flat["name"] == "Alex Chen"
    assert flat["emails"] == ["user@example.com", "alex@example.com"]
    assert flat["role"] == "CTO and Co-founder"
    assert flat["timezone"] == "America/Los_Angeles"


def test_flatten_v1_passthrough():
    """v1 format already has flat keys — returns as-is."""
    flat = flatten_profile(V1_PROFILE)
    assert flat is V1_PROFILE


def test_flatten_empty():
    assert flatten_profile({}) == {}
    assert flatten_profile(None) == {}


def test_flatten_preserves_sensitivity():
    p = dict(V2_PROFILE)
    p["sensitivity_mode"] = "local_only"
    flat = flatten_profile(p)
    assert flat["sensitivity_mode"] == "local_only"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: load_profile (with mocked filesystem)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_load_profile_yaml(tmp_path):
    """load_profile finds YAML in LOOM_DIR."""
    import yaml

    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text(yaml.dump(V2_PROFILE))

    with patch("loom.constants.LOOM_DIR", tmp_path):
        result = load_profile()

    assert result["identity"]["name"] == "Alex Chen"


def test_load_profile_json_fallback(tmp_path):
    """load_profile falls back to config.json."""
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"user": {"emails": ["test@example.com"]}}))

    with patch("loom.constants.LOOM_DIR", tmp_path), \
         patch("pathlib.Path.home", return_value=tmp_path / "nope"):
        result = load_profile()

    assert result["emails"] == ["test@example.com"]


def test_load_profile_no_files(tmp_path):
    """load_profile returns {} when no config files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with patch("loom.constants.LOOM_DIR", empty_dir), \
         patch("pathlib.Path.home", return_value=tmp_path / "nope"):
        result = load_profile()
    assert result == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tests: edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_format_partial_profile():
    """Profile with only name should still produce output."""
    result = format_profile_context({"identity": {"name": "Test User"}})
    assert "Test User" in result


def test_format_v1_family_string_children():
    """v1 format with string children (not dicts)."""
    profile = {"family": {"spouse": "Jane", "children": ["Kid1", "Kid2"]}}
    result = format_profile_context(profile)
    assert "Jane" in result
    assert "Kid1" in result


def test_colleagues_non_list():
    """Gracefully handle non-list key_colleagues."""
    profile = {"professional": {"key_colleagues": "not a list"}}
    result = get_colleague_names(profile)
    assert result == []


def test_get_user_emails_non_list():
    """Gracefully handle non-list emails."""
    profile = {"identity": {"emails": "not a list"}}
    result = get_user_emails(profile)
    assert result == []
