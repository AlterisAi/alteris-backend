"""Tests for loom.privacy: SensitivityLevel, PrivacyPolicy, redact(), classify_field().

Tests cover:
  - SensitivityLevel ordering and values
  - FIELD_SENSITIVITY completeness
  - PrivacyPolicy.max_cloud_sensitivity for all flag combos
  - PrivacyPolicy.can_send_field
  - redact() behavior under different policies
  - classify_field() known and unknown fields
  - Pre-built policy objects (LOCAL_ONLY, METADATA_CLOUD, CONTENT_CLOUD, FULL_CLOUD)
"""

import pytest

from loom.privacy import (
    CONTENT_CLOUD,
    FIELD_SENSITIVITY,
    FULL_CLOUD,
    LOCAL_ONLY,
    METADATA_CLOUD,
    PrivacyPolicy,
    SensitivityLevel,
    classify_field,
    redact,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SensitivityLevel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSensitivityLevel:
    def test_ordering(self):
        """Levels are ordered: PUBLIC < PRIVATE < SENSITIVE < CRITICAL."""
        assert SensitivityLevel.PUBLIC < SensitivityLevel.PRIVATE
        assert SensitivityLevel.PRIVATE < SensitivityLevel.SENSITIVE
        assert SensitivityLevel.SENSITIVE < SensitivityLevel.CRITICAL

    def test_values(self):
        assert SensitivityLevel.PUBLIC == 0
        assert SensitivityLevel.PRIVATE == 1
        assert SensitivityLevel.SENSITIVE == 2
        assert SensitivityLevel.CRITICAL == 3

    def test_count(self):
        assert len(SensitivityLevel) == 4

    def test_is_int(self):
        """SensitivityLevel is IntEnum, so it works as int."""
        assert SensitivityLevel.PRIVATE + 1 == SensitivityLevel.SENSITIVE

    def test_comparison_with_int(self):
        assert SensitivityLevel.PUBLIC <= 0
        assert SensitivityLevel.CRITICAL >= 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIELD_SENSITIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFieldSensitivity:
    def test_known_public_fields(self):
        assert FIELD_SENSITIVITY["source_name"] == SensitivityLevel.PUBLIC

    def test_known_private_fields(self):
        assert FIELD_SENSITIVITY["email"] == SensitivityLevel.PRIVATE
        assert FIELD_SENSITIVITY["phone"] == SensitivityLevel.PRIVATE
        assert FIELD_SENSITIVITY["display_name"] == SensitivityLevel.PRIVATE
        assert FIELD_SENSITIVITY["timestamp"] == SensitivityLevel.PRIVATE

    def test_known_sensitive_fields(self):
        assert FIELD_SENSITIVITY["body"] == SensitivityLevel.SENSITIVE
        assert FIELD_SENSITIVITY["subject"] == SensitivityLevel.SENSITIVE
        assert FIELD_SENSITIVITY["call_duration"] == SensitivityLevel.SENSITIVE

    def test_known_critical_fields(self):
        assert FIELD_SENSITIVITY["user_phone"] == SensitivityLevel.CRITICAL
        assert FIELD_SENSITIVITY["relationship_type"] == SensitivityLevel.CRITICAL
        assert FIELD_SENSITIVITY["behavioral_pattern"] == SensitivityLevel.CRITICAL
        assert FIELD_SENSITIVITY["inferred_role"] == SensitivityLevel.CRITICAL

    def test_all_values_are_sensitivity_levels(self):
        for field, level in FIELD_SENSITIVITY.items():
            assert isinstance(level, SensitivityLevel), f"{field} has invalid level"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PrivacyPolicy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPrivacyPolicy:
    def test_default_policy(self):
        """Default policy allows nothing to cloud."""
        p = PrivacyPolicy()
        assert p.allow_content_to_cloud is False
        assert p.allow_identifiers_to_cloud is False
        assert p.allow_metadata_to_cloud is False
        assert p.allow_inferences_to_cloud is False

    def test_frozen(self):
        """PrivacyPolicy is frozen dataclass."""
        p = PrivacyPolicy()
        with pytest.raises(AttributeError):
            p.allow_content_to_cloud = True

    def test_max_cloud_all_false(self):
        p = PrivacyPolicy()
        assert p.max_cloud_sensitivity() == SensitivityLevel.PUBLIC

    def test_max_cloud_metadata(self):
        p = PrivacyPolicy(allow_metadata_to_cloud=True)
        assert p.max_cloud_sensitivity() == SensitivityLevel.PRIVATE

    def test_max_cloud_identifiers(self):
        p = PrivacyPolicy(allow_identifiers_to_cloud=True)
        assert p.max_cloud_sensitivity() == SensitivityLevel.PRIVATE

    def test_max_cloud_content(self):
        p = PrivacyPolicy(allow_content_to_cloud=True)
        assert p.max_cloud_sensitivity() == SensitivityLevel.SENSITIVE

    def test_max_cloud_inferences(self):
        p = PrivacyPolicy(allow_inferences_to_cloud=True)
        assert p.max_cloud_sensitivity() == SensitivityLevel.CRITICAL

    def test_max_cloud_all_true(self):
        p = PrivacyPolicy(
            allow_content_to_cloud=True,
            allow_identifiers_to_cloud=True,
            allow_metadata_to_cloud=True,
            allow_inferences_to_cloud=True,
        )
        assert p.max_cloud_sensitivity() == SensitivityLevel.CRITICAL

    def test_can_send_field_public(self):
        """Default policy can send public fields."""
        p = PrivacyPolicy()
        assert p.can_send_field("source_name") is True

    def test_can_send_field_private_blocked(self):
        """Default policy blocks private fields."""
        p = PrivacyPolicy()
        assert p.can_send_field("email") is False

    def test_can_send_field_with_metadata(self):
        p = PrivacyPolicy(allow_metadata_to_cloud=True)
        assert p.can_send_field("email") is True
        assert p.can_send_field("phone") is True
        assert p.can_send_field("body") is False

    def test_can_send_field_with_content(self):
        p = PrivacyPolicy(allow_content_to_cloud=True)
        assert p.can_send_field("body") is True
        assert p.can_send_field("subject") is True
        assert p.can_send_field("relationship_type") is False

    def test_can_send_field_full_cloud(self):
        p = PrivacyPolicy(
            allow_content_to_cloud=True,
            allow_identifiers_to_cloud=True,
            allow_metadata_to_cloud=True,
            allow_inferences_to_cloud=True,
        )
        assert p.can_send_field("relationship_type") is True
        assert p.can_send_field("behavioral_pattern") is True

    def test_can_send_unknown_field_defaults_sensitive(self):
        """Unknown fields default to SENSITIVE."""
        p = PrivacyPolicy()
        assert p.can_send_field("totally_unknown_field_xyz") is False

    def test_retention_defaults(self):
        p = PrivacyPolicy()
        assert p.retain_raw_content is True
        assert p.content_retention_days is None
        assert p.encrypt_at_rest is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pre-built policies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPreBuiltPolicies:
    def test_local_only(self):
        assert LOCAL_ONLY.max_cloud_sensitivity() == SensitivityLevel.PUBLIC
        assert LOCAL_ONLY.allow_content_to_cloud is False

    def test_metadata_cloud(self):
        assert METADATA_CLOUD.max_cloud_sensitivity() == SensitivityLevel.PRIVATE
        assert METADATA_CLOUD.allow_metadata_to_cloud is True
        assert METADATA_CLOUD.allow_content_to_cloud is False

    def test_content_cloud(self):
        assert CONTENT_CLOUD.max_cloud_sensitivity() == SensitivityLevel.SENSITIVE
        assert CONTENT_CLOUD.allow_content_to_cloud is True
        assert CONTENT_CLOUD.allow_metadata_to_cloud is True

    def test_full_cloud(self):
        assert FULL_CLOUD.max_cloud_sensitivity() == SensitivityLevel.CRITICAL
        assert FULL_CLOUD.allow_inferences_to_cloud is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# redact()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRedact:
    def test_redact_default_policy(self):
        """Default policy strips everything except public fields."""
        data = {
            "source_name": "mail",
            "email": "a@b.com",
            "body": "Hello world",
            "relationship_type": "friend",
        }
        result = redact(data, LOCAL_ONLY)
        assert "source_name" in result
        assert "email" not in result
        assert "body" not in result
        assert "relationship_type" not in result

    def test_redact_metadata_policy(self):
        data = {
            "source_name": "mail",
            "email": "a@b.com",
            "body": "Hello world",
            "relationship_type": "friend",
        }
        result = redact(data, METADATA_CLOUD)
        assert "source_name" in result
        assert "email" in result
        assert "body" not in result
        assert "relationship_type" not in result

    def test_redact_content_policy(self):
        data = {
            "source_name": "mail",
            "email": "a@b.com",
            "body": "Hello world",
            "relationship_type": "friend",
        }
        result = redact(data, CONTENT_CLOUD)
        assert "source_name" in result
        assert "email" in result
        assert "body" in result
        assert "relationship_type" not in result

    def test_redact_full_cloud(self):
        data = {
            "source_name": "mail",
            "email": "a@b.com",
            "body": "Hello world",
            "relationship_type": "friend",
        }
        result = redact(data, FULL_CLOUD)
        assert len(result) == 4

    def test_redact_empty_dict(self):
        assert redact({}, LOCAL_ONLY) == {}

    def test_redact_preserves_values(self):
        data = {"source_name": "mail_app", "email": "test@test.com"}
        result = redact(data, METADATA_CLOUD)
        assert result["source_name"] == "mail_app"
        assert result["email"] == "test@test.com"

    def test_redact_unknown_fields_treated_as_sensitive(self):
        """Fields not in FIELD_SENSITIVITY default to SENSITIVE."""
        data = {"unknown_field": "secret"}
        result = redact(data, LOCAL_ONLY)
        assert "unknown_field" not in result
        result2 = redact(data, CONTENT_CLOUD)
        assert "unknown_field" in result2

    def test_redact_returns_new_dict(self):
        data = {"source_name": "mail"}
        result = redact(data, LOCAL_ONLY)
        assert result is not data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# classify_field()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClassifyField:
    def test_known_field(self):
        assert classify_field("body") == SensitivityLevel.SENSITIVE
        assert classify_field("email") == SensitivityLevel.PRIVATE
        assert classify_field("source_name") == SensitivityLevel.PUBLIC

    def test_unknown_field_defaults_sensitive(self):
        assert classify_field("totally_new_field") == SensitivityLevel.SENSITIVE

    def test_all_registered_fields(self):
        for field, expected in FIELD_SENSITIVITY.items():
            assert classify_field(field) == expected
