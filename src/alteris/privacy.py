"""Privacy classification, sensitivity levels, and redaction policy.

Every field in the system carries a sensitivity level. Processing decisions
(local vs cloud, retention, export) are governed by the user's PrivacyPolicy.

Sensitivity levels (ascending severity):
  PUBLIC     -- display names (as chosen by the user), publicly visible info
  PRIVATE    -- identifiers: phone numbers, emails, message counts
  SENSITIVE  -- content: message text, call durations, timing patterns
  CRITICAL   -- inferences: relationship types, behavioral patterns, health signals
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any


@unique
class SensitivityLevel(IntEnum):
    """Data sensitivity level. Higher = more sensitive."""
    PUBLIC = 0
    PRIVATE = 1
    SENSITIVE = 2
    CRITICAL = 3


FIELD_SENSITIVITY: dict[str, SensitivityLevel] = {
    # Identity
    "display_name": SensitivityLevel.PRIVATE,
    "phone": SensitivityLevel.PRIVATE,
    "email": SensitivityLevel.PRIVATE,
    "contact_jid": SensitivityLevel.PRIVATE,
    "user_phone": SensitivityLevel.CRITICAL,

    # Event content
    "body": SensitivityLevel.SENSITIVE,
    "body_preview": SensitivityLevel.SENSITIVE,
    "subject": SensitivityLevel.SENSITIVE,
    "attachment_name": SensitivityLevel.SENSITIVE,

    # Metadata
    "timestamp": SensitivityLevel.PRIVATE,
    "message_count": SensitivityLevel.PRIVATE,
    "sent_count": SensitivityLevel.PRIVATE,
    "recv_count": SensitivityLevel.PRIVATE,

    # Behavioral signals
    "timing_histogram": SensitivityLevel.SENSITIVE,
    "call_duration": SensitivityLevel.SENSITIVE,
    "language": SensitivityLevel.SENSITIVE,
    "response_time_avg": SensitivityLevel.SENSITIVE,

    # Inferences
    "relationship_type": SensitivityLevel.CRITICAL,
    "behavioral_pattern": SensitivityLevel.CRITICAL,
    "sentiment_analysis": SensitivityLevel.CRITICAL,
    "communication_style": SensitivityLevel.CRITICAL,
    "inferred_role": SensitivityLevel.CRITICAL,
    "inferred_location": SensitivityLevel.CRITICAL,

    # Aggregate (safe for cloud when anonymized)
    "message_type_distribution": SensitivityLevel.PRIVATE,
    "country_code": SensitivityLevel.PRIVATE,
    "source_name": SensitivityLevel.PUBLIC,
}


@dataclass(frozen=True)
class PrivacyPolicy:
    """Controls what data can leave the device and how it's handled.

    Default policy: nothing leaves the device. Users opt in to
    cloud processing with explicit understanding of what's sent.
    """
    allow_content_to_cloud: bool = False
    allow_identifiers_to_cloud: bool = False
    allow_metadata_to_cloud: bool = False
    allow_inferences_to_cloud: bool = False

    local_model: str = ""
    cloud_model: str = ""

    retain_raw_content: bool = True
    content_retention_days: int | None = None

    encrypt_at_rest: bool = True
    allow_third_party_inference: bool = False

    def max_cloud_sensitivity(self) -> SensitivityLevel:
        """Highest sensitivity level allowed to leave the device."""
        if self.allow_inferences_to_cloud:
            return SensitivityLevel.CRITICAL
        if self.allow_content_to_cloud:
            return SensitivityLevel.SENSITIVE
        if self.allow_identifiers_to_cloud:
            return SensitivityLevel.PRIVATE
        if self.allow_metadata_to_cloud:
            return SensitivityLevel.PRIVATE
        return SensitivityLevel.PUBLIC

    def can_send_field(self, field_name: str) -> bool:
        """Check if a specific field can be sent to cloud."""
        level = FIELD_SENSITIVITY.get(field_name, SensitivityLevel.SENSITIVE)
        return level <= self.max_cloud_sensitivity()


LOCAL_ONLY = PrivacyPolicy()

METADATA_CLOUD = PrivacyPolicy(
    allow_metadata_to_cloud=True,
)

CONTENT_CLOUD = PrivacyPolicy(
    allow_content_to_cloud=True,
    allow_metadata_to_cloud=True,
)

FULL_CLOUD = PrivacyPolicy(
    allow_content_to_cloud=True,
    allow_identifiers_to_cloud=True,
    allow_metadata_to_cloud=True,
    allow_inferences_to_cloud=True,
)


def redact(data: dict[str, Any], policy: PrivacyPolicy) -> dict[str, Any]:
    """Strip fields that exceed the policy's cloud threshold.

    Returns a new dict with sensitive fields removed entirely.
    """
    threshold = policy.max_cloud_sensitivity()
    result = {}
    for key, value in data.items():
        level = FIELD_SENSITIVITY.get(key, SensitivityLevel.SENSITIVE)
        if level <= threshold:
            result[key] = value
    return result


def classify_field(field_name: str) -> SensitivityLevel:
    """Look up the sensitivity level of a named field."""
    return FIELD_SENSITIVITY.get(field_name, SensitivityLevel.SENSITIVE)
