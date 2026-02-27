"""Shared profile loader — single source of truth for user identity.

Consolidates 5 independent profile loaders scattered across cli.py,
briefing.py, person_model.py, cq_tools.py, and cross_source.py.

Profile is loaded from ~/.alteris/profile.yaml (preferred) or
~/.alteris/profile.yaml (legacy), with JSON config fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_profile() -> dict:
    """Load profile.yaml from ~/.alteris or ~/.loom (legacy), JSON fallback.

    Returns the raw dict from YAML parsing. Handles both the v2
    hierarchical format (~/.alteris) and v1 flat format (~/.loom legacy).
    Returns empty dict on failure.
    """
    from alteris.constants import ALTERIS_DIR

    for yaml_path in [
        ALTERIS_DIR / "profile.yaml",
        Path.home() / ".alteris" / "profile.yaml",
    ]:
        if not yaml_path.exists():
            continue
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text())
            if data and isinstance(data, dict):
                return data
        except ImportError:
            # Fall back to manual parsing if PyYAML not installed
            try:
                return _parse_yaml_manual(yaml_path)
            except Exception:
                continue
        except Exception:
            continue

    # Fall back to JSON configs
    import json as _json
    for path in [ALTERIS_DIR / "config.json", Path.home() / ".alteris" / "config.json"]:
        if path.exists():
            try:
                data = _json.loads(path.read_text())
                return data.get("user", data)
            except Exception:
                continue
    return {}


def _parse_yaml_manual(yaml_path: Path) -> dict:
    """Minimal manual YAML parser for name/emails/phones when PyYAML is missing."""
    import ast

    text = yaml_path.read_text()
    result: dict = {}
    for line in text.splitlines():
        if line.startswith("name:"):
            result["name"] = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("emails:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith("["):
                result["emails"] = ast.literal_eval(raw)
        elif line.startswith("phones:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith("["):
                result["phones"] = ast.literal_eval(raw)
        elif line.startswith("sensitivity_mode:"):
            result["sensitivity_mode"] = line.split(":", 1)[1].strip().strip("'\"")
    return result


def format_profile_context(profile: dict | None) -> str:
    """Format profile into a compact text block for LLM prompt injection.

    Handles both hierarchical (.alteris format) and flat (.loom legacy format).
    Returns empty string if profile is empty/None.

    Output is ~250 tokens, negligible overhead per prompt.
    """
    if not profile:
        return ""

    parts = ["USER PROFILE:"]

    # Name
    name = get_user_name(profile)
    preferred = (
        profile.get("identity", {}).get("preferred_name")
        if isinstance(profile.get("identity"), dict) else None
    )
    if name:
        name_str = f"{name} ({preferred})" if preferred else name
        parts.append(f"  Name: {name_str}")

    # Role
    prof = profile.get("professional", {})
    role = prof.get("role", "") if isinstance(prof, dict) else ""
    company = prof.get("company", "") if isinstance(prof, dict) else ""
    context = prof.get("context", "") if isinstance(prof, dict) else ""
    # Fall back to v1 flat keys
    if not role:
        role = profile.get("role", "")
    if not context:
        context = profile.get("context", "")
    if role or company:
        parts.append(f"  Role: {role}{' at ' + company if company else ''}")
    if context:
        if len(context) > 200:
            context = context[:197] + "..."
        parts.append(f"  Context: {context}")

    # Key colleagues
    colleagues = _get_colleagues(profile)
    if colleagues:
        parts.append(f"  Key colleagues: {', '.join(colleagues)}")

    # Family
    family_str = _get_family_summary(profile)
    if family_str:
        parts.append(f"  Family: {family_str}")

    # Timezone / location
    tz = (
        profile.get("identity", {}).get("timezone")
        if isinstance(profile.get("identity"), dict)
        else profile.get("timezone")
    )
    location = None
    identity = profile.get("identity", {})
    if isinstance(identity, dict):
        loc = identity.get("location", {})
        if isinstance(loc, dict):
            city = loc.get("city", "")
            neighborhood = loc.get("neighborhood", "")
            if city:
                location = f"{neighborhood}, {city}" if neighborhood else city
    if not location and profile.get("home"):
        home = profile["home"]
        if isinstance(home, dict):
            location = home.get("city", "")
    if tz:
        loc_suffix = f" ({location})" if location else ""
        parts.append(f"  Timezone: {tz}{loc_suffix}")

    # Work patterns
    wp = profile.get("work_patterns", {})
    if isinstance(wp, dict):
        deep = wp.get("deep_work_windows", [])
        if deep and isinstance(deep, list):
            parts.append(f"  Work patterns: {'; '.join(str(w) for w in deep[:3])}")
        non_neg = wp.get("non_negotiable_blocks", [])
        if non_neg and isinstance(non_neg, list):
            parts.append(f"  Non-negotiable: {'; '.join(str(b) for b in non_neg[:6])}")

    if len(parts) <= 1:
        return ""
    return "\n".join(parts)


def get_user_emails(profile: dict) -> list[str]:
    """Extract emails from either format."""
    if not profile:
        return []
    # v2 hierarchical format
    identity = profile.get("identity", {})
    if isinstance(identity, dict):
        emails = identity.get("emails", [])
        if emails and isinstance(emails, list):
            return emails
    # v1 flat format
    emails = profile.get("emails", [])
    if isinstance(emails, list):
        return emails
    return []


def get_user_name(profile: dict) -> str:
    """Extract name from either format."""
    if not profile:
        return ""
    identity = profile.get("identity", {})
    if isinstance(identity, dict):
        name = identity.get("name", "")
        if name:
            return name
    return profile.get("name", "")


def get_colleague_names(profile: dict) -> list[str]:
    """Extract key colleague names from profile (for propagation boost)."""
    return _get_colleagues(profile)


def flatten_profile(profile: dict) -> dict:
    """Normalize hierarchical format to flat keys for backward compat.

    Maps v2 hierarchical keys to v1 flat keys so callers that expect
    the flat format (e.g. cli.py config for user_email) still work.
    """
    if not profile:
        return {}

    # Already flat format — return as-is
    if "emails" in profile or "phones" in profile:
        return profile

    result: dict = {}

    identity = profile.get("identity", {})
    if isinstance(identity, dict):
        if identity.get("name"):
            result["name"] = identity["name"]
        if identity.get("emails"):
            result["emails"] = identity["emails"]
        if identity.get("timezone"):
            result["timezone"] = identity["timezone"]

    prof = profile.get("professional", {})
    if isinstance(prof, dict):
        if prof.get("role"):
            result["role"] = prof["role"]
        if prof.get("company"):
            result["company"] = prof["company"]

    # Carry through sensitivity_mode if present
    if profile.get("sensitivity_mode"):
        result["sensitivity_mode"] = profile["sensitivity_mode"]

    return result


def _get_colleagues(profile: dict) -> list[str]:
    """Extract colleague names from either format."""
    prof = profile.get("professional", {})
    if isinstance(prof, dict):
        colleagues = prof.get("key_colleagues", [])
        if colleagues and isinstance(colleagues, list):
            return [str(c) for c in colleagues]
    # v1 flat
    colleagues = profile.get("key_colleagues", [])
    if isinstance(colleagues, list):
        return [str(c) for c in colleagues]
    return []


def _get_family_summary(profile: dict) -> str:
    """Build compact family summary string."""
    parts = []

    # v2 hierarchical format
    far = profile.get("family_and_relationships", {})
    if isinstance(far, dict):
        immediate = far.get("immediate_family", [])
        if isinstance(immediate, list):
            parts.extend(str(m) for m in immediate[:5])
        if parts:
            return "; ".join(parts)

    # v1 flat format
    family = profile.get("family", {})
    if isinstance(family, dict):
        if family.get("spouse"):
            parts.append(f"{family['spouse']} (spouse)")
        children = family.get("children", [])
        for child in children:
            if isinstance(child, dict) and child.get("name"):
                parts.append(child["name"])
            elif isinstance(child, str):
                parts.append(child)

    return "; ".join(parts)
