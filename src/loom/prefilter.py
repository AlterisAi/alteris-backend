"""Deterministic machine-generated email classifier.

Pre-filters automated/notification emails BEFORE any LLM call.
Pure Python, zero tokens, <1ms per event.

Integration: called from score.py during Stage 2 scoring. Events
classified as machine-generated get route=skip and never reach
triage or extraction.

Architecture principle (Section 3.1): "Extract maximum value before
spending LLM tokens. If you can compute it with SQL or Python,
do not call an LLM."
"""

from __future__ import annotations

import re

from loom.models import Event

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sender patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_AUTOMATED_SENDER_PATTERNS = frozenset({
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "notification", "notifications", "mailer-daemon",
    "postmaster", "bounce", "auto-confirm", "automated",
})

_NOTIFICATION_DOMAINS = frozenset({
    "linkedin.com", "facebookmail.com", "github.com",
    "accounts.google.com", "amazonses.com", "sendgrid.net",
    "mailchimp.com", "hubspot.com", "intercom.io",
    "slack.com", "asana.com", "atlassian.net",
    "zoom.us", "calendly.com", "stripe.com",
    "shopify.com", "squarespace.com", "substack.com",
    "notion.so", "figma.com", "vercel.com",
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Subject patterns (compiled once at import time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MACHINE_SUBJECT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(re:\s*)?your .+ (receipt|order|invoice|statement|confirmation)",
        r"^(re:\s*)?verify your",
        r"^(re:\s*)?confirm your",
        r"^(re:\s*)?reset your password",
        r"^(re:\s*)?security alert",
        r"^(re:\s*)?sign.in .+ new (device|location|browser)",
        r"^(re:\s*)?\d+% off",
        r"^(re:\s*)?welcome to",
        r"^(re:\s*)?thanks for (signing up|registering|subscribing)",
        r"^(re:\s*)?your .+ has (shipped|been delivered)",
        r"^(re:\s*)?payment (received|processed|confirmed)",
    ]
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Actionable-automated allowlist (promotes back to human pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ACTIONABLE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"flight.+(changed|cancelled|delayed)",
        r"your (appointment|reservation).+(changed|cancelled|rescheduled)",
        r"(confirm|confirmation).+(appointment|reservation|visit)",
        r"appointment.+(confirm|required|reminder|upcoming)",
        r"payment.+(failed|declined|overdue|past due|due today|due soon)",
        r"(bill|statement).+(ready|due|amount due|past due)",
        r"account.+(suspended|locked|compromised)",
    ]
]


def is_machine_generated(event: Event) -> tuple[bool, str]:
    """Classify an email event as machine-generated using metadata signals.

    Returns (is_machine, reason). Only applies to mail events — non-mail
    events always return (False, "").

    Detection signals checked in order (short-circuit on first match):
      1. Apple Mail 'automated' flag (strongest single signal)
      2. Sender local-part patterns (noreply, notifications, etc.)
      3. Known notification domains
      4. Subject line patterns (receipts, verifications, shipping, etc.)
      5. Body structure signals (link-dense, unsubscribe)

    Narrow actionable-automated allowlist rescues flight changes,
    payment failures, etc. back to the human pipeline.
    """
    if event.source != "mail":
        return False, ""

    meta = event.metadata

    # Safety valve: never prefilter emails the user interacted with.
    # replied=True means the user found it worth responding to.
    # is_from_me=True means the user authored it.
    if meta.get("replied") or meta.get("is_from_me") or meta.get("flagged"):
        return False, ""

    # Signal 1: Apple Mail automated flag — strongest single metadata signal
    if meta.get("automated") and meta.get("list_id"):
        return _check_actionable(event, "automated+list_id")

    # Signal 2: Sender patterns
    sender = _extract_sender_email(event)
    sender_local = sender.split("@")[0] if "@" in sender else ""
    sender_domain = sender.split("@")[-1] if "@" in sender else ""

    if sender_local:
        for pattern in _AUTOMATED_SENDER_PATTERNS:
            if pattern in sender_local:
                return _check_actionable(
                    event, f"automated sender: {sender_local}"
                )

    # Signal 3: Known notification domains
    if sender_domain in _NOTIFICATION_DOMAINS:
        # Only kill if also automated or noreply — some domains
        # send legitimate person-to-person email (e.g., github.com
        # for PR reviews vs notifications)
        if meta.get("automated") or meta.get("sender_is_noreply"):
            return _check_actionable(
                event, f"notification domain+automated: {sender_domain}"
            )

    # Signal 4: Subject line patterns
    subject = (meta.get("subject") or "").strip()
    for pattern in _MACHINE_SUBJECT_PATTERNS:
        if pattern.search(subject):
            # Subject match alone is weak — require corroboration
            # with at least one metadata signal
            if meta.get("automated") or meta.get("sender_is_noreply") or meta.get("list_id"):
                return _check_actionable(
                    event, f"machine subject+metadata: {subject[:50]}"
                )

    # Signal 5: Body structure — link-dense newsletters
    body = (event.raw_content or "")[:500]
    if body.count("http") > 5 and (meta.get("automated") or meta.get("list_id")):
        return _check_actionable(event, "link-dense+automated body")

    return False, ""


def _check_actionable(event: Event, reason: str) -> tuple[bool, str]:
    """Check if a machine-generated email is actionable (should be rescued)."""
    subject = (event.metadata.get("subject") or "")
    body = (event.raw_content or "")[:500]
    text = subject + " " + body

    for pattern in _ACTIONABLE_PATTERNS:
        if pattern.search(text):
            return False, ""  # Rescue: actionable automated email

    return True, reason


def _extract_sender_email(event: Event) -> str:
    """Extract sender email from event participants."""
    if not event.metadata.get("is_from_me") and event.participants:
        # First participant is typically the sender for inbound mail
        first = event.participants[0]
        # Handle "Name <email>" format
        if "<" in first and ">" in first:
            return first[first.index("<") + 1:first.index(">")].lower()
        if "@" in first:
            return first.lower()
    return ""
