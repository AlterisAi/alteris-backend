"""Structural annotation extraction.

Extracts lens-independent faceted observations from event metadata.
Runs after ingestion, before scoring. Every event gets annotations
regardless of how it will score — annotations are permanent, projections
are disposable.

No LLM calls. Pure metadata parsing.
"""

from __future__ import annotations

import logging
import re
import time

from alteris.models import Annotation, Event
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Email address pattern for domain extraction
_EMAIL_RE = re.compile(r"[\w.+-]+@([\w.-]+\.\w+)")


def _extract_domain(participant: str) -> str | None:
    """Extract domain from an email address or participant string."""
    m = _EMAIL_RE.search(participant)
    return m.group(1).lower() if m else None


def _mail_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    # Sender domain
    if event.participants:
        domain = _extract_domain(event.participants[0])
        if domain:
            anns.append(Annotation(event_id=eid, facet="sender_domain", value=domain))

    # Platform classifier signals
    if meta.get("automated"):
        anns.append(Annotation(event_id=eid, facet="is_automated", value="true"))
    if meta.get("list_id"):
        anns.append(Annotation(event_id=eid, facet="has_list_id", value="true"))

    cat = meta.get("model_category")
    if cat is not None:
        anns.append(Annotation(
            event_id=eid, facet="email_category", value=str(int(cat)),
            source="apple_intelligence",
        ))

    junk = meta.get("junk_level", 0)
    if junk and junk > 0:
        anns.append(Annotation(event_id=eid, facet="junk_level", value=str(junk)))

    if meta.get("high_impact"):
        anns.append(Annotation(
            event_id=eid, facet="time_sensitive", value="true",
            source="apple_intelligence",
        ))

    # Noreply detection
    if event.participants:
        sender_lower = event.participants[0].lower()
        _NOREPLY = ("noreply@", "no-reply@", "no_reply@", "donotreply@", "do-not-reply@")
        if any(sender_lower.startswith(p) or f"<{p}" in sender_lower for p in _NOREPLY):
            anns.append(Annotation(event_id=eid, facet="is_noreply", value="true"))

    # Interaction signals
    if meta.get("replied"):
        anns.append(Annotation(event_id=eid, facet="has_reply", value="true"))
    if meta.get("is_from_me"):
        anns.append(Annotation(event_id=eid, facet="is_from_me", value="true"))

    return anns


def _imessage_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    if meta.get("is_filtered"):
        anns.append(Annotation(event_id=eid, facet="is_filtered", value="true",
                               source="apple_intelligence"))
    if meta.get("delivered_quietly"):
        anns.append(Annotation(event_id=eid, facet="delivered_quietly", value="true",
                               source="apple_intelligence"))
    if meta.get("is_auto_reply"):
        anns.append(Annotation(event_id=eid, facet="is_auto_reply", value="true"))
    if meta.get("is_from_me"):
        anns.append(Annotation(event_id=eid, facet="is_from_me", value="true"))
    if meta.get("is_group"):
        anns.append(Annotation(event_id=eid, facet="is_group", value="true"))

    return anns


def _whatsapp_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    if meta.get("is_group"):
        anns.append(Annotation(event_id=eid, facet="is_group", value="true"))
        group_size = meta.get("group_size", 0)
        if group_size:
            anns.append(Annotation(event_id=eid, facet="group_size", value=str(group_size)))
    if meta.get("is_from_me"):
        anns.append(Annotation(event_id=eid, facet="is_from_me", value="true"))
    ct = meta.get("content_type")
    if ct:
        anns.append(Annotation(event_id=eid, facet="content_type", value=str(ct)))

    return anns


def _calendar_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    if meta.get("is_holiday") or meta.get("is_birthday"):
        reason = "holiday" if meta.get("is_holiday") else "birthday"
        anns.append(Annotation(event_id=eid, facet="calendar_noise", value=reason))
    # is_declined/is_accepted stripped — calendar defaults are unreliable.
    # TODO: re-enable once we can detect actual vs default RSVP responses.
    if meta.get("has_attendees"):
        anns.append(Annotation(event_id=eid, facet="has_attendees", value="true"))
    if meta.get("organizer"):
        anns.append(Annotation(event_id=eid, facet="organizer", value=str(meta["organizer"])))

    return anns


def _granola_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    if meta.get("has_transcript"):
        anns.append(Annotation(event_id=eid, facet="has_transcript", value="true"))
    participant_count = meta.get("participant_count", 0)
    if participant_count:
        anns.append(Annotation(event_id=eid, facet="participant_count",
                               value=str(participant_count)))

    return anns


def _slack_annotations(event: Event) -> list[Annotation]:
    meta = event.metadata
    eid = event.id
    anns: list[Annotation] = []

    ch = meta.get("channel_name")
    if ch:
        anns.append(Annotation(event_id=eid, facet="channel", value=ch))
    if meta.get("has_thread"):
        anns.append(Annotation(event_id=eid, facet="has_thread", value="true"))
    rc = meta.get("reply_count", 0)
    if rc:
        anns.append(Annotation(event_id=eid, facet="reply_count", value=str(rc)))
    if meta.get("is_from_me"):
        anns.append(Annotation(event_id=eid, facet="is_from_me", value="true"))

    return anns


_SOURCE_ANNOTATORS = {
    "mail": _mail_annotations,
    "imessage": _imessage_annotations,
    "whatsapp": _whatsapp_annotations,
    "calendar": _calendar_annotations,
    "granola": _granola_annotations,
    "slack": _slack_annotations,
}


def annotate_event(event: Event) -> list[Annotation]:
    """Extract structural annotations from a single event."""
    annotator = _SOURCE_ANNOTATORS.get(event.source)
    if annotator:
        return annotator(event)
    return []


def annotate_structural(store: LayeredGraphStore, events: list[Event]) -> dict:
    """Extract and write structural annotations for a batch of events.

    Returns:
        {"events_processed": N, "annotations_written": N, "elapsed_seconds": float}
    """
    t0 = time.time()
    all_anns: list[Annotation] = []
    for event in events:
        all_anns.extend(annotate_event(event))

    written = store.put_annotations_batch(all_anns) if all_anns else 0

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Structural annotation: %d events → %d annotations (%d written) in %.1fs",
        len(events), len(all_anns), written, elapsed,
    )
    return {
        "events_processed": len(events),
        "annotations_total": len(all_anns),
        "annotations_written": written,
        "elapsed_seconds": elapsed,
    }
