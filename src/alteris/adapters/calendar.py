"""Calendar adapter — reads from macOS Calendar via EventKit framework.

Uses pyobjc-framework-EventKit to access the Calendar.app database
through Apple's official API. This avoids direct SQLite access and
handles permission prompts natively.

Calendar events produce Event records with attendee lists, organizer,
recurrence info, and RSVP status in metadata.
"""

from __future__ import annotations

import hashlib
import logging
import time

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    make_source_id_hash,
)
from alteris.constants import (
    BOOTSTRAP_CALENDAR_DAYS,
    CALENDAR_QUERY_WINDOW_DAYS,
    EVENT_TYPE_CALENDAR,
    SECONDS_PER_DAY,
    SOURCE_ID_HASH_LEN,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# Maximum length for calendar notes
_CALENDAR_NOTES_MAX = 5_000


def _load_user_emails() -> set[str]:
    """Load user's known emails for attendee matching.

    Checks: persons DB (is_user=1 identifiers), then event_person_edges
    (is_from_me events), then config files.
    """
    import json
    import os
    import sqlite3
    emails: set[str] = set()

    db_path = os.path.expanduser("~/.alteris/graph.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            # Try person_identifiers for user
            rows = conn.execute(
                "SELECT pi.identifier FROM person_identifiers pi "
                "JOIN persons p ON pi.person_id = p.person_id "
                "WHERE p.is_user = 1 AND pi.identifier LIKE '%@%'"
            ).fetchall()
            emails = {r[0].lower().strip() for r in rows if r[0]}

            if not emails:
                # Fallback: scan calendar event attendees for the user's own
                # name — look for attendees matching is_user person names.
                user_rows = conn.execute(
                    "SELECT canonical_name FROM persons WHERE is_user = 1"
                ).fetchall()
                user_names = {r[0].lower() for r in user_rows if r[0]}

                if user_names:
                    # Also match on individual name parts (first/last)
                    name_parts = set()
                    for un in user_names:
                        name_parts.update(p for p in un.split() if len(p) > 2)

                    cal_rows = conn.execute(
                        "SELECT metadata FROM events WHERE source = 'calendar' LIMIT 500"
                    ).fetchall()
                    for r in cal_rows:
                        meta = json.loads(r[0] or "{}")
                        for a in meta.get("attendees", []):
                            name = (a.get("name") or "").lower()
                            email = (a.get("email") or "").lower()
                            if email and (
                                any(un in name for un in user_names)
                                or any(p in name or p in email.split("@")[0] for p in name_parts)
                            ):
                                emails.add(email)
            conn.close()
        except sqlite3.Error:
            pass

    if not emails:
        for path in ("~/.alteris/config.json", "~/.alteris/config.json"):
            try:
                with open(os.path.expanduser(path)) as f:
                    cfg = json.load(f)
                emails = {e.lower().strip() for e in cfg.get("emails", []) if e}
                if emails:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                continue

    return emails


_user_emails = _load_user_emails()


class CalendarAdapter(SourceAdapter):
    """Reads macOS Calendar via EventKit framework."""

    @property
    def source_name(self) -> str:
        return "calendar"

    def check_availability(self) -> AvailabilityResult:
        try:
            import EventKit  # noqa: F401
            return AvailabilityResult(available=True, source="calendar")
        except ImportError:
            return AvailabilityResult(
                available=False,
                source="calendar",
                reason="framework_missing",
                user_action="Install pyobjc-framework-EventKit: pip install pyobjc-framework-EventKit",
            )

    def check_schema(self) -> SchemaResult:
        # EventKit is API-based, no schema to check
        return SchemaResult(compatible=True, source="calendar")

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        try:
            import EventKit
            from Foundation import NSDate, NSCalendar, NSDateComponents
        except ImportError:
            return IngestResult(
                source="calendar",
                errors=["EventKit not available. Install: pip install pyobjc-framework-EventKit"],
            )

        ek_store = EventKit.EKEventStore.alloc().init()
        status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
        logger.debug("Calendar EventKit authorization status: %d", status)

        # Status 0 = not determined, request access
        if status == 0:
            import threading
            granted_event = threading.Event()
            grant_result = [False]

            def callback(granted, error):
                grant_result[0] = granted
                if error:
                    logger.warning("Calendar access request error: %s", error)
                granted_event.set()

            ek_store.requestFullAccessToEventsWithCompletion_(callback)
            granted_event.wait(timeout=30)

            if not grant_result[0]:
                logger.warning("Calendar access not granted after request")
                return IngestResult(
                    source="calendar",
                    errors=[
                        "Calendar access not granted. Go to: "
                        "System Settings > Privacy & Security > Calendars, "
                        "then enable access for Alteris (or alteris-cli/Python). "
                        "If you don't see it listed, try running the pipeline "
                        "once from Terminal: python -m alteris.cli ingest"
                    ],
                )
        elif status == 2:
            # Denied — tell the user how to fix
            return IngestResult(
                source="calendar",
                errors=[
                    "Calendar access denied. Go to: "
                    "System Settings > Privacy & Security > Calendars, "
                    "then enable access for Alteris (or alteris-cli/Python)."
                ],
            )
        elif status not in (3, 4):
            return IngestResult(
                source="calendar",
                errors=[
                    f"Calendar access not granted (status={status}). Go to: "
                    "System Settings > Privacy & Security > Calendars, "
                    "then enable access for Alteris."
                ],
            )

        cal = NSCalendar.currentCalendar()

        # Determine date range.
        # When since_ts is given, start from that point but always extend
        # the end bound to BOOTSTRAP_CALENDAR_DAYS from now. The old logic
        # used since_ts + 30 days which missed future events when since_ts
        # was far in the past (e.g. a 6-month lookback).
        if since_ts > 0:
            start_date = NSDate.dateWithTimeIntervalSince1970_(since_ts)
            end_comp = NSDateComponents.alloc().init()
            end_comp.setDay_(BOOTSTRAP_CALENDAR_DAYS)
            end_date = cal.dateByAddingComponents_toDate_options_(end_comp, NSDate.date(), 0)
        else:
            start_comp = NSDateComponents.alloc().init()
            start_comp.setDay_(-BOOTSTRAP_CALENDAR_DAYS)
            start_date = cal.dateByAddingComponents_toDate_options_(start_comp, NSDate.date(), 0)

            end_comp = NSDateComponents.alloc().init()
            end_comp.setDay_(BOOTSTRAP_CALENDAR_DAYS)
            end_date = cal.dateByAddingComponents_toDate_options_(end_comp, NSDate.date(), 0)

        predicate = ek_store.predicateForEventsWithStartDate_endDate_calendars_(start_date, end_date, None)
        ek_events = ek_store.eventsMatchingPredicate_(predicate)

        if not ek_events:
            return IngestResult(source="calendar", duration_seconds=time.time() - t0)

        events: list[Event] = []
        seen: set[tuple[str, int]] = set()  # (title_lower, start_ts) for dedup

        for ev in ek_events:
            start_ts_f = ev.startDate().timeIntervalSince1970()
            end_ts_f = ev.endDate().timeIntervalSince1970()
            start_ts = int(start_ts_f)
            end_ts = int(end_ts_f)

            title = str(ev.title() or "")
            notes = str(ev.notes() or "") if ev.notes() else ""
            location = str(ev.location() or "") if ev.location() else ""
            cal_name = str(ev.calendar().title()) if ev.calendar() else ""
            is_all_day = bool(ev.isAllDay())

            # Dedup across synced calendars
            dedup_key = (title.strip().lower(), start_ts)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Organizer
            organizer_name = ""
            organizer_email = ""
            if ev.organizer():
                org = ev.organizer()
                organizer_name = str(org.name() or "")
                try:
                    organizer_email = str(org.emailAddress() or "")
                except AttributeError:
                    pass

            # Attendees
            attendee_list = []
            ek_attendees = ev.attendees()
            if ek_attendees and len(ek_attendees) > 0:
                role_map = {
                    0: "unknown", 1: "required", 2: "optional",
                    3: "chair", 4: "non-participant",
                }
                for att in ek_attendees:
                    att_name = str(att.name() or "")
                    att_role = role_map.get(att.participantRole(), "unknown")
                    att_email = ""
                    att_url = att.URL()
                    if att_url:
                        url_str = str(att_url)
                        if "mailto:" in url_str:
                            att_email = url_str.split("mailto:")[-1]
                    # Strip accept/decline/tentative status — calendar defaults
                    # are unreliable (can't distinguish "never responded" from
                    # "actively declined"). TODO: revisit once we can detect
                    # actual vs default RSVP responses.
                    attendee_list.append({
                        "name": att_name,
                        "email": att_email,
                        "role": att_role,
                    })

            # Recurrence info
            is_recurring = bool(ev.hasRecurrenceRules())
            recurrence_desc = ""
            if is_recurring and ev.recurrenceRules():
                freq_map = {0: "daily", 1: "weekly", 2: "monthly", 3: "yearly"}
                for rule in ev.recurrenceRules():
                    freq = freq_map.get(rule.frequency(), "unknown")
                    interval = rule.interval()
                    recurrence_desc = f"{freq}" if interval == 1 else f"every {interval} {freq}"

            # Event status
            ev_status_map = {0: "none", 1: "confirmed", 2: "tentative", 3: "cancelled"}
            event_status = ev_status_map.get(ev.status(), "unknown")

            # Skip cancelled events (organizer cancelled, not user declining)
            if event_status == "cancelled":
                continue

            # URL
            event_url = str(ev.URL()) if ev.URL() else ""

            # Build body
            from datetime import datetime, timezone
            start_dt = datetime.fromtimestamp(start_ts_f, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ts_f, tz=timezone.utc)

            body_parts = [title]
            if location:
                body_parts.append(f"Location: {location}")
            if is_all_day:
                body_parts.append("All day event")
            else:
                body_parts.append(f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}")
            if organizer_name:
                body_parts.append(f"Organizer: {organizer_name}" +
                                  (f" ({organizer_email})" if organizer_email else ""))
            if attendee_list:
                att_strs = []
                for a in attendee_list:
                    s = a["name"] or a["email"] or "unknown"
                    # Do NOT inject accept/decline/tentative status into body text.
                    # Calendar default statuses are unreliable — a "declined" often
                    # means the person never responded, not that they actively declined.
                    att_strs.append(s)
                body_parts.append(f"Attendees: {', '.join(att_strs)}")
            if recurrence_desc:
                body_parts.append(f"Recurrence: {recurrence_desc}")
            if event_url:
                body_parts.append(f"URL: {event_url}")
            if notes:
                body_parts.append(notes[:_CALENDAR_NOTES_MAX])

            body = "\n".join(body_parts)

            # Participants: organizer + attendees
            participants = []
            if organizer_email:
                participants.append(organizer_email)
            elif organizer_name:
                participants.append(organizer_name)
            for att in attendee_list:
                ident = att["email"] or att["name"]
                if ident and ident not in participants:
                    participants.append(ident)

            # Deterministic source_id: title + start_time (RSVP-independent)
            source_id = hashlib.sha256(
                f"{title.strip().lower()}|{start_ts}".encode()
            ).hexdigest()[:SOURCE_ID_HASH_LEN]
            event_id = Event.make_id("calendar", source_id)
            content_hash = Event.content_hash_of(body) if body else ""

            # Detect calendar type for downstream filtering
            cal_name_lower = cal_name.lower()
            if any(kw in cal_name_lower for kw in ("holiday", "holidays")):
                calendar_type = "holiday"
            elif any(kw in cal_name_lower for kw in ("birthday", "birthdays")):
                calendar_type = "birthday"
            elif any(kw in cal_name_lower for kw in ("found in", "siri", "other")):
                calendar_type = "found"
            else:
                calendar_type = "regular"

            # NOTE: user_acceptance (accept/decline/tentative) is intentionally
            # NOT extracted. Calendar default statuses are unreliable — a
            # "declined" often means the person never responded, not that they
            # actively declined. Until we can distinguish actual RSVP responses
            # from defaults, surfacing this data anywhere in the pipeline would
            # poison inference quality.
            # TODO: detect actual vs default RSVP responses and re-enable.

            events.append(Event(
                id=event_id,
                source="calendar",
                source_id=source_id,
                event_type=EVENT_TYPE_CALENDAR,
                timestamp=start_ts,
                participants=tuple(participants),
                raw_content=body,
                content_hash=content_hash,
                metadata={
                    "subject": title,
                    "is_from_me": False,
                    "thread_id": "",
                    "calendar": cal_name,
                    "calendar_type": calendar_type,
                    "is_all_day": is_all_day,
                    "end": end_dt.isoformat(),
                    "location": location,
                    "organizer_name": organizer_name,
                    "organizer_email": organizer_email,
                    "attendees": attendee_list,
                    "event_url": event_url,
                    "is_recurring": is_recurring,
                    "recurrence": recurrence_desc,
                    "status": event_status,
                },
                sensitivity=SensitivityLevel.SENSITIVE,
            ))

            if limit > 0 and len(events) >= limit:
                break

        return IngestResult(
            source="calendar",
            events=events,
            duration_seconds=time.time() - t0,
        )
