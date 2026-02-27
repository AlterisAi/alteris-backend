"""Granola adapter — reads meeting transcripts via the Granola API.

Granola stores auth tokens locally at ~/Library/Application Support/Granola/.
Documents and transcripts are fetched via the Granola REST API using
WorkOS token authentication.

Auth flow: reads WorkOS refresh token from local storage, exchanges for
access token, then fetches documents and transcripts.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
)
from alteris.constants import (
    EVENT_TYPE_MEETING,
    GRANOLA_DATA_DIR,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

GRANOLA_API = "https://api.granola.ai"
WORKOS_AUTH_URL = "https://api.workos.com/user_management/authenticate"
CLIENT_VERSION = "5.354.0"


def _find_auth_token() -> dict | None:
    """Find Granola auth credentials from local storage."""
    for candidate in [
        GRANOLA_DATA_DIR / "workos_tokens.json",
        GRANOLA_DATA_DIR / "auth.json",
        GRANOLA_DATA_DIR / "tokens.json",
    ]:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
                if "refresh_token" in data:
                    return {"type": "workos", "data": data, "path": str(candidate)}
                if "access_token" in data:
                    return {"type": "direct", "data": data, "path": str(candidate)}
            except (json.JSONDecodeError, KeyError):
                continue

    # Try supabase.json
    supabase_path = GRANOLA_DATA_DIR / "supabase.json"
    if supabase_path.exists():
        try:
            data = json.loads(supabase_path.read_text())
            if isinstance(data, dict):
                workos_str = data.get("workos_tokens", "")
                if workos_str and isinstance(workos_str, str):
                    try:
                        workos_data = json.loads(workos_str)
                        if "refresh_token" in workos_data:
                            return {"type": "workos", "data": workos_data, "path": str(supabase_path)}
                        if "access_token" in workos_data:
                            return {"type": "direct", "data": workos_data, "path": str(supabase_path)}
                    except json.JSONDecodeError:
                        pass
                for val in data.values():
                    if isinstance(val, dict) and "access_token" in val:
                        return {"type": "direct", "data": val, "path": str(supabase_path)}
        except (json.JSONDecodeError, KeyError):
            pass

    # Scan all JSON files
    if GRANOLA_DATA_DIR.exists():
        for json_file in GRANOLA_DATA_DIR.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if isinstance(data, dict):
                    if "refresh_token" in data:
                        return {"type": "workos", "data": data, "path": str(json_file)}
                    if "access_token" in data:
                        return {"type": "direct", "data": data, "path": str(json_file)}
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def _refresh_access_token(refresh_token: str, client_id: str, token_path: str) -> str | None:
    """Exchange refresh token for a new access token via WorkOS."""
    import requests

    resp = requests.post(WORKOS_AUTH_URL, json={
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })

    if resp.status_code != 200:
        logger.error("WorkOS token refresh failed: %s", resp.status_code)
        return None

    token_data = resp.json()
    new_access = token_data.get("access_token")
    new_refresh = token_data.get("refresh_token")

    # Save rotated refresh token back to disk
    if new_refresh and token_path:
        try:
            path = Path(token_path)
            existing = json.loads(path.read_text())
            if path.name == "supabase.json" and "workos_tokens" in existing:
                workos_data = json.loads(existing["workos_tokens"])
                workos_data["refresh_token"] = new_refresh
                if new_access:
                    workos_data["access_token"] = new_access
                existing["workos_tokens"] = json.dumps(workos_data)
            else:
                existing["refresh_token"] = new_refresh
                if new_access:
                    existing["access_token"] = new_access
            path.write_text(json.dumps(existing, indent=2))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not save rotated token: %s", exc)

    return new_access


def _get_access_token() -> str | None:
    """Get a valid Granola API access token."""
    auth = _find_auth_token()
    if not auth:
        return None

    if auth["type"] == "direct":
        return auth["data"]["access_token"]

    data = auth["data"]
    access_token = data.get("access_token", "")
    obtained_at = data.get("obtained_at", 0)
    expires_in = data.get("expires_in", 0)

    if access_token and obtained_at and expires_in:
        expiry_ms = obtained_at + (expires_in * 1000)
        now_ms = time.time() * 1000
        if now_ms < expiry_ms - 60000:
            return access_token

    refresh_token = data.get("refresh_token")
    client_id = data.get("client_id", "")

    if not client_id:
        access_token = data.get("access_token", "")
        if access_token:
            try:
                payload = access_token.split(".")[1]
                payload += "=" * (4 - len(payload) % 4)
                jwt_data = json.loads(base64.urlsafe_b64decode(payload))
                iss = jwt_data.get("iss", "")
                if "/client_" in iss:
                    client_id = iss.rsplit("/", 1)[-1]
            except Exception:
                pass

    if not client_id:
        for candidate in GRANOLA_DATA_DIR.glob("*.json"):
            try:
                d = json.loads(candidate.read_text())
                if isinstance(d, dict) and "client_id" in d:
                    client_id = d["client_id"]
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if not client_id:
        logger.error("WorkOS client_id not found")
        return None

    return _refresh_access_token(refresh_token, client_id, auth["path"])


def _api_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": f"Granola/{CLIENT_VERSION}",
        "X-Client-Version": CLIENT_VERSION,
    }


def _fetch_documents(access_token: str, limit: int = 50) -> list[dict]:
    """Fetch recent documents from Granola API."""
    import requests

    resp = requests.post(
        f"{GRANOLA_API}/v2/get-documents",
        headers=_api_headers(access_token),
        json={"limit": limit, "offset": 0, "include_last_viewed_panel": True},
    )
    if resp.status_code != 200:
        logger.error("Failed to fetch Granola documents: %s", resp.status_code)
        return []
    return resp.json().get("docs", [])


def _fetch_transcript(access_token: str, document_id: str) -> list[dict]:
    """Fetch transcript for a specific document."""
    import requests

    resp = requests.post(
        f"{GRANOLA_API}/v1/get-document-transcript",
        headers=_api_headers(access_token),
        json={"document_id": document_id},
    )
    if resp.status_code == 404:
        return []
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


def _format_transcript(utterances: list[dict]) -> str:
    """Format transcript utterances into readable text."""
    lines = []
    for u in utterances:
        source = u.get("source", "unknown")
        text = u.get("text", "").strip()
        start = u.get("start_timestamp", "")
        if not text:
            continue
        ts_display = ""
        if start:
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                ts_display = dt.strftime("%H:%M:%S")
            except ValueError:
                ts_display = start[:8]
        speaker = "You" if source == "microphone" else "Other"
        lines.append(f"[{ts_display}] {speaker}: {text}")
    return "\n".join(lines)


def _prosemirror_to_text(node: dict) -> str:
    """Convert ProseMirror JSON to plain text."""
    if not isinstance(node, dict):
        return ""
    node_type = node.get("type", "")
    text_parts = []

    if node_type == "text":
        return node.get("text", "")
    if node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        children_text = "".join(_prosemirror_to_text(c) for c in node.get("content", []))
        if children_text:
            text_parts.append("#" * level + " " + children_text)
    elif node_type == "paragraph":
        children_text = "".join(_prosemirror_to_text(c) for c in node.get("content", []))
        if children_text:
            text_parts.append(children_text)
    elif node_type in ("bulletList", "orderedList"):
        for i, item in enumerate(node.get("content", []), 1):
            item_text = "".join(_prosemirror_to_text(c) for c in item.get("content", []))
            if item_text:
                prefix = f"{i}. " if node_type == "orderedList" else "- "
                text_parts.append(prefix + item_text)
    else:
        for child in node.get("content", []):
            child_text = _prosemirror_to_text(child)
            if child_text:
                text_parts.append(child_text)

    return "\n".join(text_parts)


class GranolaAdapter(SourceAdapter):
    """Reads meeting notes and transcripts from Granola API."""

    @property
    def source_name(self) -> str:
        return "granola"

    def check_availability(self) -> AvailabilityResult:
        if not GRANOLA_DATA_DIR.exists():
            return AvailabilityResult(
                available=False,
                source="granola",
                reason="not_installed",
                user_action="Granola not installed (directory not found at "
                            f"{GRANOLA_DATA_DIR})",
            )
        auth = _find_auth_token()
        if not auth:
            return AvailabilityResult(
                available=False,
                source="granola",
                reason="auth_missing",
                user_action="Granola installed but no auth tokens found. Are you logged in?",
            )
        return AvailabilityResult(available=True, source="granola")

    def check_schema(self) -> SchemaResult:
        # API-based, no schema to check
        return SchemaResult(compatible=True, source="granola")

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        access_token = _get_access_token()
        if not access_token:
            return IngestResult(
                source="granola",
                errors=["Cannot get Granola access token. Check that you're logged in."],
            )

        fetch_limit = limit if limit > 0 else 200
        documents = _fetch_documents(access_token, limit=fetch_limit)
        if not documents:
            return IngestResult(source="granola", duration_seconds=time.time() - t0)

        cutoff = datetime.fromtimestamp(since_ts, tz=timezone.utc) if since_ts > 0 else None

        events: list[Event] = []
        errors: list[str] = []

        # Phase 1: Parse dates, filter by cutoff, collect eligible documents.
        # API returns newest-first so we break on the first doc older than cutoff.
        eligible_docs: list[tuple[dict, datetime]] = []
        for doc in documents:
            ts_str = doc.get("created_at", "") or doc.get("updated_at", "")
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts_dt = datetime.now(timezone.utc)

            if cutoff and ts_dt < cutoff:
                break

            eligible_docs.append((doc, ts_dt))

            if limit > 0 and len(eligible_docs) >= limit:
                break

        # Phase 2: Fetch all transcripts in parallel.
        transcript_map: dict[str, list[dict]] = {}
        doc_ids_to_fetch = [doc.get("id", "") for doc, _ in eligible_docs if doc.get("id")]

        if doc_ids_to_fetch:
            with ThreadPoolExecutor(max_workers=8) as pool:
                future_to_id = {
                    pool.submit(_fetch_transcript, access_token, did): did
                    for did in doc_ids_to_fetch
                }
                for future in as_completed(future_to_id):
                    did = future_to_id[future]
                    try:
                        transcript_map[did] = future.result()
                    except Exception as exc:
                        logger.warning("Transcript fetch failed for %s: %s", did, exc)
                        transcript_map[did] = []

        # Phase 3: Build events using pre-fetched transcripts.
        for doc, ts_dt in eligible_docs:
            doc_id = doc.get("id", "")
            title = doc.get("title", "Untitled Meeting")
            created = doc.get("created_at", "")
            updated = doc.get("updated_at", "")
            ts = int(ts_dt.timestamp())

            # Extract notes from ProseMirror content
            notes = ""
            panel = doc.get("last_viewed_panel")
            if panel and isinstance(panel, dict):
                content = panel.get("content")
                if content and isinstance(content, dict):
                    notes = _prosemirror_to_text(content)

            # Use pre-fetched transcript
            transcript_text = ""
            utterances = transcript_map.get(doc_id, [])
            if utterances:
                transcript_text = _format_transcript(utterances)

            body_parts = []
            if notes:
                body_parts.append("## Meeting Notes\n" + notes)
            if transcript_text:
                body_parts.append("## Transcript\n" + transcript_text)
            body = "\n\n".join(body_parts) if body_parts else "(no content)"

            # Parse real participants from people field + google_calendar_event
            participants: list[str] = []
            people_enrichment: list[dict] = []

            people = doc.get("people") or {}
            # Creator
            creator = people.get("creator") or {}
            if creator.get("email"):
                participants.append(creator["email"])
            elif creator.get("name"):
                participants.append(creator["name"])
            # Attendees from people field
            for att in (people.get("attendees") or []):
                ident = att.get("email") or att.get("name", "")
                if ident and ident not in participants:
                    participants.append(ident)
                # Extract enrichment data (LinkedIn, employment)
                details = att.get("details") or {}
                person_details = details.get("person") or {}
                enrichment: dict = {}
                if att.get("email"):
                    enrichment["email"] = att["email"]
                if att.get("name"):
                    enrichment["name"] = att["name"]
                employment = person_details.get("employment") or {}
                if employment.get("title"):
                    enrichment["job_title"] = employment["title"]
                if employment.get("name"):
                    enrichment["company"] = employment["name"]
                linkedin = person_details.get("linkedin") or {}
                if linkedin.get("handle"):
                    enrichment["linkedin"] = linkedin["handle"]
                company = details.get("company") or {}
                if company.get("name") and "company" not in enrichment:
                    enrichment["company"] = company["name"]
                if enrichment:
                    people_enrichment.append(enrichment)

            # Merge with google_calendar_event attendees for RSVP status
            gcal_event = doc.get("google_calendar_event") or {}
            gcal_attendees = gcal_event.get("attendees") or []
            rsvp_map: dict[str, str] = {}
            for gatt in gcal_attendees:
                ge = gatt.get("email", "")
                if ge:
                    rsvp_map[ge.lower()] = gatt.get("responseStatus", "unknown")
                    if ge not in participants:
                        display = gatt.get("displayName") or ge
                        participants.append(display)

            # Fallback if no participants found
            if not participants:
                participants = ["meeting"]

            source_id = doc_id
            event_id = Event.make_id("granola", source_id)
            content_hash = Event.content_hash_of(body) if body else ""

            events.append(Event(
                id=event_id,
                source="granola",
                source_id=source_id,
                event_type=EVENT_TYPE_MEETING,
                timestamp=ts,
                participants=tuple(participants),
                raw_content=body,
                content_hash=content_hash,
                metadata={
                    "subject": title,
                    "is_from_me": False,
                    "thread_id": doc_id,
                    "document_id": doc_id,
                    "created_at": created,
                    "updated_at": updated,
                    "has_transcript": bool(transcript_text),
                    "people_enrichment": people_enrichment,
                    "rsvp": rsvp_map,
                    "has_calendar_link": bool(gcal_event),
                },
                sensitivity=SensitivityLevel.SENSITIVE,
            ))

        return IngestResult(
            source="granola",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
