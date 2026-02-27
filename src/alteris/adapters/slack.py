"""Slack adapter — reads messages via the Slack API.

Uses a bot token stored in macOS Keychain or environment variable.
Fetches messages from all channels the bot is a member of.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
)
from alteris.constants import EVENT_TYPE_MESSAGE
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "alteris"


def _get_user_name_from_profile() -> str | None:
    """Read user's name from profile.yaml for Slack user matching."""
    from pathlib import Path

    for path in [
        Path.home() / ".alteris" / "profile.yaml",
        Path.home() / ".alteris" / "profile.yaml",
    ]:
        if path.exists():
            try:
                for line in path.read_text().splitlines():
                    if line.startswith("name:"):
                        return line.split(":", 1)[1].strip().strip("'\"")
            except Exception:
                continue
    return None


def _get_slack_token() -> str | None:
    """Get Slack bot token from env or Keychain."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token

    for account, service in [
        ("slack", KEYCHAIN_SERVICE),
        ("slack", "alteris-listener"),
        ("alteris-listener", "alteris-listener-slack"),
    ]:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    return None


class SlackAdapter(SourceAdapter):
    """Reads messages from Slack via the Slack API."""

    @property
    def source_name(self) -> str:
        return "slack"

    def check_availability(self) -> AvailabilityResult:
        token = _get_slack_token()
        if not token:
            return AvailabilityResult(
                available=False,
                source="slack",
                reason="token_missing",
                user_action=(
                    "Slack token not found. Either:\n"
                    "  1. Set SLACK_BOT_TOKEN environment variable\n"
                    "  2. Or store in Keychain: security add-generic-password "
                    f"-a slack -s {KEYCHAIN_SERVICE} -w 'xoxb-...'"
                ),
            )
        return AvailabilityResult(available=True, source="slack")

    def check_schema(self) -> SchemaResult:
        # API-based, no schema
        return SchemaResult(compatible=True, source="slack")

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
        except ImportError:
            return IngestResult(
                source="slack",
                errors=["Install slack-sdk: pip install slack-sdk"],
            )

        token = _get_slack_token()
        if not token:
            return IngestResult(source="slack", errors=["No Slack token available"])

        client = WebClient(token=token)

        # Identify the authenticated user so we can set is_from_me.
        # For bot tokens, auth_test returns the bot's user_id. We also
        # check SLACK_USER_ID env var and try to match user emails from
        # the users list against user_emails (set by resolver/config).
        authed_user_id: str | None = os.environ.get("SLACK_USER_ID")
        if not authed_user_id:
            try:
                auth_resp = client.auth_test()
                if auth_resp["ok"] and not auth_resp.get("bot_id"):
                    authed_user_id = auth_resp["user_id"]
            except Exception as exc:
                logger.warning("Failed to get Slack auth info: %s", exc)

        # Build user map
        user_map: dict[str, str] = {}
        user_emails: dict[str, str] = {}  # uid → email
        try:
            resp = client.users_list()
            if resp["ok"]:
                for member in resp["members"]:
                    uid = member["id"]
                    profile = member.get("profile", {})
                    name = (
                        profile.get("display_name")
                        or profile.get("real_name")
                        or member.get("name", uid)
                    )
                    user_map[uid] = name
                    email = profile.get("email", "")
                    if email:
                        user_emails[uid] = email.lower()
        except Exception as exc:
            logger.warning("Failed to fetch Slack users: %s", exc)

        # For bot tokens: resolve the human user from known emails or name.
        # Strategy order: SLACK_USER_EMAIL env → profile.yaml name match
        if not authed_user_id:
            user_email = os.environ.get("SLACK_USER_EMAIL", "").lower()
            if user_email:
                for uid, email in user_emails.items():
                    if email == user_email:
                        authed_user_id = uid
                        logger.info("Resolved Slack user from email: %s → %s", email, uid)
                        break

        if not authed_user_id and user_map:
            # Try matching by name from profile.yaml
            user_name = _get_user_name_from_profile()
            if user_name:
                user_name_lower = user_name.lower()
                for uid, name in user_map.items():
                    if name.lower() == user_name_lower:
                        authed_user_id = uid
                        logger.info("Resolved Slack user from profile name: %s → %s", name, uid)
                        break

        # Get channels
        channels: list[dict] = []
        try:
            cursor = None
            while True:
                kwargs: dict = {"types": "public_channel,private_channel",
                                "limit": 200, "exclude_archived": True}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = client.conversations_list(**kwargs)
                if not resp["ok"]:
                    break
                for ch in resp["channels"]:
                    if ch.get("is_member", False):
                        channels.append(ch)
                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
        except Exception as exc:
            logger.warning("Failed to list Slack channels: %s", exc)

        if not channels:
            return IngestResult(source="slack", duration_seconds=time.time() - t0)

        oldest = str(since_ts) if since_ts > 0 else "0"
        per_channel_limit = limit if limit > 0 else 200

        events: list[Event] = []
        errors: list[str] = []

        for channel in channels:
            ch_id = channel["id"]
            ch_name = channel.get("name", ch_id)

            try:
                resp = client.conversations_history(
                    channel=ch_id,
                    oldest=oldest,
                    limit=per_channel_limit,
                )
                if not resp["ok"]:
                    continue

                for msg in resp["messages"]:
                    subtype = msg.get("subtype", "")
                    if subtype in ("channel_join", "channel_leave", "bot_message"):
                        continue

                    user_id = msg.get("user", "unknown")
                    sender = user_map.get(user_id, user_id)
                    is_from_me = (user_id == authed_user_id) if authed_user_id else False
                    text = msg.get("text", "")
                    ts_float = float(msg.get("ts", 0))
                    ts = int(ts_float)
                    thread_ts = msg.get("thread_ts")

                    # Resolve @mentions
                    for uid, uname in user_map.items():
                        text = text.replace(f"<@{uid}>", f"@{uname}")

                    source_id = f"{ch_id}_{msg.get('ts', '')}"
                    event_id = Event.make_id("slack", source_id)
                    content_hash = Event.content_hash_of(text) if text else ""

                    events.append(Event(
                        id=event_id,
                        source="slack",
                        source_id=source_id,
                        event_type=EVENT_TYPE_MESSAGE,
                        timestamp=ts,
                        participants=(sender, f"#{ch_name}"),
                        raw_content=text,
                        content_hash=content_hash,
                        metadata={
                            "subject": f"#{ch_name}",
                            "is_from_me": is_from_me,
                            "thread_id": thread_ts or msg.get("ts", ""),
                            "channel_id": ch_id,
                            "channel_name": ch_name,
                            "has_thread": bool(thread_ts),
                            "reply_count": msg.get("reply_count", 0),
                            "ts": msg.get("ts", ""),
                        },
                        sensitivity=SensitivityLevel.SENSITIVE,
                    ))

            except SlackApiError as exc:
                errors.append(f"Failed to read #{ch_name}: {exc.response['error']}")

            if limit > 0 and len(events) >= limit:
                break

        return IngestResult(
            source="slack",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
