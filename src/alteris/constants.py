"""Constants for the Alteris module.

All magic numbers, thresholds, and configuration defaults live here.
No literal numbers in logic code — import from this module instead.
"""

import json
import logging
import shutil
from pathlib import Path

_logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Epoch offsets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE_DATA_EPOCH = 978_307_200
"""Seconds between Unix epoch (1970-01-01) and Apple Core Data epoch (2001-01-01).
Used by WhatsApp, iMessage, Mail.app for timestamp conversion."""

APPLE_EPOCH_OFFSET = CORE_DATA_EPOCH
"""Alias: Apple's NSDate reference date is 2001-01-01."""

CHROME_EPOCH_OFFSET = 11_644_473_600_000_000
"""Microseconds between Windows/Chrome epoch (1601-01-01) and Unix epoch (1970-01-01).
Chrome stores timestamps as microseconds since 1601-01-01 UTC."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

HOURS_ALL = 87_600
"""~10 years in hours. Used as 'all available data' when no time bound is specified."""

LIMIT_ALL = 500_000
"""Effectively unlimited item count for full bootstrap."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hashing and deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HASH_PREFIX_LEN = 24
"""Length of the hex prefix used for event IDs (SHA-256 truncated).
24 hex chars = 96 bits -> collision probability < 1 in 10^14 at 10M events."""

CONTENT_HASH_PREFIX_LEN = 16
"""Length of the hex prefix for content hashes (body dedup).
16 hex chars = 64 bits -> sufficient for dedup within a single source."""

CONTENT_HASH_MIN_LENGTH = 20
"""Minimum content length to generate a content hash.
Messages shorter than this get empty hash to avoid false cross-source
duplicate matches on common short phrases like 'Sure', 'Ok', 'Yes'."""

SOURCE_ID_HASH_LEN = 20
"""Length of the hex prefix for computed source_id when no native ID exists.
20 hex chars = 80 bits -> unique within a source."""

BODY_PREVIEW_LEN = 100
"""Characters of message body used in fallback ID computation.
Long enough to disambiguate, short enough to be stable."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_QUERY_LIMIT = 100
"""Default max results for list/search queries."""

DEFAULT_EVENT_QUERY_LIMIT = 1_000
"""Default max results for event queries (larger due to volume)."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe / auto-ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBE_SAMPLE_LIMIT = 10
"""Max items to fetch during probe() for quick sampling."""

PROBE_SAMPLE_HOURS = 24
"""Hours to look back during probe() sampling."""

PROBE_MEETING_HOURS = 168
"""Hours to look back for meeting probes (7 days)."""

BOOTSTRAP_CALENDAR_DAYS = 365
"""Days to look back/forward for full calendar bootstrap."""

CALENDAR_QUERY_WINDOW_DAYS = 30
"""When since_ts is provided, cap the end date at since_ts + this many days.
Prevents EventKit from expanding recurring events indefinitely into the future."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model names
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOCAL_EMBED_MODEL = "nomic-embed-text"
LOCAL_FAST_MODEL = "qwen3:8b"
LOCAL_REASONING_MODEL = "qwen3:30b-a3b"

# Cloud model names (Gemini)
CLOUD_LITE_MODEL = "gemini-2.5-flash-lite"  # Cheap classification, no thinking
CLOUD_FAST_MODEL = "gemini-2.5-flash"
CLOUD_REASONING_MODEL = "gemini-2.5-flash"  # Use for propagate
CLOUD_DEEP_MODEL = "gemini-3-flash-preview"  # Use for extract, synthesize, brief
CLOUD_FRONTIER_MODEL = "gemini-3-flash-preview"  # Frontier: Granola transcripts, deep analysis
CLOUD_PRO_MODEL = "gemini-3.1-pro-preview"  # Pro: sandwich Surveyor + Consigliere

EMBEDDING_DIM = 768
"""nomic-embed-text produces 768-dimensional vectors."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALTERIS_DIR = Path.home() / ".alteris"
"""Root directory for all Alteris data."""

DEFAULT_DB_PATH = ALTERIS_DIR / "graph.db"
"""Default path for the knowledge graph database."""

# ── Config discovery search paths (checked in order) ──
_CONFIG_SEARCH_PATHS = [
    ALTERIS_DIR / "config.json",
    Path.home() / ".alteris" / "config.json",         # legacy
    Path.home() / "Downloads" / "config.json",         # non-techy drop zone
    Path.home() / "Desktop" / "config.json",           # non-techy drop zone
    Path.home() / "Documents" / "config.json",
]

_config_cache: dict | None = None


def load_config() -> dict:
    """Load Alteris config.json, searching multiple locations.

    Search order:
      1. ~/.alteris/config.json  (canonical)
      2. ~/.alteris/config.json  (legacy)
      3. ~/Downloads/config.json  (for non-technical users)
      4. ~/Desktop/config.json
      5. ~/Documents/config.json

    If found outside ~/.alteris/, the file is auto-copied to ~/.alteris/config.json
    (with owner-only permissions) so subsequent lookups are instant.

    Returns an empty dict if no config is found anywhere.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    canonical = _CONFIG_SEARCH_PATHS[0]

    for path in _CONFIG_SEARCH_PATHS:
        if not path.exists():
            continue
        try:
            cfg = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(cfg, dict):
            continue

        # Auto-install to canonical location if found elsewhere
        if path != canonical:
            try:
                ALTERIS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, canonical)
                canonical.chmod(0o600)
                _logger.info("Copied config from %s → %s", path, canonical)
            except OSError as exc:
                _logger.warning("Could not copy config to %s: %s", canonical, exc)

        _config_cache = cfg
        return cfg

    _config_cache = {}
    return _config_cache


WHATSAPP_BASE = Path.home() / "Library" / "Group Containers" / "group.net.whatsapp.WhatsApp.shared"
WHATSAPP_CHAT_DB = WHATSAPP_BASE / "ChatStorage.sqlite"
WHATSAPP_CALL_DB = WHATSAPP_BASE / "CallHistory.sqlite"
WHATSAPP_LID_DB = WHATSAPP_BASE / "LID.sqlite"

IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"

GRANOLA_DATA_DIR = Path.home() / "Library" / "Application Support" / "Granola"

ADDRESSBOOK_DIR = Path.home() / "Library" / "Application Support" / "AddressBook"

CALENDAR_DB = Path.home() / "Library" / "Calendars" / "Calendar.sqlitedb"

MAIL_DB = Path.home() / "Library" / "Mail" / "V10" / "MailData" / "Envelope Index"

MACOS_CALL_HISTORY_DB = Path.home() / "Library" / "Application Support" / "CallHistoryDB" / "CallHistory.storedata"

KNOWLEDGEC_DB = Path.home() / "Library" / "Application Support" / "Knowledge" / "knowledgeC.db"
SAFARI_HISTORY_DB = Path.home() / "Library" / "Safari" / "History.db"
CHROME_HISTORY_DB = Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History"
NOTES_DB = Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NoteStore.sqlite"
ZSH_HISTORY_FILE = Path.home() / ".zsh_history"
BASH_HISTORY_FILE = Path.home() / ".bash_history"

CHROMIUM_BROWSER_PATHS: dict[str, Path] = {
    "chrome": Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History",
    "arc": Path.home() / "Library" / "Application Support" / "Arc" / "User Data" / "Default" / "History",
    "brave": Path.home() / "Library" / "Application Support" / "BraveSoftware" / "Brave-Browser" / "Default" / "History",
    "edge": Path.home() / "Library" / "Application Support" / "Microsoft Edge" / "Default" / "History",
    "vivaldi": Path.home() / "Library" / "Application Support" / "Vivaldi" / "Default" / "History",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Call history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CALL_MEETING_OVERLAP_WINDOW_SECONDS = 600
"""10 min tolerance for temporal correlation between calls and meetings."""

CALL_MIN_DURATION_SECONDS = 10
"""Ignore unanswered/very short calls (< 10s)."""

OVERLAP_FRACTION_CONTAINED = 0.90
""">=90% of call within event = 'contained' (high credibility)."""

OVERLAP_FRACTION_PARTIAL = 0.50
""">=50% = 'partial', <50% = 'weak' overlap."""

SPEECH_RATE_WPM = 150
"""Effective words-per-minute for estimating meeting duration from transcript.
Conversational speech averages 130-150 WPM (National Center for Voice and Speech).
We use 150 WPM because Granola raw_content includes AI-generated meeting notes
alongside transcript, inflating word count. 150 WPM compensates for this and
yields a conservative lower-bound duration estimate."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVENT_TYPE_MESSAGE = "message"
EVENT_TYPE_CALL = "call"
EVENT_TYPE_EMAIL = "email"
EVENT_TYPE_MEETING = "meeting"
EVENT_TYPE_CALENDAR = "calendar_event"
EVENT_TYPE_IDENTITY = "identity"
EVENT_TYPE_REACTION = "reaction"
EVENT_TYPE_APP_FOCUS = "app_focus"
EVENT_TYPE_BROWSER_VISIT = "browser_visit"
EVENT_TYPE_NOTE = "note"
EVENT_TYPE_SHELL_COMMAND = "shell_command"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence and decay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CLAIM_CONFIDENCE = 0.5
"""Starting confidence for claims with no other signal."""

DEFAULT_BELIEF_CONFIDENCE = 0.5
"""Starting confidence for beliefs with no other signal."""

USER_TIMEZONE = "America/Los_Angeles"
"""Legacy fallback — no longer used as default. safe_timezone() detects system tz."""


def safe_timezone(tz_name: str | None = None):
    """Get a ZoneInfo object, falling back to UTC if tzdata is missing.

    When called with no argument, detects the system's local timezone via
    zoneinfo.ZoneInfo('localtime') rather than using a hardcoded US constant.
    This ensures UK/EU users get the correct timezone without any config.

    PyInstaller bundles may lack the tzdata package, causing
    ``zoneinfo.ZoneInfo(key)`` to raise. This helper catches all exceptions
    and returns UTC so the pipeline doesn't crash.
    """
    import zoneinfo
    from datetime import timezone

    if tz_name:
        try:
            return zoneinfo.ZoneInfo(tz_name)
        except Exception:
            _logger.warning("Timezone %r not found, using UTC", tz_name)
            return timezone.utc

    # No explicit timezone — detect from system via /etc/localtime symlink (macOS/Linux)
    try:
        import os
        link = os.readlink("/etc/localtime")
        if "/zoneinfo/" in link:
            key = link.split("/zoneinfo/")[-1]
            return zoneinfo.ZoneInfo(key)
    except Exception:
        pass

    # Final fallback: UTC
    return timezone.utc

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Score floors (post-triage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCORE_FLOOR_MEETING = 0.5
SCORE_FLOOR_CALENDAR = 0.3
SCORE_FLOOR_TIER1_SENDER = 0.3
SCORE_FLOOR_COMMITMENT = 0.7

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION_MIN_SCORE = 0.7
"""Minimum triage score for deep extraction."""

DEDUP_WHAT_PREFIX_LEN = 25
"""Characters of commitment 'what' field used for dedup matching."""

DEDUP_TOKEN_OVERLAP_THRESHOLD = 0.7
"""Jaccard similarity threshold for token-overlap dedup (P1 fix)."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment staleness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEURISTIC_SKIP_THRESHOLD = 0.1
"""Events below this score are routed to 'skip' (not sent to LLM triage)."""

HEURISTIC_FULL_TRIAGE_THRESHOLD = 0.4
"""Events at or above this score are routed to 'full_triage'."""

HEURISTIC_HIGH_IMPACT_FLOOR = 0.4
"""Apple Intelligence high_impact flag guarantees at least this score."""

HEURISTIC_UNIVERSAL_BASE = 0.3
"""Conservative base score for all sources. Unknown persons route to
low_priority (cheap LLM). Person engagement overrides for known contacts."""

DEMO_MAX_INTERESTING = 40
"""Max interesting events to display in stage2 demo output."""

COMMITMENT_DEADLINE_GRACE_DAYS = 7
"""Days after a commitment's deadline before marking it stale."""

COMMITMENT_NO_DEADLINE_STALE_DAYS = 30
"""Days after creation for non-deadlined commitments to go stale."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION_MIN_THREAD_AVG_SCORE = 0.4
"""Skip extraction for threads whose average triage score is below this."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Propagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROPAGATION_MAX_ROUNDS = 3
PROPAGATION_CONVERGENCE_THRESHOLD = 0.01

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch sizes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRIAGE_BATCH_SIZE = 20
EXTRACTION_BATCH_SIZE = 10
EMBEDDING_BATCH_SIZE = 32
SYNTHESIS_BATCH_SIZE = 200

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context enrichment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THREAD_SNAPSHOT_MAX_NODES = 100
"""Max nodes fetched for thread snapshot (100 most recent)."""

DAY_CONTEXT_MAX_EVENTS = 50
"""Max calendar events to scan for day context."""

CONTACT_DOSSIER_TIMING_LIMIT = 50_000
"""Max event rows fetched for timing histogram in contact dossier."""

PENDING_RESPONSE_MAX_THREADS = 200
"""Max threads to scan for pending user response detection."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread-based triage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MSG_BATCH_SIZE = 5
"""Messages per batch for MSG_BATCH_COMPACT strategy."""

THREAD_FULL_MAX_OUTPUT_TOKENS = 65_536
"""Max output tokens for Gemini Flash thread triage (65K output limit)."""

MSG_COMPACT_MAX_OUTPUT_TOKENS = 1024
"""Max output tokens for local model per-message triage response."""

BODY_PREVIEW_LENGTH = 500
"""Max chars of body preview in compact (Qwen) triage items."""

THREAD_INPUT_TOKEN_BUDGET = 100_000

THREAD_MAX_SCORED_MSGS = 100
"""Max messages to request per-message scores for in THREAD_FULL mode.
Older messages still contribute context but don't get individual scores.
Prevents Gemini output timeout on very large threads (500+ msgs)."""
"""Soft token budget for thread input to Gemini (1M limit minus headroom
for system prompt, context enrichment, and output). When a thread exceeds
this, we keep the TAIL (most recent messages) and discard from the head."""

MAX_BODY_CHARS = 1_000_000
"""Hard safety cap on a single message body (1M chars). Prevents malformed
events (base64 blobs, injection payloads, adapter bugs) from propagating
into prompts. Any message body exceeding this is truncated with a marker."""

CONTACT_DOSSIER_WINDOW_DAYS = 30
"""Default window for contact dossier stats (matches old system's 30d)."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Binary gate (Stage 6 replacement)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GATE_PROMPT_VERSION = "gate_v5"
"""Prompt version for the binary actionable gate."""

LOGISTICS_GATE_PROMPT_VERSION = "logistics_gate_v2"
"""Prompt version for the logistics gate."""

RELATIONAL_GATE_PROMPT_VERSION = "relational_gate_v2"
"""Prompt version for the relational gate."""

LOGISTICS_EXTRACTION_PROMPT_VERSION = "logistics_extract_v2"
"""Prompt version for logistics fact extraction."""

RELATIONAL_EXTRACTION_PROMPT_VERSION = "relational_extract_v2"
"""Prompt version for relational context extraction."""

SYNTHESIS_PROMPT_VERSION = "synthesis_v5"
"""Prompt version for per-thread commitment synthesis."""

STALENESS_THREAD_AGE_DAYS = 14
"""Threads older than this (no recent activity) get staleness_signal='old_thread'."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fact belief compilation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BELIEF_JACCARD_VOTE_THRESHOLD = 0.40
"""Jaccard similarity threshold (stop words removed) for one expert vote.
Lower than raw-token dedup threshold (0.7) because stop-word removal
concentrates on content words and belief-level merge is additive."""

BELIEF_SEQMATCH_VOTE_THRESHOLD = 0.65
"""SequenceMatcher ratio threshold for one expert vote.
Catches near-duplicates with transcription errors (e.g., "Altarus"
vs "Alteris") where token overlap misses them."""

BELIEF_MERGE_MIN_VOTES = 2
"""Minimum expert votes (out of 3: prefix, Jaccard, SeqMatch) to merge.
Requiring 2+ prevents any single noisy signal from causing false merges."""

BELIEF_COMMITMENT_EXPIRY_DAYS = 14
"""Days after a commitment's deadline before the FACT belief expires.
Longer than COMMITMENT_DEADLINE_GRACE_DAYS because beliefs persist for
review even after the underlying claims go stale."""

RELATIONAL_SKIP_GENERIC_NAMES = frozenset({
    "wife", "husband", "mom", "dad", "sister", "brother",
    "mother", "father", "son", "daughter", "uncle", "aunt",
    "cousin", "grandma", "grandpa", "partner", "spouse",
})
"""Generic relationship words that should not become relational beliefs.
These appear when calendar attendee fields contain relationship labels
instead of actual names."""

GATE_BATCH_SIZE = 20
"""Max threads per gate batch."""

MAX_THREAD_HISTORY = 500
"""Max messages to keep per thread for synthesis (after 1yr enrichment).
For threads exceeding this, only the most recent N are kept."""

WINDOW_SIZE = 50
"""Messages per window for sliding-window synthesis on long threads.
Threads with <= WINDOW_SIZE messages are processed in one shot."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Watch daemon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WATCH_DEBOUNCE_SECONDS = 5.0
"""Seconds to wait after last FSEvents notification before triggering pipeline."""

WATCH_POLL_CALENDAR_SECONDS = 300
"""Calendar polling interval (5 min). Fast local EventKit call, infrequent changes."""

WATCH_POLL_SLACK_SECONDS = 120
"""Slack polling interval (2 min). REST API, frequent during work hours."""

WATCH_POLL_GRANOLA_SECONDS = 900
"""Granola polling interval (15 min). REST API, transcripts appear post-meeting."""

WATCH_POLL_FILE_SOURCES_SECONDS = 30
"""Fallback poll interval for file-based sources (30s).
FSEvents may not fire for some databases (e.g., iMessage chat.db is written
by imagent which can bypass normal file notifications on macOS). This poll
checks mtime on watched files as a reliable backup."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread reactivation (incremental triage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REACTIVATION_THRESHOLD = 0.3
"""Prior thread score below which a thread is considered 'dormant'.
New messages in dormant threads trigger full reprocessing."""

INCREMENTAL_CONTEXT_MESSAGES = 75
"""Number of most recent messages to include in incremental triage prompts."""

MAX_THREAD_FETCH = 500
"""Max events to fetch from a thread for triage (tail, most recent)."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VC intelligence gathering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VC_INTEL_PROMPT_VERSION = "vc_intel_v1"
"""Prompt version for VC intel extraction."""

VC_INTEL_MAX_QUERIES = 6
"""Maximum number of parallel queries per VC research run."""

VC_INTEL_MAX_PARALLEL = 3
"""Maximum number of concurrent threads for VC queries."""

VC_DOSSIER_EXPIRY_DAYS = 30
"""Days before a VC dossier belief is considered stale."""

VC_KG_LOOKBACK_DAYS = 365
"""Days to look back in the knowledge graph for direct VC interactions."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCP Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP_DEFAULT_PORT = 9119
MCP_HOST = "localhost"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clarity Queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CQ_DEFAULT_LOOKBACK_DAYS = 30
"""How far back to show overdue/recent items in the Clarity Queue."""

CQ_DEFAULT_LOOKAHEAD_DAYS = 30
"""How far forward to show upcoming items in the Clarity Queue."""

CQ_BUCKETS = ["immediate", "review", "background"]
CQ_DEFAULT_SESSION_TYPE = "clarity"
CQ_COACHING_MODEL = CLOUD_FAST_MODEL
CQ_MAX_CONTEXT_EVENTS = 50
CQ_MAX_CONTEXT_BELIEFS = 30

CQ_UNDO_MAX_AGE_SECONDS = 604800  # 7 days
CQ_UNDO_MAX_ENTRIES = 200

CQ_CLUSTERING_MODEL = CLOUD_DEEP_MODEL   # gemini-3-flash, no thinking
CQ_CLUSTERING_MAX_TASKS = 80
CQ_CLUSTERING_CACHE_TTL = 3600  # 1 hour
CQ_RECURRENCE_FREQUENCIES = ["daily", "weekly", "monthly", "yearly"]
CQ_DEFAULT_CATEGORIES = [
    "work", "personal", "finance", "health",
    "home", "errands", "waiting-on", "someday",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENTS_DIR = ALTERIS_DIR / "agents"
"""Directory containing agent YAML spec files. Each .yaml file = one agent."""

BUILTIN_AGENTS_DIR = Path(__file__).parent / "agents" / "specs"
"""Directory containing built-in agent specs shipped with Alteris."""

AGENT_DEFAULT_LLM = "anthropic"
AGENT_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

AGENT_READ_TOOLS = [
    "alteris_stats", "alteris_query_events", "alteris_query_beliefs",
    "alteris_query_persons", "alteris_query_commitments",
    "alteris_person_detail", "alteris_search",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spend Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPEND_ONBOARDING_BUDGET_USD = 5.0
SPEND_DAILY_LIMIT_USD = 2.0
SPEND_BYPASS_SALT = "alteris_bypass_salt_2026"

MODEL_PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},
    "text-embedding-004": {"input": 0.006, "output": 0.0},
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ambient ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AMBIENT_INGEST_WINDOW_DAYS = 30
"""Default lookback window for ambient data sources."""

NOTES_CONTENT_MAX_LEN = 10_000
"""Max characters to extract from a single Apple Note's content."""

SHELL_HISTORY_MAX_COMMAND_LEN = 500
"""Max characters to keep from a single shell command."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pro-Lite-Pro sandwich pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SANDWICH_SURVEYOR_MODEL = CLOUD_PRO_MODEL
"""Phase 1 Surveyor model. Pro-tier for strategic reasoning over graph topology."""

SANDWICH_CONSIGLIERE_MODEL = CLOUD_PRO_MODEL
"""Phase 3 Consigliere model. Pro-tier for evidence synthesis and user questions."""

SANDWICH_MAX_INQUIRY_VECTORS = 20
"""Surveyor produces up to this many inquiry vectors."""

SANDWICH_SCOUT_MAX_RESULTS = 20
"""Max results per individual scout query."""

SANDWICH_SCOUT_CONTEXT_MAX_CHARS = 50_000
"""Cap total scout output to keep Consigliere prompt manageable."""

SANDWICH_ORACLE_MODEL = CLOUD_DEEP_MODEL
"""Oracle model for interactive question answering."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Person Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSON_MODEL_SURVEYOR_MODEL = CLOUD_PRO_MODEL
"""Surveyor model for person model estimation (Pro-tier synthesis)."""

PERSON_MODEL_CONSIGLIERE_MODEL = CLOUD_PRO_MODEL
"""Consigliere model for person model chat corrections."""

PERSON_MODEL_CHAT_MODEL = CLOUD_DEEP_MODEL
"""Chat model for person model onboarding conversation."""

PERSON_MODEL_MAX_SCOUT_RESULTS = 30
"""Max results per individual scout query."""

PERSON_MODEL_LOW_CONFIDENCE_THRESHOLD = 0.3
"""Dimensions with confidence below this are 'gaps' the chat will focus on."""
