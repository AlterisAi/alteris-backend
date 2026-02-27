"""Topic normalization: raw triage tags -> canonical topics.

Uses an LLM to semantically group raw topic tags into canonical topics.
Batch artifact detection is data-driven (frequency analysis).

Usage:
    from alteris.topic_normalize import normalize_topic, run_normalization
    canonical = normalize_topic("ai agents", store)  # -> DB lookup
    stats = run_normalization(store)                  # full pipeline
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter

from alteris.prompts.triage import SPECIFIC_TOPICS, UNIVERSAL_SPHERES
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Fast lookup set for taxonomy tags — normalize_topic passes these through unchanged
_TAXONOMY_SET = set(SPECIFIC_TOPICS) | set(UNIVERSAL_SPHERES)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Seed synonym map — LLM-discovered mappings baked in for offline use
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This map is used as a fallback when the DB synonym table is empty
# (i.e., before `topic-normalize` has been run). It's also the record
# of known-good mappings discovered by the LLM. Keys are space-form
# (normalize_topic converts underscores/hyphens to spaces before lookup).
# Regenerated via: alteris topic-normalize → extract from topic_synonyms table.

TOPIC_SYNONYMS: dict[str, str] = {
    # account management
    "account": "account management", "account activity": "account management",
    "account creation": "account management", "account monitoring": "account management",
    "funnel optimization": "account management", "onboarding": "account management",
    "onboarding optimization": "account management", "user signup": "account management",
    # administrative
    "alerts": "administrative", "assistance": "administrative",
    "discount": "administrative", "forms": "administrative",
    "receipts": "administrative", "refund": "administrative",
    "statement": "administrative", "statements": "administrative",
    "summary": "administrative",
    # agriculture
    "farming": "agriculture",
    # ai and machine learning
    "agentic briefing": "ai and machine learning", "ai": "ai and machine learning",
    "ai agents": "ai and machine learning", "ai architecture": "ai and machine learning",
    "ai assistant": "ai and machine learning", "ai development": "ai and machine learning",
    "ai engineering": "ai and machine learning", "ai integration": "ai and machine learning",
    "ai/ml": "ai and machine learning", "chatgpt": "ai and machine learning",
    "edge processing": "ai and machine learning", "gemini api": "ai and machine learning",
    "knowledge graph": "ai and machine learning", "knowledge graphs": "ai and machine learning",
    "machine learning": "ai and machine learning", "openai": "ai and machine learning",
    "task completion": "ai and machine learning",
    # chores
    "chore": "chores",
    # civics
    "activism": "civics", "election": "civics", "fundraiser": "civics",
    "politics": "civics",
    # collaboration
    "activities": "collaboration", "brainstorming": "collaboration",
    "project collaboration": "collaboration", "team": "collaboration",
    "workshop": "collaboration",
    # communication
    "catch up": "communication", "email list": "communication",
    "follow up": "communication", "mail": "communication",
    "notification": "communication", "notifications": "communication",
    "praise": "communication", "system notification": "communication",
    # contact management
    "contact": "contact management",
    # content and media
    "center updates": "content and media", "concert": "content and media",
    "entertainment": "content and media", "journalism": "content and media",
    "news": "content and media", "newsletter": "content and media",
    "recreation": "content and media", "university news": "content and media",
    "video": "content and media", "youtube": "content and media",
    # data
    "data apps": "data",
    # device protection
    "phone protection": "device protection",
    # education
    "alumni": "education", "enrollment": "education", "learning": "education",
    "preschool": "education", "school": "education", "school event": "education",
    "school events": "education", "teaching": "education", "tuition": "education",
    # events
    "conference": "events", "contests": "events", "event": "events",
    "event invitation": "events", "invitation": "events", "party": "events",
    "social event": "events", "volunteering": "events",
    # finance
    "401k": "finance", "banking": "finance", "billing": "finance",
    "bills": "finance", "credit card": "finance", "credit union": "finance",
    "crypto": "finance", "debit card": "finance", "disclosures": "finance",
    "insurance": "finance", "investment": "finance", "investments": "finance",
    "loans": "finance", "money transfer": "finance", "payment": "finance",
    "payment reminder": "finance", "payment request": "finance",
    "payments": "finance", "paypal": "finance", "payroll": "finance",
    "personal banking": "finance", "personal finance": "finance",
    "quant finance": "finance", "registration": "finance", "savings": "finance",
    "savings account": "finance", "stocks": "finance", "tax": "finance",
    "tax service": "finance", "taxes": "finance", "trading": "finance",
    "transaction": "finance", "transaction alert": "finance",
    "transactions": "finance", "transfer": "finance", "usaa": "finance",
    "usage limit": "finance", "utilities": "finance", "venmo": "finance",
    "w2": "finance", "wallet": "finance", "zelle": "finance",
    # food and dining
    "dining": "food and dining", "food": "food and dining",
    "opentable": "food and dining", "reservation": "food and dining",
    "reservations": "food and dining", "restaurant": "food and dining",
    # healthcare
    "burnout": "healthcare", "clinic policy": "healthcare",
    "health": "healthcare", "health notice": "healthcare",
    "medication": "healthcare", "medications": "healthcare",
    "pediatrics": "healthcare", "pharmacy": "healthcare",
    "prescription": "healthcare", "prescriptions": "healthcare",
    "vaccines": "healthcare",
    # holidays
    "birthday": "holidays", "holiday": "holidays", "observance": "holidays",
    "valentine's day": "holidays", "valentines day": "holidays",
    # home services
    "cleaning service": "home services", "home": "home services",
    "home maintenance": "home services",
    # investing
    "fundraising": "investing", "investor call": "investing",
    "investor feedback": "investing", "investor pitch": "investing",
    "investor relations": "investing", "pitch deck": "investing",
    "pitch video": "investing", "product pitch": "investing",
    "vc": "investing", "vc funding": "investing",
    # leadership
    "leadership change": "leadership", "university leadership": "leadership",
    # legal
    "arbitration": "legal", "attorney": "legal",
    "internal investigation": "legal", "legal amendment": "legal",
    "privacy": "legal", "tcpa": "legal", "terms and conditions": "legal",
    # marketing
    "deals": "marketing", "leads": "marketing", "promotion": "marketing",
    "webinar": "marketing",
    # miscellaneous
    "acknowledgement": "miscellaneous", "link": "miscellaneous",
    # parenting
    "babies": "parenting", "camp": "parenting", "camps": "parenting",
    "child": "parenting", "child development": "parenting",
    "childcare": "parenting", "children": "parenting", "family": "parenting",
    "family visit": "parenting", "playdates": "parenting",
    "summer camp": "parenting",
    # personal projects
    "personal website": "personal projects",
    # pets
    "dog walking": "pets",
    # planning
    "strategic planning": "planning",
    # product demo
    "product demonstration": "product demo",
    # product development
    "content creation": "product development", "market strategy": "product development",
    "product": "product development", "product improvement": "product development",
    "product strategy": "product development", "product updates": "product development",
    "research": "product development", "user engagement": "product development",
    "user experience": "product development", "user research": "product development",
    "user retention": "product development",
    # product launch
    "beta launch": "product launch", "launch": "product launch",
    # productivity
    "tasks": "productivity",
    # project management
    "timeline extension": "project management",
    # real estate
    "housing": "real estate",
    # recruiting
    "application": "recruiting", "career": "recruiting",
    "career change": "recruiting", "career development": "recruiting",
    "career planning": "recruiting", "career resources": "recruiting",
    "employment": "recruiting", "hiring": "recruiting",
    "interview": "recruiting", "interview prep": "recruiting",
    "interview scheduling": "recruiting", "job application": "recruiting",
    "job interview": "recruiting", "job search": "recruiting",
    "opportunity": "recruiting", "professional development": "recruiting",
    "recruitment": "recruiting", "reference": "recruiting",
    "resume": "recruiting", "resume strategy": "recruiting",
    "technical assessment": "recruiting", "work authorization": "recruiting",
    # referrals
    "referral": "referrals", "referral bonus": "referrals",
    "referral program": "referrals",
    # sales
    "offers": "sales",
    # scheduling
    "alert": "scheduling", "appointment": "scheduling", "calendar": "scheduling",
    "meeting": "scheduling", "meeting request": "scheduling",
    "reminder": "scheduling", "reminders": "scheduling",
    "time change": "scheduling",
    # security
    "account security": "security", "account verification": "security",
    "security alert": "security", "verification": "security",
    "verification code": "security",
    # services
    "maintenance": "services", "service": "services",
    # shopping
    "discounts": "shopping", "order": "shopping", "receipt": "shopping",
    "sale": "shopping", "shipping": "shopping",
    # social
    "career networking": "social", "community": "social",
    "community center": "social", "contacts": "social", "forum": "social",
    "forums": "social", "friends": "social", "networking": "social",
    "social media": "social", "wedding": "social",
    # software and tech
    "amazon": "software and tech", "icloud": "software and tech",
    "login": "software and tech", "meta": "software and tech",
    "microsoft": "software and tech", "mobile plan": "software and tech",
    "phones": "software and tech", "whatsapp": "software and tech",
    # software engineering
    "analytics": "software engineering", "api": "software engineering",
    "app development": "software engineering",
    "authentication": "software engineering", "bug report": "software engineering",
    "data correction": "software engineering", "deployment": "software engineering",
    "developer tools": "software engineering", "development": "software engineering",
    "incident report": "software engineering",
    "mobile development": "software engineering", "replit": "software engineering",
    "software": "software engineering", "software development": "software engineering",
    "technical feasibility": "software engineering", "testing": "software engineering",
    "upgrade": "software engineering", "web development": "software engineering",
    "web scraping": "software engineering",
    # sports
    "super bowl": "sports",
    # startups
    "company update": "startups", "startup": "startups",
    "startup pivot": "startups", "startup strategy": "startups",
    "venture capital": "startups", "yc": "startups",
    "yc application": "startups",
    # subscriptions
    "downgrade": "subscriptions", "membership": "subscriptions",
    "price increase": "subscriptions", "streaming": "subscriptions",
    "subscription": "subscriptions",
    # transportation
    "automotive": "transportation", "car": "transportation",
    "car maintenance": "transportation", "car rental": "transportation",
    "car service": "transportation", "cars": "transportation",
    "coordination": "transportation", "delivery": "transportation",
    "delivery notification": "transportation",
    "delivery notifications": "transportation",
    "delivery updates": "transportation", "package": "transportation",
    "package delivery": "transportation", "package tracking": "transportation",
    "packages": "transportation", "parking": "transportation",
    "rental": "transportation", "two wheeler": "transportation",
    "ups": "transportation", "vehicle maintenance": "transportation",
    # travel
    "accommodation": "travel", "airbnb": "travel", "airline": "travel",
    "flight": "travel", "flight confirmation": "travel", "itinerary": "travel",
    "location": "travel", "loyalty program": "travel",
    "points redemption": "travel", "travel plans": "travel", "vacation": "travel",
    # updates
    "confirmation": "updates", "personal updates": "updates",
    # website management
    "website": "website management", "website access": "website management",
    "website performance": "website management",
    "website traffic": "website management",
    # work issues
    "work issue": "work issues",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Topic normalization (runtime — uses DB synonym table + seed map)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def normalize_topic(raw: str, store: LayeredGraphStore | None = None) -> str:
    """Normalize a single topic string.

    If the raw tag is already in the two-tiered taxonomy, returns it as-is.
    Otherwise lowercases, strips whitespace/underscores, then checks:
    1. DB synonym table (if store provided)
    2. Hardcoded seed map (TOPIC_SYNONYMS)
    If no match, returns the cleaned form.
    """
    if not raw:
        return raw

    # Fast path: if it's already a canonical taxonomy tag, return as-is
    raw_clean = raw.strip().lower()
    if raw_clean in _TAXONOMY_SET:
        return raw_clean

    topic = re.sub(r"[_\-]+", " ", raw_clean).strip()

    # Check DB synonyms first
    if store:
        canonical = store.get_topic_canonical(topic)
        if canonical:
            return canonical
        if raw_clean != topic:
            canonical = store.get_topic_canonical(raw_clean)
            if canonical:
                return canonical

    # Fall back to seed map
    if topic in TOPIC_SYNONYMS:
        return TOPIC_SYNONYMS[topic]

    return topic


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch artifact detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BATCH_ARTIFACT_THRESHOLD = 100  # topic combo appearing on >100 claims = batch artifact


def detect_batch_artifacts(
    topic_lists: list[list[str]],
    threshold: int = BATCH_ARTIFACT_THRESHOLD,
) -> set[tuple[str, ...]]:
    """Detect topic combinations that appear suspiciously often (batch contamination).

    Returns set of frozen topic combos that are likely artifacts.
    """
    combo_counts: Counter[tuple[str, ...]] = Counter()
    for topics in topic_lists:
        if len(topics) >= 2:
            combo = tuple(sorted(topics))
            combo_counts[combo] += 1

    return {combo for combo, count in combo_counts.items() if count >= threshold}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM-based topic grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_GROUPING_PROMPT = """\
You are given a list of topic tags that were extracted by an LLM from a person's \
digital life — their personal and work emails (Mail.app), iMessages, WhatsApp \
conversations, calendar events, and meeting transcripts (Granola). These topics \
span personal life (family, social, health), professional work (startup, \
recruiting, software, sales), and everything in between.

Each tag is shown with its frequency count across all events.

Your job: group semantically equivalent or near-duplicate topics together, \
and choose a clear, concise canonical name for each group.

Rules:
- Topics that refer to the SAME concept should be grouped (e.g. "ai agents", \
"ai engineering", "artificial intelligence" -> "ai")
- Topics that refer to DIFFERENT activities should NOT be merged even if they \
share a word (e.g. "product demo" and "product development" are different topics)
- The canonical name should be lowercase, use spaces (not underscores), and be \
the most intuitive human-readable label for a personal knowledge graph
- Topics that are already good on their own should appear as a group of one
- Aim for roughly 40-80 canonical groups total
- Prefer broader groups for low-frequency topics (count <= 3) when a natural \
parent exists — a count-2 "airline" topic should fold into "travel"
- Be aggressive about merging plural/singular variants, underscore/space variants, \
and obvious synonyms

Return a JSON object where each key is the canonical topic name and the value \
is the list of raw topics that belong to that group. Every input topic must \
appear in exactly one group.

Example:
{
  "ai": ["ai agents", "ai engineering", "artificial intelligence", "machine learning"],
  "recruiting": ["hiring", "job search", "resume", "interview"],
  "product demo": ["product demo", "product demonstration"]
}

Here are the topics (format: count | topic):
"""


def _build_topic_grouping_prompt(topic_counts: dict[str, int]) -> str:
    """Build the LLM prompt with topic frequency data."""
    lines = []
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        lines.append(f"{count:>4} | {topic}")
    return _GROUPING_PROMPT + "\n".join(lines)


def _pre_cluster_topics(topic_counts: dict[str, int]) -> list[dict[str, int]]:
    """Pre-cluster topics into batches of related topics using shared words.

    Groups topics that share at least one word token via BFS connected components,
    then chunks large components and merges small ones into batches of ~60-80.
    """
    topics = list(topic_counts.keys())
    if len(topics) <= 80:
        return [topic_counts]

    # Tokenize each topic
    tokenized: dict[str, set[str]] = {}
    for t in topics:
        tokens = set(re.split(r"[\s_\-/]+", t.lower().strip())) - {""}
        tokenized[t] = tokens

    # Build inverted index: word → topics containing that word
    word_index: dict[str, list[str]] = {}
    for t, tokens in tokenized.items():
        for token in tokens:
            word_index.setdefault(token, []).append(t)

    # BFS connected components (topics sharing ≥1 word)
    visited: set[str] = set()
    components: list[list[str]] = []

    for t in topics:
        if t in visited:
            continue
        component = []
        queue = [t]
        visited.add(t)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for token in tokenized.get(current, set()):
                for neighbor in word_index.get(token, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        components.append(component)

    # Pack components into batches of ~60-80 topics
    target_batch_size = 70
    batches: list[dict[str, int]] = []
    current_batch: dict[str, int] = {}

    # Sort components: largest first
    components.sort(key=len, reverse=True)

    for component in components:
        component_dict = {t: topic_counts[t] for t in component}

        if len(component) > target_batch_size:
            # Large component: split into chunks
            sorted_topics = sorted(component, key=lambda t: -topic_counts[t])
            for i in range(0, len(sorted_topics), target_batch_size):
                chunk = sorted_topics[i:i + target_batch_size]
                batches.append({t: topic_counts[t] for t in chunk})
        elif len(current_batch) + len(component) > target_batch_size:
            # Would overflow current batch — flush it
            if current_batch:
                batches.append(current_batch)
            current_batch = component_dict
        else:
            current_batch.update(component_dict)

    if current_batch:
        batches.append(current_batch)

    return batches


def _llm_group_topics(
    topic_counts: dict[str, int],
    store: LayeredGraphStore,
) -> dict[str, list[str]]:
    """Call LLM to group topics into canonical groups.

    Pre-clusters topics into batches of related topics, sends each batch
    to the LLM, then merges results.

    Returns {canonical_name: [raw_topic, ...]}.
    """
    import time as _time

    from alteris.llm.gemini import GeminiClient

    from alteris.constants import CLOUD_DEEP_MODEL

    client = GeminiClient(model=CLOUD_DEEP_MODEL, store=store)
    batches = _pre_cluster_topics(topic_counts)
    logger.info(
        "Grouping %d unique topics in %d batches...",
        len(topic_counts), len(batches),
    )

    all_groups: dict[str, list[str]] = {}

    for batch_idx, batch in enumerate(batches):
        prompt = _build_topic_grouping_prompt(batch)

        logger.info(
            "  Batch %d/%d: %d topics...",
            batch_idx + 1, len(batches), len(batch),
        )

        response = client.generate(
            prompt=prompt,
            system="You are a taxonomy expert. Return valid JSON only.",
            temperature=0.1,
            max_tokens=16384,
            format_json=True,
            thinking_budget=0,
        )

        if not response:
            logger.error("Batch %d: empty LLM response", batch_idx + 1)
            continue

        try:
            groups = json.loads(response)
            if not isinstance(groups, dict):
                logger.error(
                    "Batch %d: LLM response is not a JSON object: %s",
                    batch_idx + 1, type(groups),
                )
                continue

            for canonical, members in groups.items():
                if not isinstance(members, list):
                    members = [str(members)]
                canonical_clean = canonical.strip().lower()
                if canonical_clean in all_groups:
                    all_groups[canonical_clean].extend(members)
                else:
                    all_groups[canonical_clean] = list(members)

        except json.JSONDecodeError as e:
            logger.error(
                "Batch %d: JSON parse error: %s\nResponse preview: %s",
                batch_idx + 1, e, response[-200:] if response else "empty",
            )
            continue

        # Brief pause between batches
        if batch_idx < len(batches) - 1:
            _time.sleep(1)

    if not all_groups:
        return {}

    # ── Pass 2: Merge canonical names across batches ──────────────
    # The first pass may produce overlapping canonical names from different
    # batches (e.g., "scheduling" and "calendar" as separate canonicals).
    # Send just the canonical names to the LLM for a final merge.
    if len(all_groups) > 60:
        logger.info(
            "  Pass 2: merging %d canonical topics across batches...",
            len(all_groups),
        )
        merged = _merge_canonical_names(all_groups, client)
        if merged:
            all_groups = merged

    return all_groups


_MERGE_PROMPT = """\
You previously grouped raw topic tags into canonical topics, but the grouping \
was done in batches so some canonical names overlap or are near-duplicates.

Below are the canonical topic names with their total member counts. \
Merge any that refer to the same concept into a single canonical name.

Rules:
- "scheduling", "calendar", "meeting", "appointments" -> pick ONE canonical name
- "banking", "billing & payments", "financial" -> pick ONE
- "recruiting", "job search", "resume" -> pick ONE
- Keep genuinely different topics separate (e.g. "product demo" vs "product development")
- The merged canonical name should be the most intuitive, concise label
- Target: 40-80 total groups

Return JSON: {final_canonical_name: [list of old canonical names to merge into it]}
Only include groups where merging happens (2+ old names). Canonicals that stay \
as-is don't need to appear.

Canonical topics (count | name):
"""


def _merge_canonical_names(
    groups: dict[str, list[str]],
    client,
) -> dict[str, list[str]] | None:
    """Second LLM pass: merge overlapping canonical names across batches."""
    # Build counts
    counts = {canonical: len(members) for canonical, members in groups.items()}
    lines = []
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"{count:>4} | {name}")

    prompt = _MERGE_PROMPT + "\n".join(lines)

    response = client.generate(
        prompt=prompt,
        system="You are a taxonomy expert. Return valid JSON only.",
        temperature=0.1,
        max_tokens=8192,
        format_json=True,
        thinking_budget=0,
    )

    if not response:
        logger.error("Pass 2: empty LLM response for canonical merge")
        return None

    try:
        merges = json.loads(response)
        if not isinstance(merges, dict):
            logger.error("Pass 2: response is not a JSON object")
            return None
    except json.JSONDecodeError as e:
        logger.error(
            "Pass 2: JSON parse error: %s\nResponse preview: %s",
            e, response[-200:] if response else "empty",
        )
        return None

    # Apply merges
    merged_groups = dict(groups)
    merge_count = 0
    for final_name, old_names in merges.items():
        if not isinstance(old_names, list):
            continue
        final_clean = final_name.strip().lower()

        # Collect all members from old canonical names
        all_members: list[str] = []
        for old in old_names:
            old_clean = old.strip().lower()
            if old_clean in merged_groups:
                all_members.extend(merged_groups.pop(old_clean))
                merge_count += 1

        # Add/extend the final canonical group
        if final_clean in merged_groups:
            merged_groups[final_clean].extend(all_members)
        else:
            merged_groups[final_clean] = all_members

    logger.info(
        "  Pass 2: merged %d canonical names → %d final groups",
        merge_count, len(merged_groups),
    )
    return merged_groups


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main normalization pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_normalization(store: LayeredGraphStore) -> dict:
    """Run the full topic normalization pipeline.

    1. Collect all raw topics from triage claims + existing annotations
    2. Detect and strip batch artifacts
    3. Call LLM to group topics into canonical groups
    4. Populate topic_synonyms table
    5. Re-normalize existing topic annotations

    Returns stats dict.
    """
    t0 = time.time()

    # Collect all raw topics from triage claims
    rows = store.conn.execute(
        "SELECT object FROM claims WHERE claim_type IN ('triage', 'thread_triage') "
        "AND superseded_by IS NULL"
    ).fetchall()

    all_topic_lists: list[list[str]] = []
    raw_topic_counter: Counter[str] = Counter()

    for r in rows:
        try:
            obj = json.loads(r["object"])
            topics = obj.get("topics", [])
            if isinstance(topics, list) and topics:
                all_topic_lists.append(topics)
                raw_topic_counter.update(topics)
        except (json.JSONDecodeError, TypeError):
            pass

    raw_count = len(raw_topic_counter)
    logger.info("Found %d unique raw topics from %d claims", raw_count, len(rows))

    # Detect batch artifacts
    artifacts = detect_batch_artifacts(all_topic_lists)
    logger.info("Detected %d batch artifact combos", len(artifacts))

    # Strip batch artifact annotations
    artifact_stripped = 0
    if artifacts:
        artifact_stripped = _strip_batch_artifacts(store, artifacts)
        logger.info("Stripped %d batch artifact annotations", artifact_stripped)

    # Also collect existing annotation values (post-backfill, post-artifact-strip)
    ann_rows = store.conn.execute(
        "SELECT value, COUNT(*) as cnt FROM annotations "
        "WHERE facet = 'topic' GROUP BY value"
    ).fetchall()
    for r in ann_rows:
        # Merge annotation counts with raw claim counts
        raw_topic_counter[r["value"]] = max(
            raw_topic_counter.get(r["value"], 0), r["cnt"]
        )

    # Call LLM to group topics
    groups = _llm_group_topics(dict(raw_topic_counter), store)

    if not groups:
        logger.error("LLM grouping failed — no synonym mappings written")
        return {
            "raw_unique_topics": raw_count,
            "synonym_mappings": 0,
            "canonical_topics": 0,
            "batch_artifact_combos": len(artifacts),
            "batch_artifacts_stripped": artifact_stripped,
            "llm_groups": 0,
            "annotations_renormalized": 0,
            "duration_seconds": round(time.time() - t0, 2),
        }

    # Convert LLM groups to synonym mappings
    synonym_mappings: list[tuple[str, str, str]] = []
    for canonical, members in groups.items():
        canonical_clean = canonical.strip().lower()
        for raw in members:
            raw_clean = raw.strip().lower()
            if raw_clean != canonical_clean:
                synonym_mappings.append((raw_clean, canonical_clean, "llm"))
            # Also map underscore/hyphen variants
            raw_variants = {
                raw_clean,
                raw_clean.replace(" ", "_"),
                raw_clean.replace(" ", "-"),
                raw_clean.replace("_", " "),
                raw_clean.replace("-", " "),
            }
            for variant in raw_variants:
                if variant != raw_clean and variant != canonical_clean:
                    synonym_mappings.append((variant, canonical_clean, "llm"))

    # Write to DB
    count = store.put_topic_synonyms_batch(synonym_mappings)
    logger.info("Wrote %d topic synonym mappings to DB", count)

    # Re-normalize existing topic annotations
    renorm_count = _renormalize_topic_annotations(store)

    elapsed = time.time() - t0
    stats = {
        "raw_unique_topics": raw_count,
        "synonym_mappings": count,
        "canonical_topics": len(groups),
        "batch_artifact_combos": len(artifacts),
        "batch_artifacts_stripped": artifact_stripped,
        "llm_groups": len(groups),
        "annotations_renormalized": renorm_count,
        "duration_seconds": round(elapsed, 2),
    }
    logger.info("Topic normalization complete: %s", stats)
    return stats


def _strip_batch_artifacts(
    store: LayeredGraphStore,
    artifacts: set[tuple[str, ...]],
) -> int:
    """Remove topic annotations from events that have a batch-contaminated combo.

    An event is contaminated if its set of topic annotation values is a
    superset of any artifact combo. We delete ALL topic annotations for
    these events since the entire set is unreliable.
    """
    # Normalize artifact combos the same way annotations were normalized
    norm_artifacts: list[set[str]] = []
    for combo in artifacts:
        normed = set()
        for t in combo:
            canonical = store.get_topic_canonical(t) or t.strip().lower()
            normed.add(canonical)
        norm_artifacts.append(normed)

    # Get all events with topic annotations, grouped by event
    rows = store.conn.execute(
        "SELECT event_id, GROUP_CONCAT(value, '||') as topics "
        "FROM annotations WHERE facet = 'topic' "
        "GROUP BY event_id"
    ).fetchall()

    contaminated_events: set[str] = set()
    for r in rows:
        event_topics = set(r["topics"].split("||"))
        for artifact_set in norm_artifacts:
            if artifact_set.issubset(event_topics):
                contaminated_events.add(r["event_id"])
                break

    if not contaminated_events:
        return 0

    stripped = 0
    with store.conn:
        for eid in contaminated_events:
            cursor = store.conn.execute(
                "DELETE FROM annotations WHERE event_id = ? AND facet = 'topic'",
                (eid,),
            )
            stripped += cursor.rowcount

    logger.info(
        "Stripped %d annotations from %d contaminated events",
        stripped, len(contaminated_events),
    )
    return stripped


def _renormalize_topic_annotations(store: LayeredGraphStore) -> int:
    """Re-normalize existing topic annotations using the DB synonym table.

    For each topic annotation whose value maps to a different canonical form,
    delete the old annotation and insert one with the canonical value.
    Returns count of annotations updated.
    """
    synonym_map = store.get_all_topic_synonyms()
    if not synonym_map:
        return 0

    rows = store.conn.execute(
        "SELECT event_id, value, confidence, source, created_at "
        "FROM annotations WHERE facet = 'topic'"
    ).fetchall()

    updated = 0
    now = int(time.time())
    with store.conn:
        for r in rows:
            raw_val = r["value"]
            # Check synonym map, also try cleaned variant
            canonical = synonym_map.get(raw_val)
            if not canonical:
                cleaned = re.sub(r"[_\-]+", " ", raw_val.strip().lower()).strip()
                if cleaned != raw_val:
                    canonical = synonym_map.get(cleaned)
            if canonical and canonical != raw_val:
                store.conn.execute(
                    "DELETE FROM annotations WHERE event_id = ? AND facet = 'topic' "
                    "AND value = ? AND source = ?",
                    (r["event_id"], raw_val, r["source"]),
                )
                store.conn.execute(
                    "INSERT OR IGNORE INTO annotations "
                    "(event_id, facet, value, confidence, source, created_at) "
                    "VALUES (?, 'topic', ?, ?, ?, ?)",
                    (r["event_id"], canonical, r["confidence"], r["source"], now),
                )
                updated += 1

    logger.info("Re-normalized %d topic annotations", updated)
    return updated


def topic_stats(store: LayeredGraphStore) -> dict:
    """Compute topic frequency statistics from annotations.

    Returns distribution of canonical topics with counts.
    """
    rows = store.conn.execute(
        "SELECT value, COUNT(*) as cnt FROM annotations "
        "WHERE facet = 'topic' GROUP BY value ORDER BY cnt DESC"
    ).fetchall()

    topics = {r["value"]: r["cnt"] for r in rows}
    total = sum(topics.values())
    singletons = sum(1 for c in topics.values() if c == 1)

    return {
        "total_annotations": total,
        "unique_topics": len(topics),
        "singletons": singletons,
        "top_20": dict(list(topics.items())[:20]),
    }
