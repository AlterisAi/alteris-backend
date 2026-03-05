"""Stage 0b2: Person deduplication via deterministic + LLM merge.

Three-layer approach:
  Layer 1 — Deterministic cleanup: strip prefixes ("to:", "from:"),
            suffixes ("(via X)", "(Google+)"), separators (">>"),
            normalize case. Group exact matches after cleanup.
  Layer 2 — Token-based Jaccard clustering: tokenize names, compute
            pairwise Jaccard similarity (≥0.67, min 2 shared tokens).
            BFS components, capped at 15 members.
  Layer 3 — LLM verification: send candidate clusters to Gemini Flash
            for merge/split/reject decisions.

Then apply: merge person_ids (keep the one with most event_persons edges),
reassign identifiers, re-point event_persons, update canonical_name.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Nickname → canonical first name (for Jaccard matching)
NICKNAME_MAP = {
    "rob": "robert", "bob": "robert", "bobby": "robert",
    "mike": "michael", "mikey": "michael",
    "bill": "william", "will": "william", "billy": "william", "willy": "william",
    "jim": "james", "jimmy": "james", "jamie": "james",
    "tom": "thomas", "tommy": "thomas",
    "dick": "richard", "rick": "richard", "ricky": "richard", "rich": "richard",
    "joe": "joseph", "joey": "joseph",
    "dave": "david", "davy": "david",
    "dan": "daniel", "danny": "daniel",
    "matt": "matthew",
    "chris": "christopher",
    "tony": "anthony",
    "steve": "stephen", "steven": "stephen",
    "ted": "theodore",
    "ed": "edward", "eddie": "edward",
    "al": "albert",
    "ben": "benjamin",
    "sam": "samuel",
    "pat": "patrick",
    "andy": "andrew", "drew": "andrew",
    "nick": "nicholas",
    "charlie": "charles", "chuck": "charles",
    "jeff": "jeffrey",
    "greg": "gregory",
    "lex": "alexander", "alex": "alexander",
    "liz": "elizabeth", "beth": "elizabeth", "betsy": "elizabeth",
    "kate": "katherine", "kathy": "katherine", "katie": "katherine",
    "sue": "susan", "susie": "susan",
    "jen": "jennifer", "jenny": "jennifer",
    "meg": "margaret", "maggie": "margaret", "peggy": "margaret",
    "vicky": "victoria", "vic": "victoria",
    "nicky": "nicole",
    "ari": "arianna",
}

# Prefixes to strip from names
_PREFIX_RE = re.compile(
    r"^(?:to:\s*|from:\s*|reply-to:\s*|cc:\s*|bcc:\s*)",
    re.IGNORECASE,
)

# Suffixes to strip: "(via X)", "(Google+)", "(Google Groups)", etc.
_SUFFIX_PAREN_RE = re.compile(
    r"\s*\((?:via\s+.+?|Google\+?.*?|Google\s+\w+)\)\s*$",
    re.IGNORECASE,
)

# Suffixes to strip: " via platform" without parentheses
_VIA_NO_PAREN_RE = re.compile(r"\s+via\s+.+$", re.IGNORECASE)

# "Name >> Other Name" → take second part
_REDIRECT_RE = re.compile(r"^.+?\s*>>\s*(.+)$")

# "Program Application from Name (email)" → take "Name"
_APPLICATION_RE = re.compile(
    r"^(?:Program Application from|Application from)\s+(.+?)(?:\s*\(.+\))?\s*$",
    re.IGNORECASE,
)

# Tokens to ignore when computing Jaccard similarity
_STOP_TOKENS = {
    "the", "and", "of", "for", "in", "at", "to", "from", "via", "a", "an",
    "inc", "llc", "ltd", "corp", "co", "group", "team",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1: Deterministic Cleanup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_name(name: str) -> str:
    """Deterministic name cleanup: strip prefixes, suffixes, normalize."""
    if not name:
        return ""

    cleaned = name.strip()

    # Strip prefixes
    cleaned = _PREFIX_RE.sub("", cleaned).strip()

    # "Name >> Other Name" → "Other Name"
    m = _REDIRECT_RE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    # "Program Application from Name (email)" → "Name"
    m = _APPLICATION_RE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    # Strip parenthetical suffixes
    cleaned = _SUFFIX_PAREN_RE.sub("", cleaned).strip()

    # Strip unparenthesized 'via' suffixes
    cleaned = _VIA_NO_PAREN_RE.sub("", cleaned).strip()

    # Normalize "LAST, FIRST" → "FIRST LAST" (but only if exactly 2 parts)
    if "," in cleaned and cleaned.count(",") == 1:
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            # Only flip if both parts look like names (not orgs)
            if not any(c.isdigit() for c in cleaned):
                cleaned = f"{parts[1]} {parts[0]}"

    # Title case (but preserve intentional all-caps short names like "DJ")
    if len(cleaned) > 3:
        cleaned = cleaned.title()

    return cleaned


def clean_all_canonical_names(store: LayeredGraphStore) -> int:
    """Update canonical_name for all persons whose name changes after cleanup.

    Catches singletons that still have dirty names like "nowak >> Robert Nowak"
    or "Name (via Google Docs)".

    Returns count of names updated.
    """
    conn = store.conn
    rows = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE canonical_name != ''"
    ).fetchall()

    updated = 0
    for r in rows:
        cleaned = clean_name(r[1])
        if cleaned and cleaned != r[1]:
            conn.execute(
                "UPDATE persons SET canonical_name = ? WHERE person_id = ?",
                (cleaned, r[0]),
            )
            updated += 1

    if updated:
        conn.commit()

    return updated


def find_layer1_groups(store: LayeredGraphStore) -> list[dict]:
    """Find persons whose names are identical after deterministic cleanup.

    Returns list of groups:
        [{"cleaned_name": str, "members": [{"person_id": str, "name": str, "edges": int, "is_user": bool}]}]
    """
    conn = store.conn

    rows = conn.execute("""
        SELECT p.person_id, p.canonical_name, p.is_user,
               (SELECT COUNT(*) FROM event_persons WHERE person_id = p.person_id) as edge_count
        FROM persons p
        WHERE p.canonical_name != ''
    """).fetchall()

    # Group by cleaned name
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cleaned = clean_name(r[1])
        if not cleaned:
            continue
        groups[cleaned.lower()].append({
            "person_id": r[0],
            "name": r[1],
            "edges": r[3],
            "is_user": bool(r[2]),
        })

    # Only keep groups with 2+ members
    result = []
    for cleaned, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(members) >= 2:
            result.append({
                "cleaned_name": cleaned,
                "members": sorted(members, key=lambda m: -m["edges"]),
            })

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Email parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Role addresses — local parts that encode a function, not a person
_ROLE_LOCAL_PARTS = {
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "info", "support", "help", "admin", "contact", "hello",
    "team", "mail", "email", "sales", "billing", "service",
    "notify", "notification", "notifications", "alerts",
    "newsletter", "news", "updates", "marketing", "promo",
    "feedback", "reply", "bounce", "postmaster", "webmaster",
    "web", "root", "abuse", "security", "privacy",
}


def parse_email_parts(email: str) -> tuple[set[str], str]:
    """Split email into (name_tokens_from_local, root_domain).

    Local part is split on dots/underscores/hyphens/digits into name tokens.
    Role addresses (noreply, newsletter, etc.) return empty tokens.
    Domain is reduced to root (strip subdomains like 'email.' or 'mail.').

    Returns:
        (name_tokens, root_domain)
    """
    if "@" not in email:
        return set(), ""

    local, domain = email.lower().rsplit("@", 1)

    # Strip role addresses
    local_clean = re.sub(r"[._\-+0-9]+", " ", local).strip()
    local_words = [w for w in local_clean.split() if len(w) > 1]

    if not local_words or local_clean.replace(" ", "") in _ROLE_LOCAL_PARTS:
        return set(), domain

    # Filter out role words within the local part
    name_tokens = {w for w in local_words if w not in _ROLE_LOCAL_PARTS}

    # Root domain: keep last 2 parts (or 3 for co.uk etc.)
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "ac", "edu"):
        root = ".".join(parts[-3:])
    elif len(parts) >= 2:
        root = ".".join(parts[-2:])
    else:
        root = domain

    return name_tokens, root


def _get_email_tokens_for_persons(store: LayeredGraphStore) -> dict[str, tuple[set[str], set[str]]]:
    """Get email-derived name tokens and domains for each person.

    Returns: {person_id: (name_tokens, domains)}
    """
    conn = store.conn
    rows = conn.execute("""
        SELECT person_id, identifier
        FROM person_identifiers
        WHERE identifier_type = 'email' AND identifier LIKE '%@%'
    """).fetchall()

    person_tokens: dict[str, set[str]] = defaultdict(set)
    person_domains: dict[str, set[str]] = defaultdict(set)

    for pid, email in rows:
        tokens, domain = parse_email_parts(email)
        person_tokens[pid].update(tokens)
        if domain:
            person_domains[pid].add(domain)

    return {pid: (person_tokens.get(pid, set()), person_domains.get(pid, set()))
            for pid in set(person_tokens) | set(person_domains)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1.5: Subset/Superset Name Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_subset_matches(
    store: LayeredGraphStore,
    layer1_merged_names: set[str] | None = None,
    max_tokens_for_subset: int = 4,
    min_tokens_in_smaller: int = 2,
) -> list[list[dict]]:
    """Find persons where one name is a token-subset of another.

    Only considers pairs where:
    - The smaller name has ≥ min_tokens_in_smaller tokens (avoids "Air France" matching everything)
    - Both names have ≤ max_tokens_for_subset tokens (avoids org name snowballs)
    - Email evidence corroborates (shared local-part tokens or shared domain)

    Returns list of clusters, each a list of person dicts.
    """
    conn = store.conn

    rows = conn.execute("""
        SELECT p.person_id, p.canonical_name,
               (SELECT COUNT(*) FROM event_persons WHERE person_id = p.person_id) as edge_count
        FROM persons p
        WHERE p.canonical_name != ''
    """).fetchall()

    email_data = _get_email_tokens_for_persons(store)

    # Build person records
    persons = []
    for r in rows:
        cleaned = clean_name(r[1])
        if not cleaned:
            continue
        if layer1_merged_names and cleaned.lower() in layer1_merged_names:
            continue
        tokens = _tokenize_name(cleaned)
        if len(tokens) < min_tokens_in_smaller:
            continue

        email_tokens, email_domains = email_data.get(r[0], (set(), set()))

        persons.append({
            "person_id": r[0],
            "name": r[1],
            "cleaned": cleaned,
            "name_tokens": tokens,
            "email_tokens": email_tokens,
            "email_domains": email_domains,
            "edges": r[2],
        })

    # Find subset pairs with corroborating evidence
    n = len(persons)
    adj: dict[int, set[int]] = defaultdict(set)

    for i in range(n):
        ti = persons[i]["name_tokens"]
        if len(ti) > max_tokens_for_subset:
            continue
        for j in range(i + 1, n):
            tj = persons[j]["name_tokens"]
            if len(tj) > max_tokens_for_subset:
                continue

            smaller, larger = (ti, tj) if len(ti) <= len(tj) else (tj, ti)
            if len(smaller) < min_tokens_in_smaller:
                continue

            # Check subset relationship
            if not smaller <= larger:
                continue

            # Corroborating evidence needed. Email alone is NOT enough:
            #   - Same local + different domain = different orgs (notification@a.com ≠ notification@b.com)
            #   - Same domain + different local = different people (alice@org.com ≠ bob@org.com)
            # Valid: local tokens match name tokens AND (shared domain OR strong name overlap)
            ei, ej = persons[i]["email_tokens"], persons[j]["email_tokens"]
            di, dj = persons[i]["email_domains"], persons[j]["email_domains"]

            # Email corroboration requires BOTH shared local tokens AND shared domain
            has_email_corroboration = bool(ei & ej) and bool(di & dj)

            # Strong name overlap: subset covers ≥50% of the larger name
            # (e.g., "Chen Jameson" ⊂ "Alex Chen Jameson")
            overlap_ratio = len(smaller) / len(larger) if larger else 0
            strong_name_overlap = overlap_ratio >= 0.5 and len(smaller) >= 2

            if has_email_corroboration or strong_name_overlap:
                adj[i].add(j)
                adj[j].add(i)

    # BFS components (capped)
    visited: set[int] = set()
    clusters = []

    for start in range(n):
        if start not in adj or start in visited:
            continue
        component: list[int] = []
        queue = [start]
        visited.add(start)
        while queue and len(component) < 15:
            node = queue.pop(0)
            component.append(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        if len(component) >= 2:
            cluster = [
                {
                    "person_id": persons[idx]["person_id"],
                    "name": persons[idx]["name"],
                    "cleaned": persons[idx]["cleaned"],
                    "edges": persons[idx]["edges"],
                }
                for idx in component
            ]
            clusters.append(sorted(cluster, key=lambda x: -x["edges"]))

    return clusters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2: Token-Based Jaccard Clustering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tokenize_name(name: str) -> set[str]:
    """Tokenize a name into normalized tokens for similarity matching."""
    # Split on whitespace and punctuation
    tokens = re.split(r"[\s,.\-_/]+", name.lower())
    # Remove stop tokens and empty strings
    tokens = {t for t in tokens if t and t not in _STOP_TOKENS and len(t) > 1}
    # Expand nicknames
    expanded = set()
    for t in tokens:
        expanded.add(NICKNAME_MAP.get(t, t))
    return expanded


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


def find_layer2_clusters(
    store: LayeredGraphStore,
    layer1_merged_names: set[str] | None = None,
    min_jaccard: float = 0.67,
    min_shared_tokens: int = 2,
    max_component_size: int = 15,
) -> list[list[dict]]:
    """Find fuzzy name clusters via Jaccard token similarity.

    Skips persons already handled by Layer 1 (via layer1_merged_names).
    Uses strict pairwise matching (no transitive union-find) with BFS components.

    Returns list of clusters, each a list of person dicts.
    """
    conn = store.conn

    rows = conn.execute("""
        SELECT p.person_id, p.canonical_name,
               (SELECT COUNT(*) FROM event_persons WHERE person_id = p.person_id) as edge_count
        FROM persons p
        WHERE p.canonical_name != ''
    """).fetchall()

    # Build person records with tokens
    persons = []
    for r in rows:
        cleaned = clean_name(r[1])
        if not cleaned:
            continue
        # Skip if already merged in Layer 1
        if layer1_merged_names and cleaned.lower() in layer1_merged_names:
            continue
        tokens = _tokenize_name(cleaned)
        if not tokens:
            continue
        persons.append({
            "person_id": r[0],
            "name": r[1],
            "cleaned": cleaned,
            "tokens": tokens,
            "edges": r[2],
        })

    # Build adjacency list via pairwise similarity
    n = len(persons)
    adj: dict[int, set[int]] = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            shared = persons[i]["tokens"] & persons[j]["tokens"]
            if len(shared) >= min_shared_tokens:
                sim = _jaccard(persons[i]["tokens"], persons[j]["tokens"])
                if sim >= min_jaccard:
                    adj[i].add(j)
                    adj[j].add(i)

    # BFS to find connected components
    visited: set[int] = set()
    clusters = []

    for start in range(n):
        if start not in adj or start in visited:
            continue

        component: list[int] = []
        queue = [start]
        visited.add(start)

        while queue and len(component) < max_component_size:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) >= 2:
            cluster = []
            for idx in component:
                p = persons[idx]
                cluster.append({
                    "person_id": p["person_id"],
                    "name": p["name"],
                    "cleaned": p["cleaned"],
                    "edges": p["edges"],
                })
            clusters.append(sorted(cluster, key=lambda x: -x["edges"]))

    return clusters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3: LLM Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LLM_DEDUP_PROMPT = """You are a master data architect resolving entity resolution conflicts for a personal knowledge graph.

For each candidate group below, you will see a list of names and their associated email domains (if known). Your job is to determine if they represent a single physical person/entity.

DECISION LOGIC:
- "merge": The group represents exactly ONE person or ONE organization.
- "split": The group contains multiple distinct people or distinct organizations.
- "reject": The data is too sparse or ambiguous to safely decide.

CRITICAL RULES:
1. THE ORG VS PERSON RULE: Never merge a human person with an automated system or organization. (e.g., "Misha" and "Meta Careers" must be SPLIT, even if they share an email domain).
2. THE DOMAIN RULE: If you see two identical role names (e.g., "Support") but they have different email domains (e.g., "stripe.com" vs "apple.com"), you MUST SPLIT them. They are different entities.
3. THE ORG + PERSON RULE: If one name is an Organization (e.g., "Match.Com") and the other is that Organization + a Person's Name (e.g., "Match.Com Mandy Ginsberg") → SPLIT. The person is a distinct entity from the corporate inbox.
4. THE SUB-ENTITY & UTILITY RULE: Generic sub-brands (e.g., "Tailored Hair For Men" and "Tailored Hair") → MERGE. Distinct functional utilities or government departments (e.g., "City Of Seattle" and "Seattle City Light") → SPLIT. They are functionally different organizations.
5. THE MIDDLE INITIAL RULE: "First Initial Last" (e.g., Lav R Varshney) and "First Middle Last" (e.g., Lav Raj Varshney) → MERGE if the initial matches the first letter of the middle name.
6. THE SWAPPED NAME RULE: If a multi-word human name is simply reversed (e.g., "Petrina Tutumina Johannes" vs "Johannes Petrina"), assume it is a formatting error and MERGE them.
7. THE LAB/TEAM EXTENSION: A person's name (e.g., "Heather Kirkorian") and their academic lab or team (e.g., "Heather Kirkorian Lab") → MERGE, as they represent the same primary contact node.
8. PERSON VARIANTS: "LAST, FIRST" and "FIRST LAST" → same person. "Matt X" and "Matthew X" → same person (nicknames). "Dr. Name" and "Name" → same person. "First Last - Organization" and "First Last" → same person.
9. DIFFERENT PEOPLE AT SAME ORG: "Tim Miller - Bulwark" ≠ "Charlie Sykes - Bulwark" — SPLIT. Same org does NOT mean same person.
10. CANONICAL NAMES: When merging human persons, the canonical name should be the cleanest "First Last" format (e.g., merge "LAST, FIRST" into "First Last"). Strip all domains and org titles from the canonical name.

Groups to evaluate:
"""


def _build_dedup_response_schema():
    """Build Gemini structured output schema for dedup verification.

    Guarantees valid JSON with correct field names and enum-constrained action.
    """
    from google.genai import types

    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "group": types.Schema(
                    type="INTEGER",
                    description="The group number being evaluated.",
                ),
                "action": types.Schema(
                    type="STRING",
                    enum=["merge", "split", "reject"],
                    description="merge if all represent 1 entity. split if multiple distinct entities. reject if too ambiguous.",
                ),
                "canonical": types.Schema(
                    type="STRING",
                    description="If 'merge', the best cleanest full name. Do not include email domains. Empty string if not merge.",
                ),
                "subgroups": types.Schema(
                    type="ARRAY",
                    description="If 'split', arrays of names that belong together. Empty array if not split.",
                    items=types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="STRING"),
                    ),
                ),
            },
            required=["group", "action", "canonical", "subgroups"],
        ),
    )


def verify_clusters_with_llm(
    clusters: list[list[dict]],
    llm_client,
    store: LayeredGraphStore | None = None,
    batch_size: int = 10,
    save_path: str | None = None,
    max_concurrency: int = 10,
) -> list[dict]:
    """Send candidate clusters to LLM for merge/split/reject verification.

    Fires all batches concurrently via asyncio (up to max_concurrency at once).
    Capped at 10 to stay within Gemini rate limits and avoid 429 cascades.
    """
    from alteris.constants import CLOUD_FAST_MODEL

    # Look up email domains for all persons in all clusters
    person_domains: dict[str, set[str]] = {}
    if store:
        email_data = _get_email_tokens_for_persons(store)
        for pid, (_, domains) in email_data.items():
            person_domains[pid] = domains

    schema = _build_dedup_response_schema()

    # Build all batch prompts upfront
    batches: list[tuple[int, str]] = []  # (batch_start, prompt)
    for batch_start in range(0, len(clusters), batch_size):
        batch = clusters[batch_start:batch_start + batch_size]
        prompt = _LLM_DEDUP_PROMPT
        for i, cluster in enumerate(batch):
            group_num = batch_start + i + 1
            members_with_context = []
            for m in cluster:
                domains = person_domains.get(m["person_id"], set())
                if domains:
                    domain_str = ", ".join(sorted(domains))
                    members_with_context.append(f"{m['name']} ({domain_str})")
                else:
                    members_with_context.append(f"{m['name']} (no email)")
            prompt += f"\nGroup {group_num}: {members_with_context}"
        batches.append((batch_start, prompt))

    raw_responses: list[str | None] = [None] * len(batches)
    batch_results: list[list[dict]] = [[] for _ in range(len(batches))]

    async def _call_batch(idx: int, batch_start: int, prompt: str) -> None:
        for retry in range(3):
            try:
                response = await llm_client.agenerate(
                    prompt=prompt,
                    model=CLOUD_FAST_MODEL,
                    temperature=0.1,
                    max_tokens=8192,
                    response_schema=schema,
                )
                if not response:
                    continue
                raw_responses[idx] = response
                results = json.loads(response.strip())
                if not isinstance(results, list):
                    results = [results]
                batch_results[idx] = results
                return
            except Exception as exc:
                logger.warning("LLM dedup batch %d retry %d: %s", batch_start, retry, exc)
        # All retries failed — reject all groups in this batch
        batch = clusters[batch_start:batch_start + batch_size]
        batch_results[idx] = [
            {"group": batch_start + i + 1, "action": "reject"}
            for i in range(len(batch))
        ]

    async def _run_all() -> None:
        await asyncio.gather(*[
            _call_batch(idx, batch_start, prompt)
            for idx, (batch_start, prompt) in enumerate(batches)
        ])
        # Close the async client before asyncio.run() tears down the loop,
        # otherwise GC calls aclose() on a dead loop → "Event loop is closed"
        if hasattr(llm_client, '_async_client') and llm_client._async_client is not None:
            try:
                await llm_client._async_client.aio.aclose()
            except Exception:
                pass
            llm_client._async_client = None

    asyncio.run(_run_all())

    all_results = [r for batch in batch_results for r in batch]

    if save_path:
        responses_to_save = [r for r in raw_responses if r is not None]
        if responses_to_save:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(json.dumps(responses_to_save, indent=2))
            logger.info("Saved %d raw LLM responses to %s", len(responses_to_save), save_path)

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Apply Merges
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_layer1_merges(
    store: LayeredGraphStore,
    groups: list[dict],
) -> dict:
    """Apply Layer 1 deterministic merges.

    For each group, keep the person_id with the most event_persons edges
    (preferring is_user=True). Merge all others into it.

    Returns stats dict.
    """
    merged_count = 0
    for group in groups:
        members = group["members"]
        if len(members) < 2:
            continue

        # Pick the winner: prefer is_user, then most edges
        winner = max(members, key=lambda m: (m.get("is_user", False), m["edges"]))
        losers = [m for m in members if m["person_id"] != winner["person_id"]]

        canonical = clean_name(winner["name"])
        if not canonical:
            canonical = winner["name"]

        for loser in losers:
            _merge_person_into(
                store, loser["person_id"], winner["person_id"], canonical,
            )
            merged_count += 1

    return {"merged": merged_count}


def apply_layer2_merges(
    store: LayeredGraphStore,
    clusters: list[list[dict]],
    llm_results: list[dict],
) -> dict:
    """Apply Layer 2+3 merges based on LLM decisions.

    Returns stats dict.
    """
    merged_count = 0
    split_count = 0
    reject_count = 0

    # Build a mapping from group number → cluster
    cluster_map: dict[int, list[dict]] = {}
    for i, cluster in enumerate(clusters):
        cluster_map[i + 1] = cluster

    for result in llm_results:
        action = result.get("action", "reject")
        group_num = result.get("group", 0)
        cluster = cluster_map.get(group_num, [])

        if not cluster:
            continue

        if action == "merge":
            canonical = result.get("canonical", cluster[0].get("cleaned", cluster[0]["name"]))
            # Keep the person with most edges
            winner = max(cluster, key=lambda m: m["edges"])
            losers = [m for m in cluster if m["person_id"] != winner["person_id"]]

            for loser in losers:
                _merge_person_into(
                    store, loser["person_id"], winner["person_id"], canonical,
                )
                merged_count += 1

        elif action == "split":
            split_count += 1
            # For splits, we merge within each subgroup
            subgroups = result.get("subgroups", [])
            for subgroup in subgroups:
                if isinstance(subgroup, dict) and "members" in subgroup:
                    # Format: {"canonical_name": "...", "members": ["name1", ...]}
                    sg_canonical = subgroup.get("canonical_name", subgroup.get("canonical", ""))
                    sg_names = set(n.lower() for n in subgroup.get("members", []))
                    sg_members = [m for m in cluster if m["name"].lower() in sg_names or m.get("cleaned", "").lower() in sg_names]
                    if len(sg_members) >= 2:
                        winner = max(sg_members, key=lambda m: m["edges"])
                        for loser in sg_members:
                            if loser["person_id"] != winner["person_id"]:
                                _merge_person_into(
                                    store, loser["person_id"],
                                    winner["person_id"],
                                    sg_canonical or winner.get("cleaned", winner["name"]),
                                )
                                merged_count += 1
                elif isinstance(subgroup, list):
                    sg_names = set(n.lower() for n in subgroup)
                    sg_members = [m for m in cluster if m["name"].lower() in sg_names or m.get("cleaned", "").lower() in sg_names]
                    if len(sg_members) >= 2:
                        winner = max(sg_members, key=lambda m: m["edges"])
                        for loser in sg_members:
                            if loser["person_id"] != winner["person_id"]:
                                _merge_person_into(
                                    store, loser["person_id"],
                                    winner["person_id"],
                                    winner.get("cleaned", winner["name"]),
                                )
                                merged_count += 1

        else:
            reject_count += 1

    return {
        "merged": merged_count,
        "split": split_count,
        "rejected": reject_count,
    }


def _merge_person_into(
    store: LayeredGraphStore,
    loser_id: str,
    winner_id: str,
    canonical_name: str,
) -> None:
    """Merge loser_id into winner_id.

    - Re-point all event_persons edges from loser to winner
    - Re-point all person_identifiers from loser to winner
    - Update winner's canonical_name
    - Record the alias in person_aliases table
    - Delete the loser person record
    """
    conn = store.conn

    # Re-point event_persons (ignore conflicts if winner already has that edge)
    conn.execute(
        "UPDATE OR IGNORE event_persons SET person_id = ? WHERE person_id = ?",
        (winner_id, loser_id),
    )
    # Delete any remaining duplicates
    conn.execute(
        "DELETE FROM event_persons WHERE person_id = ?",
        (loser_id,),
    )

    # Re-point person_identifiers
    conn.execute(
        "UPDATE OR IGNORE person_identifiers SET person_id = ? WHERE person_id = ?",
        (winner_id, loser_id),
    )
    conn.execute(
        "DELETE FROM person_identifiers WHERE person_id = ?",
        (loser_id,),
    )

    # Re-point person_events materialized table (if it exists)
    try:
        conn.execute(
            "UPDATE OR IGNORE person_events SET person_id = ? WHERE person_id = ?",
            (winner_id, loser_id),
        )
        conn.execute(
            "DELETE FROM person_events WHERE person_id = ?",
            (loser_id,),
        )
    except Exception:
        pass  # Table might not exist yet

    # Update canonical name
    conn.execute(
        "UPDATE persons SET canonical_name = ? WHERE person_id = ?",
        (canonical_name, winner_id),
    )

    # Record alias for future fast lookup
    loser_name = conn.execute(
        "SELECT canonical_name FROM persons WHERE person_id = ?",
        (loser_id,),
    ).fetchone()
    loser_name = loser_name[0] if loser_name else ""

    _record_alias(conn, loser_id, loser_name, winner_id, canonical_name)

    # Delete loser's person_profile (winner keeps theirs; will be recomputed)
    try:
        conn.execute("DELETE FROM person_profiles WHERE person_id = ?", (loser_id,))
    except Exception:
        pass  # Table might not exist yet

    # Delete loser's person_model (if any)
    try:
        conn.execute("DELETE FROM person_model WHERE rowid IN (SELECT rowid FROM person_model LIMIT 0)", ())
    except Exception:
        pass  # Table might not exist yet

    # Delete loser
    conn.execute("DELETE FROM persons WHERE person_id = ?", (loser_id,))


def _record_alias(conn, old_id: str, old_name: str, new_id: str, new_name: str) -> None:
    """Record a merge alias for future lookups."""
    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_aliases (
            old_person_id TEXT NOT NULL,
            old_name TEXT,
            new_person_id TEXT NOT NULL,
            new_name TEXT,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (old_person_id)
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO person_aliases VALUES (?, ?, ?, ?, ?)",
        (old_id, old_name, new_id, new_name, int(time.time())),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_dedup(
    store: LayeredGraphStore,
    llm_client=None,
    skip_llm: bool = False,
    save_dir: str | None = None,
) -> dict:
    """Run the full three-layer person deduplication.

    Args:
        store: Graph store with persons table populated.
        llm_client: LLM client for Layer 3 (Gemini). If None and not skip_llm,
            raises ValueError.
        skip_llm: If True, skip Layer 3 LLM verification (apply Layer 1 only).
        save_dir: Directory to save intermediate results for recovery.

    Returns:
        Stats dict with per-layer merge counts.
    """
    t0 = time.time()
    conn = store.conn

    before_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    print(f"  Persons before dedup: {before_count:,}")

    # ── Layer 1: Deterministic ──────────────────────────────────
    print("  Layer 1: Deterministic cleanup...")
    layer1_groups = find_layer1_groups(store)
    print(f"    Found {len(layer1_groups)} exact-match groups "
          f"({sum(len(g['members']) for g in layer1_groups)} persons)")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "layer1_groups.json").write_text(
            json.dumps(layer1_groups, indent=2)
        )

    layer1_stats = apply_layer1_merges(store, layer1_groups)
    conn.commit()
    print(f"    Merged: {layer1_stats['merged']}")

    # Clean canonical names for ALL persons (including singletons)
    names_cleaned = clean_all_canonical_names(store)
    if names_cleaned:
        print(f"    Names cleaned: {names_cleaned}")

    # Track which cleaned names were already merged
    merged_names = {g["cleaned_name"] for g in layer1_groups}

    # ── Layer 1.5: Subset/superset + email-aware matching ───────
    print("  Layer 1.5: Subset/superset name matching (email-aware)...")
    subset_clusters = find_subset_matches(store, layer1_merged_names=merged_names)
    print(f"    Found {len(subset_clusters)} subset-match clusters "
          f"({sum(len(c) for c in subset_clusters)} persons)")

    subset_stats = {"merged": 0, "split": 0, "rejected": 0}

    if subset_clusters and not skip_llm:
        if llm_client is None:
            raise ValueError("LLM client required for Layer 1.5 verification")

        print(f"    LLM verification ({len(subset_clusters)} clusters)...")
        save_path = str(Path(save_dir) / "layer1_5_llm_responses.json") if save_dir else None

        # Smaller batches for subset clusters — they're more complex
        llm_results_1_5 = verify_clusters_with_llm(
            subset_clusters, llm_client, store=store, batch_size=10, save_path=save_path,
        )

        if save_dir:
            (Path(save_dir) / "layer1_5_results.json").write_text(
                json.dumps(llm_results_1_5, indent=2)
            )

        subset_stats = apply_layer2_merges(store, subset_clusters, llm_results_1_5)
        conn.commit()
        print(f"    Merged: {subset_stats['merged']}, "
              f"Split: {subset_stats['split']}, "
              f"Rejected: {subset_stats['rejected']}")

    elif subset_clusters and skip_llm:
        print("    Skipping LLM verification (--skip-llm)")

    # ── Layer 2: Jaccard Clustering ─────────────────────────────
    print("  Layer 2: Jaccard token clustering...")
    layer2_clusters = find_layer2_clusters(store, layer1_merged_names=merged_names)
    print(f"    Found {len(layer2_clusters)} candidate clusters "
          f"({sum(len(c) for c in layer2_clusters)} persons)")

    if save_dir:
        (Path(save_dir) / "layer2_clusters.json").write_text(
            json.dumps(layer2_clusters, indent=2)
        )

    layer2_stats = {"merged": 0, "split": 0, "rejected": 0}

    if layer2_clusters and not skip_llm:
        # ── Layer 3: LLM Verification ──────────────────────────
        if llm_client is None:
            raise ValueError("LLM client required for Layer 3 verification")

        print(f"  Layer 3: LLM verification ({len(layer2_clusters)} clusters)...")
        save_path = str(Path(save_dir) / "llm_raw_responses.json") if save_dir else None

        llm_results = verify_clusters_with_llm(
            layer2_clusters, llm_client, store=store, batch_size=20, save_path=save_path,
        )

        if save_dir:
            (Path(save_dir) / "llm_results.json").write_text(
                json.dumps(llm_results, indent=2)
            )

        layer2_stats = apply_layer2_merges(store, layer2_clusters, llm_results)
        conn.commit()
        print(f"    Merged: {layer2_stats['merged']}, "
              f"Split: {layer2_stats['split']}, "
              f"Rejected: {layer2_stats['rejected']}")

    elif layer2_clusters and skip_llm:
        print("    Skipping LLM verification (--skip-llm)")

    after_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    elapsed = time.time() - t0

    total_merged = layer1_stats["merged"] + subset_stats["merged"] + layer2_stats["merged"]
    print(f"  Persons after dedup:  {after_count:,} "
          f"({total_merged} eliminated, {elapsed:.1f}s)")

    return {
        "before": before_count,
        "after": after_count,
        "layer1_groups": len(layer1_groups),
        "layer1_merged": layer1_stats["merged"],
        "layer1_5_clusters": len(subset_clusters),
        "layer1_5_merged": subset_stats["merged"],
        "layer2_clusters": len(layer2_clusters),
        "layer2_merged": layer2_stats["merged"],
        "layer2_split": layer2_stats.get("split", 0),
        "layer2_rejected": layer2_stats.get("rejected", 0),
        "total_merged": total_merged,
        "elapsed": elapsed,
    }
