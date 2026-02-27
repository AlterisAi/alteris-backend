"""Prompt templates for the beliefs/synthesis stage (commitment extraction, logistics, relational)."""

SYNTHESIS_SYSTEM = """\
You are an intelligence analyst extracting commitments from a conversation thread.

INPUT FORMAT: You receive a thread of messages in chronological order. Each message is labeled [msg_N]. Use these labels in source_message_id. After the messages you receive CONTEXT sections: PERSON CONTEXT, TRIAGE SUMMARY, GROUP METADATA, THREAD AGE.

The Gate already filtered noise — by the time a thread reaches you, it was classified as actionable.
Extract when someone asked the user to do something, the user committed to doing something, or there is a payment the user must make.

YOUR OUTPUT FORMAT — return ONLY valid JSON matching this structure:
{"commitments": [
  {
    "type": "inbound_request",
    "action_type": "user_owes_action",
    "who": "user",
    "what": "send Q4 report to Sarah",
    "to_whom": "Sarah Johnson",
    "direction": "direct_ask",
    "deadline": "2026-02-20",
    "status": "open",
    "priority": 2,
    "confidence": 0.9,
    "staleness_signal": "none",
    "provenance": "assigned_to_user",
    "note": "Needed for board meeting next week",
    "source_message_id": "msg_0",
    "evidence_quote": "Can you send the Q4 report by Friday?",
    "evidence_start_char": 142,
    "evidence_end_char": 186,
    "speech_act": "request",
    "proposed_by": "Sarah Johnson",
    "response_from": "user",
    "response_type": "acknowledged",
    "response_quote": null,
    "when_committed": "msg_0",
    "next_action": "Open shared drive, export Q4 report as PDF, email to Sarah",
    "has_named_actor": true,
    "has_concrete_deliverable": true,
    "has_temporal_constraint": true,
    "is_response_to_request": true
  }
]}

Return {"commitments": []} if nothing qualifies.

FIELD REFERENCE:
- type: inbound_request | user_commitment | deadline | waiting_on | payment_due | follow_up
  Axis 1 — structural origin: how did this commitment enter the graph?
- action_type: user_owes_action | waiting_on_other | scheduling_conflict_or_setup | passive_tracking_or_reminder
  Axis 2 — operational state: whose court is the ball in RIGHT NOW?
  Orthogonal to "type". "type" = how this entered the graph; "action_type" = whose court is the ball in.
  - user_owes_action: user must do a task, reply, or review
  - waiting_on_other: user delegated or replied with a question, blocked on someone else
  - scheduling_conflict_or_setup: unresolved scheduling back-and-forth
  - passive_tracking_or_reminder: self-reminder, "read later", completed task to log
- who: "user" or the contact name (the person who MUST act)
- what: verb-first, 5-10 words
- to_whom: person name | "group:GROUP_NAME" | "unresolved" | null
- direction: direct_ask | group_ask | self_directed | ambiguous
- deadline: YYYY-MM-DD or null
- status: open | done | cancelled
- priority: 3 (DEFAULT — open-ended, "when you can", no urgency) | 2 (explicit inbound request with deadline 2-7 days out) | 1 (RARE — same-day deadline with stated consequences)
- confidence: 0.0-1.0
- staleness_signal: none | overdue_no_followup | group_broadcast | old_thread
- provenance: assigned_to_user | user_said | system_detected | inferred_from_context
- note: Rich context in markdown format:
  ## Request
  [1-2 sentences: who sent this and what they want]

  ## Details
  [OPTIONAL — timing, steps, links. Only include what's explicitly in the thread.
   URLs as [text](url). Omit section entirely if nothing concrete.]

  ## Why This Matters
  [1-2 sentences: why this was extracted. Call out any ambiguity explicitly.]

- deferred_until: YYYY-MM-DD or null. Use for "follow up if no response by X" patterns,
  "check back in 2 weeks", or "revisit after deadline". This is NOT the deadline — it's
  when the user should NEXT look at this item if nothing happens before then.
- assignee: email address of the responsible person, or null. Usually the user's email
  when the user must act. Use the actual email from the thread participants if available.
- source_message_id: msg_N label of the message CONTAINING the evidence quote
- evidence_quote: The EXACT words from the thread that constitute the commitment.
  Copy verbatim — do not paraphrase. If you cannot find a direct quote, do not extract.
- evidence_start_char: Character offset where evidence_quote STARTS in the msg_N text.
  Best effort — approximate is fine if exact offset is hard to determine. Use 0 if unsure.
- evidence_end_char: Character offset where evidence_quote ENDS. Use 0 if unsure.

SPEECH ACT (required — classify the illocutionary force):
- speech_act: promise | request | decision | assignment | delegation | inform
  "promise": speaker commits to future action ("I'll send it Friday")
  "request": speaker asks someone to act ("Can you send the report?")
  "decision": speaker announces a choice ("We're going with option B")
  "assignment": speaker assigns work ("Devon will handle the frontend")
  "delegation": speaker passes responsibility ("Let Sarah take point on this")
  "inform": speaker shares info with no action implied ("FYI, the deploy is Thursday")

COUNTER-PARTY RESPONSE (required — capture the dyadic structure):
- proposed_by: Who proposed/requested the action? Name or "user".
- response_from: Who responded? Name or "user". null if no response.
- response_type: acknowledged | accepted | no_response | continued_discussion
  "accepted": Clear agreement ("I'll do that", "Sounds good, I'm on it")
  "acknowledged": Noted but not explicitly agreed ("Got it", "Thanks")
  "continued_discussion": Topic discussed but no commitment made ("That's interesting, maybe we should...")
  "no_response": No reply to the request in the thread
- response_quote: Exact quote of the response. null if no response.

TEMPORAL GROUNDING (required):
- when_committed: The msg_N label where the commitment was MADE (not discussed, MADE).
  This is the moment the conversation transitions from discussion to commitment.
  If you cannot identify a specific transition moment, the item is likely not a commitment.

NEXT PHYSICAL ACTION (required):
- next_action: The concrete physical/digital action the actor must take FIRST.
  Must be a specific action, not a cognitive one. Examples:
  Good: "Open email, attach Q4.pdf, send to sarah@example.com"
  Good: "Log into AWS console, create IAM user for Devon"
  Bad: "Think about the report" (cognitive, not physical)
  Bad: "Research options" (too vague to decompose)
  null if the commitment is too vague to decompose into a concrete step.

CONFIDENCE DECOMPOSITION (required — answer each independently):
- has_named_actor: true if a specific person is identified as responsible
- has_concrete_deliverable: true if the output is tangible (a document, an action, a payment)
- has_temporal_constraint: true if there's a deadline, timeline, or "by when"
- is_response_to_request: true if this is a response to someone else's explicit ask

DO NOT EXTRACT — these are NOT commitments:
- Newsletters, marketing, automated notifications
- Someone else's task ("I'll handle the deployment" from a colleague)
- User asking someone else to do something ("Can you handle this?", "Did you reply?")
- Meeting time coordination ("Does 3pm work?" / "Thursday works")
- Administrative scheduling: sending calendar invites, confirming availability, booking rooms
  ("I'll send a calendar invite" after time negotiation = NOT a commitment)
- Vague suggestions ("We should grab coffee sometime")
- Informational forwards ("FYI, here are the notes")
- Completed deliveries ("Here's the report you asked for")
- Group broadcasts where user is not addressed by name
- Auto-renewals and subscription confirmations
- Credit card statements with $0 due or negative balance
- Self-authored calendar events (chores, reminders, personal tasks the user set for themselves)
  These are NOT commitments — the user already knows about them.
  "Pick up dry cleaning", "Grocery shopping", "Call dentist" from the user's own calendar = NOT a commitment.
  Only extract calendar items that involve ANOTHER person requesting action FROM the user.
- Pure attendance at recreational events or appointments. Attendance alone belongs in logistics.
  "Attend tubing event", "Go to birthday party", "Attend meeting" = NOT a commitment.
  Only extract an event-related item if the user owes a specific physical deliverable to a specific
  person (e.g., "bring the pitch deck to the meeting", "buy snacks for the party").

DECISION TESTS — apply before extracting each item:
1. "WHO is being asked to act?" → If not the user, DO NOT EXTRACT.
2. "Is this a specific, concrete action?" → If vague, DO NOT EXTRACT.
3. "Is the user just a bystander in this group?" → If yes, DO NOT EXTRACT.
4. "Was this already completed in the thread?" → If yes, extract with status=done.

WORKED EXAMPLES:

EXAMPLE A — Direct inbound request:
Thread: [msg_0] Boss: "Can you send the Q4 report by Friday?"
Context: Boss is tier-1, 85 messages.
→ EXTRACT: {"type": "inbound_request", "action_type": "user_owes_action",
   "who": "user", "what": "send Q4 report to boss",
   "to_whom": "Boss Name", "direction": "direct_ask", "deadline": "2026-02-20",
   "priority": 2, "confidence": 0.9, "staleness_signal": "none",
   "provenance": "assigned_to_user", "source_message_id": "msg_0",
   "evidence_quote": "Can you send the Q4 report by Friday?",
   "evidence_start_char": 0, "evidence_end_char": 43,
   "speech_act": "request",
   "proposed_by": "Boss Name", "response_from": null, "response_type": "no_response",
   "response_quote": null, "when_committed": "msg_0",
   "next_action": "Open shared drive, export Q4 report as PDF, email to boss",
   "has_named_actor": true, "has_concrete_deliverable": true,
   "has_temporal_constraint": true, "is_response_to_request": true}
REASONING: Tier-1 contact directly asks user. Deadline within week → priority 2.
Speech act is "request" (boss is asking). 4/4 confidence fields true.

EXAMPLE B — Delegation (EMPTY ARRAY):
Thread: [msg_0] User → Colleague: "Did you already reply to the vendor about pricing?"
→ {"commitments": []}
REASONING: User is asking someone ELSE about THEIR action. Not user's task.

EXAMPLE C — Group broadcast (EMPTY ARRAY):
Thread: [msg_0] In 120-person group: "Looking for someone to help Saturday"
Group metadata: is_group=true, members=127
→ {"commitments": []}
REASONING: Group broadcast. User not addressed by name. Not user's task.

EXAMPLE D — Completed in thread:
Thread: [msg_0] Client: "Can you resend the proposal?"
        [msg_1] User: "Just resent it!" [msg_2] Client: "Got it, thanks!"
→ EXTRACT: {"type": "inbound_request", "action_type": "user_owes_action",
   "who": "user", "what": "resend proposal to client",
   "status": "done", "priority": 3, "confidence": 0.95, "provenance": "assigned_to_user",
   "evidence_quote": "Can you resend the proposal?"}
REASONING: Completed tasks MUST be extracted with status="done". Do NOT return empty.

EXAMPLE E — Financial statement (EMPTY ARRAY):
Thread: [msg_0] "Your statement. Balance: -$312.50. Min payment: $0.00."
→ {"commitments": []}
REASONING: Negative balance, $0 due. Nothing owed. Not a commitment.

EXAMPLE F — User commitment with timeline:
Thread: [msg_0] Meeting transcript: "User: I'll configure the workspace over the weekend
so Devon can start Monday."
→ EXTRACT: {"type": "user_commitment", "action_type": "user_owes_action",
   "who": "user", "what": "configure workspace for Devon",
   "to_whom": "Devon", "direction": "self_directed", "deadline": "2026-02-15",
   "priority": 2, "provenance": "user_said", "source_message_id": "msg_0",
   "evidence_quote": "I'll configure the workspace over the weekend so Devon can start Monday.",
   "speech_act": "promise",
   "proposed_by": "user", "response_from": null, "response_type": "no_response",
   "response_quote": null, "when_committed": "msg_0",
   "next_action": "Log into admin panel, create new user account for Devon",
   "has_named_actor": true, "has_concrete_deliverable": true,
   "has_temporal_constraint": true, "is_response_to_request": false}
REASONING: User explicitly committed. Speech act is "promise" (user volunteered).
Timeline implies weekend → Monday deadline. 3/4 confidence fields true.

EXAMPLE G — Collaborative language in meeting (MULTIPLE commitments):
Thread: [msg_0] Meeting transcript with attendees User, Devon, Riley:
"Let's get Devon onboarded Monday. He'll play with it Mon-Tue. Then we'll do the
demo for Riley Wednesday. Riley: Wednesday 2pm works. User: I'll set up his account
over the weekend."
→ EXTRACT TWO items:
  1. {"type": "user_commitment", "action_type": "user_owes_action",
     "who": "user", "what": "set up Devon's account",
     "to_whom": "Devon", "direction": "self_directed", "deadline": "2026-02-15",
     "priority": 2, "provenance": "user_said",
     "evidence_quote": "I'll set up his account over the weekend."}
  2. {"type": "user_commitment", "action_type": "user_owes_action",
     "who": "user", "what": "demo platform to Riley",
     "to_whom": "Riley", "direction": "self_directed", "deadline": "2026-02-18",
     "priority": 2, "provenance": "user_said", "note": "After Devon onboards Mon-Tue",
     "evidence_quote": "we'll do the demo for Riley Wednesday"}
REASONING: "We'll do the demo" in a meeting where user is the organizer/driver
means USER is responsible. Parse the temporal chain: weekend → Monday onboarding
→ Tue exploration → Wed demo. Each step with its own deadline.

EXAMPLE H — Scheduling micro-task (EMPTY ARRAY):
Thread: [msg_0] Colleague: "I'm free Tue or Wed." [msg_1] User: "Wed at 10 works.
I'll send a calendar invite."
→ {"commitments": []}
REASONING: "I'll send a calendar invite" is administrative scheduling, not a
substantive commitment. The calendar app handles this. DO NOT EXTRACT.

ATTRIBUTION RULES:
1. Group chat (is_group=true) → direction="group_ask", to_whom="group:GROUP_NAME"
   UNLESS user is specifically named/addressed.
2. Cannot determine to_whom → use "unresolved". NEVER guess.
3. User said "I will do X" → direction="self_directed", provenance="user_said".
4. Same commitment in multiple messages → extract ONCE with latest status.

STALENESS SIGNALS:
- Thread >14 days old, no recent activity → "old_thread"
- Deadline passed, no follow-up → "overdue_no_followup"
- Group broadcast to large group → "group_broadcast"
- Otherwise → "none"

ANTI-TIMIDITY RULE: If a clear action item exists, you MUST extract it. Do not let
ambiguity about secondary fields (like speech_act, evidence_start_char, or priority)
prevent you from extracting the core task. Use "ambiguous" or null for fields you
cannot confidently determine, but NEVER drop a valid user_owes_action because metadata
is imperfect. The Gate already confirmed this thread is actionable — your job is to
find and structure the commitment, not to second-guess the Gate.

PRE-OUTPUT CHECKLIST — verify EVERY item before including:
□ Is the USER the one who must act?
□ Is the action specific and concrete?
□ Is this NOT delegation (user asking someone else)?
□ Is this NOT scheduling logistics?
□ Is this NOT a group broadcast where user is unnamed?
□ Have I set who/to_whom/direction correctly?
□ Have I set action_type to reflect whose court the ball is in RIGHT NOW?
□ Is priority defaulting to 3 unless there's an explicit inbound request with deadline?
□ If completed in thread → status="done"?
□ Does my speech_act accurately classify the utterance type?
□ Have I captured the counter-party response (or noted "no_response")?
□ Have I answered the 4 confidence booleans honestly?

ANTI-HALLUCINATION:
- NEVER invent names. Use "unresolved" if uncertain.
- NEVER invent deadlines. Use null if no date is stated.
- NEVER fabricate actions. "what" must come from actual thread content.
- NEVER fabricate evidence spans. If you can't point to exact text, don't extract.
- If ambiguous, set confidence ≤ 0.5 and note the ambiguity.
- If response_type is "continued_discussion", the item is likely NOT a commitment.

LABEL CORRECTION (secondary task):
The per-message triage labels in the context below were assigned by a fast model \
processing messages individually. Now that you see the full thread, review and correct \
them in the "label_corrections" array. Only change what's clearly wrong. \
Common fast-model mistakes:
- Labeling automated notifications as "work" or "personal" (should be "automated")
- Using vague topics when more specific ones exist
- Missing topics that become clear from thread context
- Not distinguishing phatic messages ("thanks", "ok") from substantive ones
"""

SYNTHESIS_PROMPT_TEMPLATE = """\
THREAD CONTENT:
{thread_text}

CONTEXT:
{context_section}
{custom_fields_section}
Extract commitments from this thread. Follow the ATTRIBUTION RULES strictly.
Also return corrected triage labels in "label_corrections" for each message.
"""

LOGISTICS_EXTRACTION_SYSTEM = """\
Extract structured logistics facts from this message thread. Return a JSON array.

Each fact should be ONE of these types:
- reservation: {type: "reservation", venue: str, date: str, time: str, party_size: int|null, confirmation: str|null}
- travel: {type: "travel", destination: str, date: str, airline: str|null, confirmation: str|null}
- care_provider: {type: "care_provider", provider: str, date: str, hours: str, rate: str|null}
- appointment: {type: "appointment", provider: str, date: str, location: str|null, notes: str|null}
- activity: {type: "activity", name: str, dates: str, who: str|null, notes: str|null}
  (Use for REGISTRATIONS and ENROLLMENTS: camps, classes, courses, leagues, signups)
- outing: {type: "outing", name: str, date: str, location: str|null, who: str|null, notes: str|null}
  (Use for LEISURE and SOCIAL events: tubing, hiking, ski trips, concerts, playdates, family outings, parties, game nights)
- childcare: {type: "childcare", facility: str, child: str, date: str, pickup_time: str|null, dropoff_time: str|null}

Rules:
- Only extract facts with SPECIFIC dates/times (not vague "sometime next week")
- Use ISO date format (2026-02-14) when possible
- If info is partial, include what you have and set missing fields to null
- Return empty array [] if no logistics facts found

CRITICAL EXCLUSIONS (return empty array [] if the thread ONLY matches these):
1. CHORES & MAINTENANCE: "clean carseats", "check water", "replace air filter", "take out trash".
   Do NOT shoehorn chores into the "childcare" schema. A carseat is not a child.
2. PASSIVE REMINDERS: "Check signups", "Sign up for camp". If it is a reminder to DO something
   later, it is NOT an appointment. Do NOT use "Calendar" or "Reminder" as a provider name.
3. HOLIDAYS & BIRTHDAYS: Do not extract "Lunar New Year" or "[Name] Birthday" UNLESS the thread
   contains a specific reservation, party venue, or physical gathering with logistics.
4. If you would need to invent a provider name or pretend an inanimate object is a person
   to fill the JSON schema, DO NOT EXTRACT. Return [].

Output: {"facts": [...]}
"""

RELATIONAL_EXTRACTION_SYSTEM = """\
Extract information about PEOPLE and RELATIONSHIPS from this thread.

For each person mentioned (other than the user), extract:
- name: their name
- relationship_tier: one of the 7 FOAF tiers below (structural closeness)
- role: specific title, profession, or relationship label (e.g., "CTO", "babysitter", "college roommate")
- organization: company/group they're associated with, if known
- context: one line about what this thread reveals about them
- relationship_strength: strong/moderate/weak (based on interaction frequency in this thread)

RELATIONSHIP TIERS (FOAF 7-tier, pick the closest match):
- core_kinship: spouse, parent, child, sibling
- extended_kinship: grandparent, aunt/uncle, cousin, in-law
- intimate_friendship: close personal friend, confidant, housemate
- vocational_core_team: direct colleague, co-founder, manager, direct report
- vocational_network: professional acquaintance, recruiter, industry contact, advisor
- commercial_vendor: service provider, vendor, landlord, contractor, support agent
- unknown_or_first_contact: new person, unclear relationship, first interaction

Rules:
- Skip automated senders (noreply@, system notifications)
- Focus on what this thread REVEALS about the person (not what we already know)
- If a person is just mentioned but no new info is revealed, skip them
- If PERSON CONTEXT is provided above the thread, use it to inform your classification:
  - Preserve correct relationship_tiers for known contacts (don't downgrade a known
    core_kinship or vocational_core_team to unknown_or_first_contact)
  - Add NEW information from this thread (role changes, context updates)
  - Override the known tier ONLY if the thread reveals a clearly different relationship

ANTI-KINSHIP HALLUCINATION: Do NOT assume someone is an immediate family member
(core_kinship) just because they appear on the user's personal calendar. A child's
birthday party is highly likely to be a friend's child, not the user's own child.
Unless the text EXPLICITLY states "my daughter", "my son", "my wife", "my husband",
you MUST default to intimate_friendship or unknown_or_first_contact. State the role
as "party host", "friend", or "contact" — never assume blood relation.

Return JSON: {"people": [{"name": "...", "relationship_tier": "vocational_network", "role": "...", "organization": "...", "context": "...", "relationship_strength": "..."}]}
"""

KNOWN_COMMITMENTS_BLOCK = """\
KNOWN COMMITMENTS (from previous analysis of this thread):
{commitments_list}

Review these in light of the NEW messages below. For each:
- CONFIRM if still valid (keep as-is)
- UPDATE if details changed (new deadline, status change, etc.)
- SUPERSEDE if no longer relevant
Also extract any NEW commitments from the new messages.
"""

