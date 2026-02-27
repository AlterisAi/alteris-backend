"""Prompt templates for the triage stage (thread classification, summarization).

Two-tiered topic taxonomy:
  - universal_spheres: 18 structural tags (schema-enforced via enum)
  - specific_topics: ~95 semantic tags (text-prompt-enforced)
"""

# ─── Two-tiered topic taxonomy ────────────────────────────────────────

UNIVERSAL_SPHERES = [
    "primary_vocational_execution",
    "strategic_planning_and_research",
    "organizational_administration",
    "professional_network_and_development",
    "dependent_care_and_development",
    "intimate_partnership_sync",
    "extended_family_relations",
    "friendship_and_hobby_communities",
    "medical_and_healthcare_admin",
    "outdoor_recreation_and_fitness",
    "personal_enrichment_and_learning",
    "financial_institution_communications",
    "vehicle_and_transit_maintenance",
    "property_and_household_admin",
    "consumer_goods_and_logistics",
    "geographic_travel_coordination",
    "automated_system_telemetry",
    "phatic_acknowledgments",
]

SPECIFIC_TOPICS = [
    # Vocational
    "software_backend_development", "software_frontend_development",
    "mobile_app_development", "cloud_infrastructure_operations",
    "api_and_integration_design", "devops_and_ci_cd_pipelines",
    "system_architecture_design",
    "machine_learning_model_training", "llm_prompt_engineering_and_evals",
    "data_pipeline_engineering", "analytics_and_business_intelligence",
    "academic_literature_review",
    "product_roadmap_planning", "user_experience_research",
    "competitor_and_market_analysis", "go_to_market_strategy",
    "startup_fundraising_and_pitching", "investor_and_board_relations",
    "cofounder_strategic_alignment",
    "talent_sourcing_and_recruiting", "candidate_interviewing",
    "employee_onboarding_and_training",
    "b2b_sales_outreach", "client_success_and_support",
    "marketing_campaign_management",
    "legal_contract_drafting", "legal_ip_and_patent_filing",
    "curriculum_and_lesson_planning",
    "event_planning_and_conference_ops",
    # Social
    "child_school_enrollment_and_admin", "child_extracurricular_activities",
    "childcare_and_babysitter_logistics", "child_medical_coordination",
    "pet_care_and_veterinary",
    "spousal_schedule_alignment", "shared_financial_decisions",
    "extended_family_holiday_planning", "sibling_catchups_and_support",
    "close_friend_catchups", "hobby_group_coordination",
    "neighborhood_and_hoa_communication",
    "volunteer_and_philanthropic_work", "alumni_network_engagement",
    "sports_league_participation", "party_and_event_hosting",
    "local_politics_and_activism",
    "personal_birthdays_and_milestones",
    "public_and_religious_holidays",
    "casual_social_chatter",
    # Embodied
    "primary_care_checkups", "dental_and_orthodontic_care",
    "pharmaceutical_and_prescription_management",
    "specialist_medical_consults", "vaccination_and_immunization_records",
    "gym_and_weightlifting_routines", "outdoor_hiking_and_trail_running",
    "camping_and_wilderness_permits", "winter_sports_and_skiing",
    "team_sports_and_intramurals",
    "meditation_and_mindfulness_practice",
    "dietary_and_nutrition_planning",
    "leisure_reading_and_books",
    # Infrastructure
    "personal_budgeting_and_cashflow", "tax_preparation_and_filing",
    "investment_portfolio_management", "credit_card_and_debt_management",
    "insurance_policy_renewals",
    "mortgage_and_rent_payments", "real_estate_transactions",
    "internet_and_telecom_services",
    "plumbing_and_hvac_maintenance", "landscaping_and_exterior_care",
    "interior_cleaning_and_housekeeping",
    "home_security_and_monitoring",
    "car_purchase_and_leasing", "vehicle_dealership_service",
    "auto_insurance_and_registration",
    "ride_share_and_taxi_receipts", "parking_and_transportation_fees",
    "flight_booking_and_itineraries", "hotel_and_lodging_reservations",
    "travel_reward_point_management",
    "grocery_and_supermarket_runs", "restaurant_and_takeout_orders",
    "electronics_and_hardware_upgrades",
    "software_subscription_renewals", "package_delivery_tracking",
    "return_and_refund_processing",
    # Telemetry
    "software_webhook_alerts", "api_usage_and_billing_warnings",
    "security_login_notifications", "newsletter_and_marketing_broadcasts",
    "calendar_reminder_pings",
    "scheduling_poll_or_doodle",
    # Escape hatch
    "taxonomy_expansion_required",
]

_SPECIFIC_TOPICS_CSV = ", ".join(SPECIFIC_TOPICS)

# ─── Topic classification rules (shared by both prompt paths) ─────────

_TOPIC_RULES = """\
Topic classification (two-tiered taxonomy):
- universal_spheres: 1-2 tags describing the structural nature of the event.
  MUST be from: {spheres}
- specific_topics: 0-3 tags for exact semantic context. Select ONLY from this list:
  {specific_topics}
  Use empty array [] for pure phatic acknowledgments or automated system telemetry.
- For birthday calendar events, use personal_birthdays_and_milestones.
- For public/religious holidays, use public_and_religious_holidays.
- For brief social banter without coordination intent, use casual_social_chatter.
- Reserve close_friend_catchups for actual meetup coordination or substantive life updates.
- taxonomy_expansion_required: Use ONLY as an absolute last resort when an event genuinely
  does not fit ANY of the 90+ specific tags. Expect <2% usage.
- You MUST NOT invent tags. Select ONLY from the lists above.\
""".format(spheres=", ".join(UNIVERSAL_SPHERES), specific_topics=_SPECIFIC_TOPICS_CSV)

# ─── Thread-full prompts ──────────────────────────────────────────────

THREAD_FULL_SYSTEM = """\
You are a triage classifier for a personal knowledge graph. You classify THREADS (entire conversations), not individual messages.

The core question: "Does the USER need to THINK, DECIDE, or ACT on anything in this thread?"

Thread-level score from 0.0 to 1.0 in increments of 0.1:

SCORE 0.1 = IGNORE — Automated noise, no human action
SCORE 0.3 = LIGHTWEIGHT — Worth indexing, not worth deep analysis
SCORE 0.5 = MEDIUM — Coordination, soft requests, scheduling
SCORE 0.7 = DEEP — Clear action required from a human
SCORE 0.9 = CRITICAL — Urgent/imminent deadline, blocking issue

Key rules:
- Automated senders (noreply, alerts) = 0.1 unless financial data (then 0.3)
- A human asking the user to do something specific = 0.7+
- Sender tier matters: tier-1 acks in active threads = 0.3, tier-3 acks = 0.1
- Give each message its OWN score. Thread score = max of message scores.

{topic_rules}

Keep thread_summary to 1 sentence. Keep each message reason to 15 words max.
Respond with ONLY valid JSON, no other text. No markdown fences.
""".format(topic_rules=_TOPIC_RULES)

THREAD_FULL_SUFFIX = """\
---
Classify this thread. Return a JSON object:
{
  "thread_score": <0.0-1.0>,
  "domain": "<work|personal|family|financial|health|legal|travel|shopping|automated>",
  "universal_spheres": ["1-2 structural tags from the taxonomy"],
  "specific_topics": ["0-3 semantic tags from the taxonomy"],
  "thread_status": "<active_conversation|awaiting_user|awaiting_them|stale|one_shot>",
  "relationship": "<description of sender-user relationship>",
  "thread_summary": "<1 sentence summary>",
  "message_scores": [
    {"id": "<event_id>", "score": <0.0-1.0>, "reason": "<15 words max>"}
  ],
  "extraction_candidates": ["<event_ids worth deep extraction>"],
  "pii": ["<financial|medical|legal|credentials|travel_docs or empty>"],
  "sensitivity": ["<health_discussion|relationship_conflict|financial_distress|legal_matter|intimate_content|child_info|grief or empty>"],
  "commitment_type": "<inbound_request|user_commitment|deadline|waiting_on|payment_due|follow_up|null>"
}
"""

# ─── Thread summary prompts ──────────────────────────────────────────

THREAD_SUMMARY_SYSTEM = """\
You are summarizing a conversation thread for a personal knowledge graph.
Your summary will be used as context for triage classification of recent messages.

Focus on:
- Key topics, decisions, and commitments discussed
- Who asked whom to do what (action items, requests)
- Any deadlines, dates, or time-sensitive items
- The overall relationship dynamic and thread status
- Any unresolved questions or pending items

Be factual and concise. Do NOT classify or score — just summarize.
"""

THREAD_SUMMARY_SUFFIX = """\
---
Summarize this conversation. Return a JSON object:
{
  "summary": "<3-5 sentence summary of the conversation so far>",
  "key_topics": ["<topic tags>"],
  "open_items": ["<any unresolved requests, questions, or commitments>"],
  "participants_summary": "<who is involved and their roles>"
}
"""

# ─── Compact (batch) prompts ─────────────────────────────────────────

MSG_COMPACT_SYSTEM = """\
You are a triage classifier for a personal knowledge graph. Score 0.0-1.0.

0.1=noise 0.3=index-only 0.5=coordination 0.7=action-needed 0.9=urgent

{topic_rules}

Respond with ONLY valid JSON, no other text.
""".format(topic_rules=_TOPIC_RULES)

MSG_COMPACT_SUFFIX = """\
---
Classify ALL items above. Return a JSON array with one object per item, in order.
Each object must have:
- "id": the item id string
- "score": 0.0-1.0 (triage importance)
- "reason": 25 words or less
- "domain": one of: work, personal, family, financial, health, legal, travel, shopping, automated
- "universal_spheres": 1-2 structural tags from the taxonomy in the system prompt
- "specific_topics": 0-3 semantic tags from the taxonomy in the system prompt. Empty array [] is valid.
- "entities": companies, products, projects mentioned IN THE BODY
- "pii": array of PII types present, or empty. Types: financial, medical, legal, credentials, travel_docs
- "sensitivity": array of sensitivity flags, or empty. Types: health_discussion, relationship_conflict, financial_distress, legal_matter, intimate_content, child_info, grief
- "commitment_type": one of: inbound_request, user_commitment, deadline, waiting_on, payment_due, follow_up, or null if none
"""

# ─── Prior thread context ────────────────────────────────────────────

PRIOR_THREAD_CONTEXT_BLOCK = """\
PRIOR THREAD CONTEXT (from previous analysis):
- Summary: {thread_summary}
- Score: {thread_score}
- Status: {thread_status}
- Domain: {domain}
- Commitment type: {commitment_type}

The following messages include both CONTEXT (previously analyzed) and NEW messages.
Focus your analysis on what the NEW messages change about this thread.
Update the thread score, status, and summary based on the new information.
"""
