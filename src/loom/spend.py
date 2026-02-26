"""API spend tracking with daily limits and onboarding budget.

Append-only spend ledger backed by the api_spend table in the graph store.
Supports:
- Per-call cost recording based on model pricing
- Daily spend limits for bundled Gemini key ($2/day)
- Separate onboarding budget ($5) that doesn't count against daily limit
- Unlimited usage for user-provided keys
- Secret bypass code to reset daily counters
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import date, datetime, timezone

from loom.constants import (
    MODEL_PRICING,
    SPEND_BYPASS_SALT,
    SPEND_DAILY_LIMIT_USD,
    SPEND_ONBOARDING_BUDGET_USD,
)

logger = logging.getLogger(__name__)


def _today_str(dt: date | None = None) -> str:
    """Return YYYY-MM-DD string for a date, defaulting to today (UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc).date()
    return dt.isoformat()


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    thinking_tokens: int = 0,
) -> float:
    """Compute USD cost from token counts using the model pricing table.

    cached_input_tokens: tokens served from Gemini context cache (90% cheaper).
    thinking_tokens: reasoning tokens (charged at output rate for thinking models).
    Returns 0.0 for unknown models (local models, etc.).
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    # Fresh input tokens (subtract cached portion — Gemini includes cached in prompt_token_count)
    fresh_input = max(0, input_tokens - cached_input_tokens)
    cost = (
        fresh_input * pricing["input"]
        + cached_input_tokens * pricing["input"] * 0.1  # 90% discount
        + output_tokens * pricing["output"]
        + thinking_tokens * pricing["output"]  # thinking billed at output rate
    ) / 1_000_000
    return round(cost, 8)


def record_usage(
    store,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    source: str = "",
    cached_input_tokens: int = 0,
    thinking_tokens: int = 0,
) -> int:
    """Record an API call's token usage and computed cost.

    cached_input_tokens: tokens served from context cache (90% cheaper).
    thinking_tokens: reasoning tokens (charged at output rate).
    Returns the row ID of the inserted spend record.
    """
    cost = _compute_cost(model, input_tokens, output_tokens, cached_input_tokens, thinking_tokens)
    today = _today_str()
    return store.insert_spend(today, provider, model, input_tokens, output_tokens, cost, source)


def record_onboarding_usage(
    store,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    thinking_tokens: int = 0,
) -> int:
    """Record API usage against the onboarding budget pool.

    Uses source='onboarding' to distinguish from regular usage.
    Returns the row ID of the inserted spend record.
    """
    cost = _compute_cost(model, input_tokens, output_tokens, cached_input_tokens, thinking_tokens)
    today = _today_str()
    return store.insert_spend(today, provider, model, input_tokens, output_tokens, cost, "onboarding")


def check_limit(
    store,
    provider: str,
    has_own_key: bool = False,
) -> dict:
    """Check whether the provider is within its daily spend limit.

    User-provided keys are always unlimited. The bundled Gemini key
    has a $2/day cap. Onboarding spend is excluded from the daily limit.

    Returns:
        {
            "within_limit": bool,
            "usage_pct": float (0.0-1.0+),
            "daily_total": float (USD spent today, excluding onboarding),
            "limit": float (daily limit in USD, or -1 for unlimited),
        }
    """
    if has_own_key:
        return {
            "within_limit": True,
            "usage_pct": 0.0,
            "daily_total": 0.0,
            "limit": -1.0,
        }

    today = _today_str()
    spend_data = store.get_daily_spend(today, provider)

    # Exclude onboarding spend from the daily total
    daily_total = 0.0
    if spend_data:
        by_source = spend_data.get("by_source", {})
        total = spend_data.get("total_usd", 0.0)
        onboarding_spend = by_source.get("onboarding", 0.0)
        daily_total = total - onboarding_spend

    limit = SPEND_DAILY_LIMIT_USD
    usage_pct = daily_total / limit if limit > 0 else 0.0

    return {
        "within_limit": daily_total < limit,
        "usage_pct": round(usage_pct, 4),
        "daily_total": round(daily_total, 8),
        "limit": limit,
    }


def get_daily_spend(
    store,
    dt: date | None = None,
    provider: str | None = None,
) -> dict:
    """Get spend summary for a single day.

    Returns:
        {
            "date": str,
            "total_usd": float,
            "by_provider": {provider: float, ...},
            "by_source": {source: float, ...},
        }
    """
    day_str = _today_str(dt)
    raw = store.get_daily_spend(day_str, provider)
    if not raw:
        return {
            "date": day_str,
            "total_usd": 0.0,
            "by_provider": {},
            "by_source": {},
        }
    return raw


def get_spend_summary(store, days: int = 30) -> list[dict]:
    """Get daily spend summaries for the last N days.

    Returns a list of daily summaries ordered by date descending.
    """
    today = datetime.now(timezone.utc).date()
    start = date(today.year, today.month, today.day)
    # Walk backwards
    from datetime import timedelta
    start_date = (today - timedelta(days=days - 1)).isoformat()
    end_date = today.isoformat()
    return store.get_spend_range(start_date, end_date)


def verify_bypass_code(code: str, dt: date | None = None) -> bool:
    """Verify a daily bypass code.

    The code is HMAC-SHA256(SPEND_BYPASS_SALT, "YYYY-MM-DD")[:8].
    """
    day_str = _today_str(dt)
    expected = hmac.new(
        SPEND_BYPASS_SALT.encode(),
        day_str.encode(),
        hashlib.sha256,
    ).hexdigest()[:8]
    return hmac.compare_digest(code.lower(), expected.lower())


def reset_daily_spend(store, dt: date | None = None) -> int:
    """Delete all spend records for a given date. Returns rows deleted."""
    day_str = _today_str(dt)
    return store.delete_spend_for_date(day_str)


def check_onboarding_budget(store) -> dict:
    """Check the onboarding budget status.

    Sums all spend records with source='onboarding' across all dates.

    Returns:
        {
            "used": float (USD spent from onboarding pool),
            "remaining": float (USD remaining),
            "exhausted": bool,
        }
    """
    # Query all onboarding spend across all dates
    rows = store.conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM api_spend WHERE source = 'onboarding'"
    ).fetchone()
    used = rows[0] if rows else 0.0
    remaining = max(0.0, SPEND_ONBOARDING_BUDGET_USD - used)
    return {
        "used": round(used, 8),
        "remaining": round(remaining, 8),
        "exhausted": used >= SPEND_ONBOARDING_BUDGET_USD,
    }
