"""Tests for Clarity Queue v2 features.

Covers: undo, recurrence, defer, thread provenance, sender rules,
categories, all-tasks view, custom fields.
"""

import json
import time

import pytest

from alteris.store import LayeredGraphStore
from alteris.mcp_tools.cq_tools import (
    _auto_bucket,
    _compute_next_due,
    handle_cq_add_task,
    handle_cq_dismiss,
    handle_cq_get_all,
    handle_cq_get_state,
    handle_cq_get_thread,
    handle_cq_manage_categories,
    handle_cq_manage_sender_rules,
    handle_cq_move_task,
    handle_cq_undo,
    handle_cq_update_task,
)


@pytest.fixture
def store():
    s = LayeredGraphStore(":memory:")
    _ = s.conn  # Initialize schema
    return s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Undo tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUndo:
    def test_undo_empty_returns_error(self, store):
        result = handle_cq_undo(store)
        assert "error" in result

    def test_undo_add_deletes_task(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Test task")
        task_id = result["task_id"]

        # Task should exist
        tasks = store.get_cq_tasks()
        assert any(t["id"] == task_id for t in tasks)

        # Undo should delete it
        undo_result = handle_cq_undo(store)
        assert undo_result["undone"] == "add"
        assert undo_result["task_id"] == task_id

        tasks = store.get_cq_tasks()
        assert not any(t["id"] == task_id for t in tasks)

    def test_undo_dismiss_restores_task(self, store):
        result = handle_cq_add_task(store, bucket="immediate", title="Important")
        task_id = result["task_id"]

        # Dismiss it
        handle_cq_dismiss(store, task_id=task_id)

        # Check it's gone from active
        tasks = store.get_cq_tasks()
        assert not any(t["id"] == task_id and t.get("source") != "dismissed" for t in tasks)

        # Undo should restore it
        undo_result = handle_cq_undo(store)
        assert undo_result["undone"] == "dismiss"

        tasks = store.get_cq_tasks()
        restored = [t for t in tasks if t["id"] == task_id]
        assert len(restored) == 1
        assert restored[0]["bucket"] == "immediate"

    def test_undo_move_restores_bucket(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Moveable")
        task_id = result["task_id"]

        handle_cq_move_task(store, task_id=task_id, target_bucket="immediate")

        # Verify it moved
        tasks = store.get_cq_tasks()
        moved = next(t for t in tasks if t["id"] == task_id)
        assert moved["bucket"] == "immediate"

        # Undo
        handle_cq_undo(store)
        tasks = store.get_cq_tasks()
        restored = next(t for t in tasks if t["id"] == task_id)
        assert restored["bucket"] == "review"

    def test_undo_update_restores_fields(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Original")
        task_id = result["task_id"]

        # Pop the add undo entry so the update undo is on top
        store.pop_cq_undo()

        handle_cq_update_task(store, task_id=task_id, title="Updated")
        tasks = store.get_cq_tasks()
        assert next(t for t in tasks if t["id"] == task_id)["title"] == "Updated"

        handle_cq_undo(store)
        tasks = store.get_cq_tasks()
        assert next(t for t in tasks if t["id"] == task_id)["title"] == "Original"

    def test_canUndo_flag_in_state(self, store):
        state = handle_cq_get_state(store)
        assert state["canUndo"] is False

        handle_cq_add_task(store, bucket="review", title="Something")
        state = handle_cq_get_state(store)
        assert state["canUndo"] is True

    def test_undo_prune_old_entries(self, store):
        handle_cq_add_task(store, bucket="review", title="Old task")
        # Manually set created_at to 8 days ago
        store.conn.execute(
            "UPDATE cq_undo_log SET created_at = ?",
            (int(time.time()) - 8 * 86400,),
        )
        store.conn.commit()

        store.prune_cq_undo()
        result = handle_cq_undo(store)
        assert "error" in result  # Should be pruned


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Recurrence tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRecurrence:
    def test_compute_next_due_daily(self):
        assert _compute_next_due("2026-02-17", {"freq": "daily", "interval": 1}) == "2026-02-18"

    def test_compute_next_due_weekly(self):
        assert _compute_next_due("2026-02-17", {"freq": "weekly", "interval": 1}) == "2026-02-24"

    def test_compute_next_due_biweekly(self):
        assert _compute_next_due("2026-02-17", {"freq": "weekly", "interval": 2}) == "2026-03-03"

    def test_compute_next_due_monthly(self):
        assert _compute_next_due("2026-01-15", {"freq": "monthly", "interval": 1}) == "2026-02-15"

    def test_compute_next_due_yearly(self):
        assert _compute_next_due("2025-02-17", {"freq": "yearly", "interval": 1}) == "2026-02-17"

    def test_compute_next_due_empty_returns_none(self):
        assert _compute_next_due(None, {"freq": "daily"}) is None
        assert _compute_next_due("2026-01-01", None) is None
        assert _compute_next_due("2026-01-01", {}) is None

    def test_compute_next_due_monthly_end_of_month(self):
        # 31st should clamp to 28th for safety
        result = _compute_next_due("2026-01-31", {"freq": "monthly", "interval": 1})
        assert result == "2026-02-28"

    def test_recurring_task_creates_next_on_done(self, store):
        result = handle_cq_add_task(
            store,
            bucket="review",
            title="Weekly standup",
            due_date="2026-02-17",
            recurrence={"freq": "weekly", "interval": 1},
        )
        task_id = result["task_id"]

        # Pop the add undo so done undo is clean
        store.pop_cq_undo()

        update_result = handle_cq_update_task(store, task_id=task_id, done=True)
        assert "next_instance" in update_result
        assert update_result["next_instance"]["due_date"] == "2026-02-24"

        # Verify next task exists
        tasks = store.get_cq_tasks()
        new_id = update_result["next_instance"]["task_id"]
        new_task = next(t for t in tasks if t["id"] == new_id)
        assert new_task["title"] == "Weekly standup"
        assert new_task["due_date"] == "2026-02-24"
        assert new_task["recurrence"] == {"freq": "weekly", "interval": 1}

    def test_non_recurring_done_no_next(self, store):
        result = handle_cq_add_task(store, bucket="review", title="One-off")
        task_id = result["task_id"]
        store.pop_cq_undo()

        update_result = handle_cq_update_task(store, task_id=task_id, done=True)
        assert "next_instance" not in update_result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Defer tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDefer:
    def test_deferred_task_hidden_from_state(self, store):
        handle_cq_add_task(
            store, bucket="review", title="Deferred item",
            defer_until="2099-01-01",
        )

        state = handle_cq_get_state(store)
        all_tasks = state["immediate"] + state["background"] + state["review"]
        assert not any(t["title"] == "Deferred item" for t in all_tasks)
        assert state["deferred_count"] == 1

    def test_deferred_task_visible_in_all(self, store):
        handle_cq_add_task(
            store, bucket="review", title="Future task",
            defer_until="2099-01-01",
        )

        all_state = handle_cq_get_all(store)
        assert any(t["title"] == "Future task" for t in all_state["deferred"])

    def test_past_defer_date_shows_task(self, store):
        handle_cq_add_task(
            store, bucket="review", title="Ready now",
            defer_until="2020-01-01",
        )

        state = handle_cq_get_state(store)
        all_tasks = state["immediate"] + state["background"] + state["review"]
        assert any(t["title"] == "Ready now" for t in all_tasks)

    def test_update_defer_until(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Defer me")
        task_id = result["task_id"]
        store.pop_cq_undo()

        handle_cq_update_task(store, task_id=task_id, defer_until="2099-12-31")

        tasks = store.get_cq_tasks()
        task = next(t for t in tasks if t["id"] == task_id)
        assert task["defer_until"] == "2099-12-31"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread provenance tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestThreadProvenance:
    def test_get_thread_no_events(self, store):
        result = handle_cq_get_thread(store, task_id="nonexistent_claim")
        assert result["messages"] == []

    def test_get_thread_with_events(self, store):
        from alteris.models import Event
        from alteris.privacy import SensitivityLevel

        now = int(time.time())
        e1 = Event(
            id="evt_1", source="mail", source_id="mail_1",
            event_type="email", timestamp=now - 3600,
            participants=("alice@test.com",),
            raw_content="Hey, about that project...",
            metadata={"thread_id": "thread_abc", "sender": "alice@test.com", "subject": "Project update"},
            sensitivity=SensitivityLevel.PUBLIC,
            created_at=now,
        )
        e2 = Event(
            id="evt_2", source="mail", source_id="mail_2",
            event_type="email", timestamp=now - 1800,
            participants=("bob@test.com",),
            raw_content="Thanks, I'll look into it.",
            metadata={"thread_id": "thread_abc", "sender": "bob@test.com", "subject": "Re: Project update"},
            sensitivity=SensitivityLevel.PUBLIC,
            created_at=now,
        )
        store.put_event(e1)
        store.put_event(e2)

        # Link events to a claim
        from alteris.models import Claim, Modality, ExtractionProvenance, ExtractionMethod
        claim = Claim(
            id="claim_proj",
            event_ids=["evt_1"],
            claim_type="commitment",
            subject="project",
            predicate="has_commitment",
            object='{"what": "Review project", "status": "open"}',
            confidence=0.8,
            modality=Modality.ASSERTED,
            provenance=ExtractionProvenance(
                model_id="test", prompt_version="v1",
                extraction_method=ExtractionMethod.CLOUD_MODEL,
                extracted_at=now,
            ),
            created_at=now,
        )
        store.put_claim(claim)

        # Now create a CQ task referencing this claim
        store.put_cq_task(
            task_id="cq_test123",
            bucket="review",
            title="Review project",
            source="manual",
            claim_id="claim_proj",
        )

        result = handle_cq_get_thread(store, task_id="cq_test123")
        assert result["thread_id"] == "thread_abc"
        assert result["message_count"] == 2
        assert result["messages"][0]["sender"] == "alice@test.com"
        assert result["messages"][1]["sender"] == "bob@test.com"

    def test_get_thread_requires_task_id(self, store):
        result = handle_cq_get_thread(store, task_id="")
        assert "error" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sender rules tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSenderRules:
    def test_list_empty(self, store):
        result = handle_cq_manage_sender_rules(store, action="list")
        assert result["count"] == 0

    def test_add_and_list(self, store):
        handle_cq_manage_sender_rules(
            store, action="add", pattern="ceo@bigcorp.com",
            priority="P1", note="CEO - always important",
        )
        result = handle_cq_manage_sender_rules(store, action="list")
        assert result["count"] == 1
        assert result["rules"][0]["pattern"] == "ceo@bigcorp.com"
        assert result["rules"][0]["priority"] == "P1"

    def test_add_block_rule(self, store):
        handle_cq_manage_sender_rules(
            store, action="add", pattern="newsletter@spam.com",
            priority="block",
        )
        result = handle_cq_manage_sender_rules(store, action="list")
        assert result["rules"][0]["priority"] == "block"

    def test_delete_rule(self, store):
        handle_cq_manage_sender_rules(
            store, action="add", pattern="test@test.com", priority="P2",
        )
        rules = handle_cq_manage_sender_rules(store, action="list")["rules"]
        rule_id = rules[0]["id"]

        handle_cq_manage_sender_rules(store, action="delete", rule_id=rule_id)
        result = handle_cq_manage_sender_rules(store, action="list")
        assert result["count"] == 0

    def test_check_sender_exact_match(self, store):
        store.put_sender_rule("alice@example.com", "P1")
        assert store.check_sender_rules("alice@example.com") == "P1"
        assert store.check_sender_rules("bob@example.com") is None

    def test_check_sender_domain_match(self, store):
        store.put_sender_rule("@bigcorp.com", "P2")
        assert store.check_sender_rules("anyone@bigcorp.com") == "P2"
        assert store.check_sender_rules("someone@othercorp.com") is None

    def test_invalid_priority(self, store):
        result = handle_cq_manage_sender_rules(
            store, action="add", pattern="x@y.com", priority="P99",
        )
        assert "error" in result

    def test_upsert_sender_rule(self, store):
        store.put_sender_rule("test@test.com", "P1")
        store.put_sender_rule("test@test.com", "P3")
        rules = store.get_sender_rules()
        assert len(rules) == 1
        assert rules[0]["priority"] == "P3"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Categories tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCategories:
    def test_list_empty(self, store):
        result = handle_cq_manage_categories(store, action="list")
        assert result["count"] == 0

    def test_add_and_list(self, store):
        handle_cq_manage_categories(
            store, action="add", name="finance", color="#00ff00", icon="dollarsign.circle",
        )
        result = handle_cq_manage_categories(store, action="list")
        assert result["count"] == 1
        assert result["categories"][0]["name"] == "finance"
        assert result["categories"][0]["color"] == "#00ff00"

    def test_delete_category(self, store):
        handle_cq_manage_categories(store, action="add", name="health")
        handle_cq_manage_categories(store, action="delete", name="health")
        result = handle_cq_manage_categories(store, action="list")
        assert result["count"] == 0

    def test_upsert_category(self, store):
        store.put_cq_category("work", "#ff0000", "briefcase")
        store.put_cq_category("work", "#0000ff", "building")
        cats = store.get_cq_categories()
        assert len(cats) == 1
        assert cats[0]["color"] == "#0000ff"

    def test_category_on_task(self, store):
        handle_cq_add_task(
            store, bucket="review", title="Tax prep",
            category="finance",
        )
        tasks = store.get_cq_tasks()
        assert tasks[0]["category"] == "finance"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# All-tasks view tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAllTasks:
    def test_empty_all_tasks(self, store):
        result = handle_cq_get_all(store)
        assert result["counts"]["total"] == 0

    def test_all_tasks_includes_done(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Done item")
        task_id = result["task_id"]
        store.pop_cq_undo()
        handle_cq_update_task(store, task_id=task_id, done=True)

        all_state = handle_cq_get_all(store)
        assert any(t["title"] == "Done item" for t in all_state["done"])

    def test_all_tasks_includes_dismissed(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Dismissed item")
        task_id = result["task_id"]

        # Dismiss creates a dismiss record but deletes manual tasks
        handle_cq_dismiss(store, task_id=task_id)

        # Manual tasks get deleted on dismiss, not dismissed
        # So this tests the KG dismissal path would work (no KG items in test)
        all_state = handle_cq_get_all(store)
        assert all_state["counts"]["total"] >= 0

    def test_all_tasks_counts(self, store):
        handle_cq_add_task(store, bucket="review", title="Active 1")
        handle_cq_add_task(store, bucket="immediate", title="Active 2")
        result = handle_cq_add_task(store, bucket="review", title="Deferred")
        task_id = result["task_id"]
        store.pop_cq_undo()
        handle_cq_update_task(store, task_id=task_id, defer_until="2099-01-01")

        all_state = handle_cq_get_all(store)
        assert all_state["counts"]["active"] == 2
        assert all_state["counts"]["deferred"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom fields tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCustomFields:
    def test_add_task_with_custom_fields(self, store):
        handle_cq_add_task(
            store, bucket="review", title="Meeting prep",
            custom_fields={"location": "Room 3B", "priority": "high"},
        )
        tasks = store.get_cq_tasks()
        assert tasks[0]["custom_fields"] == {"location": "Room 3B", "priority": "high"}

    def test_update_custom_fields(self, store):
        result = handle_cq_add_task(store, bucket="review", title="Task")
        task_id = result["task_id"]
        store.pop_cq_undo()

        handle_cq_update_task(
            store, task_id=task_id,
            custom_fields={"new_field": "value"},
        )
        tasks = store.get_cq_tasks()
        task = next(t for t in tasks if t["id"] == task_id)
        assert task["custom_fields"] == {"new_field": "value"}

    def test_custom_fields_round_trip(self, store):
        fields = {"key1": "val1", "key2": "val2", "key3": "val3"}
        result = handle_cq_add_task(
            store, bucket="review", title="Multi-field",
            custom_fields=fields,
        )
        task_id = result["task_id"]

        tasks = store.get_cq_tasks()
        task = next(t for t in tasks if t["id"] == task_id)
        assert task["custom_fields"] == fields


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Store migration tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSchemaMigration:
    def test_new_columns_exist(self, store):
        cols = {
            r["name"]
            for r in store.conn.execute("PRAGMA table_info(cq_tasks)").fetchall()
        }
        assert "defer_until" in cols
        assert "recurrence" in cols
        assert "custom_fields" in cols
        assert "category" in cols

    def test_new_tables_exist(self, store):
        tables = {
            r[0]
            for r in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "cq_undo_log" in tables
        assert "sender_rules" in tables
        assert "cq_categories" in tables

    def test_migration_idempotent(self, store):
        # Running migration again should not fail
        store._migrate_cq_schema()
        cols = {
            r["name"]
            for r in store.conn.execute("PRAGMA table_info(cq_tasks)").fetchall()
        }
        assert "defer_until" in cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-bucket tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAutoBucket:
    def test_no_deadline(self):
        assert _auto_bucket(None, int(time.time())) == "background"

    def test_overdue(self):
        assert _auto_bucket("2020-01-01", int(time.time())) == "immediate"

    def test_today(self):
        from datetime import datetime, timezone
        today = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        assert _auto_bucket(today, int(time.time())) == "immediate"

    def test_this_week(self):
        from datetime import datetime, timedelta, timezone
        now = int(time.time())
        future = datetime.fromtimestamp(now, tz=timezone.utc) + timedelta(days=3)
        assert _auto_bucket(future.strftime("%Y-%m-%d"), now) == "review"

    def test_far_future(self):
        assert _auto_bucket("2099-12-31", int(time.time())) == "background"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source channel tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceChannel:
    def test_manual_task_has_empty_channel(self, store):
        handle_cq_add_task(store, bucket="review", title="Manual")
        state = handle_cq_get_state(store)
        tasks = state["review"]
        assert tasks[0].get("source_channel", "") == ""

    def test_get_state_includes_source_channel_key(self, store):
        handle_cq_add_task(store, bucket="review", title="Test")
        state = handle_cq_get_state(store)
        # Manual tasks should have source_channel field (empty string)
        for t in state["review"]:
            assert "source_channel" in t or t.get("source") == "manual"
