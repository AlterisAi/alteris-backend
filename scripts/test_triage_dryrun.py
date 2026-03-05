"""Dry-run triage against real Gemini to verify the async hang fix.

Copies ~/.alteris/graph.db to a temp location, runs Stage 4 triage
with real Gemini, prints timing and results, then deletes the copy.
Your real graph.db is never touched.

Usage:
    python scripts/test_triage_dryrun.py [--threads N] [--hours H]

    --threads N   Cap number of threads to triage (default: 20)
    --hours H     Only triage events from the last H hours (default: 168 = 1 week)
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Dry-run triage with real Gemini")
    parser.add_argument("--threads", type=int, default=20, help="Max threads to triage")
    parser.add_argument("--hours", type=int, default=168, help="Hours of events to include")
    args = parser.parse_args()

    src_db = os.path.expanduser("~/.alteris/graph.db")
    if not os.path.exists(src_db):
        print(f"ERROR: {src_db} not found")
        sys.exit(1)

    # Copy DB to temp location
    tmp_dir = tempfile.mkdtemp(prefix="alteris_dryrun_")
    tmp_db = os.path.join(tmp_dir, "graph.db")
    print(f"Copying {src_db} to {tmp_db} ...")
    t0 = time.time()
    shutil.copy2(src_db, tmp_db)
    # Also copy WAL/SHM if they exist
    for ext in ("-wal", "-shm"):
        src_extra = src_db + ext
        if os.path.exists(src_extra):
            shutil.copy2(src_extra, tmp_db + ext)
    print(f"  Copied in {time.time() - t0:.1f}s")

    try:
        from alteris.store import LayeredGraphStore
        from alteris.llm.gemini import GeminiClient
        from alteris.triage import (
            get_triageable_events,
            group_events_by_thread,
            build_sender_cache,
            _async_triage_one_thread,
            THREAD_TIMEOUT,
        )
        from alteris.profile import format_profile_context, load_profile
        import asyncio

        store = LayeredGraphStore(tmp_db)
        llm = GeminiClient(store=store)

        since_ts = int(time.time()) - (args.hours * 3600)
        now = int(time.time())

        print(f"\nFetching events (since {args.hours}h ago) ...")
        event_rows = get_triageable_events(store, resume=False, since_ts=since_ts)
        if not event_rows:
            print("  No events to triage.")
            store.close()
            return

        thread_groups = group_events_by_thread(event_rows)
        total_threads = len(thread_groups)
        total_events = len(event_rows)
        print(f"  {total_events} events in {total_threads} threads")

        # Cap threads for the test
        thread_items = list(thread_groups.items())[:args.threads]
        print(f"  Testing with {len(thread_items)} threads (capped at {args.threads})")

        sender_cache = build_sender_cache(store)
        profile = load_profile()
        profile_ctx = format_profile_context(profile)

        # Run triage with real Gemini -- collect results but don't write
        results = []
        done = 0
        failed = 0
        timed_out = 0
        start = time.time()

        async def _triage_one(tid, tevts):
            nonlocal done, failed, timed_out
            label = tid[:40]
            n = len(tevts)
            t_start = time.time()

            def on_status(status):
                pass  # silent

            try:
                result = await _async_triage_one_thread(
                    tid, tevts, store, sender_cache, llm, now,
                    profile_context=profile_ctx,
                    on_status=on_status,
                )
                elapsed = time.time() - t_start
                if result[1] is None:
                    failed += 1
                    print(f"  FAIL  {label} ({n} msgs) {elapsed:.1f}s")
                else:
                    score = result[1].thread_score if result[1] else 0
                    print(f"  OK    {label} ({n} msgs) -> score={score:.2f} {elapsed:.1f}s")
                results.append(result)
            except asyncio.TimeoutError:
                elapsed = time.time() - t_start
                timed_out += 1
                print(f"  TIMEOUT {label} ({n} msgs) {elapsed:.1f}s (limit={THREAD_TIMEOUT}s)")
                results.append((tid, None))

            done += 1

        async def _run_all():
            tasks = [_triage_one(tid, tevts) for tid, tevts in thread_items]
            await asyncio.gather(*tasks, return_exceptions=False)

        print(f"\nRunning triage against real Gemini (max 10 concurrent) ...\n")
        asyncio.run(_run_all())

        elapsed = time.time() - start
        succeeded = len(thread_items) - failed - timed_out

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Threads:   {len(thread_items)}")
        print(f"  Succeeded: {succeeded}")
        print(f"  Failed:    {failed}")
        print(f"  Timed out: {timed_out}")
        print(f"  Elapsed:   {elapsed:.1f}s")
        if succeeded > 0:
            print(f"  Avg/thread: {elapsed / len(thread_items):.1f}s")
        print(f"\n  No results written to any database.")

        store.close()

    finally:
        # Clean up temp DB
        print(f"\nCleaning up {tmp_dir} ...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Done.")


if __name__ == "__main__":
    main()
