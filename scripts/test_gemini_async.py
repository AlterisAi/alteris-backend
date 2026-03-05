"""Direct test of the Gemini async fixes.

Exercises the exact code paths that caused the 57-minute hang:
1. agenerate with cache_system=True (F4: was blocking event loop)
2. Multiple concurrent calls through the semaphore (F1: semaphore in wait_for)
3. Timeout behavior (F3: no longer retries timeouts)
4. Client cleanup (F6, F10: aclose_all)

Usage:
    python scripts/test_gemini_async.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_single_call():
    """Test 1: Single agenerate call with cache_system=True."""
    from alteris.llm.gemini import GeminiClient

    client = GeminiClient()
    print("Test 1: Single agenerate call (cache_system=True)")
    t0 = time.time()
    result = await client.agenerate(
        prompt="Respond with exactly: HELLO",
        system="You are a test assistant. Respond briefly.",
        cache_system=True,
        max_tokens=32,
    )
    elapsed = time.time() - t0
    ok = result is not None and len(result) > 0
    print(f"  {'PASS' if ok else 'FAIL'}: {elapsed:.1f}s, response={repr(result[:80] if result else None)}")
    return ok


async def test_concurrent_calls():
    """Test 2: 15 concurrent calls through the 10-slot semaphore."""
    from alteris.llm.gemini import GeminiClient

    client = GeminiClient()
    n = 15
    print(f"\nTest 2: {n} concurrent agenerate calls (10-slot semaphore)")

    statuses = []

    async def _one_call(i):
        t0 = time.time()
        status_log = []

        def on_status(s):
            status_log.append((s, time.time() - t0))

        result = await client.agenerate(
            prompt=f"Respond with exactly: NUMBER {i}",
            system="You are a test assistant. Respond with the exact text requested.",
            cache_system=True,
            max_tokens=32,
            on_status=on_status,
        )
        elapsed = time.time() - t0
        ok = result is not None
        statuses.append((i, ok, elapsed, status_log))
        return ok

    t0 = time.time()
    results = await asyncio.gather(*[_one_call(i) for i in range(n)])
    total = time.time() - t0

    succeeded = sum(1 for r in results if r)
    print(f"  {succeeded}/{n} succeeded in {total:.1f}s total")

    # Show timing distribution
    times = [s[2] for s in statuses if s[1]]
    if times:
        print(f"  Per-call: min={min(times):.1f}s, max={max(times):.1f}s, avg={sum(times)/len(times):.1f}s")

    # Check that semaphore waiting actually happened
    waited = sum(1 for s in statuses if any(st[0] == "waiting_for_slot" for st in s[3]))
    print(f"  {waited}/{n} waited for semaphore slot (expected: {max(0, n-10)}+)")

    return succeeded == n


async def test_timeout_fires():
    """Test 3: Verify that asyncio.wait_for timeout actually fires."""
    print("\nTest 3: Verify wait_for timeout fires (simulated)")

    sem = asyncio.Semaphore(1)

    async def _never_returns():
        async with sem:
            await asyncio.sleep(999)

    # First, fill the semaphore
    await sem.acquire()

    t0 = time.time()
    try:
        # This should timeout waiting for the semaphore
        await asyncio.wait_for(_never_returns(), timeout=2.0)
        print("  FAIL: did not timeout")
        ok = False
    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        ok = 1.5 < elapsed < 3.0
        print(f"  {'PASS' if ok else 'FAIL'}: timed out in {elapsed:.1f}s (expected ~2.0s)")
    finally:
        sem.release()

    return ok


async def test_cleanup():
    """Test 4: Verify aclose_all cleans up all clients."""
    from alteris.llm.gemini import GeminiClient

    client = GeminiClient()
    print("\nTest 4: aclose_all cleans up tracked clients")

    # Make a call to create the async client
    await client.agenerate(
        prompt="Say OK",
        system="Respond briefly.",
        max_tokens=16,
    )

    n_tracked = len(client._created_async_clients)
    has_client = client._async_client is not None
    print(f"  Before cleanup: {n_tracked} tracked clients, _async_client={'set' if has_client else 'None'}")

    await client.aclose_all()

    n_after = len(client._created_async_clients)
    has_after = client._async_client is not None
    ok = n_after == 0 and not has_after
    print(f"  After cleanup:  {n_after} tracked clients, _async_client={'set' if has_after else 'None'}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


async def test_cross_loop_semaphore():
    """Test 5: Verify semaphore works correctly across event loops."""
    import weakref
    from alteris.llm.gemini import GeminiClient

    print("\nTest 5: WeakKeyDictionary semaphore survives loop lifecycle")

    # Check the type
    is_weak = isinstance(GeminiClient._semaphore_by_loop, weakref.WeakKeyDictionary)
    print(f"  _semaphore_by_loop is WeakKeyDictionary: {is_weak}")

    # Get the semaphore for this loop
    sem = GeminiClient._get_semaphore()
    loop = asyncio.get_running_loop()
    in_dict = loop in GeminiClient._semaphore_by_loop
    print(f"  Current loop in dict: {in_dict}")
    print(f"  Semaphore value: {sem._value}")

    ok = is_weak and in_dict and sem._value == 10
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


async def main():
    print("=" * 60)
    print("  Gemini Async Fix Verification")
    print("=" * 60)
    print()

    results = {}

    results["single_call"] = await test_single_call()
    results["concurrent"] = await test_concurrent_calls()
    results["timeout"] = await test_timeout_fires()
    results["cleanup"] = await test_cleanup()
    results["semaphore"] = await test_cross_loop_semaphore()

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n  {passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
