# V12 Priority-Freshness Live Engine

## Scope

V12 changes live execution architecture only. The frozen Candidate-C strategy, nine-series universe, Q1/Q10 sizing, M0-M5 window, opposite-BBO exit rule, M5 flatten rule, fee assumptions, V10 recording bundle, and V11 V2-only/idempotent order transport remain unchanged.

The first real V12 run is Q1 one-window only. The V12 Q10 launcher is intentionally disabled until a fresh Q1 latency audit passes.

## Why V12 exists

The interrupted V11 session showed two dominant execution defects:

1. ENTRY decisions were often made from a book state that the raw feed had already superseded.
2. Resting orders often became obsolete and filled before the main control loop revisited that ticker and sent a cancel.

The root architecture issue was that the quote-critical path shared a synchronous loop with balance, position, queue, resting-order, create/cancel, and historical-key work.

## Architecture

### 1. Independent raw-book watchdog

A dedicated thread tails the same persisted V5 `book_top3_events.jsonl` from byte zero. It maintains a latest-row cache and a short per-ticker recent history.

The watchdog is allowed to do only one trading action: cancel an already-authorized resting strategy order when a new raw row makes the tracked role/side/price obsolete under the exact frozen public-book mechanics.

It never creates orders and never mutates strategy inventory state.

### 2. Dedicated priority-cancel writers

When the watchdog detects invalidation, it dispatches a V2 singular cancel through a small dedicated worker pool. Each worker owns its own `LiveClient`, so a main-thread REST request cannot block the cancel send.

The main thread later reconciles the cancel response, fills, and authoritative position before any replacement is allowed.

### 3. Create freshness guard

Before every CREATE, the main engine compares the decision source row against the watchdog's newest raw row.

If the watchdog has already seen a newer row, or the watchdog-current row invalidates the intended quote, CREATE is aborted. A bounded recent-row history also audits the tiny guard-to-POST race and records whether a newer raw row existed before the actual request-send timestamp.

### 4. Current-window-only actionable set

Finalized/historical ticker rows are pruned from `latest_rows`. Trading passes visit only active or current eligible M0-M5 tickers.

`risk_tick()` is no longer called once per historical key. It remains once per outer loop so the proven loss/position/order/queue safety logic remains in place.

### 5. Concurrency safety

The watchdog sees a copied snapshot of `active` orders protected by a lock. A `(ticker, order_id)` may have at most one priority cancel in flight.

While a priority cancel is pending, normal reconcile/poll paths do not send a duplicate cancel or replacement. M5/shutdown retains the existing fail-closed group-trigger and cleanup protection.

## Latency instrumentation

V12 writes `latency_events_v12.jsonl` with wall and monotonic timestamps for:

- raw receipt -> watchdog detect;
- watchdog detect -> priority cancel request send;
- raw obsolete receipt -> priority cancel request send;
- priority cancel RTT;
- every main-thread HTTP call class and RTT;
- main raw-book read and ingest duration;
- decision -> create request send;
- source-row age at create request send;
- create freshness-guard blocks;
- any create detected as superseded at actual request send;
- main loop duration and actionable ticker count.

## Q1 smoke SLOs

Engineering targets, not alpha parameters:

- target obsolete -> cancel send: < 25 ms;
- p95 obsolete -> cancel send: <= 50 ms;
- hard observed max: <= 100 ms;
- superseded-at-send CREATE count: 0;
- current actionable ticker set: <= 18 (two adjacent 9-series windows at a boundary);
- final account flat;
- zero strategy resting orders;
- no unrecovered error.

`audit_v12_smoke(session_dir)` writes `v12_smoke_latency_audit.json` and reports separate safety, create-freshness, actionable-set, and cancel-latency gates. If no real stale-order invalidation occurs during Q1, the run can be operationally safe but the latency proof is marked incomplete rather than passed.

## Promotion policy

Do not treat Q1 PnL as the primary acceptance criterion. Promote only if the execution gates pass. V12 Q10 remains disabled until a fresh Q1 run has measured at least one real invalidation and passed the latency SLOs.
