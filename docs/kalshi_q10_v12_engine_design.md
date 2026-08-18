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

A dedicated thread tails the same persisted V5 `book_top3_events.jsonl`. At startup it reads existing content to the current EOF with cancellation disabled, emits `WATCHDOG_CAUGHT_UP`, and only then becomes eligible to certify creates or invalidate resting quotes. This prevents historical pre-order rows from causing false cancels.

The watchdog maintains a latest-row cache plus bounded recent history per ticker. It is allowed to do only one trading action: cancel an already-authorized resting strategy order when a new raw row makes the tracked role/side/price obsolete under the exact frozen public-book mechanics.

It never creates orders and never mutates strategy inventory state.

### 2. Dedicated priority-cancel writers

When the watchdog detects invalidation, it dispatches a V2 singular cancel through nine dedicated worker slots, one per frozen series at maximum simultaneous breadth. Each worker owns its own `LiveClient`, so a main-thread REST request cannot occupy the cancel request's Python call path.

The main thread later reconciles the cancel response, fills, and authoritative position before any replacement is allowed. A `(ticker, order_id)` may have at most one priority cancel in flight.

### 3. Double create-freshness certification

No CREATE is allowed until the watchdog has caught up and reached at least the main decision source row.

The intended role/side/price is certified against the watchdog-current raw state twice: once before CREATE preparation and again immediately before the decision is logged and POST is invoked. If the watchdog is behind, has a newer row, or the current row invalidates the quote, CREATE is aborted and retried only from a later loop state.

A bounded recent-row history also audits the remaining guard-to-request-send race and records whether a newer raw row existed before the actual request-send timestamp. Once the POST response exposes the order id, publishing the active order wakes the watchdog so a change during create flight can be canceled immediately.

### 4. Current-window-only actionable set

Finalized/historical ticker rows are pruned from `latest_rows`. Trading passes visit only active or current eligible M0-M5 tickers.

`risk_tick()` is no longer called once per historical key. It remains once per outer loop so the proven loss/position/order/queue safety logic remains in place. Priority cancellation remains independent of those synchronous housekeeping calls.

### 5. Concurrency and shutdown safety

The watchdog sees a copied snapshot of `active` orders protected by a lock. While a priority cancel is pending, normal reconcile/poll paths do not send a duplicate cancel or replacement.

M5/shutdown retains the existing V11/V1 fail-closed order-group trigger, V2-only cancel safety, authoritative position refresh, and reduce-only flatten logic.

## Latency instrumentation

V12 writes `latency_events_v12.jsonl` with wall and monotonic timestamps for:

- watchdog startup catch-up;
- raw receipt -> watchdog detect;
- watchdog detect -> priority cancel request send;
- raw obsolete receipt -> priority cancel request send;
- priority cancel RTT;
- cancel response -> main reconciliation start;
- main reconciliation duration;
- every main-thread HTTP call class, local call duration, request-send, response-receive, and RTT;
- main raw-book read and ingest duration;
- decision -> create request send;
- create RTT;
- source-row age at create request send;
- create freshness-guard blocks;
- any create detected as superseded at actual request send;
- main loop duration and actionable ticker count.

## Q1 smoke SLOs

Engineering targets, not alpha parameters:

- median obsolete -> cancel send: <= 25 ms;
- p95 obsolete -> cancel send: <= 50 ms;
- hard observed max: <= 100 ms;
- every observed fast invalidation gets a successful fast-cancel result;
- superseded-at-send CREATE count: 0;
- watchdog catches up with zero JSON-tail errors;
- current actionable ticker set: <= 18 (two adjacent nine-series windows at a boundary);
- final account flat;
- zero strategy resting orders;
- no unrecovered error.

`audit_v12_smoke(session_dir)` writes `v12_smoke_latency_audit.json` and reports separate safety, watchdog, create-freshness, actionable-set, cancel-completion, and cancel-latency gates. If no real stale-order invalidation occurs during Q1, the run can be operationally safe but the latency proof is marked incomplete rather than passed.

## Promotion policy

Do not treat Q1 PnL as the primary acceptance criterion. Promote only if the execution gates pass. V12 Q10 remains disabled until a fresh Q1 run has measured at least one real invalidation and passed the latency SLOs. A larger Q3/Q5 execution smoke should precede any future fresh Q10.
