from __future__ import annotations

"""Disorder-tolerant wrapper for the frozen Q10 JOIN_ASK 15h secondary check.

The validation book JSONL is mostly receipt-time ordered but the coarse mmap audit found
a small -4.815s reversal.  The original V1 intentionally refused any reversal larger
than 2s.  V1.1 changes NO trading/economic rule.  It only widens the book extraction
safety margin so the mmap fast path remains usable without pretending the file is
perfectly ordered.

Safety policy
-------------
- Allow coarse book-order reversal only up to 60 seconds; anything worse still aborts.
- Expand every relevant M1-M5 book interval by 60 seconds on both sides.
- V6.3 already takes an extra coarse-index sample before/after each padded interval.
- The base V1 loader still applies exact ticker + elapsed_s filtering and audits the
  reconstructed receipt clock against persisted receipt_time (<=1ms).
- Strategy, Q10 size, 5c entry, 100ms action latency, JOIN_ASK queue model, M5 fallback,
  fees, universe, and no-retuning guardrails are unchanged.

This is still a SECONDARY HISTORICAL ROBUSTNESS CHECK, not independent validation.
READ ONLY. NO API. NO ORDERS.
"""

from . import mm_deep_tail_join_ask_q10_secondary_validation_v1 as BASE
from . import mm_deep_tail_trailing_passive_exit_dev_v6_3 as V63

VERSION = "MM_DEEP_TAIL_JOIN_ASK_Q10_SECONDARY_VALIDATION_V1_1_DISORDER_GUARD"
BOOK_DISORDER_GUARD_S = 60.0


def run_q10_join_ask_secondary_validation(source_session, *, hard_bind=True, show=True):
    old_base_version = BASE.VERSION
    old_pad = BASE.TIME_PAD_S
    old_max_disorder = V63.MAX_COARSE_DISORDER_S
    try:
        BASE.VERSION = VERSION
        BASE.TIME_PAD_S = BOOK_DISORDER_GUARD_S
        V63.MAX_COARSE_DISORDER_S = BOOK_DISORDER_GUARD_S

        if show:
            print(
                f"V1.1 BOOK DISORDER GUARD: allowing <= {BOOK_DISORDER_GUARD_S:.0f}s coarse reversal "
                f"and padding each relevant M1-M5 interval by +/-{BOOK_DISORDER_GUARD_S:.0f}s."
            )
            print("Observed prior coarse reversal was 4.815s; strategy/economics are unchanged.")

        return BASE.run_q10_join_ask_secondary_validation(
            source_session,
            hard_bind=hard_bind,
            show=show,
        )
    finally:
        BASE.VERSION = old_base_version
        BASE.TIME_PAD_S = old_pad
        V63.MAX_COARSE_DISORDER_S = old_max_disorder
