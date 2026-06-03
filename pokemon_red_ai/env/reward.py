"""Reward functions, one per env.

v1 is intentionally minimal — see CLAUDE.md: "treat the first version
as a starting point, not a target." Subsequent branches add the
exploration bonus, HP penalty, event-flag bonus, etc. Keeping the
function pure (taking two metrics dicts, returning a float) makes
those iterations testable in isolation.
"""

from __future__ import annotations

Metrics = dict[str, float]


def compute_overworld_reward(prev: Metrics, curr: Metrics) -> float:
    """Minimal v1 overworld reward.

    Two components:
      - 0.01 per party-level gained (dense-ish, fires when a mon levels up)
      - 100  on earning the Boulder Badge (sparse, the v1 success signal)

    The badge term is huge relative to the level term so the optimal
    policy unambiguously prefers progress to grinding.
    """
    level_delta = curr["party/level_sum"] - prev["party/level_sum"]
    badge_delta = curr["progress/boulder_badge"] - prev["progress/boulder_badge"]
    return 0.01 * level_delta + 100.0 * badge_delta
