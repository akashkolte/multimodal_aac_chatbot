# Communication efficiency — SLO pass/fail on response latency.
from __future__ import annotations


def compute_efficiency(latency_log: dict, slo_target: float = 6.0) -> dict:
    """Check if total response time meets the SLO target."""
    t_total = latency_log.get("t_total", 0.0)
    return {
        "t_total": round(t_total, 3),
        "slo_target": slo_target,
        "slo_passed": t_total < slo_target,
        "margin_s": round(slo_target - t_total, 3),
    }
