"""Outer-iteration schedule; independent of native optimizer mini-updates."""
import math

MODE = "complete_trajectory_reasoning_action"


def coefficient(options, iteration):
    if type(iteration) is not int or iteration < 1:
        raise ValueError("SFT schedule requires a one-based rollout iteration")
    initial = float(options["coef"])
    end = options["decay_end_iteration"]
    if not math.isfinite(initial) or initial < 0 or type(end) is not int or end < 2:
        raise ValueError("Invalid SFT decay schedule")
    schedule = options.get("decay_schedule", "linear")
    if schedule == "cosine":
        # Native rollout iterations are one-based. Keep the established
        # endpoints: iteration 1 = initial; iteration end = exactly zero.
        if iteration >= end:
            return 0.0
        progress = (iteration - 1) / (end - 1)
        return initial * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule != "linear":
        raise ValueError(f"Unknown SFT decay schedule: {schedule}")
    return initial * max(0.0, (end - iteration) / (end - 1))
