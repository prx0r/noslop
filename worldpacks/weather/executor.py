"""Deterministic executor for weather worldpack.

Executes FORMAT_RESPONSE actions by returning the pre-formatted response.
"""
from __future__ import annotations

import time

from cogym_kernel.kernel.contracts import ActionResult, ActionSpec


class WeatherExecutor:
    """Executor that passes through formatted responses."""

    def execute(self, action: ActionSpec) -> ActionResult:
        now = time.time_ns()
        if action.kind == "FORMAT_RESPONSE":
            response = action.payload.get("response", "")
            return ActionResult(
                action_id=action.action_id,
                status="ok",
                payload={"response": response},
                started_ns=now,
                finished_ns=time.time_ns(),
                wall_ms=0.1,
            )
        return ActionResult(
            action_id=action.action_id,
            status="error",
            error=f"unknown action: {action.kind}",
            started_ns=now,
            finished_ns=time.time_ns(),
        )
