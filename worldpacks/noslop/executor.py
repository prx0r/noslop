"""Deterministic executor for noslop classifier worldpack."""
import time
from cogym_kernel.kernel.contracts import ActionResult, ActionSpec


class ClassifierExecutor:
    def execute(self, action: ActionSpec) -> ActionResult:
        now = time.time_ns()
        if action.kind == "CLASSIFY":
            return ActionResult(
                action_id=action.action_id,
                status="ok",
                payload=action.payload,
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
