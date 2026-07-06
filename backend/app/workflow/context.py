"""Workflow execution context.

Threads the resolved services and request through a workflow, and provides the
step runner that logs, times, records and error-handles each step.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TypeVar

from app.core.clock import Clock
from app.core.logging import get_logger
from app.workflow.errors import WorkflowStepError
from app.workflow.models import StepResult, WorkflowRequest
from app.workflow.services import WorkflowServices

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class WorkflowContext:
    """Per-run context passed to a workflow."""

    services: WorkflowServices
    request: WorkflowRequest
    clock: Clock
    steps: list[StepResult] = field(default_factory=list)

    async def run_step(
        self, name: str, awaitable: Awaitable[T], *, detail: str | None = None
    ) -> T:
        """Execute one step: log it, time it, record it, and handle failure.

        Args:
            name: Human-readable step name.
            awaitable: The step's work.
            detail: Optional success detail for the report.

        Returns:
            The step's result.

        Raises:
            WorkflowStepError: If the step raises; the failure is recorded first.
        """
        logger.info("Workflow step started: %s", name)
        started = self.clock.now()
        try:
            result = await awaitable
        except Exception as exc:
            duration = (self.clock.now() - started).total_seconds()
            self.steps.append(
                StepResult(
                    name=name, ok=False, duration_seconds=duration, error=str(exc)
                )
            )
            logger.error("Workflow step failed: %s (%s)", name, exc)
            raise WorkflowStepError(name, exc) from exc
        duration = (self.clock.now() - started).total_seconds()
        self.steps.append(
            StepResult(name=name, ok=True, duration_seconds=duration, detail=detail)
        )
        logger.info("Workflow step finished: %s (%.3fs)", name, duration)
        return result
