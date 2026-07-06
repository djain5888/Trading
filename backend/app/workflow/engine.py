"""The workflow engine.

Resolves services through DI, runs a named workflow with per-step logging and
timing, and returns a :class:`WorkflowRun`. On any step failure it reports the
error and exits gracefully, leaving no partial report. It contains no business
logic — only coordination.
"""

from __future__ import annotations

from app.core.clock import Clock
from app.core.logging import get_logger
from app.workflow.context import WorkflowContext
from app.workflow.errors import WorkflowStepError
from app.workflow.models import WorkflowReport, WorkflowRequest, WorkflowRun
from app.workflow.registry import WorkflowRegistry, default_registry
from app.workflow.services import WorkflowServices

logger = get_logger(__name__)


class WorkflowEngine:
    """Runs registered workflows using DI-resolved services."""

    def __init__(
        self,
        services: WorkflowServices,
        clock: Clock,
        registry: WorkflowRegistry | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            services: The resolved service container.
            clock: Time source for run/step timing.
            registry: Workflow registry; a default one is built if omitted.
        """
        self._services = services
        self._clock = clock
        self._registry = registry or default_registry()

    @property
    def registry(self) -> WorkflowRegistry:
        """Return the workflow registry."""
        return self._registry

    async def run(self, name: str, request: WorkflowRequest) -> WorkflowRun:
        """Run a workflow by name and return its outcome.

        Args:
            name: The registered workflow name.
            request: The workflow parameters.

        Returns:
            A :class:`WorkflowRun` describing the execution.

        Raises:
            UnknownWorkflowError: If ``name`` is not registered.
        """
        workflow = self._registry.create(name)
        context = WorkflowContext(
            services=self._services, request=request, clock=self._clock
        )
        logger.info("Workflow started: %s", name)
        started = self._clock.now()
        report: WorkflowReport | None = None
        success = True
        error: str | None = None
        try:
            report = await workflow.run(context)
        except WorkflowStepError as exc:
            success, error = False, str(exc)
        except Exception as exc:  # noqa: BLE001 - graceful catch-all
            success, error = False, str(exc)
            logger.exception("Workflow failed unexpectedly: %s", name)
        elapsed = (self._clock.now() - started).total_seconds()
        logger.info("Workflow finished: %s success=%s (%.3fs)", name, success, elapsed)
        return WorkflowRun(
            workflow=name,
            success=success,
            elapsed_seconds=elapsed,
            steps=tuple(context.steps),
            error=error,
            report=report,
        )


def build_workflow_engine(
    services: WorkflowServices | None = None,
    clock: Clock | None = None,
    registry: WorkflowRegistry | None = None,
) -> WorkflowEngine:
    """Construct a workflow engine from DI-resolved services.

    Args:
        services: Optional service container; resolved from DI when omitted.
        clock: Optional clock; taken from the services when omitted.
        registry: Optional workflow registry.

    Returns:
        A ready-to-use :class:`WorkflowEngine`.
    """
    resolved = services or WorkflowServices.resolve()
    return WorkflowEngine(
        services=resolved,
        clock=clock or resolved.clock,
        registry=registry or default_registry(),
    )
