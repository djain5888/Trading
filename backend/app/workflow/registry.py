"""The workflow registry."""

from __future__ import annotations

from collections.abc import Callable

from app.workflow.base import Workflow
from app.workflow.errors import UnknownWorkflowError
from app.workflow.workflows import (
    CollectWorkflow,
    ImportWorkflow,
    IndicatorsWorkflow,
    MorningWorkflow,
    ScanWorkflow,
)

WorkflowFactory = Callable[[], Workflow]


class WorkflowRegistry:
    """A registry of workflow factories."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._factories: dict[str, WorkflowFactory] = {}

    def register(self, name: str, factory: WorkflowFactory) -> None:
        """Register a workflow factory under ``name``."""
        self._factories[name] = factory

    def create(self, name: str) -> Workflow:
        """Create a workflow by name.

        Raises:
            UnknownWorkflowError: If ``name`` is not registered.
        """
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise UnknownWorkflowError(f"Unknown workflow '{name}'.") from exc

    def available(self) -> list[str]:
        """Return the sorted registered workflow names."""
        return sorted(self._factories)


def default_registry() -> WorkflowRegistry:
    """Return a registry populated with the built-in workflows."""
    registry = WorkflowRegistry()
    registry.register("morning", MorningWorkflow)
    registry.register("import", ImportWorkflow)
    registry.register("indicators", IndicatorsWorkflow)
    registry.register("scan", ScanWorkflow)
    registry.register("collect", CollectWorkflow)
    return registry
