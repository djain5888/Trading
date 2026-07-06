"""The workflow interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.workflow.context import WorkflowContext
from app.workflow.models import WorkflowReport


class Workflow(ABC):
    """Orchestrates existing modules to accomplish a user-facing task.

    A workflow contains no business logic; it only coordinates services via the
    context's step runner and returns a report.
    """

    name: ClassVar[str]

    @abstractmethod
    async def run(self, context: WorkflowContext) -> WorkflowReport:
        """Execute the workflow and return its report."""
