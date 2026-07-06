"""Workflow-engine exceptions."""

from __future__ import annotations


class WorkflowError(Exception):
    """Base class for all workflow-engine errors."""


class UnknownWorkflowError(WorkflowError):
    """Raised when a workflow name is not registered."""


class WorkflowStepError(WorkflowError):
    """Raised when a workflow step fails; carries the step name and cause."""

    def __init__(self, step: str, cause: BaseException) -> None:
        """Initialise the error.

        Args:
            step: The name of the failed step.
            cause: The underlying exception.
        """
        super().__init__(f"Step '{step}' failed: {cause}")
        self.step = step
        self.cause = cause
