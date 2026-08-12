from __future__ import annotations


class StudioError(Exception):
    """Structured error that may safely be returned by the HTTP API."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message
        self.details = details or {}


class RevisionConflict(StudioError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(
            409,
            "REVISION_CONFLICT",
            "The label changed in another browser tab or process.",
            details={"current_revision": int(current_revision)},
        )


class SourceIntegrityError(StudioError):
    def __init__(self, message: str = "The immutable source blob failed verification.") -> None:
        super().__init__(409, "SOURCE_INTEGRITY_ERROR", message)
