class CheckpointError(RuntimeError):
    """Base error for checkpoint persistence and restoration failures."""


class CheckpointNotSupportedError(CheckpointError):
    """Raised when a component does not support checkpointing."""


class CheckpointStateError(CheckpointError):
    """Raised when a component is not at a checkpoint-safe boundary."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the configured session."""


class CheckpointFormatError(CheckpointError):
    """Raised when checkpoint content is corrupt or has an unknown schema."""


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "CheckpointNotSupportedError",
    "CheckpointStateError",
]
