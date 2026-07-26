"""Typed failures used across EngVit boundaries."""


class EngVitError(Exception):
    """Base class for expected EngVit failures."""


class ConfigurationError(EngVitError, ValueError):
    """Configuration is valid structurally but violates an application limit."""


class UnsafePathError(EngVitError):
    """A path escapes or aliases an approved root."""


class CanonicalizationError(EngVitError, ValueError):
    """A value cannot be represented by EngVit's canonical JSON contract."""
