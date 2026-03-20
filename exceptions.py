"""Explicit errors so failures are actionable instead of silent empty dicts."""


class PipelineError(Exception):
    """Base class for pipeline failures."""


class ConfigurationError(PipelineError):
    """Missing or invalid configuration (env vars, paths)."""
