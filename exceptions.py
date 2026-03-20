"""Explicit errors so failures are actionable instead of silent empty dicts."""


class PipelineError(Exception):
    """Base class for pipeline failures."""


class ConfigurationError(PipelineError):
    """Missing or invalid configuration (env vars, paths)."""


class ZyteExtractError(PipelineError):
    """Zyte Extract API returned an unusable payload after retries."""


class RegistryAPIError(PipelineError):
    """PR registry response was missing or malformed."""
