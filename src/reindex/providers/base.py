"""
Provider contract: one required structured-output call, optional batch capability.

A provider turns (system_prompt, user_content, schema) into a validated
Pydantic payload plus cost accounting. Implement `Provider.invoke()` and
register in `providers/__init__.py:_REGISTRY`. Implement `BatchCapable`
ONLY if the provider has a real server-side batch product — the capability
check everywhere is `isinstance(provider, BatchCapable)`, no flags.

See providers/_template.py for the new-provider checklist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from reindex import coerce, log
from reindex.config import ProviderConfig


@dataclass
class InvokeRequest:
    step: str                       # "leaf" | "project" | "root"
    slug: str
    model: str
    system_prompt: str
    user_content: str
    schema_cls: type[BaseModel]
    work_dir: Path | None = None
    allow_filesystem: bool = False  # honored only when supports_file_transcripts


@dataclass
class InvokeResult:
    payload: BaseModel
    cost: float                     # USD; provider computes (envelope or token table)
    turns: int = 1
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderFailure(RuntimeError):
    """Structured failure from any provider.

    Carries the diagnostic context the caller (runner / failure log) needs
    to triage later. `kind` slugs are stable across providers so the
    failure log stays greppable: process_exit, envelope_parse,
    result_parse, schema_violation, http_error, no_structured_output.

    retryable: the provider's own judgment of "would a stronger model
        likely fix this?" — drives the runner's escalation retry. Process-
        level failures are not retryable; schema/parse failures are.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        provider: str = "unknown",
        retryable: bool = False,
        exit_code: int | None = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.retryable = retryable
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout

    def to_context(self) -> dict[str, object]:
        """Serialize to a dict suitable for the failure log."""
        return {
            "kind": self.kind,
            "provider": self.provider,
            "exit_code": self.exit_code,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "message": str(self),
        }


class Provider(ABC):
    """One structured-output call. Everything else is optional."""

    name: ClassVar[str]
    # claude-cli only today: deliver oversized transcripts via the Read
    # tool (--add-dir) instead of inlining. Providers without it always
    # get INLINE transcripts from prepare_leaf.
    supports_file_transcripts: ClassVar[bool] = False

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def model_for(self, step: str, *, size_bytes: int = 0) -> str:
        """Tier lookup from config; replaces workers._pick_leaf_model."""
        return self.config.models.for_step(step, size_bytes=size_bytes)

    async def preflight(self) -> None:
        """Fail fast on missing binaries/keys. Default: nothing to check."""
        return

    @abstractmethod
    async def invoke(self, req: InvokeRequest) -> InvokeResult: ...

    async def aclose(self) -> None:
        return


@dataclass
class BatchStatus:
    """Provider-neutral status row for `csindex batches list/show --live`."""

    status: str                     # provider's own vocabulary ("in_progress", "ended", ...)
    done: bool
    counts: dict[str, int] = field(default_factory=dict)
    expires_at: str = ""


class BatchCapable(ABC):
    """Optional second base class for providers with a real batch product."""

    @abstractmethod
    async def run_batches(
        self,
        *,
        step: str,
        model: str,
        schema_cls: type[BaseModel],
        tasks: list,                # batch.BatchTask
        batch_size: int,
        state: Any,                 # state.BatchState
        finalize: Callable[..., Awaitable[None]],
        serialize_context: Callable[[Any], dict],
        max_in_flight: int = 4,
        wait: bool = False,
    ) -> None: ...

    @abstractmethod
    async def resume_pending(
        self,
        *,
        state: Any,
        schema_for_step: Callable[[str], type[BaseModel]],
        finalize_persisted: Callable[..., Awaitable[None]],
        max_in_flight: int = 4,
    ) -> None: ...

    # Management primitives — the only three things batches_cli needs.
    @abstractmethod
    async def batch_status(self, batch_id: str) -> BatchStatus: ...

    @abstractmethod
    async def batch_cancel(self, batch_id: str) -> str: ...

    @abstractmethod
    async def batch_exists(self, batch_id: str) -> bool: ...

    async def aclose(self) -> None:
        # batches_cli holds providers as BatchCapable; concrete classes also
        # subclass Provider, whose aclose (earlier in the MRO) does the work.
        return


def validate_or_coerce(
    data: Any,
    schema_cls: type[BaseModel],
    *,
    log_bind=None,
    custom_id: str = "",
) -> BaseModel:
    """Validate `data` against `schema_cls`; on failure auto-coerce
    mechanical shape errors (string-where-array, nulls, extra keys) and
    re-validate. Raises the second ValidationError if coercion didn't fix
    it — callers map that onto kind="schema_violation".
    """
    blog = log_bind or log.get("provider")
    primary_err: ValidationError | None = None
    try:
        return schema_cls.model_validate(data)
    except ValidationError as e:
        primary_err = e

    coerced = coerce.coerce_for_model(data, schema_cls)
    try:
        payload = schema_cls.model_validate(coerced)
        blog.info("coerce_success", custom_id=custom_id,
                  original_errors=summarize_validation(primary_err))
        return payload
    except ValidationError as e:
        blog.warn(
            "schema_violation",
            custom_id=custom_id,
            errors=summarize_validation(e),
            tried_coerce=True,
        )
        raise


def summarize_validation(err: ValidationError | None) -> str:
    """Compact one-line summary of a ValidationError for log lines."""
    if err is None:
        return ""
    parts = []
    for e in err.errors()[:5]:
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc}: {e.get('type', '?')}")
    more = len(err.errors()) - 5
    if more > 0:
        parts.append(f"(+{more} more)")
    return "; ".join(parts)
