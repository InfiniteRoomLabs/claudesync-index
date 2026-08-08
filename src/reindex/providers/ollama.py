"""
Ollama provider: local httpx POST to /api/chat (no subscription/API key).

Targets Ollama >= 0.5 structured-output support: the `format` field accepts a
full JSON Schema object, which the server grammar-samples at generation time.
The model still occasionally produces valid JSON that doesn't match the schema,
so we validate_or_coerce and retry once with a reminder message before giving up.

Transport is a lazily-created shared httpx.AsyncClient (call `_get_client()` to
access it). That function is the patchable seam for tests: patch
`reindex.providers.ollama._get_client` to return a mock AsyncClient.

Default target: http://127.0.0.1:11434 (a local Ollama server). Override via
`options.base_url` in reindex.toml or BUILTIN_DEFAULTS.
"""

from __future__ import annotations

import json
import time

import httpx
from pydantic import ValidationError

from reindex import log
from reindex.batch import sanitize_user_content
from reindex.providers.base import (
    InvokeRequest,
    InvokeResult,
    Provider,
    ProviderFailure,
    validate_or_coerce,
)

# Local models are slow — quantized 8B+ can take 30-60s per generation on
# CPU, and a shared/remote GPU host may queue requests. 600s is generous
# but avoids silent hangs.
_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# Cap response body snippets in error messages; Ollama error bodies are
# usually short JSON, but a proxy or nginx error page could be larger.
_BODY_CAP = 4096

_shared_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return (and lazily create) the shared AsyncClient.

    Module-level function so tests can patch it cleanly:
        monkeypatch.setattr("reindex.providers.ollama._get_client", lambda: mock_client)
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _shared_client


class OllamaProvider(Provider):
    name = "ollama"
    # Ollama has no Read-tool filesystem loop — transcripts are always inlined.
    supports_file_transcripts = False

    async def preflight(self) -> None:
        """Verify the Ollama server is reachable before the pipeline starts."""
        base_url = self.config.options.get("base_url", "http://127.0.0.1:11434")
        try:
            resp = await _get_client().get(
                f"{base_url}/api/tags",
                timeout=httpx.Timeout(10.0),
            )
            resp.raise_for_status()
        except Exception as exc:
            raise ProviderFailure(
                f"Ollama not reachable at {base_url}: {exc!s}. "
                "Start the Ollama server or set [providers.ollama] base_url in reindex.toml.",
                kind="http_error",
                provider=self.name,
            ) from exc

    async def invoke(self, req: InvokeRequest) -> InvokeResult:
        base_url = self.config.options.get("base_url", "http://127.0.0.1:11434")
        blog = log.get("provider.ollama").bind(step=req.step, slug=req.slug)

        # Wrap transcript in <conversation> tags and sanitize XML-marker
        # characters that could confuse the model — same framing the CLI and
        # batch providers use.
        safe_content = (
            "<conversation>\n"
            + sanitize_user_content(req.user_content)
            + "\n</conversation>"
        )

        body = {
            "model": req.model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": safe_content},
            ],
            # Ollama >= 0.5: full JSON Schema in `format` enables structured output.
            "format": req.schema_cls.model_json_schema(),
            "stream": False,
        }

        t0 = time.monotonic()
        raw_content, in_tok, out_tok = await self._post_chat(base_url, body, blog)
        duration_ms = int((time.monotonic() - t0) * 1000)

        # One in-provider retry if the model returned non-JSON prose. Ollama
        # structured output is best-effort — grammar sampling occasionally
        # loses and produces an explanation instead of JSON. A reminder message
        # is cheap and fixes most of these without burning an escalation slot.
        if raw_content is None:
            body["messages"].append({
                "role": "assistant",
                "content": "(previous response was not valid JSON)",
            })
            body["messages"].append({
                "role": "user",
                "content": (
                    "Respond with ONLY a JSON object conforming to the schema, "
                    "no prose, no markdown fences."
                ),
            })
            blog.info("ollama_retry_non_json", slug=req.slug)
            t1 = time.monotonic()
            raw_content, in_tok2, out_tok2 = await self._post_chat(base_url, body, blog)
            duration_ms += int((time.monotonic() - t1) * 1000)
            in_tok += in_tok2
            out_tok += out_tok2
            if raw_content is None:
                raise ProviderFailure(
                    "Ollama response is not JSON after retry",
                    kind="result_parse",
                    provider=self.name,
                    retryable=True,
                )

        # validate_or_coerce handles string-where-array and similar shape
        # errors silently (coerce_success log line). If it still fails, one
        # in-provider retry with a reminder — same pattern as the prose path.
        try:
            payload = validate_or_coerce(raw_content, req.schema_cls, log_bind=blog, custom_id=req.slug)
        except ValidationError as e:
            body["messages"].append({
                "role": "assistant",
                "content": json.dumps(raw_content),
            })
            body["messages"].append({
                "role": "user",
                "content": (
                    "Respond with ONLY a JSON object conforming to the schema, "
                    "no prose, no markdown fences."
                ),
            })
            blog.info("ollama_retry_schema_violation", slug=req.slug)
            t2 = time.monotonic()
            raw_content2, in_tok3, out_tok3 = await self._post_chat(base_url, body, blog)
            duration_ms += int((time.monotonic() - t2) * 1000)
            in_tok += in_tok3
            out_tok += out_tok3
            if raw_content2 is None:
                raise ProviderFailure(
                    "Ollama retry after schema violation returned non-JSON",
                    kind="schema_violation",
                    provider=self.name,
                    retryable=True,
                ) from e
            try:
                payload = validate_or_coerce(raw_content2, req.schema_cls, log_bind=blog, custom_id=req.slug)
            except ValidationError as e:
                raise ProviderFailure(
                    f"schema violation after retry: {e}",
                    kind="schema_violation",
                    provider=self.name,
                    retryable=True,
                ) from e

        return InvokeResult(
            payload=payload,
            cost=self.config.compute_cost(req.model, in_tok, out_tok),
            duration_ms=duration_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    async def _post_chat(
        self,
        base_url: str,
        body: dict,
        blog,
    ) -> tuple[dict | None, int, int]:
        """POST to /api/chat. Returns (parsed_content_dict | None, in_tok, out_tok).

        Returns None for content when the response body is not JSON or when
        `message.content` is not a JSON object — caller handles retry logic.

        Raises ProviderFailure(kind="http_error") on network or HTTP errors;
        retryable=False because an HTTP 500 / connection failure won't be
        fixed by a stronger model.
        """
        try:
            resp = await _get_client().post(
                f"{base_url}/api/chat",
                json=body,
            )
        except Exception as exc:
            raise ProviderFailure(
                f"Ollama HTTP request failed: {exc!s}",
                kind="http_error",
                provider=self.name,
                retryable=False,
            ) from exc

        if resp.status_code >= 400:
            snippet = resp.text[:_BODY_CAP]
            blog.error("ollama_http_error", status=resp.status_code, body=snippet[:300])
            raise ProviderFailure(
                f"Ollama returned HTTP {resp.status_code}",
                kind="http_error",
                provider=self.name,
                retryable=False,
                stdout=snippet,
            )

        try:
            data = resp.json()
        except Exception as exc:
            snippet = resp.text[:_BODY_CAP]
            blog.error("ollama_response_not_json", body=snippet[:300])
            raise ProviderFailure(
                f"Ollama response body is not JSON: {exc!s}",
                kind="http_error",
                provider=self.name,
                retryable=False,
                stdout=snippet,
            ) from exc

        # Ollama non-stream: {"message": {"role": "assistant", "content": "..."}, ...}
        in_tok = int(data.get("prompt_eval_count", 0))
        out_tok = int(data.get("eval_count", 0))
        content_str = data.get("message", {}).get("content", "")

        try:
            parsed = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            blog.warning("ollama_content_not_json", preview=str(content_str)[:200])
            return None, in_tok, out_tok

        if not isinstance(parsed, dict):
            blog.warning("ollama_content_not_object", type=type(parsed).__name__)
            return None, in_tok, out_tok

        return parsed, in_tok, out_tok

    async def aclose(self) -> None:
        global _shared_client
        if _shared_client is not None:
            await _shared_client.aclose()
            _shared_client = None
