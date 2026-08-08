# Adding an AI Provider

A provider turns `(system_prompt, user_content, schema)` into a validated Pydantic payload plus cost accounting. The whole surface is one required async method.

## The 6 steps

1. **Name it.** Add a member to `config.ProviderName` and a defaults block in `config._BUILTIN_DEFAULTS` (model tiers + pricing). An empty pricing dict means "free provider" — `compute_cost` returns 0 silently. A non-empty dict warns once per unknown model instead of silently undercounting.
2. **Copy `src/reindex/providers/_template.py`** to `src/reindex/providers/<name>.py` and implement `invoke()`. Set `name = "<name>"` matching the enum value.
3. **Map failures** onto `ProviderFailure` with a stable `kind` (`process_exit`, `envelope_parse`, `result_parse`, `schema_violation`, `http_error`, `no_structured_output`) and an honest `retryable` — the runner retries retryable failures once on `config.models.escalation` (set it to `None` in your defaults to disable escalation entirely).
4. **Validate with `validate_or_coerce()`** from `providers.base`. It auto-repairs mechanical shape errors (string-where-array, nulls, extra keys) before failing, so don't pre-clean the payload yourself.
5. **Register** in `providers/__init__.py:_registry()` (one line).
6. **Add a transport mock** to `tests/test_provider_contract.py` `_TRANSPORTS` — the parametrized contract test then covers the valid-payload roundtrip and the garbage-output failure path. The test fails loudly if a registered provider has no mock.

## Capabilities (opt-in, don't fake them)

- `BatchCapable` (second base class): only if the provider has a real server-side batch product — own batch IDs, polling, result retrieval, and the five methods (`run_batches`, `resume_pending`, `batch_status`, `batch_cancel`, `batch_exists`). The runner's invoke path is already a concurrent loop; wrapping it in a fake "batch" buys nothing.
- `supports_file_transcripts = True`: only if `invoke()` can read files in `req.work_dir` through a tool (claude-cli's `--add-dir` + Read). Without it, `prepare_leaf` inlines every transcript — including >200KB ones, which may exceed your provider's context and land in the failure log. That's the honest outcome; don't point the model at files it can't open.

## Selection & config

```
--provider <name>  >  $CSINDEX_PROVIDER  >  reindex.toml [reindex].provider  >  claude-cli
```

Per-provider model tiers and pricing are overridable without code in `reindex.toml` at the export root — see [`reindex.example.toml`](../reindex.example.toml) for the full shape:

```toml
[reindex]
provider = "ollama"

[reindex.providers.ollama]
base_url = "http://localhost:11434"        # lands in config.options

[reindex.providers.ollama.models]
leaf = "llama3.2"
project = "llama3.2"
root = "llama3.3:70b"
```

## Cost accounting

Call `self.config.compute_cost(model, in_tok, out_tok)` with whatever token counts your transport reports (0s are fine — claude-cli gets cost from its envelope instead and passes it directly). The cost log applies no further math; batch discounts are the provider's job.
