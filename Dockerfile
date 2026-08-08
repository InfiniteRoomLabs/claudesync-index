FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS build
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md LICENSE ./
RUN uv sync --frozen --no-dev && uv build --wheel
# Export the locked, non-dev dependency set so the final stage installs the
# exact versions uv.lock pinned instead of letting pip/uv re-resolve the
# wheel's loose PEP 440 ranges against whatever is newest on the index at
# build time.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt

FROM python:3.13-slim-trixie
RUN useradd --create-home --uid 1000 csindex
# Reuse the build stage's uv binary to install with the frozen lockfile
# instead of invoking pip; --no-deps on the wheel keeps this from
# re-resolving its own (range) dependency specifiers against the index --
# only requirements.txt's exact pins are ever installed.
COPY --from=build /usr/local/bin/uv /usr/local/bin/uv
COPY --from=build /app/dist/*.whl /app/requirements.txt /tmp/
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt --no-deps /tmp/*.whl \
    && rm -rf /tmp/*.whl /tmp/requirements.txt \
    && rm /usr/local/bin/uv
USER csindex
ENV CSINDEX_PROVIDER=anthropic
ENTRYPOINT ["csindex"]
CMD ["--help"]
