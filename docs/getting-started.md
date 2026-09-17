# Getting started

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- podman or docker for the database container.
- About 2.5 GB of disk for CPU torch plus the BGE-M3 model on first use.

## Install

    uv tool install memini-ai
    memini-ai db up            # pgvector/pgvector:pg18 on 127.0.0.1:5555
    memini-ai migrate          # optional; serve applies migrations on first connection

## Connect a client

    cd ~/Projects/my-repo
    memini-ai init --client claude-code --project my-repo

Restart the client, then call `orient`. Expect `status.db == "ok"`.

## From a source checkout

    memini-ai init --client opencode --command "uv run --directory /path/to/memini-ai memini-ai serve"

## First-call latency

The first `remember` or `recall` loads BGE-M3, which takes a few seconds on CPU. After that,
embedding one paragraph takes about 50 ms.
