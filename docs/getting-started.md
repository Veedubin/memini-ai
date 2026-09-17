# Getting started

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- podman or docker for the database container.
- About 2.5 GB of disk: CPU torch, plus a one-time ~2.2 GB BGE-M3 download on first use.

## Install

    uv tool install memini-ai --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    memini-ai db up            # pgvector/pgvector:pg18 on 127.0.0.1:5555
    memini-ai migrate          # optional; serve applies migrations on first connection
    memini-ai warm             # one-time ~2.2 GB BGE-M3 download, then embeds one string

The two `--index` flags pull CPU torch. Without them pip resolves the default CUDA build: a
several-gigabyte download you have no use for unless you set `MEMINI_DEVICE=cuda`.

`warm` is worth running before you wire up a client. The first `remember` or `recall` downloads
BGE-M3 (about 2.2 GB) and can exceed `MEMINI_TIMEOUT_S`, returning `{"error": "timeout after 30s"}`
until the download finishes. `warm` pays that cost once, in the foreground, and prints the model
name and embedding dimension when it is ready.

## Connect a client

    cd ~/Projects/my-repo
    memini-ai init --client claude-code --project my-repo

Restart the client, then call `orient`. Expect `status.db == "ok"`.

## From a source checkout

`init` writes files into the current directory, so run it with `--project`, which keeps the cwd,
and not `--directory`, which changes it into the checkout:

    uv run --project /path/to/memini-ai memini-ai init --client opencode \
        --command "uv run --directory /path/to/memini-ai memini-ai serve"

The `--command` it writes keeps `--directory`: the server does not care about the cwd, and
`--directory` is what makes the client run the checkout's environment.

## First-call latency

After `warm`, the first `remember` or `recall` in a process still loads BGE-M3 from disk, which
takes a few seconds on CPU. After that, embedding one paragraph takes about 50 ms.
