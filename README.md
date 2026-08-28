# kaggle-mcp

An MCP (Model Context Protocol) server that exposes Kaggle dataset search,
inspection, download, and quick EDA as tools for LLM clients (Claude Desktop,
Claude Code, and any other MCP-compatible client). Runs over stdio — the
client launches it as a subprocess, there's nothing to keep running yourself.

## Tools

| Tool | What it does |
|---|---|
| `search_datasets(query, sort_by="hottest", max_results=10)` | Search Kaggle for datasets matching a query. |
| `get_dataset_info(dataset_ref)` | Get metadata for one dataset — size, license, description, file list. |
| `download_dataset(dataset_ref, path=None, max_size_mb=500, force=False)` | Download a dataset's files locally. Refuses above `max_size_mb` unless `force=True`. |
| `preview_dataset(dataset_ref, n_rows=10, file_name=None)` | Download if needed, read with pandas, return schema + sample rows. |
| `run_quick_eda(dataset_ref)` | Null counts, dtypes, `describe()` stats, and a correlation matrix (numeric columns, capped at 20). |

`dataset_ref` is always Kaggle's `owner/dataset-slug` form, e.g.
`"zynicide/wine-reviews"` — get it from `search_datasets`' `"ref"` field.

## Setup

### 1. Get a Kaggle API key

1. Go to [kaggle.com/settings/account](https://www.kaggle.com/settings/account).
2. Under **API**, click **Create New Token**. This downloads `kaggle.json`.
3. Save it to `~/.kaggle/kaggle.json` and lock down its permissions:
   ```
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

Alternatively, skip the file and set `KAGGLE_USERNAME` and `KAGGLE_KEY`
environment variables directly (useful for containers/CI). A custom
`kaggle.json` location can be set via `KAGGLE_CONFIG_DIR`.

If neither is found, every tool call returns a clear error explaining exactly
what's missing and how to fix it — no stack traces, no silent failures.

### 2. Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.10+.

```
git clone <this-repo>
cd kaggle_mcp
uv sync
```

This creates a local `.venv` and installs everything pinned in `uv.lock`
(including `kaggle<2`, deliberately — see "Why `kaggle<2`" below).

### 3. Try it locally (optional but recommended)

```
uv run mcp dev src/kaggle_mcp/server.py
```

Opens the MCP Inspector in your browser. Connect, and you should see all 5
tools listed. Try `search_datasets` with `query: "titanic"` to confirm your
credentials work end-to-end.

### 4. Add it to Claude Desktop

Edit Claude Desktop's config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/kaggle_mcp",
        "kaggle-mcp"
      ]
    }
  }
}
```

Use the absolute path to wherever you cloned this repo. If your Kaggle
credentials aren't in the default location, add an `"env"` block here too —
Claude Desktop launches the server without your shell's environment, so
anything you rely on being "already exported" needs to be set explicitly:

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/kaggle_mcp", "kaggle-mcp"],
      "env": {
        "KAGGLE_USERNAME": "your-username",
        "KAGGLE_KEY": "your-key"
      }
    }
  }
}
```

Restart Claude Desktop. You should see "kaggle" connected with 5 tools
available.

### Adding it to Claude Code instead

```
claude mcp add kaggle -- uv run --project /absolute/path/to/kaggle_mcp kaggle-mcp
```

Then start a new Claude Code session — the tools will be available there.

You never need to manually start or keep the server running: whichever
client you configure spawns its own subprocess on startup and kills it when
it disconnects.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | — | Kaggle credentials (alternative to `kaggle.json`). |
| `KAGGLE_CONFIG_DIR` | `~/.kaggle` | Where to look for `kaggle.json`. |
| `KAGGLE_MCP_DATA_DIR` | `~/.cache/kaggle-mcp/data` | Where datasets are cached (one folder per `owner/slug`). `download_dataset`'s default, and where `preview_dataset`/`run_quick_eda` look for/save files, all share this — pass an explicit `path` to `download_dataset` to store elsewhere for a specific call. |
| `KAGGLE_MCP_MAX_DOWNLOAD_MB` | `500` | Default size threshold above which `download_dataset` refuses (per-call `max_size_mb`/`force` override this). |
| `KAGGLE_MCP_EDA_MAX_ROWS` | `100000` | Row cap for `run_quick_eda` — larger files are analyzed on a truncated sample (reported via `"truncated": true`), not read in full, to keep memory/latency bounded. |

## Notes on error handling

Every tool distinguishes:
- **Missing/invalid credentials** — actionable setup instructions.
- **Rate limiting (HTTP 429)** — a clear "wait and retry" message.
- **Dataset not found / private / no access (HTTP 403)** — Kaggle returns the
  same 403 for all three cases (it won't reveal which datasets exist
  privately), so the message says exactly that rather than incorrectly
  blaming your credentials.
- **Size-limit refusals** — tells you the actual size, the limit that was
  hit, and how to override it.

All of these come back as proper MCP tool errors (`is_error: true` with a
readable message) — never a raw Python traceback, and never a silent no-op.

## Why `kaggle<2`

The `kaggle` PyPI package's 2.x line is a recent OAuth-first rewrite that
prints diagnostic text directly to stdout in several places (including at
`import` time when credentials are missing). Since this server talks
JSON-RPC over stdout, any stray print would corrupt the protocol stream. The
1.x line (`kaggle.json`-based, matching Kaggle's classic documented API) has
the same problem in a couple of spots — an outdated-version nag, an
auth-failure message — so `kaggle_client.py` redirects stdout around every
call into the library, on top of pinning to 1.x. If you ever bump this
dependency, re-verify that redirect still covers whatever the newer version
prints.
