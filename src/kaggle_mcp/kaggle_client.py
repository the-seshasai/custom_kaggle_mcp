"""Shared Kaggle API access for all tools.

IMPORTANT: `import kaggle` has a side effect — kaggle/__init__.py eagerly
constructs an API client and calls `.authenticate()` at import time, which
calls `sys.exit(1)` (after printing a plain-text message to stdout) if no
credentials are found. That is fatal for this server: stdio MCP transport
uses stdout for JSON-RPC framing, so a stray print corrupts the protocol
stream, and sys.exit(1) would kill the whole process on the first tool call
that happens to touch a missing-credentials path.

So credential presence must be verified *before* `kaggle` is ever imported,
using the exact same discovery rules kaggle itself uses (KAGGLE_USERNAME /
KAGGLE_KEY env vars, or a kaggle.json file under KAGGLE_CONFIG_DIR / ~/.kaggle).
`get_kaggle_api()` is the only place in this codebase allowed to import kaggle.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")

DEFAULT_MAX_DOWNLOAD_MB = int(os.environ.get("KAGGLE_MCP_MAX_DOWNLOAD_MB", "500"))


class KaggleCredentialsError(RuntimeError):
    """Raised when Kaggle API credentials cannot be found or are invalid."""


class KaggleRateLimitError(RuntimeError):
    """Raised when the Kaggle API responds with 429 Too Many Requests."""


class KaggleDatasetNotFoundError(RuntimeError):
    """Raised when a dataset ref doesn't resolve: missing, private, or no access.

    Kaggle's API returns a bare 403 PERMISSION_DENIED for all three cases
    (it doesn't distinguish "doesn't exist" from "exists but private", to
    avoid leaking which private datasets exist), so the message can't be
    more specific than that.
    """


def _config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))


def _config_file() -> Path:
    return _config_dir() / "kaggle.json"


def check_kaggle_credentials() -> None:
    """Verify Kaggle credentials are discoverable, without importing `kaggle`.

    Raises:
        KaggleCredentialsError: with actionable setup instructions if neither
            KAGGLE_USERNAME/KAGGLE_KEY env vars nor a kaggle.json file are found.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    config_path = _config_file()
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        if data.get("username") and data.get("key"):
            return

    raise KaggleCredentialsError(
        "Kaggle API credentials not found or incomplete.\n\n"
        "This tool needs one of the following:\n"
        "  1. Environment variables KAGGLE_USERNAME and KAGGLE_KEY, or\n"
        f"  2. A kaggle.json file at {config_path} with 'username' and 'key' "
        "fields (set KAGGLE_CONFIG_DIR to use a different directory).\n\n"
        "To fix this:\n"
        "  1. Go to https://www.kaggle.com/settings/account\n"
        "  2. Under the 'API' section, click 'Create New Token'\n"
        "  3. Save the downloaded kaggle.json to "
        f"{_config_dir()}/ and run: chmod 600 {config_path}"
    )


_api = None


def get_kaggle_api():
    """Return a lazily-constructed, authenticated KaggleApi instance.

    This is the only function in the codebase permitted to `import kaggle`.
    Always call check_kaggle_credentials() first (or rely on this function's
    own internal check) so kaggle's own import-time authenticate()/exit(1)
    side effect never fires against missing credentials.
    """
    global _api
    if _api is not None:
        return _api

    check_kaggle_credentials()

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

    _api = api
    return _api


@contextmanager
def kaggle_errors() -> Iterator[None]:
    """Convert our Kaggle exceptions into an MCP ToolError inside a tool body.

    Usage: `with kaggle_errors(): ...tool logic that calls into kaggle_client...`
    Every tool should wrap its Kaggle-touching code in this so credential and
    rate-limit failures surface to the model as a clean message instead of a
    generic "Error executing tool" crash.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    try:
        yield
    except (KaggleCredentialsError, KaggleRateLimitError, KaggleDatasetNotFoundError) as e:
        raise ToolError(str(e)) from e


def parse_dataset_ref(dataset_ref: str) -> tuple[str, str]:
    """Split "owner/dataset-slug" into (owner_slug, dataset_slug).

    Raises:
        ValueError: if dataset_ref isn't in "owner/dataset-slug" form.
    """
    parts = dataset_ref.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"dataset_ref must be in the form 'owner/dataset-slug', got {dataset_ref!r}"
        )
    return parts[0], parts[1]


def _call_dataset_scoped(dataset_ref: str, fn: Callable[[], T]) -> T:
    """Like call_kaggle, but translates a 403 into KaggleDatasetNotFoundError.

    A bare 403 from a per-dataset endpoint means the ref doesn't exist, is
    private, or isn't shared with you — not a credentials problem (unlike a
    403 in other contexts, which call_kaggle would otherwise blame on auth).
    """
    import requests

    try:
        return call_kaggle(fn)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise KaggleDatasetNotFoundError(
                f"Dataset '{dataset_ref}' was not found, is private, or you don't "
                "have access to it. Double-check the ref (owner/dataset-slug) — "
                "get it from search_datasets' \"ref\" field to be sure it's exact."
            ) from e
        raise


def get_dataset_details(dataset_ref: str):
    """Fetch full dataset metadata (ApiDataset) directly by owner/slug.

    Unlike `dataset_metadata()` on KaggleApi, this doesn't write anything to
    disk — it calls the underlying get_dataset RPC and returns the ApiDataset
    object, which includes files (each with inferred columns), license,
    description, size, and timestamps.
    """
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    owner_slug, dataset_slug = parse_dataset_ref(dataset_ref)
    api = get_kaggle_api()

    def _call():
        with api.build_kaggle_client() as kaggle_client:
            request = ApiGetDatasetRequest()
            request.owner_slug = owner_slug
            request.dataset_slug = dataset_slug
            return kaggle_client.datasets.dataset_api_client.get_dataset(request)

    return _call_dataset_scoped(dataset_ref, _call)


def list_dataset_files(dataset_ref: str) -> list:
    """List files in a dataset (name, size, and columns if Kaggle inferred any).

    In practice Kaggle's metadata API rarely populates column info here even
    for tabular files — that's why preview_dataset/run_quick_eda download and
    read files with pandas instead of relying on this.
    """
    api = get_kaggle_api()
    resp = _call_dataset_scoped(dataset_ref, lambda: api.dataset_list_files(dataset_ref))
    return resp.dataset_files or []


def default_data_dir() -> Path:
    """Base directory for cached dataset downloads.

    Deliberately NOT relative to the current working directory: an MCP
    client (Claude Desktop, Claude Code) launches this server with whatever
    cwd it happens to use — often unrelated to this project, and not
    something you control per-call — so a "./data"-style relative default
    would put files somewhere different, and hard to find, every time the
    client's launch cwd changes. Defaults to a fixed per-user location,
    overridable with KAGGLE_MCP_DATA_DIR for anyone who wants control over it.
    """
    configured = os.environ.get("KAGGLE_MCP_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "kaggle-mcp" / "data"


def dataset_cache_dir(dataset_ref: str, base: str | Path | None = None) -> Path:
    """Where a dataset's files are cached locally (base/owner/slug).

    One directory per dataset so multiple datasets don't collide, and so a
    repeat call can detect files are already there and skip re-downloading.
    `base` defaults to default_data_dir(); download_dataset and the
    implicit downloads in preview_dataset/run_quick_eda all resolve to the
    same place by default, so files fetched one way are visible the other.
    """
    owner_slug, dataset_slug = parse_dataset_ref(dataset_ref)
    root = Path(base).expanduser().resolve() if base is not None else default_data_dir()
    return root / owner_slug / dataset_slug


def download_dataset_files(dataset_ref: str, dest_path: str) -> None:
    """Download and unzip all files for a dataset into dest_path.

    Creates dest_path if it doesn't exist. Skips re-downloading if the
    destination zip already matches what's on Kaggle (kaggle's own
    download_needed() check, via the underlying dataset_download_files call).
    Caller is responsible for checking size against a threshold first.
    """
    api = get_kaggle_api()
    _call_dataset_scoped(
        dataset_ref,
        lambda: api.dataset_download_files(dataset_ref, path=dest_path, unzip=True, quiet=True),
    )


def ensure_dataset_downloaded(
    dataset_ref: str, base: str | Path | None = None, max_size_mb: int = DEFAULT_MAX_DOWNLOAD_MB
) -> Path:
    """Download a dataset into its cache dir if not already there; return that dir.

    Used by tools that need dataset files on disk but weren't explicitly
    asked to download (preview_dataset, run_quick_eda). Applies the same
    size guard as the download_dataset tool so an implicit download can't
    silently pull down something huge.

    Raises:
        KaggleDatasetNotFoundError / KaggleCredentialsError / KaggleRateLimitError
        RuntimeError: if the dataset exceeds max_size_mb.
    """
    dest = dataset_cache_dir(dataset_ref, base)
    if dest.exists() and any(p.is_file() for p in dest.rglob("*")):
        return dest

    ds = get_dataset_details(dataset_ref)
    size_mb = ds.total_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise RuntimeError(
            f"Dataset '{dataset_ref}' is {size_mb:.1f} MB, which exceeds the "
            f"{max_size_mb} MB auto-download limit. Call download_dataset directly "
            "with a higher max_size_mb or force=True first, then retry this call."
        )

    download_dataset_files(dataset_ref, str(dest))
    return dest


def call_kaggle(fn: Callable[[], T]) -> T:
    """Run a Kaggle API call, translating transport errors into clean messages.

    Wrap every Kaggle API call site with this (e.g. `call_kaggle(lambda: api.dataset_list(...))`)
    so 401s (bad/expired credentials) and 429s (rate limiting) surface as
    actionable errors instead of a raw requests traceback.

    Also redirects stdout for the duration of the call: the `kaggle` package
    has several direct `print()` calls buried in it (an outdated-version nag
    on API responses, a confirmation() prompt helper, etc.) that would corrupt
    the stdio JSON-RPC stream if they ever fired. None of that output is
    useful to us, so it's discarded rather than logged.
    """
    import contextlib
    import io

    import requests

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 429:
            raise KaggleRateLimitError(
                "Kaggle API rate limit exceeded (HTTP 429). Wait a bit before "
                "retrying, or reduce how many requests you're making in a short window."
            ) from e
        if status == 401:
            raise KaggleCredentialsError(
                f"Kaggle API rejected the request (HTTP {status}) — your credentials "
                "may be invalid or expired. Regenerate a token at "
                "https://www.kaggle.com/settings/account (API section) and replace "
                f"{_config_file()}."
            ) from e
        raise
