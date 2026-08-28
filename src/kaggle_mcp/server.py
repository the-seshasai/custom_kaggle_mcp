import math
import os
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from kaggle_mcp.kaggle_client import (
    DEFAULT_MAX_DOWNLOAD_MB,
    call_kaggle,
    dataset_cache_dir,
    download_dataset_files,
    ensure_dataset_downloaded,
    get_dataset_details,
    get_kaggle_api,
    kaggle_errors,
    list_dataset_files,
)

mcp = MCPServer("kaggle-mcp")

VALID_SORT_BYS = ("hottest", "votes", "updated", "active", "published")

READABLE_EXTENSIONS = (".csv", ".tsv", ".json")

EDA_MAX_ROWS = int(os.environ.get("KAGGLE_MCP_EDA_MAX_ROWS", "100000"))
EDA_MAX_CORR_COLUMNS = 20


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@mcp.tool()
def search_datasets(query: str, sort_by: str = "hottest", max_results: int = 10) -> dict:
    """Search Kaggle for datasets matching a query.

    Use this to discover datasets on Kaggle before deciding which one to
    inspect further with get_dataset_info or download. Good for open-ended
    questions like "find me a dataset about X".

    Args:
        query: Free-text search term, e.g. "housing prices" or "covid19".
        sort_by: One of "hottest" (default, trending), "votes", "updated",
            "active", or "published".
        max_results: Maximum number of datasets to return (default 10).

    Returns:
        A dict with:
          - "count": number of results returned
          - "results": list of datasets, each with "ref" (owner/dataset-slug,
            pass this to other tools), "title", "size" (human-readable),
            "download_count", "license", "last_updated", and "subtitle".

    Raises:
        Tool error if Kaggle credentials are missing/invalid, or if the
        Kaggle API rate limit is hit.
    """
    if sort_by not in VALID_SORT_BYS:
        raise ToolError(f"sort_by must be one of {VALID_SORT_BYS}, got {sort_by!r}")

    with kaggle_errors():
        api = get_kaggle_api()
        datasets = call_kaggle(
            lambda: api.dataset_list(search=query, sort_by=sort_by, page=1)
        )

    results = []
    for ds in (datasets or [])[:max_results]:
        results.append(
            {
                "ref": ds.ref,
                "title": ds.title,
                "size": _human_size(ds.total_bytes),
                "download_count": ds.download_count,
                "license": ds.license_name,
                "last_updated": str(ds.last_updated) if ds.last_updated else None,
                "subtitle": ds.subtitle,
            }
        )

    return {"count": len(results), "results": results}


@mcp.tool()
def get_dataset_info(dataset_ref: str) -> dict:
    """Get detailed metadata for one specific Kaggle dataset.

    Use this once you know which dataset you want (e.g. from search_datasets
    results) and need details before downloading — size, license, when it was
    last updated, its description, and per-file column schema if Kaggle has
    inferred one.

    Args:
        dataset_ref: Dataset identifier in "owner/dataset-slug" form, e.g.
            "zynicide/wine-reviews". Get this from search_datasets' "ref" field.

    Returns:
        A dict with "ref", "title", "subtitle", "description", "size"
        (human-readable), "size_bytes", "license", "last_updated",
        "download_count", "vote_count", and "files": a list of
        {"name", "size", "columns"} per file in the dataset. "columns" is
        usually an empty list — Kaggle's metadata API rarely infers a schema
        here; use preview_dataset to actually see column names and dtypes.

    Raises:
        Tool error if dataset_ref is malformed, the dataset doesn't exist, or
        Kaggle credentials are missing/invalid/rate-limited.
    """
    with kaggle_errors():
        try:
            ds = get_dataset_details(dataset_ref)
            dataset_files = list_dataset_files(dataset_ref)
        except ValueError as e:
            raise ToolError(str(e)) from e

    files = [
        {
            "name": f.name,
            "size": _human_size(f.total_bytes),
            "columns": [{"name": c.name, "type": c.type} for c in (f.columns or [])],
        }
        for f in dataset_files
    ]

    return {
        "ref": ds.ref,
        "title": ds.title,
        "subtitle": ds.subtitle,
        "description": ds.description,
        "size": _human_size(ds.total_bytes),
        "size_bytes": ds.total_bytes,
        "license": ds.license_name,
        "last_updated": str(ds.last_updated) if ds.last_updated else None,
        "download_count": ds.download_count,
        "vote_count": ds.vote_count,
        "files": files,
    }


@mcp.tool()
def download_dataset(
    dataset_ref: str,
    path: str | None = None,
    max_size_mb: int = DEFAULT_MAX_DOWNLOAD_MB,
    force: bool = False,
) -> dict:
    """Download a Kaggle dataset's files to a local directory.

    Always checks the dataset's total size first and refuses to download if
    it exceeds max_size_mb — use get_dataset_info beforehand if you just want
    to check size without downloading anything.

    Args:
        dataset_ref: "owner/dataset-slug", e.g. "zynicide/wine-reviews".
        path: Local directory to download into (created if it doesn't exist).
            Files are extracted here directly, not left as a zip. Defaults to
            a per-dataset folder under this server's cache directory (see
            KAGGLE_MCP_DATA_DIR) — the same location preview_dataset and
            run_quick_eda use, so files downloaded here are picked up by
            those tools too without re-downloading. Pass an explicit path to
            put files somewhere else instead.
        max_size_mb: Refuse to download if the dataset exceeds this many
            megabytes (default 500, or the KAGGLE_MCP_MAX_DOWNLOAD_MB env var
            if the server was started with it set). Pass a higher value or
            force=True to override for a specific call.
        force: If True, download even though it exceeds max_size_mb.

    Returns:
        A dict with "dataset_ref", "path" (absolute download directory),
        "total_size" (human-readable), and "files": a list of
        {"name", "size", "path"} for each file now on disk.

    Raises:
        Tool error if dataset_ref is malformed or not found, the dataset
        exceeds max_size_mb and force is not set, or Kaggle credentials are
        missing/invalid/rate-limited.
    """
    with kaggle_errors():
        try:
            ds = get_dataset_details(dataset_ref)
            dataset_files = list_dataset_files(dataset_ref)
        except ValueError as e:
            raise ToolError(str(e)) from e

    size_mb = ds.total_bytes / (1024 * 1024)
    if size_mb > max_size_mb and not force:
        raise ToolError(
            f"Dataset '{dataset_ref}' is {_human_size(ds.total_bytes)}, which "
            f"exceeds the {max_size_mb} MB limit for this call. Pass a higher "
            "max_size_mb or force=True to download it anyway."
        )

    dest = Path(path).expanduser().resolve() if path else dataset_cache_dir(dataset_ref)

    with kaggle_errors():
        download_dataset_files(dataset_ref, str(dest))

    files = []
    for f in dataset_files:
        fpath = dest / f.name
        files.append(
            {
                "name": f.name,
                "size": _human_size(fpath.stat().st_size) if fpath.exists() else None,
                "path": str(fpath) if fpath.exists() else None,
            }
        )

    return {
        "dataset_ref": dataset_ref,
        "path": str(dest),
        "total_size": _human_size(ds.total_bytes),
        "files": files,
    }


def _pick_preview_file(dest: Path, file_name: str | None = None) -> Path:
    candidates = [p for p in dest.rglob("*") if p.is_file()]
    if not candidates:
        raise ToolError(f"No files found in {dest} after download.")

    if file_name:
        matches = [p for p in candidates if p.name == file_name]
        if not matches:
            raise ToolError(
                f"File {file_name!r} not found in dataset. Available files: "
                f"{', '.join(sorted(p.name for p in candidates))}"
            )
        return matches[0]

    readable = [p for p in candidates if p.suffix.lower() in READABLE_EXTENSIONS]
    if not readable:
        raise ToolError(
            "No CSV/TSV/JSON file found to preview. Files in dataset: "
            f"{', '.join(sorted(p.name for p in candidates))}"
        )

    def priority(p: Path) -> tuple[int, int]:
        return (READABLE_EXTENSIONS.index(p.suffix.lower()), -p.stat().st_size)

    readable.sort(key=priority)
    return readable[0]


def _read_dataframe(path: Path, n_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=n_rows)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=n_rows)
        if suffix == ".json":
            return pd.read_json(path).head(n_rows)
    except Exception as e:
        raise ToolError(f"Failed to read {path.name} with pandas: {e}") from e
    raise ToolError(f"Unsupported file type: {suffix}")


@mcp.tool()
def preview_dataset(dataset_ref: str, n_rows: int = 10, file_name: str | None = None) -> dict:
    """Preview a Kaggle dataset's schema and first few rows.

    Downloads the dataset to a local cache directory if not already present
    (subject to the same size limit as download_dataset), then reads it with
    pandas. For multi-file datasets, previews the largest CSV by default,
    falling back to TSV then JSON, unless file_name is given.

    Args:
        dataset_ref: "owner/dataset-slug", e.g. "zynicide/wine-reviews".
        n_rows: Number of sample rows to return (default 10).
        file_name: Optional exact file name to preview instead of the
            auto-picked one, e.g. "train.csv". Use get_dataset_info first to
            see available file names.

    Returns:
        A dict with "dataset_ref", "file_previewed", "other_files" (names of
        other files in the dataset, if any), "schema" (list of
        {"column", "dtype"}, dtypes inferred from the sampled rows), and
        "sample" (the first n_rows as a list of row dicts).

    Raises:
        Tool error if the dataset/file can't be found, the file isn't a
        CSV/TSV/JSON pandas can read, downloading it would exceed the
        auto-download size limit, or Kaggle credentials are
        missing/invalid/rate-limited.
    """
    try:
        dest = ensure_dataset_downloaded(dataset_ref)
    except (ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from e

    target = _pick_preview_file(dest, file_name)
    df = _read_dataframe(target, n_rows)

    all_files = sorted(p.name for p in dest.rglob("*") if p.is_file())
    other_files = [f for f in all_files if f != target.name]

    return {
        "dataset_ref": dataset_ref,
        "file_previewed": target.name,
        "other_files": other_files,
        "schema": [{"column": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()],
        "sample": df.head(n_rows).to_dict(orient="records"),
    }


def _clean_nans(obj):
    """Replace float('nan') with None recursively so results are valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


@mcp.tool()
def run_quick_eda(dataset_ref: str) -> dict:
    """Run a quick exploratory data analysis on a Kaggle dataset.

    Downloads the dataset if not already present (subject to the same size
    limit as download_dataset), picks the same file preview_dataset would
    (largest CSV, falling back to TSV then JSON), and summarizes it: null
    counts, dtypes, basic describe() stats, and a correlation matrix for
    numeric columns. This returns a compact summary meant for an LLM context
    window, not a full dataframe dump — use preview_dataset if you need to
    see actual row data.

    Args:
        dataset_ref: "owner/dataset-slug", e.g. "zynicide/wine-reviews".

    Returns:
        A dict with "dataset_ref", "file_analyzed", "rows_analyzed",
        "truncated" (true if the file has more rows than were analyzed — see
        KAGGLE_MCP_EDA_MAX_ROWS), "columns" (list of {"name", "dtype",
        "null_count", "null_pct", "unique_count"} per column), "numeric_describe"
        (pandas describe() stats — count/mean/std/min/25%/50%/75%/max — keyed
        by column, numeric columns only), and "correlation": a
        column-by-column correlation matrix for numeric columns, or null with
        a "correlation_note" explaining why (no numeric columns, or more than
        20 of them — kept concise rather than dumping a huge matrix).

    Raises:
        Tool error if the dataset/file can't be found, the file isn't a
        CSV/TSV/JSON pandas can read, downloading it would exceed the
        auto-download size limit, or Kaggle credentials are
        missing/invalid/rate-limited.
    """
    try:
        dest = ensure_dataset_downloaded(dataset_ref)
    except (ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from e

    target = _pick_preview_file(dest)
    df = _read_dataframe(target, EDA_MAX_ROWS)
    truncated = len(df) >= EDA_MAX_ROWS

    columns = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": round(100 * null_count / len(df), 2) if len(df) else 0.0,
                "unique_count": int(series.nunique()),
            }
        )

    numeric_df = df.select_dtypes(include="number")
    numeric_describe = _clean_nans(numeric_df.describe().round(4).to_dict())

    correlation = None
    correlation_note = None
    if numeric_df.shape[1] == 0:
        correlation_note = "No numeric columns to correlate."
    elif numeric_df.shape[1] > EDA_MAX_CORR_COLUMNS:
        correlation_note = (
            f"Skipped: {numeric_df.shape[1]} numeric columns exceeds the "
            f"{EDA_MAX_CORR_COLUMNS}-column limit for a correlation matrix."
        )
    else:
        correlation = _clean_nans(numeric_df.corr().round(3).to_dict())

    return {
        "dataset_ref": dataset_ref,
        "file_analyzed": target.name,
        "rows_analyzed": len(df),
        "truncated": truncated,
        "columns": columns,
        "numeric_describe": numeric_describe,
        "correlation": correlation,
        "correlation_note": correlation_note,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
