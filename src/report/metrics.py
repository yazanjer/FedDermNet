"""Load scalar evaluation metrics from ``results/<run>/`` JSON artifacts."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RoundSelection = Literal["last", "best_macro_f1"]

_ROUND_JSON = re.compile(r"^round_\d{3}\.json$")

# Prefix scan avoids multi-megabyte ``summary.json`` parses on federated runs
# that store only a ``history`` array at the top level.
_PREFIX_READ = 65536


def _terminal_summary_hint(summary_path: Path) -> bool | None:
    """Whether root-level metrics likely appear before a ``history`` key.

    ``True`` → load JSON and treat as terminal snapshot when ``macro_f1`` is present.
    ``False`` → defer to ``round_*.json`` when those exist (typical federated layout).
    ``None`` → inconclusive (very unusual layout); caller may fully parse.
    """
    try:
        with open(summary_path, "rb") as f:
            chunk = f.read(_PREFIX_READ).decode("utf-8", errors="ignore")
    except OSError:
        return None
    i_m = chunk.find('"macro_f1"')
    i_h = chunk.find('"history"')
    if i_h != -1 and (i_m == -1 or i_h < i_m):
        return False
    if i_m != -1 and (i_h == -1 or i_m < i_h):
        return True
    return None


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _metric_row(obj: dict[str, Any]) -> bool:
    return "macro_f1" in obj and _is_scalar(obj.get("macro_f1"))


def _load_round_tuples(run_dir: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for f in sorted(run_dir.glob("round_*.json")):
        if not _ROUND_JSON.match(f.name):
            continue
        idx = int(f.stem.split("_")[1])
        with open(f, encoding="utf-8") as fp:
            rows.append((idx, json.load(fp)))
    rows.sort(key=lambda x: x[0])
    return rows


def _candidates_from_summary_dict(data: dict[str, Any]) -> list[tuple[int | None, dict[str, Any], str]]:
    out: list[tuple[int | None, dict[str, Any], str]] = []

    if _metric_row(data) and "history" not in data:
        out.append((data.get("round") if _is_scalar(data.get("round")) else None, data, "summary_flat"))
        return out

    hist = data.get("history")
    if isinstance(hist, list):
        for i, entry in enumerate(hist):
            if not isinstance(entry, dict):
                continue
            if not _metric_row(entry):
                continue
            r = entry.get("round")
            round_idx = int(r) if _is_scalar(r) else i
            out.append((round_idx, entry, "summary_history"))
        if out:
            return out

    if _metric_row(data):
        out.append((data.get("round") if _is_scalar(data.get("round")) else None, data, "summary_flat"))

    return out


@dataclass(frozen=True)
class RunMetrics:
    """One selected evaluation snapshot for a run directory."""

    missing: bool
    round_index: int | None
    source: str
    data: dict[str, Any]

    @staticmethod
    def not_found() -> RunMetrics:
        return RunMetrics(missing=True, round_index=None, source="none", data={})


def load_run_metrics(run_dir: Path | str, selection: RoundSelection = "best_macro_f1") -> RunMetrics:
    """Pick *last* global round or *best_macro_f1* over available snapshots.

    Precedence:
    1. ``summary.json`` with **top-level** ``macro_f1`` (terminal evaluation snapshot; centralized runs).
    2. Else ``round_NNN.json`` files (federated global evaluation per round).
    3. Else ``summary.json`` ``history`` entries (legacy / round files missing).
    """
    rd = Path(run_dir)
    if not rd.is_dir():
        return RunMetrics.not_found()

    summary_path = rd / "summary.json"
    round_tuples = _load_round_tuples(rd)

    if summary_path.exists():
        hint = _terminal_summary_hint(summary_path)
        load_flat = hint is True or (hint is None and not round_tuples)
        if load_flat:
            with open(summary_path, encoding="utf-8") as f:
                summary_root: dict[str, Any] = json.load(f)
            if _metric_row(summary_root):
                r = summary_root.get("round")
                ri = int(r) if _is_scalar(r) else None
                return RunMetrics(missing=False, round_index=ri, source="summary_flat", data=summary_root)

    candidates: list[tuple[int, dict[str, Any], str]] = []

    for idx, blob in round_tuples:
        if _metric_row(blob):
            candidates.append((idx, blob, "round_file"))

    if not candidates and summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)
        for t in _candidates_from_summary_dict(summary_data):
            idx, blob, src = t
            if not _metric_row(blob):
                continue
            ri = idx if idx is not None else -1
            candidates.append((ri, blob, src))

    if not candidates:
        return RunMetrics.not_found()

    def macro_f1(blob: dict[str, Any]) -> float:
        v = blob.get("macro_f1")
        return float(v) if _is_scalar(v) else float("nan")

    if selection == "last":
        idx, blob, src = max(candidates, key=lambda x: x[0])
        return RunMetrics(missing=False, round_index=idx, source=src, data=blob)

    # best_macro_f1 — break ties by later round
    best: tuple[float, int, dict[str, Any], str] | None = None
    for idx, blob, src in candidates:
        f1 = macro_f1(blob)
        if best is None or f1 > best[0] or (f1 == best[0] and idx > best[1]):
            best = (f1, idx, blob, src)

    assert best is not None
    f1, idx, blob, src = best
    if f1 != f1:  # NaN
        warnings.warn(f"No valid macro_f1 in candidates for {rd}; using last round.", stacklevel=2)
        idx2, blob2, src2 = max(candidates, key=lambda x: x[0])
        return RunMetrics(missing=False, round_index=idx2, source=src2, data=blob2)

    return RunMetrics(missing=False, round_index=idx, source=src, data=blob)


def get_scalar(m: RunMetrics, key: str) -> float | None:
    if m.missing:
        return None
    v = m.data.get(key)
    if not _is_scalar(v):
        return None
    return float(v)
