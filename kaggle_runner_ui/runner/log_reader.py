import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def find_job_files(output_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Finds and discovers all job log files, summary CSVs, error logs, and result zips.
    Safely auto-extracts zip files and handles corrupted or missing archives.
    """
    if not output_dir:
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}

    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "zips": []}

    try:
        zips = sorted(output_dir.rglob("*.zip"))
        for zip_path in zips:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(zip_path.parent)
            except (zipfile.BadZipFile, OSError, Exception):
                pass

        return {
            "logs": sorted(set(output_dir.rglob("*.log"))),
            "summaries": sorted(set(output_dir.rglob("summary.csv"))),
            "errors": sorted(set(output_dir.rglob("errors.csv"))),
            "prefetch": sorted(set(output_dir.rglob("prefetch_status.jsonl"))),
            "finetune": sorted(set(output_dir.rglob("finetune_status.csv"))),
            "zips": zips,
        }
    except Exception:
        return {"logs": [], "summaries": [], "errors": [], "prefetch": [], "finetune": [], "zips": []}


def read_text_tail(path: Union[str, Path], max_chars: int = 20000, max_lines: Optional[int] = None) -> str:
    """
    Reads the tail of a text or log file safely and efficiently.
    Supports both max_chars and max_lines constraints.
    Handles large files, non-existent files, empty files, and binary content safely.
    """
    if not path:
        return ""

    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""

        file_size = p.stat().st_size
        if file_size == 0:
            return ""

        # Efficient tail reading for large files
        bytes_to_read = max(max_chars * 4, 65536)
        if max_lines and max_lines > 0:
            bytes_to_read = max(bytes_to_read, max_lines * 512)

        if file_size > bytes_to_read:
            with open(p, "rb") as f:
                f.seek(file_size - bytes_to_read)
                raw = f.read()
                # Decode ignoring errors
                text = raw.decode("utf-8", errors="ignore")
                # Drop possible partial line at the start if seek was midway
                first_nl = text.find("\n")
                if first_nl != -1 and first_nl < 100:
                    text = text[first_nl + 1:]
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")

        if max_lines is not None and max_lines > 0:
            lines = text.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            text = "\n".join(lines)

        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            text = text[-max_chars:]

        return text
    except Exception:
        return ""


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Safely reads a CSV file into a pandas DataFrame.
    Returns an empty DataFrame if file is missing, empty, or corrupted.
    """
    if not path:
        return pd.DataFrame()

    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def read_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    """
    Safely reads a JSONL file into a pandas DataFrame.
    Skips invalid / corrupted lines and returns empty DataFrame if file is missing or empty.
    """
    if not path:
        return pd.DataFrame()

    try:
        p = Path(path)
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return pd.DataFrame()

        rows = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    rows.append(json.loads(line_str))
                except Exception:
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def combine_summaries(project_output_dir: Union[str, Path], deduplicate_engine: bool = True) -> pd.DataFrame:
    """
    Combines all summary.csv files under a project directory into a unified DataFrame.
    Deduplicates repeated engines (such as baseline paddleocr evaluated across multiple jobs).
    """
    if not project_output_dir:
        return pd.DataFrame()

    try:
        project_dir = Path(project_output_dir)
        if not project_dir.exists() or not project_dir.is_dir():
            return pd.DataFrame()

        # Unpack any archives first
        find_job_files(project_dir)

        rows = []
        for summary_path in project_dir.rglob("summary.csv"):
            try:
                df = read_csv(summary_path)
                if not df.empty:
                    rel = summary_path.relative_to(project_dir)
                    parts = rel.parts
                    if len(parts) >= 2:
                        job_name = parts[0]
                    else:
                        job_name = summary_path.parent.name
                    df.insert(0, "job", job_name)
                    rows.append(df)
            except Exception:
                pass

        if not rows:
            return pd.DataFrame()

        combined = pd.concat(rows, ignore_index=True)
        if "engine" in combined.columns:
            def _rank_priority(row):
                is_direct = 2 if str(row.get("job", "")).lower() == str(row.get("engine", "")).lower() else 0
                is_ok = 1 if row.get("status") == "ok" else 0
                success = float(row.get("success_pages", 0) or 0)
                return is_direct + is_ok * 0.5 + success * 0.01

            combined["_prio"] = combined.apply(_rank_priority, axis=1)
            combined = combined.sort_values(by=["_prio"], ascending=False).drop_duplicates(subset=["engine"], keep="first")
            combined = combined.drop(columns=["_prio"]).reset_index(drop=True)

        return combined
    except Exception:
        return pd.DataFrame()


def collect_finetune_status(project_output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Finds and collects all finetune_status.csv files under the output directory.
    """
    if not project_output_dir:
        return pd.DataFrame()

    try:
        project_dir = Path(project_output_dir)
        if not project_dir.exists() or not project_dir.is_dir():
            return pd.DataFrame()

        rows = []
        for ft_path in project_dir.rglob("finetune_status.csv"):
            try:
                df = read_csv(ft_path)
                if not df.empty:
                    rel = ft_path.relative_to(project_dir)
                    parts = rel.parts
                    job_name = parts[0] if len(parts) >= 2 else ft_path.parent.name
                    df["job"] = job_name
                    df["source_path"] = str(ft_path)
                    rows.append(df)
            except Exception:
                pass
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_comparison_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Builds head-to-head comparison metrics for Pretrained vs Finetuned PaddleOCR.
    """
    if df.empty or "engine" not in df.columns:
        return []

    base_rows = df[df["engine"] == "paddleocr"]
    ft_rows = df[df["engine"] == "paddleocr_ft"]
    if base_rows.empty and ft_rows.empty:
        return []

    base = base_rows.iloc[0] if not base_rows.empty else {}
    ft = ft_rows.iloc[0] if not ft_rows.empty else {}

    def fmt_num(val, dec=4):
        try:
            if pd.isna(val) or val is None:
                return "N/A"
            return f"{float(val):.{dec}f}"
        except Exception:
            return "N/A"

    def fmt_diff(ft_val, base_val, lower_is_better=True, dec=4):
        try:
            if pd.isna(ft_val) or pd.isna(base_val) or ft_val is None or base_val is None:
                return "Pending / N/A"
            fv, bv = float(ft_val), float(base_val)
            diff = fv - bv
            pct = (diff / bv * 100) if bv != 0 else 0
            sign = "+" if diff > 0 else ""
            better = (diff < 0) if lower_is_better else (diff > 0)
            tag = " [Improved]" if better else (" [Degraded]" if diff != 0 else " [Same]")
            return f"{sign}{diff:.{dec}f} ({sign}{pct:.1f}%){tag}"
        except Exception:
            return "Pending / N/A"

    return [
        {
            "metric": "Character Error Rate (CER)",
            "pretrained": fmt_num(base.get("cer") if isinstance(base, pd.Series) else None),
            "finetuned": fmt_num(ft.get("cer") if isinstance(ft, pd.Series) else None),
            "difference": fmt_diff(ft.get("cer") if isinstance(ft, pd.Series) else None, base.get("cer") if isinstance(base, pd.Series) else None, lower_is_better=True),
        },
        {
            "metric": "Word Error Rate (WER)",
            "pretrained": fmt_num(base.get("wer") if isinstance(base, pd.Series) else None),
            "finetuned": fmt_num(ft.get("wer") if isinstance(ft, pd.Series) else None),
            "difference": fmt_diff(ft.get("wer") if isinstance(ft, pd.Series) else None, base.get("wer") if isinstance(base, pd.Series) else None, lower_is_better=True),
        },
        {
            "metric": "Character F1 Score",
            "pretrained": fmt_num(base.get("char_f1") if isinstance(base, pd.Series) else None),
            "finetuned": fmt_num(ft.get("char_f1") if isinstance(ft, pd.Series) else None),
            "difference": fmt_diff(ft.get("char_f1") if isinstance(ft, pd.Series) else None, base.get("char_f1") if isinstance(base, pd.Series) else None, lower_is_better=False),
        },
        {
            "metric": "Latency per Page (s)",
            "pretrained": fmt_num(base.get("latency_s") if isinstance(base, pd.Series) else None, 2),
            "finetuned": fmt_num(ft.get("latency_s") if isinstance(ft, pd.Series) else None, 2),
            "difference": fmt_diff(ft.get("latency_s") if isinstance(ft, pd.Series) else None, base.get("latency_s") if isinstance(base, pd.Series) else None, lower_is_better=True, dec=2),
        },
        {
            "metric": "Throughput (Chars / Second)",
            "pretrained": fmt_num(base.get("chars_per_second") if isinstance(base, pd.Series) else None, 1),
            "finetuned": fmt_num(ft.get("chars_per_second") if isinstance(ft, pd.Series) else None, 1),
            "difference": fmt_diff(ft.get("chars_per_second") if isinstance(ft, pd.Series) else None, base.get("chars_per_second") if isinstance(base, pd.Series) else None, lower_is_better=False, dec=1),
        },
    ]

