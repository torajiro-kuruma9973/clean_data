#!/usr/bin/env python3
"""
Traverse a PSMA/NBIA-style root directory, read all CT DICOM slices, convert
pixels to HU, compute global HU statistics, and save a polished PNG HU
distribution plot.

The directory classifier treats folders containing "Segmentation" as SEG first,
so a segmentation folder whose name also contains "ct" is not mistaken for CT.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pydicom


@dataclass
class HuStats:
    min_hu: float
    max_hu: float
    mean_hu: float
    median_hu: float
    voxel_count: int
    dicom_count: int
    ct_folder_count: int
    median_method: str


def analyze_ct_hu_distribution(
    root_dir: Union[str, Path],
    output_png: Union[str, Path],
    bins: int = 700,
    verbose: bool = False,
) -> HuStats:
    """Compute HU statistics for all CT DICOM files and save a distribution PNG.

    Args:
        root_dir: Root directory containing PSMA project/study folders.
        output_png: PNG file path to save the normalized HU distribution plot.
        bins: Number of histogram bins when plotting non-integer-like HU data.
        verbose: Print skipped files and progress to stderr.

    Returns:
        HuStats with global min, max, mean, median, and counts.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    ct_dirs = list(_find_ct_series_dirs(root))
    if not ct_dirs:
        raise RuntimeError(f"No CT series folders found under: {root}")

    ct_files: List[Path] = []
    for ct_dir in ct_dirs:
        files = _iter_dicom_files(ct_dir)
        if files:
            ct_files.extend(files)
        else:
            _log(verbose, f"[skip] no DICOM files in CT folder: {ct_dir}")

    if not ct_files:
        raise RuntimeError(f"No CT DICOM files found under: {root}")

    first = _first_pass_stats(ct_files, verbose=verbose)
    min_hu, max_hu, total_sum, voxel_count, dicom_count, integer_like = first

    if voxel_count == 0 or dicom_count == 0:
        raise RuntimeError("No readable CT pixel data found.")

    mean_hu = total_sum / voxel_count

    if integer_like and _integer_range_is_reasonable(min_hu, max_hu):
        values, counts = _integer_hu_counts(ct_files, verbose=verbose)
        median_hu = _median_from_counts(values, counts, voxel_count)
        hist_x = values.astype(np.float64)
        hist_y = counts.astype(np.float64)
        median_method = "exact integer HU count"
    else:
        hist_x, hist_y = _histogram_counts(ct_files, min_hu, max_hu, bins=bins, verbose=verbose)
        median_hu = _median_from_histogram(hist_x, hist_y, voxel_count)
        median_method = "histogram approximation"

    stats = HuStats(
        min_hu=float(min_hu),
        max_hu=float(max_hu),
        mean_hu=float(mean_hu),
        median_hu=float(median_hu),
        voxel_count=int(voxel_count),
        dicom_count=int(dicom_count),
        ct_folder_count=len(ct_dirs),
        median_method=median_method,
    )

    _plot_hu_histogram(hist_x, hist_y, stats, output_png)
    return stats


def _find_ct_series_dirs(root: Path) -> Iterable[Path]:
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue

        name = directory.name.lower()
        if "segmentation" in name:
            continue
        if "pet" in name:
            continue
        if "ct" not in name:
            continue

        if _has_direct_dicom_like_files(directory):
            yield directory


def _has_direct_dicom_like_files(directory: Path) -> bool:
    for child in directory.iterdir():
        if child.is_file() and not child.name.startswith("."):
            suffixes = "".join(child.suffixes).lower()
            if suffixes in {".dcm", ""} or child.suffix.lower() == ".dcm":
                return True
    return False


def _iter_dicom_files(directory: Path) -> List[Path]:
    dcm_files = sorted(p for p in directory.rglob("*.dcm") if p.is_file())
    if dcm_files:
        return dcm_files

    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and ".nii" not in "".join(p.suffixes).lower()
    )


def _first_pass_stats(
    files: List[Path],
    verbose: bool,
) -> Tuple[float, float, float, int, int, bool]:
    min_hu = math.inf
    max_hu = -math.inf
    total_sum = 0.0
    voxel_count = 0
    dicom_count = 0
    integer_like = True

    for index, path in enumerate(files, start=1):
        if verbose and index % 500 == 0:
            _log(verbose, f"[progress] first pass {index}/{len(files)}")

        hu = _read_ct_hu(path, verbose=verbose)
        if hu is None:
            continue

        dicom_count += 1
        voxel_count += int(hu.size)
        total_sum += float(np.sum(hu, dtype=np.float64))
        min_hu = min(min_hu, float(np.min(hu)))
        max_hu = max(max_hu, float(np.max(hu)))

        if integer_like and not _array_is_integer_like(hu):
            integer_like = False

    return min_hu, max_hu, total_sum, voxel_count, dicom_count, integer_like


def _integer_hu_counts(files: List[Path], verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    counts_by_hu: Dict[int, int] = {}

    for index, path in enumerate(files, start=1):
        if verbose and index % 500 == 0:
            _log(verbose, f"[progress] integer count pass {index}/{len(files)}")

        hu = _read_ct_hu(path, verbose=verbose)
        if hu is None:
            continue

        hu_int = np.rint(hu).astype(np.int32, copy=False)
        values, counts = np.unique(hu_int, return_counts=True)
        for value, count in zip(values, counts):
            counts_by_hu[int(value)] = counts_by_hu.get(int(value), 0) + int(count)

    sorted_values = np.array(sorted(counts_by_hu), dtype=np.int32)
    sorted_counts = np.array([counts_by_hu[int(value)] for value in sorted_values], dtype=np.int64)
    return sorted_values, sorted_counts


def _histogram_counts(
    files: List[Path],
    min_hu: float,
    max_hu: float,
    bins: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(min_hu, max_hu, bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)

    for index, path in enumerate(files, start=1):
        if verbose and index % 500 == 0:
            _log(verbose, f"[progress] histogram pass {index}/{len(files)}")

        hu = _read_ct_hu(path, verbose=verbose)
        if hu is None:
            continue

        slice_counts, _ = np.histogram(hu, bins=edges)
        counts += slice_counts.astype(np.int64, copy=False)

    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts.astype(np.float64)


def _read_ct_hu(path: Path, verbose: bool) -> Optional[np.ndarray]:
    try:
        ds = pydicom.dcmread(path)
        modality = str(getattr(ds, "Modality", "")).upper()
        if modality and modality != "CT":
            _log(verbose, f"[skip] not CT modality ({modality}): {path}")
            return None

        pixels = np.asarray(ds.pixel_array, dtype=np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        return pixels * slope + intercept
    except Exception as exc:
        _log(verbose, f"[skip] unreadable CT DICOM {path}: {exc}")
        return None


def _array_is_integer_like(values: np.ndarray, atol: float = 1e-3) -> bool:
    sample = values
    if values.size > 200_000:
        sample = values.reshape(-1)[:: max(1, values.size // 200_000)]
    return bool(np.all(np.abs(sample - np.rint(sample)) <= atol))


def _integer_range_is_reasonable(min_hu: float, max_hu: float) -> bool:
    return (math.ceil(max_hu) - math.floor(min_hu)) <= 100_000


def _median_from_counts(values: np.ndarray, counts: np.ndarray, total_count: int) -> float:
    if total_count <= 0:
        raise ValueError("total_count must be positive")

    cumulative = np.cumsum(counts)
    left_rank = (total_count - 1) // 2
    right_rank = total_count // 2
    left_value = values[int(np.searchsorted(cumulative, left_rank + 1, side="left"))]
    right_value = values[int(np.searchsorted(cumulative, right_rank + 1, side="left"))]
    return float(left_value + right_value) / 2.0


def _median_from_histogram(x: np.ndarray, counts: np.ndarray, total_count: int) -> float:
    cumulative = np.cumsum(counts)
    target = total_count / 2.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(x) - 1)
    return float(x[index])


def _plot_hu_histogram(
    x: np.ndarray,
    counts: np.ndarray,
    stats: HuStats,
    output_png: Union[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nonzero = counts > 0
    plot_x = x[nonzero]
    plot_counts = counts[nonzero]
    if plot_x.size == 0:
        raise RuntimeError("Histogram is empty; cannot plot.")

    p_low = _percentile_from_counts(plot_x, plot_counts, 0.5)
    p_high = _percentile_from_counts(plot_x, plot_counts, 99.5)
    if p_high <= p_low:
        p_low, p_high = stats.min_hu, stats.max_hu

    plot_percent = plot_counts / float(stats.voxel_count) * 100.0

    fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=180)
    fig.patch.set_facecolor("#F7F4EE")
    ax.set_facecolor("#FBFAF6")

    width = _bar_width(plot_x)
    ax.bar(
        plot_x,
        plot_percent,
        width=width,
        color="#1F6F8B",
        alpha=0.82,
        edgecolor="#0D3B4C",
        linewidth=0.15,
    )

    ax.axvline(stats.mean_hu, color="#C98A2E", linewidth=2.2, label=f"Mean {stats.mean_hu:,.1f}")
    ax.axvline(stats.median_hu, color="#A94B4B", linewidth=2.2, linestyle="--", label=f"Median {stats.median_hu:,.1f}")

    ax.set_yscale("log")
    ax.set_xlim(p_low, p_high)
    ax.grid(True, axis="y", color="#DDD4C7", linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="x")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#B8AD9E")
    ax.spines["bottom"].set_color("#B8AD9E")

    ax.set_title("Global CT HU Distribution", loc="left", fontsize=22, fontweight="bold", color="#18212B", pad=18)
    ax.text(
        0.0,
        1.015,
        "Bars show the percentage distribution of all CT voxels by HU; display range is clipped to P0.5-P99.5 for readability.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#4C5A66",
    )
    ax.set_xlabel("HU value", fontsize=12, color="#18212B", labelpad=10)
    ax.set_ylabel("Voxel proportion (%)", fontsize=12, color="#18212B", labelpad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_tick))
    ax.tick_params(axis="both", labelsize=10, colors="#36424C")

    stats_text = (
        f"Min HU       {stats.min_hu:,.1f}\n"
        f"Max HU       {stats.max_hu:,.1f}\n"
        f"Median HU    {stats.median_hu:,.1f}\n"
        f"Mean HU      {stats.mean_hu:,.1f}\n"
        f"Voxels       {stats.voxel_count:,}\n"
        f"CT DICOMs    {stats.dicom_count:,}\n"
        f"CT folders   {stats.ct_folder_count:,}\n"
        f"Median       {stats.median_method}"
    )
    ax.text(
        0.985,
        0.965,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#18212B",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.12",
            "facecolor": "#FFFFFF",
            "edgecolor": "#D3C8B8",
            "linewidth": 1.0,
            "alpha": 0.94,
        },
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.935), frameon=False, fontsize=10.5)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _percentile_from_counts(x: np.ndarray, counts: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(counts)
    target = cumulative[-1] * percentile / 100.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(x) - 1)
    return float(x[index])


def _format_percent_tick(value: float, _: object) -> str:
    if value >= 10:
        return f"{value:,.0f}%"
    if value >= 1:
        return f"{value:,.1f}%"
    if value >= 0.01:
        return f"{value:,.2f}%"
    return f"{value:,.3g}%"


def _bar_width(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    diffs = np.diff(np.sort(x))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute global CT HU statistics and save a normalized PNG distribution plot."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders.")
    parser.add_argument("output_png", help="Where to save the HU histogram PNG.")
    parser.add_argument("--bins", type=int, default=700, help="Histogram bins for non-integer HU data.")
    parser.add_argument("--verbose", action="store_true", help="Print progress and skipped files.")
    args = parser.parse_args()

    stats = analyze_ct_hu_distribution(
        args.root_dir,
        args.output_png,
        bins=args.bins,
        verbose=args.verbose,
    )

    print(f"CT folders: {stats.ct_folder_count:,}")
    print(f"CT DICOMs:  {stats.dicom_count:,}")
    print(f"Voxels:     {stats.voxel_count:,}")
    print(f"Min HU:     {stats.min_hu:,.3f}")
    print(f"Max HU:     {stats.max_hu:,.3f}")
    print(f"Median HU:  {stats.median_hu:,.3f} ({stats.median_method})")
    print(f"Mean HU:    {stats.mean_hu:,.3f}")
    print(f"Saved PNG:  {Path(args.output_png)}")


if __name__ == "__main__":
    main()