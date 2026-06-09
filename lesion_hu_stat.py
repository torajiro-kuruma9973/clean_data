#!/usr/bin/env python3
"""
Map DICOM SEG foreground voxels to same-study CT DICOM slices and export per-CT
slice HU statistics.

Expected PSMA/NBIA-style structure:
  root/
    PSMA_xxx/
      study-id/
        *PET*/
        *Segmentation*/
        *CT*/

The SEG mask is interpreted in patient physical coordinates using its DICOM
functional group geometry. Each foreground voxel is mapped to the nearest CT
slice and nearest CT row/column using CT DICOM geometry, then converted to HU by
RescaleSlope/RescaleIntercept.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom


@dataclass
class ImagePlane:
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    row_spacing: float
    col_spacing: float


@dataclass
class CtSlice:
    path: Path
    filename: str
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    row_spacing: float
    col_spacing: float
    rows: int
    cols: int
    slope: float
    intercept: float
    slice_position: float


@dataclass
class CtSeries:
    directory: Path
    slices: List[CtSlice]
    positions: np.ndarray
    median_spacing: float
    slice_tolerance_mm: float


@dataclass
class SliceAccumulator:
    hu_values: List[float]
    source_mask_files: set
    mapped_points: int = 0
    skipped_out_of_bounds: int = 0
    skipped_no_ct_slice: int = 0


def export_seg_foreground_ct_hu_stats(
    root_dir: Union[str, Path],
    output_csv: Union[str, Path],
    slice_tolerance_mm: Optional[float] = None,
    normal_dot_threshold: float = 0.95,
    verbose: bool = False,
) -> int:
    """Export HU statistics for SEG foreground points mapped onto CT slices.

    Args:
        root_dir: Root directory containing PSMA project folders.
        output_csv: CSV path to write per-study/per-CT-slice HU statistics.
        slice_tolerance_mm: Max distance from a foreground point to the nearest
            CT slice plane. If None, use about half the CT slice spacing.
        normal_dot_threshold: Minimum abs(dot(SEG normal, CT normal)) accepted
            before a frame is considered geometrically inconsistent.
        verbose: Print progress and skip diagnostics to stderr.

    Returns:
        Number of output CSV rows written.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seg_dir in _find_segmentation_dirs(root):
        study_dir = seg_dir.parent
        project_id = study_dir.parent.name
        study_id = study_dir.name

        ct_dir = _find_ct_dir(study_dir)
        if ct_dir is None:
            _log(verbose, f"[skip] CT folder not found for study: {study_dir}")
            continue

        try:
            ct_series = _build_ct_series(ct_dir, slice_tolerance_mm=slice_tolerance_mm, verbose=verbose)
        except Exception as exc:
            _log(verbose, f"[skip] failed to build CT series for {study_dir}: {exc}")
            continue

        study_rows = _process_study_segmentation(
            project_id=project_id,
            study_id=study_id,
            seg_dir=seg_dir,
            ct_series=ct_series,
            normal_dot_threshold=normal_dot_threshold,
            verbose=verbose,
        )
        rows.extend(study_rows)

    fieldnames = [
        "Project_ID",
        "Study_ID",
        "CT_Series",
        "CT_Filename",
        "CT_SlicePosition_mm",
        "Foreground_Point_Count",
        "Min_HU",
        "Max_HU",
        "Mean_HU",
        "Median_HU",
        "Mask_Filenames",
        "Skipped_No_CT_Slice",
        "Skipped_Out_Of_Bounds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _find_segmentation_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir() and "segmentation" in path.name.lower():
            yield path


def _find_ct_dir(study_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for child in study_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "segmentation" in name or "pet" in name:
            continue
        if "ct" in name:
            candidates.append(child)
    return sorted(candidates)[0] if candidates else None


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


def _build_ct_series(
    ct_dir: Path,
    slice_tolerance_mm: Optional[float],
    verbose: bool,
) -> CtSeries:
    slices: List[CtSlice] = []
    for path in _iter_dicom_files(ct_dir):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", "")).upper()
            if modality and modality != "CT":
                _log(verbose, f"[skip] not CT modality ({modality}): {path}")
                continue
            plane = _root_image_plane(ds)
            rows = int(ds.Rows)
            cols = int(ds.Columns)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            slice_position = float(np.dot(plane.ipp, plane.normal))
            slices.append(
                CtSlice(
                    path=path,
                    filename=path.name,
                    ipp=plane.ipp,
                    row_dir=plane.row_dir,
                    col_dir=plane.col_dir,
                    normal=plane.normal,
                    row_spacing=plane.row_spacing,
                    col_spacing=plane.col_spacing,
                    rows=rows,
                    cols=cols,
                    slope=slope,
                    intercept=intercept,
                    slice_position=slice_position,
                )
            )
        except Exception as exc:
            _log(verbose, f"[skip] unreadable CT header {path}: {exc}")

    if not slices:
        raise RuntimeError(f"No readable CT slices in {ct_dir}")

    slices.sort(key=lambda item: item.slice_position)
    positions = np.asarray([item.slice_position for item in slices], dtype=np.float64)
    median_spacing = _median_positive_spacing(positions)
    if slice_tolerance_mm is None:
        tolerance = max(0.5, 0.55 * median_spacing)
    else:
        tolerance = float(slice_tolerance_mm)

    return CtSeries(
        directory=ct_dir,
        slices=slices,
        positions=positions,
        median_spacing=median_spacing,
        slice_tolerance_mm=tolerance,
    )


def _process_study_segmentation(
    project_id: str,
    study_id: str,
    seg_dir: Path,
    ct_series: CtSeries,
    normal_dot_threshold: float,
    verbose: bool,
) -> List[Dict[str, object]]:
    accumulators: Dict[int, SliceAccumulator] = {}
    ct_pixel_cache: Dict[int, np.ndarray] = {}

    for mask_path in _iter_dicom_files(seg_dir):
        try:
            ds = pydicom.dcmread(mask_path)
            mask = _mask_pixel_array_3d(ds)
        except Exception as exc:
            _log(verbose, f"[skip] unreadable SEG mask {mask_path}: {exc}")
            continue

        frame_count = mask.shape[0]
        for frame_idx in range(frame_count):
            frame = mask[frame_idx]
            foreground_rows, foreground_cols = np.nonzero(frame > 0)
            if foreground_rows.size == 0:
                continue

            try:
                seg_plane = _seg_frame_plane(ds, frame_idx)
            except Exception as exc:
                _log(verbose, f"[skip] SEG frame geometry missing {mask_path} frame {frame_idx + 1}: {exc}")
                continue

            normal_dot = abs(float(np.dot(seg_plane.normal, ct_series.slices[0].normal)))
            if normal_dot < normal_dot_threshold:
                _log(
                    verbose,
                    f"[skip] SEG/CT normals inconsistent dot={normal_dot:.4f}: "
                    f"{mask_path} frame {frame_idx + 1}",
                )
                continue

            points = _mask_pixels_to_patient_points(seg_plane, foreground_rows, foreground_cols)
            slice_indices, matched = _nearest_ct_slice_indices(points, ct_series)

            if np.any(~matched):
                _add_skipped_no_ct_slice(accumulators, int(np.count_nonzero(~matched)))

            if not np.any(matched):
                continue

            matched_points = points[matched]
            matched_indices = slice_indices[matched]

            for ct_idx in np.unique(matched_indices):
                ct_idx_int = int(ct_idx)
                point_subset = matched_points[matched_indices == ct_idx_int]
                ct_slice = ct_series.slices[ct_idx_int]

                row_idx, col_idx, in_bounds = _patient_points_to_ct_indices(point_subset, ct_slice)
                accumulator = accumulators.setdefault(
                    ct_idx_int,
                    SliceAccumulator(hu_values=[], source_mask_files=set()),
                )
                accumulator.source_mask_files.add(mask_path.name)

                out_of_bounds_count = int(np.count_nonzero(~in_bounds))
                accumulator.skipped_out_of_bounds += out_of_bounds_count

                if not np.any(in_bounds):
                    continue

                pixels = ct_pixel_cache.get(ct_idx_int)
                if pixels is None:
                    pixels = _read_ct_pixels(ct_slice.path)
                    ct_pixel_cache[ct_idx_int] = pixels

                valid_rows = row_idx[in_bounds]
                valid_cols = col_idx[in_bounds]
                raw_values = pixels[valid_rows, valid_cols].astype(np.float64, copy=False)
                hu_values = raw_values * ct_slice.slope + ct_slice.intercept

                accumulator.hu_values.extend(float(value) for value in hu_values)
                accumulator.mapped_points += int(hu_values.size)

    output_rows: List[Dict[str, object]] = []
    global_skipped_no_ct_slice = accumulators.pop(-1, None)
    skipped_no_ct_slice = global_skipped_no_ct_slice.skipped_no_ct_slice if global_skipped_no_ct_slice else 0

    for ct_idx in sorted(accumulators):
        accumulator = accumulators[ct_idx]
        if not accumulator.hu_values:
            continue
        ct_slice = ct_series.slices[ct_idx]
        values = np.asarray(accumulator.hu_values, dtype=np.float64)
        output_rows.append(
            {
                "Project_ID": project_id,
                "Study_ID": study_id,
                "CT_Series": ct_series.directory.name,
                "CT_Filename": ct_slice.filename,
                "CT_SlicePosition_mm": f"{ct_slice.slice_position:.6f}",
                "Foreground_Point_Count": int(values.size),
                "Min_HU": f"{float(np.min(values)):.6f}",
                "Max_HU": f"{float(np.max(values)):.6f}",
                "Mean_HU": f"{float(np.mean(values)):.6f}",
                "Median_HU": f"{float(np.median(values)):.6f}",
                "Mask_Filenames": ";".join(sorted(accumulator.source_mask_files)),
                "Skipped_No_CT_Slice": skipped_no_ct_slice,
                "Skipped_Out_Of_Bounds": accumulator.skipped_out_of_bounds,
            }
        )

    return output_rows


def _mask_pixel_array_3d(ds: pydicom.dataset.Dataset) -> np.ndarray:
    arr = np.asarray(ds.pixel_array)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"expected 2D/3D SEG pixel array, got shape {arr.shape}")
    return arr


def _mask_pixels_to_patient_points(
    plane: ImagePlane,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    row_offsets = rows.astype(np.float64, copy=False) * plane.row_spacing
    col_offsets = cols.astype(np.float64, copy=False) * plane.col_spacing
    return (
        plane.ipp[np.newaxis, :]
        + col_offsets[:, np.newaxis] * plane.row_dir[np.newaxis, :]
        + row_offsets[:, np.newaxis] * plane.col_dir[np.newaxis, :]
    )


def _nearest_ct_slice_indices(points: np.ndarray, ct_series: CtSeries) -> Tuple[np.ndarray, np.ndarray]:
    normal = ct_series.slices[0].normal
    point_positions = points @ normal
    insertion = np.searchsorted(ct_series.positions, point_positions)

    left = np.clip(insertion - 1, 0, len(ct_series.positions) - 1)
    right = np.clip(insertion, 0, len(ct_series.positions) - 1)

    left_dist = np.abs(point_positions - ct_series.positions[left])
    right_dist = np.abs(point_positions - ct_series.positions[right])
    choose_right = right_dist < left_dist
    indices = np.where(choose_right, right, left).astype(np.int32, copy=False)
    distances = np.where(choose_right, right_dist, left_dist)
    matched = distances <= ct_series.slice_tolerance_mm
    return indices, matched


def _patient_points_to_ct_indices(
    points: np.ndarray,
    ct_slice: CtSlice,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = points - ct_slice.ipp[np.newaxis, :]
    cols_float = (delta @ ct_slice.row_dir) / ct_slice.col_spacing
    rows_float = (delta @ ct_slice.col_dir) / ct_slice.row_spacing

    cols = np.rint(cols_float).astype(np.int32)
    rows = np.rint(rows_float).astype(np.int32)
    in_bounds = (
        (rows >= 0)
        & (rows < ct_slice.rows)
        & (cols >= 0)
        & (cols < ct_slice.cols)
    )
    return rows, cols, in_bounds


def _read_ct_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D CT slice, got shape {arr.shape}: {path}")
    return arr


def _root_image_plane(ds: pydicom.dataset.Dataset) -> ImagePlane:
    ipp = _vector3(getattr(ds, "ImagePositionPatient"))
    orientation = _float_sequence(getattr(ds, "ImageOrientationPatient"), expected_len=6)
    pixel_spacing = _float_sequence(getattr(ds, "PixelSpacing"), expected_len=2)
    return _make_image_plane(ipp, orientation, pixel_spacing)


def _seg_frame_plane(ds: pydicom.dataset.Dataset, frame_idx: int) -> ImagePlane:
    ipp = _seg_frame_image_position(ds, frame_idx)
    orientation = _seg_frame_image_orientation(ds, frame_idx)
    pixel_spacing = _seg_frame_pixel_spacing(ds, frame_idx)
    return _make_image_plane(ipp, orientation, pixel_spacing)


def _seg_frame_image_position(ds: pydicom.dataset.Dataset, frame_idx: int) -> np.ndarray:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(
            frame_group,
            ("PlanePositionSequence", "PlanePositionPatientSequence", "PlanePositionSlideSequence"),
        )
        if item is not None and hasattr(item, "ImagePositionPatient"):
            return _vector3(getattr(item, "ImagePositionPatient"))

    if hasattr(ds, "ImagePositionPatient"):
        return _vector3(getattr(ds, "ImagePositionPatient"))
    raise ValueError("ImagePositionPatient not found for SEG frame")


def _seg_frame_image_orientation(ds: pydicom.dataset.Dataset, frame_idx: int) -> List[float]:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(
            frame_group,
            ("PlaneOrientationSequence", "PlaneOrientationPatientSequence"),
        )
        if item is not None and hasattr(item, "ImageOrientationPatient"):
            return _float_sequence(getattr(item, "ImageOrientationPatient"), expected_len=6)

    shared_group = _shared_group(ds)
    if shared_group is not None:
        item = _first_sequence_item(
            shared_group,
            ("PlaneOrientationSequence", "PlaneOrientationPatientSequence"),
        )
        if item is not None and hasattr(item, "ImageOrientationPatient"):
            return _float_sequence(getattr(item, "ImageOrientationPatient"), expected_len=6)

    if hasattr(ds, "ImageOrientationPatient"):
        return _float_sequence(getattr(ds, "ImageOrientationPatient"), expected_len=6)
    raise ValueError("ImageOrientationPatient not found for SEG frame")


def _seg_frame_pixel_spacing(ds: pydicom.dataset.Dataset, frame_idx: int) -> List[float]:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(frame_group, ("PixelMeasuresSequence",))
        if item is not None and hasattr(item, "PixelSpacing"):
            return _float_sequence(getattr(item, "PixelSpacing"), expected_len=2)

    shared_group = _shared_group(ds)
    if shared_group is not None:
        item = _first_sequence_item(shared_group, ("PixelMeasuresSequence",))
        if item is not None and hasattr(item, "PixelSpacing"):
            return _float_sequence(getattr(item, "PixelSpacing"), expected_len=2)

    if hasattr(ds, "PixelSpacing"):
        return _float_sequence(getattr(ds, "PixelSpacing"), expected_len=2)
    raise ValueError("PixelSpacing not found for SEG frame")


def _make_image_plane(
    ipp: np.ndarray,
    orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> ImagePlane:
    row_dir = _unit(np.asarray(orientation[:3], dtype=np.float64))
    col_dir = _unit(np.asarray(orientation[3:6], dtype=np.float64))
    normal = _unit(np.cross(row_dir, col_dir))
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    if row_spacing <= 0 or col_spacing <= 0:
        raise ValueError(f"invalid PixelSpacing: {pixel_spacing}")
    return ImagePlane(
        ipp=ipp.astype(np.float64, copy=False),
        row_dir=row_dir,
        col_dir=col_dir,
        normal=normal,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
    )


def _per_frame_group(ds: pydicom.dataset.Dataset, frame_idx: int) -> Optional[pydicom.dataset.Dataset]:
    seq = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if seq is not None and frame_idx < len(seq):
        return seq[frame_idx]
    return None


def _shared_group(ds: pydicom.dataset.Dataset) -> Optional[pydicom.dataset.Dataset]:
    seq = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if seq is not None and len(seq) > 0:
        return seq[0]
    return None


def _first_sequence_item(
    ds: pydicom.dataset.Dataset,
    names: Sequence[str],
) -> Optional[pydicom.dataset.Dataset]:
    for name in names:
        seq = getattr(ds, name, None)
        if seq is not None and len(seq) > 0:
            return seq[0]
    return None


def _vector3(value: object) -> np.ndarray:
    values = _float_sequence(value, expected_len=3)
    return np.asarray(values, dtype=np.float64)


def _float_sequence(value: object, expected_len: int) -> List[float]:
    values = [float(item) for item in value]
    if len(values) < expected_len:
        raise ValueError(f"expected at least {expected_len} values, got {values}")
    return values[:expected_len]


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("zero-length direction vector")
    return value / norm


def _median_positive_spacing(positions: np.ndarray) -> float:
    if positions.size < 2:
        return 1.0
    diffs = np.diff(np.unique(np.round(positions, decimals=6)))
    diffs = diffs[diffs > 1e-6]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _add_skipped_no_ct_slice(accumulators: Dict[int, SliceAccumulator], count: int) -> None:
    if count <= 0:
        return
    accumulator = accumulators.setdefault(-1, SliceAccumulator(hu_values=[], source_mask_files=set()))
    accumulator.skipped_no_ct_slice += count


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

def plot_single_hu_stat(data, title, xlabel, output_png_path):
    """
    核心绘图引擎：接收一维数组，计算统计量，并绘制高颜值的核密度分布图。
    """
    if len(data) == 0:
        print(f"❌ 数据为空，无法绘制 {title}")
        return

    # 1. 计算统计量
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    median_val = np.median(data)

    # 2. 计算 KDE 峰值
    # 如果数据完全一致（极差为0），加上极小的扰动防止 KDE 报错
    if min_val == max_val:
        data = data + np.random.normal(0, 1e-5, len(data))
        max_val += 1e-5
        
    kde = gaussian_kde(data)
    x_range = np.linspace(min_val, max_val, 1000)
    kde_values = kde(x_range)
    peak_x = x_range[np.argmax(kde_values)]
    peak_y = np.max(kde_values)

    # 3. 初始化高颜值画布
    plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    ax.grid(color='#E9ECEF', linestyle='-', linewidth=1, alpha=0.8)

    # 绘制带填充的 KDE 密度图 (使用经典的 Steel Blue)
    sns.kdeplot(data, color='#3B5B88', linewidth=2, fill=True, alpha=0.8, bw_adjust=0.5, ax=ax)

    # 设置标题和坐标轴
    plt.title(title, fontsize=18, fontweight='bold', pad=20, color='#2B3A42')
    plt.xlabel(xlabel, fontsize=14, labelpad=12, color='#2B3A42')
    plt.ylabel('Probability Density', fontsize=14, labelpad=12, color='#2B3A42')
    
    # 动态调整 X 轴和 Y 轴范围
    x_span = max_val - min_val if max_val > min_val else 1.0
    plt.xlim(min_val - x_span * 0.05, max_val + x_span * 0.05)
    plt.ylim(0, peak_y * 1.3) # 顶部留出 30% 空间放标签

    # ==========================================
    # 🌟 气泡注释系统 (自动避让重叠)
    # ==========================================
    def add_annotation(text, xy, xytext, edge_color, text_color):
        ax.annotate(text, xy=xy, xytext=xytext,
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=edge_color, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9FA", ec=edge_color, lw=1.5, alpha=0.9),
                    fontsize=10, color=text_color, fontweight='bold', ha='center', zorder=5)

    # 峰值注释 (红色) - 放在最高点上方
    add_annotation(f"Peak Density\nHU: {peak_x:.1f}", 
                   xy=(peak_x, peak_y), xytext=(peak_x, peak_y * 1.15), 
                   edge_color="#C0504D", text_color="#C0504D")

    # 最小值注释 (灰色) - 左侧偏下
    add_annotation(f"Min\nHU: {min_val:.1f}", 
                   xy=(min_val, 0), xytext=(min_val + (x_span * 0.05), peak_y * 0.1), 
                   edge_color="#7F7F7F", text_color="#595959")

    # 平均值注释 (橙色) - 中间偏上
    add_annotation(f"Mean\nHU: {mean_val:.1f}", 
                   xy=(mean_val, 0), xytext=(mean_val + (x_span * 0.08), peak_y * 0.25), 
                   edge_color="#E36C0A", text_color="#E36C0A")

    # 中位值注释 (绿色) - 中间偏下 (与 Mean 错开高度)
    add_annotation(f"Median\nHU: {median_val:.1f}", 
                   xy=(median_val, 0), xytext=(median_val - (x_span * 0.06), peak_y * 0.12), 
                   edge_color="#9BBB59", text_color="#76923C")

    # 最大值注释 (深灰色) - 右侧
    add_annotation(f"Max\nHU: {max_val:.1f}", 
                   xy=(max_val, 0), xytext=(max_val - (x_span * 0.1), peak_y * 0.1), 
                   edge_color="#4F6272", text_color="#3A4B56")

    # 美化边框：隐藏上方和右方的实线
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"✅ 生成完毕 -> {output_png_path}")


def analyze_and_plot_lesion_stats(csv_path, out_min_png, out_max_png, out_mean_png, out_median_png):
    """
    主调度函数：读取 CSV，提取 4 个核心字段，并触发绘图。
    """
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {csv_path}")
        return

    print(f"🚀 正在加载数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 需要处理的字段与对应的输出路径映射
    targets = [
        ('Min_HU', out_min_png, "Distribution of Lesion Minimum HU"),
        ('Max_HU', out_max_png, "Distribution of Lesion Maximum HU"),
        ('Mean_HU', out_mean_png, "Distribution of Lesion Mean HU"),
        ('Median_HU', out_median_png, "Distribution of Lesion Median HU")
    ]

    print("-" * 50)
    for col_name, out_path, title in targets:
        if col_name not in df.columns:
            print(f"⚠️ 警告: CSV 中找不到列 '{col_name}'，已跳过。")
            continue
            
        # 提取数据，剔除空值
        data = pd.to_numeric(df[col_name], errors='coerce').dropna().values
        
        print(f"📊 正在处理 [{col_name}] ... (有效切片数: {len(data)})")
        # 调用核心绘图引擎
        plot_single_hu_stat(data, title, f"{col_name} Value", out_path)
        
    print("-" * 50)
    print("🎉 恭喜！所有 4 张统计图表均已成功生成。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map DICOM SEG foreground voxels to CT and export per-slice HU statistics."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders.")
    parser.add_argument("output_csv", help="CSV output path.")
    parser.add_argument(
        "--slice-tolerance-mm",
        type=float,
        default=None,
        help="Max SEG point to CT slice-plane distance. Default: about half CT slice spacing.",
    )
    parser.add_argument(
        "--normal-dot-threshold",
        type=float,
        default=0.95,
        help="Minimum abs(dot(SEG normal, CT normal)) accepted. Default: 0.95.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress and skipped files.")
    args = parser.parse_args()

    row_count = export_seg_foreground_ct_hu_stats(
        args.root_dir,
        args.output_csv,
        slice_tolerance_mm=args.slice_tolerance_mm,
        normal_dot_threshold=args.normal_dot_threshold,
        verbose=args.verbose,
    )
    print(f"Wrote {row_count} rows to {Path(args.output_csv)}")


if __name__ == "__main__":
    #main()
    INPUT_CSV = "lesion_hu_stat.csv" 
    IMG_MIN = "plot_lesion_min_hu.png"
    IMG_MAX = "plot_lesion_max_hu.png"
    IMG_MEAN = "plot_lesion_mean_hu.png"
    IMG_MEDIAN = "plot_lesion_median_hu.png"
    analyze_and_plot_lesion_stats(INPUT_CSV, IMG_MIN, IMG_MAX, IMG_MEAN, IMG_MEDIAN)