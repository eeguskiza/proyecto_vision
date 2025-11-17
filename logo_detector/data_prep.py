"""Data preparation utilities extracted from the original notebooks."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd

from . import paths


def parse_voc_file(xml_path: Path) -> List[Dict[str, int | str]]:
    """Parse a Pascal/VOC style annotation file."""
    rows: List[Dict[str, int | str]] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Error parsing {xml_path}") from exc

    root = tree.getroot()
    filename = root.findtext("filename", default="")
    size = root.find("size")
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))

    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        rows.append(
            {
                "filename": filename,
                "class": obj.findtext("name", default="unknown"),
                "xmin": int(bb.findtext("xmin", default="0")),
                "ymin": int(bb.findtext("ymin", default="0")),
                "xmax": int(bb.findtext("xmax", default="0")),
                "ymax": int(bb.findtext("ymax", default="0")),
                "width": width,
                "height": height,
            }
        )
    return rows


def parse_split(split_dir: Path, split_name: str) -> List[Dict[str, str | int]]:
    """Parse all annotations inside a directory."""
    rows: List[Dict[str, str | int]] = []
    for xml_path in sorted(split_dir.rglob("*.xml")):
        try:
            annots = parse_voc_file(xml_path)
        except RuntimeError as exc:
            print(exc)
            continue
        for ann in annots:
            ann["split"] = split_name
            ann["path"] = str((xml_path.parent / ann["filename"]).resolve())
            rows.append(ann)
    return rows


def build_annotation_table(
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Parse the VOC folders and create data/interim/annotations.csv."""
    paths.ensure_structure()
    rows: List[Dict[str, str | int]] = []
    for split_name, directory in (
        ("train", paths.TRAIN_DIR),
        ("val", paths.VAL_DIR),
        ("test", paths.TEST_DIR),
    ):
        if not directory.exists():
            print(f"[WARN] {directory} does not exist, skipping.")
            continue
        split_rows = parse_split(directory, split_name)
        rows.extend(split_rows)
        print(f"{split_name}: {len(split_rows)} annotations.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No annotations were found.")

    output_csv = output_csv or paths.ANNOTATIONS_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved annotations to {output_csv.resolve()}")
    return df


def crop_logo_patches(
    annotations_csv: Path | None = None,
    patch_size: Tuple[int, int] = (128, 128),
    max_per_class: int | None = None,
    output_manifest: Path | None = None,
) -> pd.DataFrame:
    """Crop patches for every annotation and build a manifest."""
    paths.ensure_structure()
    annotations_csv = annotations_csv or paths.ANNOTATIONS_CSV
    output_manifest = output_manifest or paths.PATCH_MANIFEST

    df = pd.read_csv(annotations_csv)
    if df.empty:
        raise RuntimeError("Annotation CSV is empty.")

    patch_dir = paths.PATCH_DIR
    patch_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = defaultdict(int)
    manifest: List[Dict[str, str]] = []

    for row in df.to_dict("records"):
        cls = row["class"]
        if max_per_class is not None and counts[cls] >= max_per_class:
            continue
        img = cv2.imread(row["path"])
        if img is None:
            print(f"[WARN] Could not read {row['path']}")
            continue
        h, w = img.shape[:2]
        x1 = max(0, min(int(row["xmin"]), w - 1))
        y1 = max(0, min(int(row["ymin"]), h - 1))
        x2 = max(1, min(int(row["xmax"]), w))
        y2 = max(1, min(int(row["ymax"]), h))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_AREA)
        cls_dir = patch_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(row["path"]).stem
        patch_name = f"{stem}_{x1}_{y1}_{x2}_{y2}.jpg"
        out_path = cls_dir / patch_name
        cv2.imwrite(str(out_path), patch)
        manifest.append(
            {
                "patch_path": str(out_path.resolve()),
                "class": cls,
                "split": row["split"],
            }
        )
        counts[cls] += 1

    manifest_df = pd.DataFrame(manifest)
    if manifest_df.empty:
        raise RuntimeError("No patches were created.")

    manifest_df.to_csv(output_manifest, index=False)
    print(f"Saved {len(manifest_df)} patches to {output_manifest.resolve()}")
    return manifest_df
