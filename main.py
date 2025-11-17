"""Command line interface for the classical logo detection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from logo_detector import classifier, data_prep, detector, paths
from logo_detector.features import FeatureParams


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must be formatted as WIDTHxHEIGHT, e.g. 128x128")
    return int(parts[0]), int(parts[1])


def cmd_prepare_annotations(args: argparse.Namespace) -> None:
    output = Path(args.output) if args.output else paths.ANNOTATIONS_CSV
    data_prep.build_annotation_table(output_csv=output)


def cmd_crop_patches(args: argparse.Namespace) -> None:
    size = parse_size(args.size)
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    output = Path(args.output) if args.output else paths.PATCH_MANIFEST
    data_prep.crop_logo_patches(
        annotations_csv=annotations,
        patch_size=size,
        max_per_class=args.max_per_class,
        output_manifest=output,
    )


def cmd_train_classifier(args: argparse.Namespace) -> None:
    params = FeatureParams(
        vocab_size=args.vocab_size,
        orb_features=args.orb_features,
        desc_limit=args.desc_limit,
        patch_size=parse_size(args.patch_size),
        nmax_keypoints=args.nmax_keypoints,
        use_hog=not args.no_hog,
    )
    manifest = Path(args.manifest) if args.manifest else paths.PATCH_MANIFEST
    metrics = classifier.train_classifier(
        manifest_csv=manifest,
        params=params,
        c_values=args.c_values,
        gamma_multipliers=args.gamma_multipliers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def cmd_train_detector(args: argparse.Namespace) -> None:
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    models = Path(args.models) if args.models else paths.MODELS_DIR
    metrics = detector.train_binary_filter(
        annotations_csv=annotations,
        models_dir=models,
        neg_per_image=args.neg_per_image,
        max_train_images=args.max_train_images,
        iou_thr_neg=args.iou_thr_neg,
        candidate_mode=args.candidate_mode,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def cmd_detect(args: argparse.Namespace) -> None:
    params = detector.DetectorParams(
        pad=args.pad,
        min_keypoints=args.min_kp,
        top_k_per_class=args.topk,
        bin_threshold=args.bin_thresh,
        candidate_mode=args.candidate_mode,
        use_keypoint_props=not args.no_keyprops,
        use_text_props=not args.no_textprops,
        use_sliding_windows=not args.no_slideprops,
        global_nms_iou=args.global_nms,
    )
    det = detector.LogoDetector(models_dir=args.models)
    if args.count and args.count > 0:
        _run_detection_sampler(det, params, args)
        return
    if not args.image:
        raise ValueError("Provide --image when count is 0.")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    detections = det.detect(img, params=params)
    if not detections:
        print("No detections.")
        return
    vis = detector.draw_detections(img, detections)
    for label, score, box in detections:
        print(f"{label:20s} score={score:8.3f} box={box}")
    if args.show:
        cv2.imshow("detections", vis)
        cv2.waitKey(args.wait if args.wait >= 0 else 0)
        cv2.destroyAllWindows()
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (Path(args.image).stem + "_det.jpg")
        cv2.imwrite(str(out_file), vis)
        print(f"Saved visualization to {out_file}")


def _run_detection_sampler(det, params, args):
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    df = pd.read_csv(annotations)
    if args.split:
        df = df[df["split"] == args.split]
    paths_arr = df["path"].dropna().unique()
    if len(paths_arr) == 0:
        raise RuntimeError("No images available for sampling.")
    n = min(args.count, len(paths_arr))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(paths_arr), size=n, replace=False)
    chosen = paths_arr[idx]
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(chosen, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue
        gts_df = df[df["path"] == img_path][["class", "xmin", "ymin", "xmax", "ymax"]]
        gt_boxes = [
            (
                row["class"],
                (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])),
            )
            for _, row in gts_df.iterrows()
        ]
        detections = det.detect(img, params=params)
        vis_pred = detector.draw_detections(img, detections)
        vis_gt = detector.draw_ground_truth(img, gt_boxes)
        vis = np.concatenate([vis_gt, vis_pred], axis=1)
        print(f"[{i}/{n}] {img_path}")
        if gt_boxes:
            print("   Ground-truth:")
            for label, box in gt_boxes:
                print(f"      {label:20s} box={box}")
        else:
            print("   Ground-truth: none")
        print(f"   Predictions ({len(detections)}):")
        for label, score, box in detections:
            print(f"      {label:20s} score={score:8.3f} box={box}")
        if args.show:
            cv2.imshow("detections", vis)
            key = cv2.waitKey(args.wait if args.wait >= 0 else 0)
            if key == 27:  # ESC para salir antes
                break
        if out_dir:
            out_file = out_dir / (Path(img_path).stem + "_det.jpg")
            cv2.imwrite(str(out_file), vis)
    if args.show:
        cv2.destroyAllWindows()


def cmd_evaluate(args: argparse.Namespace) -> None:
    params = detector.DetectorParams(
        pad=args.pad,
        min_keypoints=args.min_kp,
        top_k_per_class=args.topk,
        bin_threshold=args.bin_thresh,
        candidate_mode=args.candidate_mode,
        limit_images=args.limit,
        iou_threshold=args.iou,
        use_keypoint_props=not args.no_keyprops,
        use_text_props=not args.no_textprops,
        use_sliding_windows=not args.no_slideprops,
        global_nms_iou=args.global_nms,
    )
    det = detector.LogoDetector(models_dir=args.models)
    metrics = detector.evaluate_detector(det, split=args.split, params=params)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("prepare-annotations", help="Parse VOC XML files into a CSV manifest.")
    p_ann.add_argument("--output", type=str, default=None, help="Destination CSV (default: data/interim/annotations.csv).")
    p_ann.set_defaults(func=cmd_prepare_annotations)

    p_crop = sub.add_parser("crop-patches", help="Crop logo patches and build a manifest.")
    p_crop.add_argument("--annotations", type=str, default=None, help="Annotation CSV (default: data/interim/annotations.csv).")
    p_crop.add_argument("--size", type=str, default="128x128", help="Patch size, e.g. 128x128.")
    p_crop.add_argument("--max-per-class", type=int, default=None, help="Optional limit per class.")
    p_crop.add_argument("--output", type=str, default=None, help="Destination CSV (default: data/processed/patches_manifest.csv).")
    p_crop.set_defaults(func=cmd_crop_patches)

    p_train = sub.add_parser("train-classifier", help="Train the BoW+HSV OpenCV SVM.")
    p_train.add_argument("--manifest", type=str, default=None, help="Patch manifest CSV.")
    p_train.add_argument("--vocab-size", type=int, default=400)
    p_train.add_argument("--orb-features", type=int, default=1000)
    p_train.add_argument("--desc-limit", type=int, default=120_000)
    p_train.add_argument("--patch-size", type=str, default="128x128")
    p_train.add_argument("--nmax-keypoints", type=int, default=800)
    p_train.add_argument("--no-hog", action="store_true", help="Disable HOG descriptors.")
    p_train.add_argument("--c-values", type=float, nargs="+", default=[1, 5, 10, 25, 50])
    p_train.add_argument("--gamma-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train_classifier)

    p_det = sub.add_parser("train-detector", help="Train the MSER proposal binary filter.")
    p_det.add_argument("--annotations", type=str, default=None)
    p_det.add_argument("--models", type=str, default=None)
    p_det.add_argument("--neg-per-image", type=int, default=40)
    p_det.add_argument("--max-train-images", type=int, default=300)
    p_det.add_argument("--iou-thr-neg", type=float, default=0.2)
    p_det.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_det.add_argument("--seed", type=int, default=42)
    p_det.set_defaults(func=cmd_train_detector)

    p_detect = sub.add_parser("detect", help="Run the detector on a single image or sample random ones.")
    p_detect.add_argument("count", nargs="?", type=int, default=0, help="Sample this many images (default 0 = use --image).")
    p_detect.add_argument("--image", help="Path to a single image (required if count=0).")
    p_detect.add_argument("--models", type=str, default=None, help="Model directory (default: models/).")
    p_detect.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_detect.add_argument("--topk", type=int, default=1, help="Boxes to keep per class.")
    p_detect.add_argument("--min-kp", type=int, default=8, help="Minimum ORB keypoints.")
    p_detect.add_argument("--pad", type=float, default=0.1, help="Relative padding around proposals.")
    p_detect.add_argument("--bin-thresh", type=float, default=0.85, help="Probability threshold for the binary filter.")
    p_detect.add_argument("--global-nms", type=float, default=0.5, help="IoU for cross-class NMS (0 disables).")
    p_detect.add_argument("--no-keyprops", action="store_true", help="Disable keypoint-based candidates.")
    p_detect.add_argument("--no-textprops", action="store_true", help="Disable text-based candidates.")
    p_detect.add_argument("--no-slideprops", action="store_true", help="Disable sliding-window candidates.")
    p_detect.add_argument("--split", choices=["train", "val", "test"], default="test", help="Split to sample images from.")
    p_detect.add_argument("--annotations", type=str, default=None, help="Annotation CSV to sample (default: data/interim/annotations.csv).")
    p_detect.add_argument("--output-dir", type=str, default=None, help="Optional folder to save visualizations.")
    p_detect.add_argument("--show", action="store_true", help="Display detections with cv2.imshow.")
    p_detect.add_argument("--wait", type=int, default=0, help="Delay in ms for --show (0 waits for key, negative uses default).")
    p_detect.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p_detect.set_defaults(func=cmd_detect)

    p_eval = sub.add_parser("evaluate", help="Evaluate recall/precision on a dataset split.")
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_eval.add_argument("--models", type=str, default=None)
    p_eval.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_eval.add_argument("--topk", type=int, default=1)
    p_eval.add_argument("--min-kp", type=int, default=8)
    p_eval.add_argument("--pad", type=float, default=0.1)
    p_eval.add_argument("--bin-thresh", type=float, default=0.85)
    p_eval.add_argument("--global-nms", type=float, default=0.5, help="IoU for cross-class NMS (0 disables).")
    p_eval.add_argument("--no-keyprops", action="store_true", help="Disable keypoint-based candidates.")
    p_eval.add_argument("--no-textprops", action="store_true", help="Disable text-based candidates.")
    p_eval.add_argument("--no-slideprops", action="store_true", help="Disable sliding-window candidates.")
    p_eval.add_argument("--limit", type=int, default=None, help="Optional max images.")
    p_eval.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
