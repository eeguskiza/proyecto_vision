"""Punto de entrada único para preparar datos, entrenar y evaluar detector + clasificador."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logo_detector import classifier, data_prep, detector, paths
from logo_detector.features import FeatureParams


def parse_size(value: str) -> tuple[int, int]:
    # Interpreta tamaños como Ancho x Alto.
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Usa formato ANCHOxALTO, ej. 128x128")
    return int(parts[0]), int(parts[1])


def main() -> None:
    # Script todo-en-uno: prepara datos, entrena y evalúa el pipeline.
    parser = argparse.ArgumentParser(
        description="Prepara datos, entrena clasificador y detector, y evalúa el pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations-out", type=str, default=None, help="Destino de annotations.csv.")
    parser.add_argument("--manifest-out", type=str, default=None, help="Destino del manifest de parches.")
    parser.add_argument("--patch-size", type=str, default="128x128", help="Tamano de los parches recortados.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Limite opcional de parches por clase.")
    parser.add_argument("--vocab-size", type=int, default=400)
    parser.add_argument("--orb-features", type=int, default=1000)
    parser.add_argument("--desc-limit", type=int, default=120_000)
    parser.add_argument("--nmax-keypoints", type=int, default=800)
    parser.add_argument("--no-hog", action="store_true", help="Desactiva HOG en los descriptores.")
    parser.add_argument("--c-values", type=float, nargs="+", default=[1, 5, 10, 25, 50])
    parser.add_argument("--gamma-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-per-image", type=int, default=40)
    parser.add_argument("--max-train-images", type=int, default=300)
    parser.add_argument("--iou-thr-neg", type=float, default=0.2)
    parser.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    parser.add_argument("--mser-preset", choices=["strict", "balanced", "loose", "tight"], default="strict")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--pad", type=float, default=0.1)
    parser.add_argument("--min-kp", type=int, default=8)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--bin-thresh", type=float, default=0.85)
    parser.add_argument("--min-var", type=float, default=8.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-area-ratio", type=float, default=0.3)
    parser.add_argument("--min-ar", type=float, default=0.25)
    parser.add_argument("--max-ar", type=float, default=4.0)
    parser.add_argument("--min-sat", type=float, default=15.0)
    parser.add_argument("--global-nms", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU para la evaluacion del pipeline.")
    parser.add_argument("--no-keyprops", action="store_true")
    parser.add_argument("--no-textprops", action="store_true")
    parser.add_argument("--no-slideprops", action="store_true")
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximo de imagenes para evaluar.")
    parser.add_argument("--models", type=str, default=None, help="Directorio de modelos (default: models/).")
    parser.add_argument("--skip-eval", action="store_true", help="Omite la fase de evaluacion final.")
    args = parser.parse_args()

    ann_csv = Path(args.annotations_out) if args.annotations_out else paths.ANNOTATIONS_CSV
    manifest_csv = Path(args.manifest_out) if args.manifest_out else paths.PATCH_MANIFEST
    models_dir = Path(args.models) if args.models else paths.MODELS_DIR
    patch_size = parse_size(args.patch_size)

    print("==> 1) Preparando anotaciones")
    data_prep.build_annotation_table(output_csv=ann_csv)

    print("==> 2) Recortando parches")
    data_prep.crop_logo_patches(
        annotations_csv=ann_csv,
        patch_size=patch_size,
        max_per_class=args.max_per_class,
        output_manifest=manifest_csv,
    )

    print("==> 3) Entrenando clasificador (BoW+HSV+HOG + SVM)")
    feat_params = FeatureParams(
        vocab_size=args.vocab_size,
        orb_features=args.orb_features,
        desc_limit=args.desc_limit,
        patch_size=patch_size,
        nmax_keypoints=args.nmax_keypoints,
        use_hog=not args.no_hog,
    )
    cls_metrics = classifier.train_classifier(
        manifest_csv=manifest_csv,
        params=feat_params,
        c_values=args.c_values,
        gamma_multipliers=args.gamma_multipliers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps({"classifier": cls_metrics}, indent=2))

    print("==> 4) Entrenando filtro binario del detector")
    det_metrics = detector.train_binary_filter(
        annotations_csv=ann_csv,
        models_dir=models_dir,
        neg_per_image=args.neg_per_image,
        max_train_images=args.max_train_images,
        iou_thr_neg=args.iou_thr_neg,
        candidate_mode=args.candidate_mode,
        seed=args.seed,
    )
    print(json.dumps({"detector_binary": det_metrics}, indent=2))

    if args.skip_eval:
        return

    print("==> 5) Evaluando pipeline completo")
    det = detector.LogoDetector(models_dir=models_dir, candidate_preset=args.mser_preset)
    det_params = detector.DetectorParams(
        pad=args.pad,
        min_keypoints=args.min_kp,
        top_k_per_class=args.topk,
        bin_threshold=args.bin_thresh,
        min_variance=args.min_var,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_aspect_ratio=args.min_ar,
        max_aspect_ratio=args.max_ar,
        min_saturation=args.min_sat,
        candidate_mode=args.candidate_mode,
        limit_images=args.limit,
        iou_threshold=args.iou,
        use_keypoint_props=not args.no_keyprops,
        use_text_props=not args.no_textprops,
        use_sliding_windows=not args.no_slideprops,
        global_nms_iou=args.global_nms,
        max_total_detections=args.max_total,
    )
    eval_metrics = detector.evaluate_detector(det, split=args.eval_split, params=det_params)
    print(json.dumps({"pipeline_eval": eval_metrics}, indent=2))


if __name__ == "__main__":
    main()
