"""CLI sencillo para preparar datos, entrenar y ejecutar el detector/clasificador clásico."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

from logo_detector import classifier, data_prep, detector, paths
from logo_detector.features import FeatureParams


# Construye los parámetros del detector con pocos ajustes expuestos.
def _build_det_params(args: argparse.Namespace, include_limit: bool = False) -> detector.DetectorParams:
    defaults = detector.DetectorParams()
    kwargs = dict(
        min_keypoints=args.min_kp,
        bin_threshold=args.bin_thresh,
        global_nms_iou=args.global_nms,
        max_total_detections=args.max_total,
        candidate_mode=args.candidate_mode,
        use_keypoint_props=not args.no_keyprops,
        use_text_props=not args.no_textprops,
        use_sliding_windows=not args.no_slideprops,
    )
    if include_limit:
        kwargs["limit_images"] = args.limit
        kwargs["iou_threshold"] = args.iou
    return detector.DetectorParams(**kwargs)


def parse_size(value: str) -> tuple[int, int]:
    # Interpreta tamaños como Ancho x Alto.
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Usa formato ANCHOxALTO, ej: 128x128")
    return int(parts[0]), int(parts[1])


def cmd_prepare_annotations(args: argparse.Namespace) -> None:
    # Convierte los XML VOC en un CSV único.
    output = Path(args.output) if args.output else paths.ANNOTATIONS_CSV
    data_prep.build_annotation_table(output_csv=output)


def cmd_crop_patches(args: argparse.Namespace) -> None:
    # Recorta parches y crea el manifest.
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
    # Entrena el SVM multiclase con BoW+HSV+HOG.
    params = FeatureParams(
        vocab_size=args.vocab_size,
        orb_features=args.orb_features,
        desc_limit=120_000,
        patch_size=parse_size(args.patch_size),
        nmax_keypoints=800,
        use_hog=not args.no_hog,
    )
    manifest = Path(args.manifest) if args.manifest else paths.PATCH_MANIFEST
    metrics = classifier.train_classifier(
        manifest_csv=manifest,
        params=params,
        c_values=(1, 5, 10, 25, 50),
        gamma_multipliers=(0.5, 1.0, 2.0),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def cmd_train_detector(args: argparse.Namespace) -> None:
    # Entrena el filtro binario logo/no-logo.
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    models = Path(args.models) if args.models else paths.MODELS_DIR
    metrics = detector.train_binary_filter(
        annotations_csv=annotations,
        models_dir=models,
        neg_per_image=args.neg_per_image,
        max_train_images=args.max_train_images,
        iou_thr_neg=0.2,
        candidate_mode=args.candidate_mode,
        seed=42,
    )
    print(json.dumps(metrics, indent=2))


def cmd_detect(args: argparse.Namespace) -> None:
    # Ejecuta detección (y opcionalmente clasificación).
    params = _build_det_params(args)
    det = detector.LogoDetector(models_dir=args.models, candidate_preset=args.mser_preset)
    classify = getattr(args, "with_classification", False) or not getattr(args, "detector_only", False)
    if args.count and args.count > 0:
        _run_detection_sampler(det, params, args, classify=classify)
        return
    if not args.image:
        raise ValueError("Usa --image cuando count=0.")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    detections = det.detect(img, params=params, classify=classify)
    if not detections:
        print("No detecciones.")
        return
    vis = detector.draw_detections(img, detections)
    for label, score, box in detections:
        print(f"{label:20s} score={score:8.3f} box={box}")
    if args.show:
        cv2.imshow("detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_file = Path(args.output_dir) / (Path(args.image).stem + "_det.jpg")
        cv2.imwrite(str(out_file), vis)
        print(f"Guardado en {out_file}")


def _run_detection_sampler(det, params, args, classify: bool) -> None:
    # Muestra detecciones en imágenes aleatorias de un split.
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    df = pd.read_csv(annotations)
    df = df[df["split"] == args.split]
    paths_arr = df["path"].dropna().unique()
    if len(paths_arr) == 0:
        raise RuntimeError("No hay imágenes disponibles.")
    n = min(args.count, len(paths_arr))
    rng = np.random.default_rng(42)
    chosen = rng.choice(paths_arr, size=n, replace=False)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in chosen:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gts_df = df[df["path"] == img_path][["class", "xmin", "ymin", "xmax", "ymax"]]
        gt_boxes = []
        for _, row in gts_df.iterrows():
            label = row["class"] if classify else "logo"
            gt_boxes.append((label, (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))))
        detections = det.detect(img, params=params, classify=classify)
        vis = np.concatenate([detector.draw_ground_truth(img, gt_boxes), detector.draw_detections(img, detections)], axis=1)
        print(f"{img_path} → {len(detections)} detecciones")
        if args.show:
            cv2.imshow("detections", vis)
            key = cv2.waitKey(0)
            if key == 27:
                break
        if out_dir:
            out_file = out_dir / (Path(img_path).stem + "_det.jpg")
            cv2.imwrite(str(out_file), vis)
    if args.show:
        cv2.destroyAllWindows()


def cmd_evaluate(args: argparse.Namespace) -> None:
    # Evalúa precisión/recall del detector.
    params = _build_det_params(args, include_limit=True)
    det = detector.LogoDetector(models_dir=args.models, candidate_preset=args.mser_preset)
    metrics = detector.evaluate_detector(det, split=args.split, params=params)
    print(json.dumps(metrics, indent=2))


def cmd_oracle_classify(args: argparse.Namespace) -> None:
    # Evalúa la clasificación usando las cajas GT (sin detección).
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    df = pd.read_csv(annotations)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise RuntimeError("No hay anotaciones.")

    det = detector.LogoDetector(models_dir=args.models)
    total = correct = 0
    per_class = {}
    per_class_total = {}
    for img_path, group in df.groupby("path"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        for row in group.to_dict("records"):
            cls = row["class"]
            x1, y1, x2, y2 = map(int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            pred_label, _ = det.classify_patch(patch)
            total += 1
            per_class_total[cls] = per_class_total.get(cls, 0) + 1
            is_correct = int(pred_label == cls)
            correct += is_correct
            per_class[cls] = per_class.get(cls, 0) + is_correct

    acc = correct / max(1, total)
    print(f"Accuracy oracle ({args.split}): {correct}/{total} = {acc:.3f}")
    for cls, tot in sorted(per_class_total.items(), key=lambda kv: kv[0]):
        acc_cls = per_class.get(cls, 0) / tot
        print(f"{cls:15s} {per_class.get(cls, 0):4d}/{tot:<4d} = {acc_cls:.3f}")


def cmd_oracle_visualize(args: argparse.Namespace) -> None:
    # Visualiza predicciones de clasificación sobre las cajas GT.
    annotations = Path(args.annotations) if args.annotations else paths.ANNOTATIONS_CSV
    df = pd.read_csv(annotations)
    if args.split:
        df = df[df["split"] == args.split]
    if df.empty:
        raise RuntimeError("No hay anotaciones.")
    groups = df.groupby("path")
    paths_list = list(groups.groups.keys())[: args.limit or 10]

    det = detector.LogoDetector(models_dir=args.models)
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in paths_list:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt_rows = df[df["path"] == img_path]
        vis = img.copy()
        for _, row in gt_rows.iterrows():
            cls = row["class"]
            x1, y1, x2, y2 = map(int, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
            patch = img[y1:y2, x1:x2]
            pred_label, color_score = det.classify_patch(patch)
            ok = pred_label == cls
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{cls}->{pred_label} ({color_score:.2f})", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if args.show:
            cv2.imshow("oracle-visualize", vis)
            key = cv2.waitKey(0)
            if key == 27:
                break
        if out_dir:
            out_file = out_dir / f"oracle_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_file), vis)
    if args.show:
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("prepare-annotations", help="Genera data/interim/annotations.csv")
    p_ann.add_argument("--output", type=str, default=None)
    p_ann.set_defaults(func=cmd_prepare_annotations)

    p_crop = sub.add_parser("crop-patches", help="Recorta parches y crea manifest.")
    p_crop.add_argument("--annotations", type=str, default=None)
    p_crop.add_argument("--size", type=str, default="128x128")
    p_crop.add_argument("--max-per-class", type=int, default=None)
    p_crop.add_argument("--output", type=str, default=None)
    p_crop.set_defaults(func=cmd_crop_patches)

    p_train = sub.add_parser("train-classifier", help="Entrena el SVM multiclase.")
    p_train.add_argument("--manifest", type=str, default=None)
    p_train.add_argument("--vocab-size", type=int, default=400)
    p_train.add_argument("--orb-features", type=int, default=1000)
    p_train.add_argument("--patch-size", type=str, default="128x128")
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-hog", action="store_true")
    p_train.set_defaults(func=cmd_train_classifier)

    p_det_train = sub.add_parser("train-detector", help="Entrena el filtro binario logo/no-logo.")
    p_det_train.add_argument("--annotations", type=str, default=None)
    p_det_train.add_argument("--models", type=str, default=None)
    p_det_train.add_argument("--neg-per-image", type=int, default=60)
    p_det_train.add_argument("--max-train-images", type=int, default=500)
    p_det_train.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_det_train.set_defaults(func=cmd_train_detector)

    p_detect = sub.add_parser("detect", help="Detecta (y opcionalmente clasifica).")
    p_detect.add_argument("count", nargs="?", type=int, default=0)
    p_detect.add_argument("--image", help="Imagen única si count=0.")
    p_detect.add_argument("--models", type=str, default=None)
    p_detect.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_detect.add_argument("--mser-preset", choices=["strict", "balanced", "loose", "tight"], default="strict")
    p_detect.add_argument("--min-kp", type=int, default=10)
    p_detect.add_argument("--bin-thresh", type=float, default=0.9)
    p_detect.add_argument("--global-nms", type=float, default=0.35)
    p_detect.add_argument("--max-total", type=int, default=2)
    p_detect.add_argument("--no-keyprops", action="store_true")
    p_detect.add_argument("--no-textprops", action="store_true")
    p_detect.add_argument("--no-slideprops", action="store_true")
    p_detect.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_detect.add_argument("--annotations", type=str, default=None)
    p_detect.add_argument("--output-dir", type=str, default=None)
    p_detect.add_argument("--show", action="store_true")
    p_detect.set_defaults(func=cmd_detect, detector_only=False, with_classification=True)

    p_eval = sub.add_parser("evaluate", help="Evalúa el detector en un split.")
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_eval.add_argument("--models", type=str, default=None)
    p_eval.add_argument("--candidate-mode", choices=["mser", "combined"], default="combined")
    p_eval.add_argument("--mser-preset", choices=["strict", "balanced", "loose", "tight"], default="strict")
    p_eval.add_argument("--min-kp", type=int, default=10)
    p_eval.add_argument("--bin-thresh", type=float, default=0.9)
    p_eval.add_argument("--global-nms", type=float, default=0.35)
    p_eval.add_argument("--max-total", type=int, default=2)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--iou", type=float, default=0.5)
    p_eval.add_argument("--no-keyprops", action="store_true")
    p_eval.add_argument("--no-textprops", action="store_true")
    p_eval.add_argument("--no-slideprops", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_oracle = sub.add_parser("oracle-classify", help="Clasifica usando cajas GT (sin detección).")
    p_oracle.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_oracle.add_argument("--annotations", type=str, default=None)
    p_oracle.add_argument("--models", type=str, default=None)
    p_oracle.add_argument("--limit", type=int, default=None)
    p_oracle.set_defaults(func=cmd_oracle_classify)

    p_oracle_vis = sub.add_parser("oracle-visualize", help="Visualiza clasificación sobre GT.")
    p_oracle_vis.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_oracle_vis.add_argument("--annotations", type=str, default=None)
    p_oracle_vis.add_argument("--models", type=str, default=None)
    p_oracle_vis.add_argument("--limit", type=int, default=10)
    p_oracle_vis.add_argument("--output-dir", type=str, default="reports/figures/oracle")
    p_oracle_vis.add_argument("--show", action="store_true")
    p_oracle_vis.set_defaults(func=cmd_oracle_visualize)

    return parser


def _prompt(msg: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    return input(f"{msg}{suffix}: ").strip() or (default or "")


def _prompt_int(msg: str, default: int) -> int:
    val = _prompt(msg, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def run_interactive_menu() -> None:
    # Menú rápido: detectar, clasificar o pipeline completo.
    print("\n¿Qué quieres hacer?")
    print(" 1) Detectar (logo/no-logo)")
    print(" 2) Clasificar (GT, sin detección)")
    print(" 3) Pipeline completo (detección + clasificación)")
    print(" 4) Salir")
    choice = _prompt("Elige opción", "3")
    if choice == "4" or choice.lower() == "salir":
        return

    show = _prompt("¿Ventana con resultado? (s/n)", "n").lower().startswith("s")
    count = _prompt_int("¿Cuántas imágenes aleatorias?", 3)

    if choice == "1":
        args = argparse.Namespace(
            count=count,
            image=None,
            models=None,
            candidate_mode="combined",
            mser_preset="balanced",
            min_kp=10,
            bin_thresh=0.9,
            global_nms=0.35,
            max_total=2,
            no_keyprops=True,
            no_textprops=False,
            no_slideprops=True,
            split="test",
            annotations=None,
            output_dir="reports/figures/demo",
            show=show,
            detector_only=True,
            with_classification=False,
        )
        cmd_detect(args)
        return

    if choice == "2":
        if show:
            args = argparse.Namespace(
                split="test",
                annotations=None,
                models=None,
                limit=count,
                output_dir="reports/figures/oracle",
                show=True,
            )
            cmd_oracle_visualize(args)
        else:
            args = argparse.Namespace(
                split="test",
                annotations=None,
                models=None,
                limit=count,
            )
            cmd_oracle_classify(args)
        return

    if choice == "3":
        args = argparse.Namespace(
            count=count,
            image=None,
            models=None,
            candidate_mode="combined",
            mser_preset="strict",
            min_kp=10,
            bin_thresh=0.9,
            global_nms=0.35,
            max_total=2,
            no_keyprops=False,
            no_textprops=False,
            no_slideprops=False,
            split="test",
            annotations=None,
            output_dir="reports/figures/demo",
            show=show,
            detector_only=False,
            with_classification=True,
        )
        cmd_detect(args)
        return

    print("Opción no reconocida.")


def main() -> None:
    # Si no hay argumentos, lanzamos un menú sencillo.
    if len(sys.argv) == 1:
        run_interactive_menu()
        return
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
