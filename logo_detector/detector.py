"""Logo detection helpers based on MSER proposals + SVM classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from . import paths
from .features import FeatureExtractor, FeatureParams, load_vocabulary


KP_ORB = cv2.ORB_create(nfeatures=1500)


CandidateFn = Callable[[np.ndarray], List[Tuple[int, int, int, int]]]


@dataclass
class DetectorParams:
    pad: float = 0.1
    min_keypoints: int = 8
    top_k_per_class: int = 1
    bin_threshold: float = 0.85
    candidate_mode: str = "combined"  # 'mser' or 'combined'
    limit_images: int | None = None
    iou_threshold: float = 0.5
    mser_scales: Tuple[float, ...] = (1.0, 0.85, 0.7)
    contour_scales: Tuple[float, ...] = (1.0, 0.75, 0.6, 0.5)
    keypoint_scales: Tuple[float, ...] = (1.0, 0.7, 0.5)
    use_keypoint_props: bool = True
    global_nms_iou: float = 0.5


class LogoDetector:
    """Utility class that loads all persisted artifacts for inference."""

    def __init__(
        self,
        models_dir: Path | None = None,
        feature_params: FeatureParams | None = None,
        candidate_preset: str = "loose",
        load_binary: bool = True,
    ):
        self.models_dir = Path(models_dir or paths.MODELS_DIR)
        self.feature_params = feature_params or FeatureParams()
        self.vocab = load_vocabulary(self.models_dir / "bow_dict.yml")
        self.extractor = FeatureExtractor(self.vocab, params=self.feature_params)
        with open(self.models_dir / "labels.json", "r", encoding="utf-8") as f:
            self.classes = json.load(f)["classes"]
        self.mu = np.load(self.models_dir / "mu.npy").ravel()
        self.sigma = np.load(self.models_dir / "sigma.npy").ravel()
        self.svm = cv2.ml.SVM_load(str(self.models_dir / "svm_bow_hsv_hellinger.xml"))
        self.binary_model_path = self.models_dir / "logo_filter.joblib"
        self.binary_filter = None
        if load_binary and self.binary_model_path.exists():
            self.binary_filter = joblib.load(self.binary_model_path)
            print(f"[INFO] Binary filter loaded from {self.binary_model_path}")
        self._mser = _build_mser(preset=candidate_preset)

    def detect(
        self,
        img_bgr: np.ndarray,
        params: DetectorParams | None = None,
        candidate_fn: CandidateFn | None = None,
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        params = params or DetectorParams()
        boxes = candidate_fn(img_bgr) if candidate_fn else self._generate_candidates(img_bgr, params)
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = pad_box((x1, y1, x2, y2), H, W, pad=params.pad)
            patch = img_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patch_gray = gray[y1:y2, x1:x2]
            kps = self.extractor.orb.detect(patch_gray, None)
            if not kps or len(kps) < params.min_keypoints:
                continue

            feat = self._standardized_features(patch)

            p_logo = 1.0
            if self.binary_filter is not None:
                p_logo = float(self.binary_filter.predict_proba(feat)[0, 1])
                if p_logo < params.bin_threshold:
                    continue

            _, pred = self.svm.predict(feat)
            cls_id = int(pred.ravel()[0])
            resp = float(np.mean([kp.response for kp in kps]))
            score = p_logo * max(1e-3, resp) * len(kps)
            detections.append((cls_id, score, (x1, y1, x2, y2)))

        final: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        for cls_id in set(det[0] for det in detections):
            cls_boxes = [d[2] for d in detections if d[0] == cls_id]
            cls_scores = [d[1] for d in detections if d[0] == cls_id]
            keep = nms(cls_boxes, cls_scores, iou_thr=0.4)[: params.top_k_per_class]
            for idx in keep:
                final.append((self.classes[cls_id], cls_scores[idx], cls_boxes[idx]))
        if params.global_nms_iou > 0 and len(final) > 1:
            boxes = [d[2] for d in final]
            scores = [d[1] for d in final]
            keep = nms(boxes, scores, iou_thr=params.global_nms_iou)
            final = [final[i] for i in keep]
        return final

    def detect_file(self, image_path: Path, **kwargs) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        return self.detect(img, **kwargs)

    def _standardized_features(self, img: np.ndarray) -> np.ndarray:
        vec = self.extractor.describe(img)
        return ((vec - self.mu) / self.sigma).astype(np.float32).reshape(1, -1)

    def _generate_candidates(self, img: np.ndarray, params: DetectorParams | str) -> List[Tuple[int, int, int, int]]:
        cfg = params if isinstance(params, DetectorParams) else DetectorParams(candidate_mode=params)
        boxes: List[Tuple[int, int, int, int]] = []
        if cfg.candidate_mode in ("mser", "combined"):
            boxes.extend(_mser_candidates_multiscale(img, self._mser, cfg.mser_scales))
        if cfg.candidate_mode == "combined":
            boxes.extend(contour_candidates(img, scales=cfg.contour_scales))
            if cfg.use_keypoint_props:
                boxes.extend(keypoint_candidates(img, scales=cfg.keypoint_scales))
        if cfg.candidate_mode not in ("mser", "combined"):
            raise ValueError(f"Unknown candidate mode '{cfg.candidate_mode}'")
        return _deduplicate_boxes(boxes)


def train_binary_filter(
    annotations_csv: Path | None = None,
    models_dir: Path | None = None,
    neg_per_image: int = 40,
    max_train_images: int = 300,
    iou_thr_neg: float = 0.2,
    candidate_mode: str = "combined",
    seed: int = 42,
) -> Dict[str, float]:
    """Train the LinearSVC-based binary filter that prunes proposals."""
    annotations_csv = annotations_csv or paths.ANNOTATIONS_CSV
    models_dir = Path(models_dir or paths.MODELS_DIR)
    detector = LogoDetector(models_dir=models_dir, load_binary=False)
    gen_params = DetectorParams(candidate_mode=candidate_mode)
    ann = pd.read_csv(annotations_csv)
    train_paths = ann[ann["split"] == "train"]["path"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(train_paths)
    train_paths = train_paths[:max_train_images]

    gt_map = {
        path: ann[(ann["path"] == path) & (ann["split"] == "train")][["class", "xmin", "ymin", "xmax", "ymax"]].values
        for path in train_paths
    }

    candidates = []
    labels = []
    neg_ratio_limit = 2.0

    for path in train_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gts = gt_map[path]
        # Positive samples
        for (_, x1, y1, x2, y2) in gts:
            patch = img[int(y1) : int(y2), int(x1) : int(x2)]
            if patch.size == 0:
                continue
            candidates.append(detector._standardized_features(patch).ravel())
            labels.append(1)

        neg_added = 0
        boxes = detector._generate_candidates(img, gen_params)
        rng.shuffle(boxes)
        for (x1, y1, x2, y2) in boxes:
            if neg_added >= neg_per_image:
                break
            if any(iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2)) >= iou_thr_neg for (_, gx1, gy1, gx2, gy2) in gts):
                continue
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            candidates.append(detector._standardized_features(patch).ravel())
            labels.append(0)
            neg_added += 1

    if not candidates:
        raise RuntimeError("No samples collected for the binary filter.")

    X = np.vstack(candidates).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    positives = (y == 1).sum()
    negatives = (y == 0).sum()
    max_negatives = int(min(negatives, neg_ratio_limit * positives))
    if negatives > max_negatives:
        idx = rng.choice(np.where(y == 0)[0], size=max_negatives, replace=False)
        keep = np.concatenate([np.where(y == 1)[0], idx])
        X = X[keep]
        y = y[keep]

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=seed,
        max_iter=20000,
        dual=False,
        tol=1e-3,
    )
    try:
        clf = CalibratedClassifierCV(base_estimator=base, cv=3)
    except TypeError:
        clf = CalibratedClassifierCV(estimator=base, cv=3)
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xva)[:, 1]
    ap = average_precision_score(yva, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(yva, (probs >= 0.5).astype(int), average="binary")

    joblib.dump(clf, models_dir / "logo_filter.joblib")
    print(f"Binary filter saved to {models_dir / 'logo_filter.joblib'}")

    return {"ap": float(ap), "precision": float(prec), "recall": float(rec), "f1": float(f1), "samples": float(len(X))}


def evaluate_detector(
    detector: LogoDetector,
    annotations_csv: Path | None = None,
    split: str = "test",
    params: DetectorParams | None = None,
) -> Dict[str, float]:
    annotations_csv = annotations_csv or paths.ANNOTATIONS_CSV
    df = pd.read_csv(annotations_csv)
    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise RuntimeError(f"No samples found for split '{split}'.")
    params = params or DetectorParams()
    paths_list = split_df["path"].unique().tolist()
    if params.limit_images:
        paths_list = paths_list[: params.limit_images]

    TP = FP = FN = 0
    for path in paths_list:
        img = cv2.imread(path)
        if img is None:
            continue
        gts = split_df[split_df["path"] == path][["class", "xmin", "ymin", "xmax", "ymax"]].values
        preds = detector.detect(img, params=params)
        used = np.zeros(len(gts), dtype=bool)
        for label, score, box in preds:
            matched = False
            for idx, (cls, x1, y1, x2, y2) in enumerate(gts):
                if used[idx] or cls != label:
                    continue
                if iou(box, (x1, y1, x2, y2)) >= params.iou_threshold:
                    TP += 1
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                FP += 1
        FN += int((~used).sum())

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    metrics = {"precision": precision, "recall": recall, "f1": f1, "TP": TP, "FP": FP, "FN": FN}
    print(f"Evaluation → P={precision:.3f} R={recall:.3f} F1={f1:.3f} (TP={TP} FP={FP} FN={FN})")
    return metrics


def _build_mser(preset: str = "loose"):
    cfg = {
        "tight": dict(delta=6, min_area=150, max_area=18000),
        "loose": dict(delta=4, min_area=50, max_area=35000),
        "balanced": dict(delta=5, min_area=80, max_area=25000),
    }.get(preset, dict(delta=5, min_area=80, max_area=25000))
    try:
        return cv2.MSER_create(_delta=cfg["delta"], _min_area=cfg["min_area"], _max_area=cfg["max_area"])
    except TypeError:
        mser = cv2.MSER_create(cfg["delta"], cfg["min_area"], cfg["max_area"])
        try:
            mser.setDelta(cfg["delta"])
            mser.setMinArea(cfg["min_area"])
            mser.setMaxArea(cfg["max_area"])
        except Exception:
            pass
        return mser


def _mser_candidates_single(img_bgr: np.ndarray, mser) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    regions, _ = mser.detectRegions(gray)
    H, W = gray.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    img_area = H * W
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        if w * h < 120 or w < 10 or h < 10:
            continue
        ar = w / max(1, h)
        if not (0.2 <= ar <= 5.0):
            continue
        area = w * h
        area_ratio = area / max(1, img_area)
        if area_ratio < 5e-5 or area_ratio > 0.35:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _mser_candidates_multiscale(img_bgr: np.ndarray, mser, scales: Sequence[float]) -> List[Tuple[int, int, int, int]]:
    H, W = img_bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for s in scales or (1.0,):
        s = float(s)
        if s <= 0:
            continue
        if abs(s - 1.0) < 1e-3:
            scaled = img_bgr
        else:
            scaled = cv2.resize(img_bgr, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        scaled_boxes = _mser_candidates_single(scaled, mser)
        if abs(s - 1.0) < 1e-3:
            boxes.extend(scaled_boxes)
        else:
            inv = 1.0 / s
            for (x1, y1, x2, y2) in scaled_boxes:
                boxes.append(
                    (
                        max(0, min(W, int(x1 * inv))),
                        max(0, min(H, int(y1 * inv))),
                        max(0, min(W, int(x2 * inv))),
                        max(0, min(H, int(y2 * inv))),
                    )
                )
    return boxes


def contour_candidates(img_bgr: np.ndarray, scales: Sequence[float] = (1.0, 0.75, 0.6), canny: Tuple[int, int] = (80, 160)) -> List[Tuple[int, int, int, int]]:
    boxes = []
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for s in scales:
        g = gray if s == 1.0 else cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(g, canny[0], canny[1])
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 150 or w < 10 or h < 10:
                continue
            ar = w / max(1, h)
            if not (0.2 <= ar <= 5.0):
                continue
            area = w * h
            area_ratio = area / max(1, (W if s == 1.0 else int(W * s)) * (H if s == 1.0 else int(H * s)))
            if area_ratio < 2e-4 or area_ratio > 0.35:
                continue
            if s != 1.0:
                x = int(x / s)
                y = int(y / s)
                w = int(w / s)
                h = int(h / s)
            boxes.append((max(0, x), max(0, y), min(W, x + w), min(H, y + h)))
    return boxes


def keypoint_candidates(
    img_bgr: np.ndarray,
    scales: Sequence[float] = (1.0, 0.7, 0.5),
    orb: cv2.ORB = KP_ORB,
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    H0, W0 = img_bgr.shape[:2]
    for s in scales or (1.0,):
        if s <= 0:
            continue
        if abs(s - 1.0) < 1e-3:
            scaled = img_bgr
            W, H = W0, H0
        else:
            W = max(1, int(W0 * s))
            H = max(1, int(H0 * s))
            scaled = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        kps = orb.detect(gray, None)
        if not kps or len(kps) < 4:
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if not (0 <= x < W and 0 <= y < H):
                continue
            radius = int(max(2, kp.size * 0.4))
            cv2.circle(mask, (x, y), radius, 255, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 60 or w < 6 or h < 6:
                continue
            ar = w / max(1, h)
            if not (0.2 <= ar <= 5.0):
                continue
            area_ratio = (w * h) / float(W * H)
            if area_ratio < 4e-5 or area_ratio > 0.35:
                continue
            if abs(s - 1.0) >= 1e-3:
                inv = 1.0 / s
                x = int(x * inv)
                y = int(y * inv)
                w = int(w * inv)
                h = int(h * inv)
            boxes.append((max(0, x), max(0, y), min(W0, x + w), min(H0, y + h)))
    return boxes


def pad_box(box: Tuple[int, int, int, int], H: int, W: int, pad: float = 0.15) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    nw = int(w * (1 + 2 * pad))
    nh = int(h * (1 + 2 * pad))
    nx1 = max(0, int(cx - nw / 2))
    ny1 = max(0, int(cy - nh / 2))
    nx2 = min(W, int(cx + nw / 2))
    ny2 = min(H, int(cy + nh / 2))
    return nx1, ny1, nx2, ny2


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thr: float = 0.4) -> List[int]:
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        rest = np.array([j for j in rest if iou(boxes[i], boxes[j]) <= iou_thr], dtype=int)
        order = rest
    return keep


def _deduplicate_boxes(boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.9) -> List[Tuple[int, int, int, int]]:
    unique: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(iou(box, b) >= iou_thr for b in unique):
            continue
        unique.append(box)
    return unique


def draw_detections(img: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
    """Return a copy of the image with detection boxes and labels overlayed."""
    vis = img.copy()
    for label, score, (x1, y1, x2, y2) in detections:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        caption = f"{label} {score:.2f}"
        cv2.putText(
            vis,
            caption,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return vis


def draw_ground_truth(img: np.ndarray, gt_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
    """Overlay ground-truth boxes (class, box)."""
    vis = img.copy()
    for label, (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )
    return vis
