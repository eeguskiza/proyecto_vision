"""Training routine for the BoW + HSV multi-class SVM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from . import paths
from .features import (
    FeatureExtractor,
    FeatureParams,
    build_feature_matrix,
    compute_stats,
    save_vocabulary,
    standardize,
    train_vocabulary,
)


def stratified_split(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts = []
    for cls, sub in df.groupby("class"):
        n_val = max(1, int(len(sub) * val_fraction))
        idx = rng.choice(sub.index, size=n_val, replace=False)
        parts.append(df.loc[idx])
    val_df = pd.concat(parts).sample(frac=1.0, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df, val_df


def train_classifier(
    manifest_csv: Path | None = None,
    params: FeatureParams | None = None,
    c_values: Sequence[float] = (1, 5, 10, 25, 50),
    gamma_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, float]:
    """Train the BoW+HSV classifier and persist its artifacts under models/."""
    paths.ensure_structure()
    manifest_csv = manifest_csv or paths.PATCH_MANIFEST
    params = params or FeatureParams()

    df = pd.read_csv(manifest_csv)
    pool = df[df["split"].isin(["train", "val"])].copy()
    if pool.empty:
        raise RuntimeError("Manifest does not contain train/val splits.")

    classes = sorted(pool["class"].unique())
    label2id = {cls: i for i, cls in enumerate(classes)}

    train_df, val_df = stratified_split(pool, val_fraction=val_fraction, seed=seed)
    print(f"Train={len(train_df)} | Val={len(val_df)} | Classes={len(classes)}")

    vocab = train_vocabulary(train_df["patch_path"].tolist(), params=params)
    save_vocabulary(vocab, paths.MODELS_DIR / "bow_dict.yml")

    extractor = FeatureExtractor(vocab, params=params)
    Xtr, ytr = build_feature_matrix(train_df, extractor, label2id)
    Xva, yva = build_feature_matrix(val_df, extractor, label2id)
    mu, sigma = compute_stats(Xtr)
    Xtr_s = standardize(Xtr, mu, sigma)
    Xva_s = standardize(Xva, mu, sigma)
    np.save(paths.MODELS_DIR / "mu.npy", mu)
    np.save(paths.MODELS_DIR / "sigma.npy", sigma)
    print(f"Feature stats saved to {paths.MODELS_DIR}")

    def train_eval(C: float, gamma: float):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setC(float(C))
        svm.setGamma(float(gamma))
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 2000, 1e-3))
        svm.train(Xtr_s, cv2.ml.ROW_SAMPLE, ytr)
        _, pred = svm.predict(Xva_s)
        acc = float((pred.astype(np.int32).ravel() == yva).mean())
        return acc, svm

    D = Xtr_s.shape[1]
    gamma_base = 1.0 / D
    best = (-1.0, None, None, None)
    for C in c_values:
        for gm in gamma_multipliers:
            gamma = gamma_base * gm
            acc, model = train_eval(C, gamma)
            print(f"C={C:<6} gamma={gamma:.6f} acc={acc:.4f}")
            if acc > best[0]:
                best = (acc, C, gamma, model)

    best_acc, best_C, best_gamma, best_svm = best
    if best_svm is None:
        raise RuntimeError("Failed to train SVM.")

    svm_path = paths.MODELS_DIR / "svm_bow_hsv_hellinger.xml"
    best_svm.save(str(svm_path))

    with open(paths.MODELS_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)

    cm = _confusion_matrix(best_svm, Xva_s, yva, len(classes))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_path = paths.MODELS_DIR / "cm_val_bow_hsv.csv"
    cm_df.to_csv(cm_path)

    metrics = {
        "val_accuracy": best_acc,
        "best_C": float(best_C),
        "best_gamma": float(best_gamma),
        "train_samples": float(len(train_df)),
        "val_samples": float(len(val_df)),
    }

    print(f"Model saved to {svm_path}")
    print(f"Label order saved to {paths.MODELS_DIR / 'labels.json'}")
    print(f"Confusion matrix saved to {cm_path}")
    proto_path = paths.MODELS_DIR / "color_prototypes.json"
    build_color_prototypes(train_df, proto_path)
    print(f"Color prototypes saved to {proto_path}")
    return metrics


def _confusion_matrix(model: cv2.ml_SVM, X: np.ndarray, y_true: np.ndarray, n_classes: int) -> np.ndarray:
    _, pred = model.predict(X)
    pred = pred.astype(np.int32).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1
    return cm


def build_color_prototypes(df: pd.DataFrame, out_path: Path) -> None:
    prototypes = {}
    for cls, sub in df.groupby("class"):
        hists = []
        for path in sub["patch_path"]:
            img = cv2.imread(path)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32)
            hist = hist.flatten()
            s = hist.sum()
            if s > 0:
                hist /= s
            hists.append(hist)
        if hists:
            mean_hist = np.mean(hists, axis=0)
            s = mean_hist.sum()
            if s > 0:
                mean_hist /= s
            prototypes[cls] = {"hist": mean_hist.tolist()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=2)
