"""Feature extraction helpers (ORB + Bag of Visual Words + HSV histograms)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class FeatureParams:
    vocab_size: int = 400
    orb_features: int = 1000
    desc_limit: int = 120_000
    patch_size: Tuple[int, int] = (128, 128)
    hsv_bins: Tuple[int, int, int] = (8, 8, 8)
    nmax_keypoints: int | None = 800
    use_hog: bool = True

    @property
    def hsv_dim(self) -> int:
        return int(np.prod(self.hsv_bins))

    @property
    def dim(self) -> int:
        return self.vocab_size + self.hsv_dim


class FeatureExtractor:
    """Compute the BoW+HSV descriptor used throughout the project."""

    def __init__(self, vocab: np.ndarray, params: FeatureParams | None = None):
        self.params = params or FeatureParams()
        self.orb = cv2.ORB_create(nfeatures=self.params.orb_features)
        self.hog = None
        self.hog_dim = 0
        if self.params.use_hog:
            self._init_hog()
        self.set_vocabulary(vocab)

    def set_vocabulary(self, vocab: np.ndarray) -> None:
        if vocab is None or len(vocab) == 0:
            raise ValueError("Vocabulary cannot be empty.")
        self.vocab = vocab.astype(np.float32)
        self._vocab_T = self.vocab.T
        self._vocab_sq = (self.vocab ** 2).sum(axis=1)

    def describe(self, img_bgr: np.ndarray) -> np.ndarray:
        """Compute the concatenated descriptor and apply Hellinger sqrt."""
        if self.params.patch_size is not None:
            img_bgr = cv2.resize(img_bgr, self.params.patch_size, interpolation=cv2.INTER_AREA)
        bow = self._bow_hist(img_bgr)
        hsv = self._hsv_hist(img_bgr)
        parts = [self._hellinger(bow), self._hellinger(hsv)]
        hog = self._hog_features(img_bgr)
        if hog is not None:
            parts.append(hog)
        feat = np.concatenate(parts, axis=0).astype(np.float32)
        return feat

    def _init_hog(self) -> None:
        w, h = self.params.patch_size
        block = (32, 32)
        block_stride = (16, 16)
        cell = (16, 16)
        nbins = 9
        win_size = (int(w), int(h))
        self.hog = cv2.HOGDescriptor(win_size, block, block_stride, cell, nbins)
        self.hog_dim = int(self.hog.getDescriptorSize())

    def _bow_hist(self, img: np.ndarray) -> np.ndarray:
        kps, des = self.orb.detectAndCompute(img, None)
        if des is None or des.size == 0:
            return np.zeros((self.vocab.shape[0],), dtype=np.float32)
        D = des.astype(np.float32)
        if self.params.nmax_keypoints and D.shape[0] > self.params.nmax_keypoints:
            D = D[: self.params.nmax_keypoints]
        D_sq = (D ** 2).sum(axis=1, keepdims=True)
        DCt = D @ self._vocab_T
        dist2 = D_sq + self._vocab_sq.reshape(1, -1) - 2.0 * DCt
        idx = dist2.argmin(axis=1)
        hist = np.bincount(idx, minlength=self.vocab.shape[0]).astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _hsv_hist(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.params.hsv_bins),
            ranges=[0, 180, 0, 256, 0, 256],
        ).astype(np.float32)
        hist = hist.flatten()
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _hog_features(self, img: np.ndarray) -> np.ndarray | None:
        if self.hog is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_vec = self.hog.compute(gray)
        if hog_vec is None:
            return None
        return hog_vec.reshape(-1).astype(np.float32)

    @staticmethod
    def _hellinger(vec: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(vec, 0.0, None)).astype(np.float32)


def train_vocabulary(
    image_paths: Sequence[str],
    params: FeatureParams | None = None,
) -> np.ndarray:
    """Train a BoW vocabulary with ORB descriptors."""
    params = params or FeatureParams()
    orb = cv2.ORB_create(nfeatures=params.orb_features)
    trainer = cv2.BOWKMeansTrainer(clusterCount=params.vocab_size)

    added = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        _, des = orb.detectAndCompute(img, None)
        if des is None or des.size == 0:
            continue
        des = des.astype(np.float32)
        if added + len(des) > params.desc_limit:
            remaining = max(0, params.desc_limit - added)
            if remaining <= 0:
                break
            des = des[:remaining]
        trainer.add(des)
        added += len(des)
        if added >= params.desc_limit:
            break

    if added < params.vocab_size:
        raise RuntimeError(
            f"Insufficient descriptors: added={added} < vocab_size={params.vocab_size}. "
            "Increase desc_limit or reduce vocab_size."
        )

    vocab = trainer.cluster().astype(np.float32)
    print(f"Vocabulary trained with {added} descriptors → shape {vocab.shape}.")
    return vocab


def build_feature_matrix(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
    label2id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the X/y arrays given a manifest subset."""
    feats = []
    labels = []
    for row in df.to_dict("records"):
        img = cv2.imread(row["patch_path"])
        if img is None:
            continue
        feat = extractor.describe(img)
        feats.append(feat)
        labels.append(label2id[row["class"]])
    if not feats:
        raise RuntimeError("Could not compute any features for the provided subset.")
    X = np.vstack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def compute_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and std used for feature standardization."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def save_vocabulary(vocab: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    fs.write("vocabulary", vocab.astype(np.float32))
    fs.release()


def load_vocabulary(path: Path) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    vocab = fs.getNode("vocabulary").mat()
    fs.release()
    if vocab is None:
        raise FileNotFoundError(f"Could not read vocabulary from {path}")
    return vocab.astype(np.float32)
