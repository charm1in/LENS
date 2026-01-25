import cv2
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

from .utils import logger


class SaliencyMode(Enum):
    GRADIENT = "grad"
    LAPLACIAN = "lap"
    HYBRID = "gradlap"
    CANNY = "canny"


@dataclass
class Proposal:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale_idx: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h


class EvidenceMiner:
    def __init__(
            self,
            scales: List[float],
            top_k: int,
            mode: str = "gradlap",
            stride: int = 16,
            refine_steps: int = 0
    ):
        self.scales = scales
        self.top_k = top_k
        self.stride = stride
        self.mode = SaliencyMode(mode)
        self.refine_steps = refine_steps

        # Internal configuration
        self._min_box_size = 32
        self._nms_iou_thresh = 0.3

    def _compute_saliency(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale float32
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0

        if self.mode == SaliencyMode.CANNY:
            uint_img = (gray * 255).astype(np.uint8)
            edges = cv2.Canny(uint_img, 100, 200)
            return edges.astype(np.float32) / 255.0

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        if self.mode == SaliencyMode.GRADIENT:
            return grad_mag

        # Compute Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_mag = np.abs(lap)

        if self.mode == SaliencyMode.LAPLACIAN:
            return lap_mag

        if self.mode == SaliencyMode.HYBRID:
            return 0.5 * grad_mag + 0.5 * lap_mag

        return grad_mag  # Default fallback

    def _integral_image(self, saliency: np.ndarray) -> np.ndarray:
        return cv2.integral(saliency)

    def _query_integral(self, ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        # Standard integral image box query: A + D - B - C
        # ii shape is (H+1, W+1)
        y1, x1 = y, x
        y2, x2 = y + h, x + w

        # Boundary checks
        h_ii, w_ii = ii.shape
        x2 = min(x2, w_ii - 1)
        y2 = min(y2, h_ii - 1)

        total = ii[y2, x2] + ii[y1, x1] - ii[y1, x2] - ii[y2, x1]
        return float(total)

    def _compute_iou(self, a: Proposal, b: Proposal) -> float:
        xx1 = max(a.x, b.x)
        yy1 = max(a.y, b.y)
        xx2 = min(a.x + a.w, b.x + b.w)
        yy2 = min(a.y + a.h, b.y + b.h)

        inter_w = max(0, xx2 - xx1)
        inter_h = max(0, yy2 - yy1)
        intersection = inter_w * inter_h

        union = a.area + b.area - intersection
        return intersection / (union + 1e-6)

    def _apply_nms(self, proposals: List[Proposal]) -> List[Proposal]:
        if not proposals:
            return []

        # Sort by score descending
        proposals.sort(key=lambda p: p.score, reverse=True)

        keep = []
        while len(proposals) > 0:
            current = proposals.pop(0)
            keep.append(current)

            if len(keep) >= self.top_k:
                break

            # Filter remaining
            proposals = [
                p for p in proposals
                if self._compute_iou(current, p) < self._nms_iou_thresh
            ]

        return keep

    def mine_proposals(self, image: np.ndarray) -> List[Proposal]:
        h_img, w_img = image.shape[:2]
        saliency_map = self._compute_saliency(image)
        ii = self._integral_image(saliency_map)

        raw_candidates = []

        # Multi-scale sliding window
        for s_idx, scale in enumerate(self.scales):
            # Calculate box dimensions
            box_w = int(w_img * scale)
            box_h = int(h_img * scale)

            # Enforce minimum size constraint
            if box_w < self._min_box_size or box_h < self._min_box_size:
                continue

            # Sliding window loops
            # Using stride for efficiency
            for y in range(0, h_img - box_h + 1, self.stride):
                for x in range(0, w_img - box_w + 1, self.stride):
                    raw_score = self._query_integral(ii, x, y, box_w, box_h)
                    # Normalize score by area to ensure fairness across scales
                    density = raw_score / (box_w * box_h)

                    raw_candidates.append(Proposal(
                        x=x, y=y, w=box_w, h=box_h,
                        score=density, scale_idx=s_idx
                    ))

        # Apply Global NMS across all scales
        final_proposals = self._apply_nms(raw_candidates)

        # Optional Refinement (Local Optimization)
        if self.refine_steps > 0:
            refined = []
            for p in final_proposals:
                refined.append(self._refine_proposal(p, ii, h_img, w_img))
            return refined

        return final_proposals

    def _refine_proposal(self, p: Proposal, ii: np.ndarray, max_h: int, max_w: int) -> Proposal:
        # Local search around the candidate to maximize score
        best_p = p
        search_r = self.stride  # Search radius

        y_start = max(0, p.y - search_r)
        y_end = min(max_h - p.h, p.y + search_r)
        x_start = max(0, p.x - search_r)
        x_end = min(max_w - p.w, p.x + search_r)

        # Fine-grained grid search
        for y in range(y_start, y_end, 2):
            for x in range(x_start, x_end, 2):
                s = self._query_integral(ii, x, y, p.w, p.h)
                d = s / (p.w * p.h)
                if d > best_p.score:
                    best_p = Proposal(x, y, p.w, p.h, d, p.scale_idx)

        return best_p

    def get_random_proposals(self, image: np.ndarray, count: int, seed: int) -> List[Proposal]:
        rng = np.random.default_rng(seed)
        h, w = image.shape[:2]
        res = []

        for _ in range(count):
            # Pick a random scale from available configs
            s = rng.choice(self.scales)
            bw, bh = int(w * s), int(h * s)

            if bw >= w or bh >= h:
                continue

            rx = rng.integers(0, w - bw)
            ry = rng.integers(0, h - bh)

            # Score is 0.0 for random patches as we don't calculate it
            res.append(Proposal(rx, ry, bw, bh, 0.0, -1))

        return res