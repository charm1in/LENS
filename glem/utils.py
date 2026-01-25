import os
import cv2
import torch
import random
import logging
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

# Configure structured logging
logger = logging.getLogger("GLEM")


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None):
    logger.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


@dataclass
class ProcessingStats:
    processed_count: int = 0
    failed_count: int = 0
    gate_early_exits: int = 0
    mining_triggers: int = 0
    total_inference_time: float = 0.0

    def update(self, gate_passed: bool, time_delta: float):
        self.processed_count += 1
        self.total_inference_time += time_delta
        if not gate_passed:
            self.gate_early_exits += 1
        else:
            self.mining_triggers += 1

    @property
    def avg_time(self) -> float:
        return self.total_inference_time / max(1, self.processed_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.processed_count,
            "errors": self.failed_count,
            "gate_exit_rate": self.gate_early_exits / max(1, self.processed_count),
            "avg_latency": self.avg_time
        }


def deterministic_setup(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed set to {seed}")


def read_image_safe(path: Union[str, Path], color_mode: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    path_str = str(path)
    if not os.path.exists(path_str):
        logger.error(f"Image not found: {path_str}")
        return None

    try:
        img = cv2.imread(path_str, color_mode)
        if img is None:
            raise ValueError("Decoded image is None")

        if color_mode == cv2.IMREAD_COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.warning(f"Failed to read {path_str}: {str(e)}")
        return None


def softmin(values: Union[List[float], np.ndarray], temperature: float = 0.05) -> float:
    if len(values) == 0:
        return 0.0

    v = np.array(values, dtype=np.float64)
    t = max(1e-6, temperature)

    # Stability shift
    v_shift = -v / t
    v_max = np.max(v_shift)

    exp_v = np.exp(v_shift - v_max)
    numerator = np.sum(exp_v * v)
    denominator = np.sum(exp_v)

    # Reconstruct softmin approximation via LogSumExp logic if needed,
    # but here we use standard weighted average interpretation
    # Actually, standard softmin is just: -T * log(mean(exp(-x/T)))
    log_sum_exp = v_max + np.log(np.mean(exp_v))
    return float(-t * log_sum_exp)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class MetricAggregator:
    def __init__(self, mode: str = "softmin", **kwargs):
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, values: List[float]) -> float:
        if not values:
            return float('nan')

        if self.mode == "softmin":
            return softmin(values, self.kwargs.get('temp', 0.05))
        elif self.mode == "min":
            return float(np.min(values))
        elif self.mode == "mean":
            return float(np.mean(values))
        elif self.mode == "percentile":
            q = self.kwargs.get('q', 50)
            return float(np.percentile(values, q))
        else:
            raise NotImplementedError(f"Aggregation {self.mode} unknown")